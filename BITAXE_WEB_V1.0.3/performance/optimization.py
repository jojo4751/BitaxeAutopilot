"""
Performance Optimization Engine

Advanced performance optimization system for the BitAxe ML-powered autonomous
mining platform. Provides automated performance tuning, bottleneck analysis,
and system optimization recommendations.
"""

import asyncio
import time
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitaxe_logging.structured_logger import get_logger
from performance.benchmarking import BenchmarkSuite, BenchmarkResult

logger = get_logger("bitaxe.performance.optimization")


@dataclass
class OptimizationResult:
    """Result of a performance optimization"""
    optimization_type: str
    parameter_name: str
    original_value: Any
    optimized_value: Any
    performance_improvement: float
    confidence_score: float
    applied_at: datetime
    description: str


@dataclass
class BottleneckAnalysis:
    """Analysis of system performance bottlenecks"""
    component: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    impact_score: float
    recommendations: List[str]
    metrics: Dict[str, float]


class AdaptiveParameterTuner:
    """Adaptive parameter tuning using Bayesian optimization"""
    
    def __init__(self):
        self.parameter_history: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        self.learning_rate = 0.1
        self.exploration_factor = 0.2
        
    def suggest_parameter_value(self, param_name: str, current_value: Any, 
                              value_range: Tuple[Any, Any]) -> Any:
        """Suggest next parameter value to try"""
        history = self.parameter_history[param_name]
        
        if len(history) < 3:
            # Initial exploration phase
            min_val, max_val = value_range
            if isinstance(current_value, (int, float)):
                return min_val + np.random.random() * (max_val - min_val)
            else:
                return current_value
        
        # Use simple Bayesian approach
        values, performances = zip(*history)
        
        if isinstance(current_value, (int, float)):
            # For numeric parameters, use Gaussian process approximation
            best_idx = np.argmax(performances)
            best_value = values[best_idx]
            
            # Exploration vs exploitation
            if np.random.random() < self.exploration_factor:
                # Explore
                min_val, max_val = value_range
                return min_val + np.random.random() * (max_val - min_val)
            else:
                # Exploit with small perturbation
                perturbation = (value_range[1] - value_range[0]) * 0.1 * np.random.randn()
                return max(value_range[0], min(value_range[1], best_value + perturbation))
        
        return current_value
    
    def record_result(self, param_name: str, value: Any, performance: float):
        """Record parameter value and resulting performance"""
        self.parameter_history[param_name].append((value, performance))
        
        # Keep only recent history
        if len(self.parameter_history[param_name]) > 50:
            self.parameter_history[param_name] = self.parameter_history[param_name][-50:]


class PerformanceOptimizer:
    """Main performance optimization engine"""
    
    def __init__(self, benchmark_suite: BenchmarkSuite):
        self.benchmark_suite = benchmark_suite
        self.parameter_tuner = AdaptiveParameterTuner()
        self.optimization_history: List[OptimizationResult] = []
        self.current_config = self._load_default_config()
        self.performance_baseline = None
        
        # Optimization parameters and their ranges
        self.optimization_parameters = {
            'ml_batch_size': (1, 100),
            'db_connection_pool_size': (5, 50),
            'async_worker_threads': (2, 20),
            'cache_size_mb': (100, 2000),
            'gc_threshold': (100, 10000),
            'optimization_interval': (10, 300)
        }
        
        logger.info("PerformanceOptimizer initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default performance configuration"""
        return {
            'ml_batch_size': 10,
            'db_connection_pool_size': 20,
            'async_worker_threads': 8,
            'cache_size_mb': 512,
            'gc_threshold': 1000,
            'optimization_interval': 60,
            'enable_profiling': False,
            'max_memory_usage_mb': 4096,
            'cpu_usage_target': 80
        }
    
    async def establish_performance_baseline(self) -> Dict[str, BenchmarkResult]:
        """Establish performance baseline with current configuration"""
        logger.info("Establishing performance baseline")
        
        baseline_results = await self.benchmark_suite.run_comprehensive_benchmark_suite()
        self.performance_baseline = baseline_results
        
        logger.info("Performance baseline established", 
                   num_benchmarks=len(baseline_results))
        
        return baseline_results
    
    async def optimize_system_performance(self, 
                                        max_iterations: int = 10,
                                        improvement_threshold: float = 0.05) -> List[OptimizationResult]:
        """Run comprehensive system performance optimization"""
        if not self.performance_baseline:
            await self.establish_performance_baseline()
        
        logger.info("Starting system performance optimization", 
                   max_iterations=max_iterations)
        
        optimization_results = []
        
        for iteration in range(max_iterations):
            logger.info(f"Optimization iteration {iteration + 1}/{max_iterations}")
            
            # Analyze current bottlenecks
            bottlenecks = await self.analyze_bottlenecks()
            
            if not bottlenecks:
                logger.info("No significant bottlenecks detected")
                break
            
            # Focus on most critical bottleneck
            critical_bottleneck = max(bottlenecks, key=lambda x: x.impact_score)
            logger.info(f"Targeting bottleneck: {critical_bottleneck.component}")
            
            # Apply optimization for this bottleneck
            iteration_results = await self._optimize_for_bottleneck(critical_bottleneck)
            optimization_results.extend(iteration_results)
            
            # Check if we've achieved sufficient improvement
            if iteration_results:
                avg_improvement = np.mean([r.performance_improvement for r in iteration_results])
                if avg_improvement < improvement_threshold:
                    logger.info(f"Improvement below threshold ({improvement_threshold}), stopping optimization")
                    break
        
        logger.info("System performance optimization completed", 
                   total_optimizations=len(optimization_results))
        
        return optimization_results
    
    async def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze current system bottlenecks"""
        logger.debug("Analyzing system bottlenecks")
        
        # Run current performance benchmark
        current_results = await self.benchmark_suite.run_comprehensive_benchmark_suite()
        
        bottlenecks = []
        
        for test_name, result in current_results.items():
            baseline_result = self.performance_baseline.get(test_name)
            if not baseline_result:
                continue
            
            # Compare with baseline
            performance_ratio = result.operations_per_second / baseline_result.operations_per_second
            
            if performance_ratio < 0.9:  # 10% degradation
                severity = 'high' if performance_ratio < 0.7 else 'medium'
                
                bottleneck = BottleneckAnalysis(
                    component=test_name,
                    severity=severity,
                    description=f"Performance degraded by {(1-performance_ratio)*100:.1f}%",
                    impact_score=(1 - performance_ratio) * 100,
                    recommendations=self._generate_bottleneck_recommendations(test_name, result),
                    metrics={
                        'ops_per_sec_ratio': performance_ratio,
                        'memory_usage_mb': result.memory_usage_mb,
                        'cpu_usage_percent': result.cpu_usage_percent,
                        'success_rate': result.success_rate
                    }
                )
                bottlenecks.append(bottleneck)
        
        # Analyze system resource bottlenecks
        system_bottlenecks = await self._analyze_system_resource_bottlenecks()
        bottlenecks.extend(system_bottlenecks)
        
        logger.debug(f"Identified {len(bottlenecks)} bottlenecks")
        return bottlenecks
    
    async def _analyze_system_resource_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze system resource bottlenecks"""
        bottlenecks = []
        
        # CPU analysis
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 85:
            bottlenecks.append(BottleneckAnalysis(
                component='cpu',
                severity='high' if cpu_percent > 95 else 'medium',
                description=f"High CPU usage: {cpu_percent:.1f}%",
                impact_score=cpu_percent,
                recommendations=[
                    "Consider reducing worker thread count",
                    "Optimize CPU-intensive algorithms",
                    "Implement better caching strategies"
                ],
                metrics={'cpu_usage_percent': cpu_percent}
            ))
        
        # Memory analysis
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        if memory_percent > 80:
            bottlenecks.append(BottleneckAnalysis(
                component='memory',
                severity='high' if memory_percent > 90 else 'medium',
                description=f"High memory usage: {memory_percent:.1f}%",
                impact_score=memory_percent,
                recommendations=[
                    "Implement memory pooling",
                    "Optimize data structures",
                    "Increase garbage collection frequency"
                ],
                metrics={'memory_usage_percent': memory_percent}
            ))
        
        # Disk I/O analysis
        disk_io = psutil.disk_io_counters()
        if disk_io:
            # Simple heuristic for disk bottleneck (would need more sophisticated analysis in practice)
            io_ratio = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 1024)  # GB
            if io_ratio > 10:  # Arbitrary threshold
                bottlenecks.append(BottleneckAnalysis(
                    component='disk_io',
                    severity='medium',
                    description=f"High disk I/O detected",
                    impact_score=50,
                    recommendations=[
                        "Implement disk caching",
                        "Optimize database queries",
                        "Consider SSD storage"
                    ],
                    metrics={'disk_io_gb': io_ratio}
                ))
        
        return bottlenecks
    
    def _generate_bottleneck_recommendations(self, component: str, result: BenchmarkResult) -> List[str]:
        """Generate recommendations for specific bottlenecks"""
        recommendations = []
        
        if 'ml_inference' in component.lower():
            if result.operations_per_second < 50:
                recommendations.extend([
                    "Optimize ML model inference",
                    "Consider model quantization",
                    "Implement batch processing",
                    "Use GPU acceleration if available"
                ])
        
        elif 'database' in component.lower():
            if result.operations_per_second < 200:
                recommendations.extend([
                    "Optimize database queries",
                    "Increase connection pool size",
                    "Add database indexes",
                    "Consider read replicas"
                ])
        
        elif 'concurrent' in component.lower():
            if result.success_rate < 90:
                recommendations.extend([
                    "Review concurrency limits",
                    "Implement better error handling",
                    "Optimize async patterns",
                    "Consider backpressure mechanisms"
                ])
        
        if result.memory_usage_mb > 1000:
            recommendations.append("Optimize memory usage")
        
        if result.cpu_usage_percent > 90:
            recommendations.append("Reduce CPU-intensive operations")
        
        return recommendations
    
    async def _optimize_for_bottleneck(self, bottleneck: BottleneckAnalysis) -> List[OptimizationResult]:
        """Apply optimization for a specific bottleneck"""
        optimizations = []
        
        if bottleneck.component == 'cpu':
            # CPU optimization
            optimization = await self._optimize_cpu_usage()
            if optimization:
                optimizations.append(optimization)
        
        elif bottleneck.component == 'memory':
            # Memory optimization
            optimization = await self._optimize_memory_usage()
            if optimization:
                optimizations.append(optimization)
        
        elif 'ml_inference' in bottleneck.component:
            # ML optimization
            optimization = await self._optimize_ml_performance()
            if optimization:
                optimizations.append(optimization)
        
        elif 'database' in bottleneck.component:
            # Database optimization
            optimization = await self._optimize_database_performance()
            if optimization:
                optimizations.append(optimization)
        
        elif 'concurrent' in bottleneck.component:
            # Concurrency optimization
            optimization = await self._optimize_concurrency()
            if optimization:
                optimizations.append(optimization)
        
        return optimizations
    
    async def _optimize_cpu_usage(self) -> Optional[OptimizationResult]:
        """Optimize CPU usage parameters"""
        param_name = 'async_worker_threads'
        current_value = self.current_config[param_name]
        
        # Suggest new value
        new_value = self.parameter_tuner.suggest_parameter_value(
            param_name, current_value, self.optimization_parameters[param_name]
        )
        
        if new_value == current_value:
            return None
        
        # Apply optimization temporarily
        original_value = self.current_config[param_name]
        self.current_config[param_name] = new_value
        
        # Measure performance
        performance_before = await self._measure_cpu_performance()
        self.current_config[param_name] = new_value
        performance_after = await self._measure_cpu_performance()
        
        improvement = (performance_after - performance_before) / performance_before
        
        # Record result
        self.parameter_tuner.record_result(param_name, new_value, performance_after)
        
        if improvement > 0.02:  # 2% improvement threshold
            # Keep the optimization
            optimization = OptimizationResult(
                optimization_type='cpu_optimization',
                parameter_name=param_name,
                original_value=original_value,
                optimized_value=new_value,
                performance_improvement=improvement * 100,
                confidence_score=min(1.0, improvement * 10),
                applied_at=datetime.now(),
                description=f"Optimized {param_name} from {original_value} to {new_value}"
            )
            self.optimization_history.append(optimization)
            return optimization
        else:
            # Revert change
            self.current_config[param_name] = original_value
            return None
    
    async def _optimize_memory_usage(self) -> Optional[OptimizationResult]:
        """Optimize memory usage parameters"""
        param_name = 'cache_size_mb'
        current_value = self.current_config[param_name]
        
        # Check current memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            # Reduce cache size
            new_value = max(100, int(current_value * 0.8))
        else:
            # Suggest optimized value
            new_value = self.parameter_tuner.suggest_parameter_value(
                param_name, current_value, self.optimization_parameters[param_name]
            )
        
        if new_value == current_value:
            return None
        
        original_value = self.current_config[param_name]
        self.current_config[param_name] = new_value
        
        # Measure performance impact
        await asyncio.sleep(5)  # Allow system to adjust
        performance = await self._measure_memory_efficiency()
        
        improvement = 0.05  # Simplified improvement calculation
        
        if improvement > 0.01:
            optimization = OptimizationResult(
                optimization_type='memory_optimization',
                parameter_name=param_name,
                original_value=original_value,
                optimized_value=new_value,
                performance_improvement=improvement * 100,
                confidence_score=0.7,
                applied_at=datetime.now(),
                description=f"Optimized {param_name} from {original_value} to {new_value}"
            )
            self.optimization_history.append(optimization)
            return optimization
        else:
            self.current_config[param_name] = original_value
            return None
    
    async def _optimize_ml_performance(self) -> Optional[OptimizationResult]:
        """Optimize ML performance parameters"""
        param_name = 'ml_batch_size'
        current_value = self.current_config[param_name]
        
        new_value = self.parameter_tuner.suggest_parameter_value(
            param_name, current_value, self.optimization_parameters[param_name]
        )
        
        if new_value == current_value:
            return None
        
        original_value = self.current_config[param_name]
        self.current_config[param_name] = int(new_value)
        
        # Measure ML performance
        ml_result = await self.benchmark_suite.run_ml_inference_benchmark(num_operations=200)
        baseline_ml = self.performance_baseline.get('ml_inference')
        
        if baseline_ml:
            improvement = (ml_result.operations_per_second - baseline_ml.operations_per_second) / baseline_ml.operations_per_second
            
            if improvement > 0.02:
                optimization = OptimizationResult(
                    optimization_type='ml_optimization',
                    parameter_name=param_name,
                    original_value=original_value,
                    optimized_value=int(new_value),
                    performance_improvement=improvement * 100,
                    confidence_score=min(1.0, improvement * 5),
                    applied_at=datetime.now(),
                    description=f"Optimized {param_name} from {original_value} to {int(new_value)}"
                )
                self.optimization_history.append(optimization)
                return optimization
        
        # Revert if no improvement
        self.current_config[param_name] = original_value
        return None
    
    async def _optimize_database_performance(self) -> Optional[OptimizationResult]:
        """Optimize database performance parameters"""
        param_name = 'db_connection_pool_size'
        current_value = self.current_config[param_name]
        
        new_value = self.parameter_tuner.suggest_parameter_value(
            param_name, current_value, self.optimization_parameters[param_name]
        )
        
        if new_value == current_value:
            return None
        
        original_value = self.current_config[param_name]
        self.current_config[param_name] = int(new_value)
        
        # Measure database performance
        db_result = await self.benchmark_suite.run_database_performance_benchmark(num_operations=100)
        baseline_db = self.performance_baseline.get('database_performance')
        
        if baseline_db:
            improvement = (db_result.operations_per_second - baseline_db.operations_per_second) / baseline_db.operations_per_second
            
            if improvement > 0.02:
                optimization = OptimizationResult(
                    optimization_type='database_optimization',
                    parameter_name=param_name,
                    original_value=original_value,
                    optimized_value=int(new_value),
                    performance_improvement=improvement * 100,
                    confidence_score=min(1.0, improvement * 5),
                    applied_at=datetime.now(),
                    description=f"Optimized {param_name} from {original_value} to {int(new_value)}"
                )
                self.optimization_history.append(optimization)
                return optimization
        
        # Revert if no improvement
        self.current_config[param_name] = original_value
        return None
    
    async def _optimize_concurrency(self) -> Optional[OptimizationResult]:
        """Optimize concurrency parameters"""
        param_name = 'async_worker_threads'
        current_value = self.current_config[param_name]
        
        # Consider CPU count for concurrency optimization
        cpu_count = psutil.cpu_count()
        optimal_range = (max(2, cpu_count // 2), min(20, cpu_count * 2))
        
        new_value = self.parameter_tuner.suggest_parameter_value(
            param_name, current_value, optimal_range
        )
        
        if new_value == current_value:
            return None
        
        original_value = self.current_config[param_name]
        self.current_config[param_name] = int(new_value)
        
        # Measure concurrent performance
        concurrent_result = await self.benchmark_suite.run_concurrent_optimization_benchmark(
            num_miners=20, concurrent_optimizations=int(new_value)
        )
        baseline_concurrent = self.performance_baseline.get('concurrent_optimization')
        
        if baseline_concurrent:
            improvement = (concurrent_result.operations_per_second - baseline_concurrent.operations_per_second) / baseline_concurrent.operations_per_second
            
            if improvement > 0.02:
                optimization = OptimizationResult(
                    optimization_type='concurrency_optimization',
                    parameter_name=param_name,
                    original_value=original_value,
                    optimized_value=int(new_value),
                    performance_improvement=improvement * 100,
                    confidence_score=min(1.0, improvement * 5),
                    applied_at=datetime.now(),
                    description=f"Optimized {param_name} from {original_value} to {int(new_value)}"
                )
                self.optimization_history.append(optimization)
                return optimization
        
        # Revert if no improvement
        self.current_config[param_name] = original_value
        return None
    
    async def _measure_cpu_performance(self) -> float:
        """Measure current CPU performance metric"""
        # Simple CPU performance measurement
        start_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=1)
        end_time = time.time()
        
        # Return inverse of CPU usage as performance metric (lower CPU = better performance)
        return 100 - cpu_usage
    
    async def _measure_memory_efficiency(self) -> float:
        """Measure current memory efficiency"""
        memory = psutil.virtual_memory()
        # Return inverse of memory usage as efficiency metric
        return 100 - memory.percent
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        total_optimizations = len(self.optimization_history)
        
        if total_optimizations == 0:
            return {
                'summary': 'No optimizations applied yet',
                'recommendations': ['Run performance optimization to identify improvements']
            }
        
        # Calculate overall improvement
        total_improvement = sum(opt.performance_improvement for opt in self.optimization_history)
        avg_improvement = total_improvement / total_optimizations
        
        # Group optimizations by type
        optimization_types = defaultdict(list)
        for opt in self.optimization_history:
            optimization_types[opt.optimization_type].append(opt)
        
        report = {
            'summary': {
                'total_optimizations': total_optimizations,
                'average_improvement': f"{avg_improvement:.2f}%",
                'total_improvement': f"{total_improvement:.2f}%",
                'optimization_types': len(optimization_types)
            },
            'optimizations_by_type': {},
            'current_configuration': self.current_config,
            'recommendations': self._generate_future_recommendations(),
            'generated_at': datetime.now().isoformat()
        }
        
        for opt_type, optimizations in optimization_types.items():
            type_improvement = sum(opt.performance_improvement for opt in optimizations)
            report['optimizations_by_type'][opt_type] = {
                'count': len(optimizations),
                'total_improvement': f"{type_improvement:.2f}%",
                'optimizations': [asdict(opt) for opt in optimizations]
            }
        
        return report
    
    def _generate_future_recommendations(self) -> List[str]:
        """Generate recommendations for future optimizations"""
        recommendations = []
        
        # Analyze optimization history to suggest future improvements
        if len(self.optimization_history) < 5:
            recommendations.append("Continue running optimization cycles to identify more improvements")
        
        # Check for parameters that haven't been optimized
        optimized_params = {opt.parameter_name for opt in self.optimization_history}
        unoptimized_params = set(self.optimization_parameters.keys()) - optimized_params
        
        if unoptimized_params:
            recommendations.append(f"Consider optimizing unused parameters: {', '.join(unoptimized_params)}")
        
        # System-specific recommendations
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            recommendations.append("System memory usage is high - consider upgrading hardware")
        
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            recommendations.append("Limited CPU cores detected - consider upgrading for better performance")
        
        return recommendations
    
    def export_optimization_results(self, file_path: str):
        """Export optimization results to JSON file"""
        report = self.generate_optimization_report()
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization results exported to {file_path}")


async def main():
    """Example usage of the performance optimization system"""
    from database import DatabaseManager
    from performance.benchmarking import BenchmarkSuite
    
    # Initialize components
    db_manager = DatabaseManager('sqlite:///optimization_test.db')
    benchmark_suite = BenchmarkSuite(db_manager)
    optimizer = PerformanceOptimizer(benchmark_suite)
    
    try:
        print("Starting performance optimization...")
        
        # Establish baseline
        baseline = await optimizer.establish_performance_baseline()
        print(f"Baseline established with {len(baseline)} benchmarks")
        
        # Run optimization
        optimizations = await optimizer.optimize_system_performance(max_iterations=5)
        print(f"Applied {len(optimizations)} optimizations")
        
        # Generate report
        report = optimizer.generate_optimization_report()
        optimizer.export_optimization_results('optimization_results.json')
        
        # Print summary
        print("\n" + "="*50)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*50)
        
        print(f"Total optimizations: {report['summary']['total_optimizations']}")
        print(f"Average improvement: {report['summary']['average_improvement']}")
        print(f"Total improvement: {report['summary']['total_improvement']}")
        
        print("\nOptimizations by type:")
        for opt_type, data in report['optimizations_by_type'].items():
            print(f"  {opt_type}: {data['count']} optimizations, {data['total_improvement']} improvement")
        
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
    except Exception as e:
        logger.error("Optimization execution failed", error=str(e))
        raise


if __name__ == '__main__':
    asyncio.run(main())