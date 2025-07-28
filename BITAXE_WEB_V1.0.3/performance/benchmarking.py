"""
Performance Benchmarking Framework

Comprehensive performance testing and benchmarking system for the BitAxe ML-powered
autonomous mining optimization platform. Provides detailed performance analysis,
bottleneck identification, and optimization recommendations.
"""

import asyncio
import time
import statistics
import json
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import gc
import resource
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitaxe_logging.structured_logger import get_logger
from ml_engine.optimization_engine import OptimizationEngine
from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.reinforcement_learning import RLAgent
from database import DatabaseManager
from models.miner_models import Miner, MiningStats

logger = get_logger("bitaxe.performance.benchmarking")


@dataclass
class BenchmarkResult:
    """Container for benchmark test results"""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    operations_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_count: int
    error_count: int
    metrics: Dict[str, Any]
    percentiles: Dict[str, float]
    
    @property
    def total_operations(self) -> int:
        return self.success_count + self.error_count
    
    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return (self.success_count / self.total_operations) * 100


@dataclass
class PerformanceProfile:
    """System performance profile during benchmark"""
    cpu_usage: List[float]
    memory_usage: List[float]
    disk_io: Dict[str, List[float]]
    network_io: Dict[str, List[float]]
    gc_collections: List[int]
    thread_count: List[int]
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for all metrics"""
        return {
            'cpu_usage': self._calculate_stats(self.cpu_usage),
            'memory_usage': self._calculate_stats(self.memory_usage),
            'disk_read': self._calculate_stats(self.disk_io.get('read_bytes', [])),
            'disk_write': self._calculate_stats(self.disk_io.get('write_bytes', [])),
            'network_sent': self._calculate_stats(self.network_io.get('bytes_sent', [])),
            'network_recv': self._calculate_stats(self.network_io.get('bytes_recv', []))
        }
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a list of values"""
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }


class PerformanceMonitor:
    """Real-time performance monitoring during benchmarks"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.profile = PerformanceProfile(
            cpu_usage=[],
            memory_usage=[],
            disk_io={'read_bytes': [], 'write_bytes': []},
            network_io={'bytes_sent': [], 'bytes_recv': []},
            gc_collections=[],
            thread_count=[]
        )
        self._monitor_task = None
        
        # Baseline system metrics
        self._baseline_disk_io = psutil.disk_io_counters()
        self._baseline_network_io = psutil.net_io_counters()
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.debug("Performance monitoring started")
    
    async def stop_monitoring(self) -> PerformanceProfile:
        """Stop monitoring and return performance profile"""
        if not self.monitoring:
            return self.profile
        
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.debug("Performance monitoring stopped")
        return self.profile
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.profile.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.profile.memory_usage.append(memory_mb)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io and self._baseline_disk_io:
                    self.profile.disk_io['read_bytes'].append(
                        disk_io.read_bytes - self._baseline_disk_io.read_bytes
                    )
                    self.profile.disk_io['write_bytes'].append(
                        disk_io.write_bytes - self._baseline_disk_io.write_bytes
                    )
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io and self._baseline_network_io:
                    self.profile.network_io['bytes_sent'].append(
                        network_io.bytes_sent - self._baseline_network_io.bytes_sent
                    )
                    self.profile.network_io['bytes_recv'].append(
                        network_io.bytes_recv - self._baseline_network_io.bytes_recv
                    )
                
                # Garbage collection
                gc_stats = gc.get_stats()
                total_collections = sum(stat['collections'] for stat in gc_stats)
                self.profile.gc_collections.append(total_collections)
                
                # Thread count
                self.profile.thread_count.append(threading.active_count())
                
                await asyncio.sleep(self.sampling_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Performance monitoring error", error=str(e))
                await asyncio.sleep(self.sampling_interval)


class BenchmarkSuite:
    """Comprehensive benchmarking suite for BitAxe system"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.results: List[BenchmarkResult] = []
        self.monitor = PerformanceMonitor()
        
        # Initialize components for testing
        self.optimization_engine = OptimizationEngine(db_manager)
        self.feature_engineer = FeatureEngineer()
        self.rl_agent = RLAgent()
        
        logger.info("BenchmarkSuite initialized")
    
    @asynccontextmanager
    async def benchmark_context(self, test_name: str):
        """Context manager for benchmark execution"""
        start_time = datetime.now()
        
        # Start performance monitoring
        await self.monitor.start_monitoring()
        
        # Force garbage collection before test
        gc.collect()
        
        # Capture initial system state
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        initial_cpu = psutil.cpu_percent(interval=1)
        
        success_count = 0
        error_count = 0
        operation_times = []
        
        class BenchmarkTracker:
            def record_success(self, duration: float):
                nonlocal success_count
                success_count += 1
                operation_times.append(duration)
            
            def record_error(self):
                nonlocal error_count
                error_count += 1
        
        try:
            yield BenchmarkTracker()
        finally:
            # Stop monitoring
            performance_profile = await self.monitor.stop_monitoring()
            
            # Calculate final metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_memory = psutil.virtual_memory().used / (1024 * 1024)
            final_cpu = psutil.cpu_percent(interval=1)
            
            # Calculate percentiles for operation times
            percentiles = {}
            if operation_times:
                percentiles = {
                    'p50': np.percentile(operation_times, 50),
                    'p75': np.percentile(operation_times, 75),
                    'p90': np.percentile(operation_times, 90),
                    'p95': np.percentile(operation_times, 95),
                    'p99': np.percentile(operation_times, 99)
                }
            
            # Create benchmark result
            result = BenchmarkResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                operations_per_second=(success_count + error_count) / duration if duration > 0 else 0,
                memory_usage_mb=final_memory - initial_memory,
                cpu_usage_percent=(final_cpu + initial_cpu) / 2,
                success_count=success_count,
                error_count=error_count,
                metrics={
                    'operation_times': operation_times,
                    'performance_profile': performance_profile.get_statistics()
                },
                percentiles=percentiles
            )
            
            self.results.append(result)
            logger.info(f"Benchmark '{test_name}' completed", 
                       duration=duration, ops_per_sec=result.operations_per_second)
    
    async def run_ml_inference_benchmark(self, num_operations: int = 1000) -> BenchmarkResult:
        """Benchmark ML model inference performance"""
        async with self.benchmark_context("ML Inference Performance") as tracker:
            
            # Generate test data
            test_features = []
            for i in range(num_operations):
                features = {
                    'hashrate': 500 + (i % 100),
                    'temperature': 65 + (i % 15),
                    'power': 3000 + (i % 500),
                    'ambient_temp': 25 + (i % 10),
                    'humidity': 50 + (i % 30)
                }
                test_features.append(features)
            
            # Run inference benchmark
            for i, features in enumerate(test_features):
                start_time = time.time()
                
                try:
                    # Simulate ML inference
                    engineered_features = self.feature_engineer.engineer_features(features)
                    
                    # RL agent decision
                    action = self.rl_agent.get_action(engineered_features)
                    
                    # Record success
                    duration = time.time() - start_time
                    tracker.record_success(duration)
                    
                except Exception as e:
                    logger.error(f"ML inference error in operation {i}", error=str(e))
                    tracker.record_error()
                
                # Brief pause to prevent overwhelming the system
                if i % 100 == 0:
                    await asyncio.sleep(0.01)
        
        return self.results[-1]
    
    async def run_database_performance_benchmark(self, num_operations: int = 500) -> BenchmarkResult:
        """Benchmark database operations performance"""
        async with self.benchmark_context("Database Performance") as tracker:
            
            # Test data
            test_miners = []
            for i in range(min(num_operations, 100)):  # Limit miners created
                miner = Miner(
                    ip_address=f"192.168.1.{100 + i}",
                    name=f"benchmark_miner_{i}",
                    model="BitAxe Ultra",
                    pool_url="stratum+tcp://test.pool.com:4334",
                    pool_username=f"test_user_{i}",
                    pool_password="x"
                )
                test_miners.append(miner)
            
            # Database operations benchmark
            for i in range(num_operations):
                start_time = time.time()
                
                try:
                    # Simulate various database operations
                    operation_type = i % 4
                    
                    if operation_type == 0:  # Create
                        miner = test_miners[i % len(test_miners)]
                        session = self.db_manager.get_session()
                        try:
                            session.add(miner)
                            session.commit()
                        finally:
                            session.close()
                    
                    elif operation_type == 1:  # Read
                        session = self.db_manager.get_session()
                        try:
                            miners = session.query(Miner).limit(10).all()
                        finally:
                            session.close()
                    
                    elif operation_type == 2:  # Update
                        session = self.db_manager.get_session()
                        try:
                            miner = session.query(Miner).first()
                            if miner:
                                miner.name = f"updated_miner_{i}"
                                session.commit()
                        finally:
                            session.close()
                    
                    else:  # Complex query
                        session = self.db_manager.get_session()
                        try:
                            # Simulate complex analytics query
                            result = session.query(Miner).filter(
                                Miner.name.like('%benchmark%')
                            ).limit(50).all()
                        finally:
                            session.close()
                    
                    duration = time.time() - start_time
                    tracker.record_success(duration)
                    
                except Exception as e:
                    logger.error(f"Database operation error in operation {i}", error=str(e))
                    tracker.record_error()
                
                # Brief pause
                if i % 50 == 0:
                    await asyncio.sleep(0.01)
        
        return self.results[-1]
    
    async def run_concurrent_optimization_benchmark(self, num_miners: int = 50, 
                                                  concurrent_optimizations: int = 10) -> BenchmarkResult:
        """Benchmark concurrent optimization performance"""
        async with self.benchmark_context("Concurrent Optimization Performance") as tracker:
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(concurrent_optimizations)
            
            async def optimize_miner(miner_id: int):
                async with semaphore:
                    start_time = time.time()
                    
                    try:
                        # Simulate optimization process
                        miner_data = {
                            'id': miner_id,
                            'ip': f"192.168.1.{100 + miner_id}",
                            'current_hashrate': 500 + (miner_id % 100),
                            'current_temp': 65 + (miner_id % 15),
                            'current_power': 3000 + (miner_id % 500)
                        }
                        
                        # Feature engineering
                        features = self.feature_engineer.engineer_features(miner_data)
                        
                        # RL decision
                        action = self.rl_agent.get_action(features)
                        
                        # Simulate optimization application
                        await asyncio.sleep(0.1)  # Simulate network delay
                        
                        duration = time.time() - start_time
                        tracker.record_success(duration)
                        
                    except Exception as e:
                        logger.error(f"Concurrent optimization error for miner {miner_id}", error=str(e))
                        tracker.record_error()
            
            # Run concurrent optimizations
            tasks = [optimize_miner(i) for i in range(num_miners)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.results[-1]
    
    async def run_memory_stress_test(self, duration_seconds: int = 60) -> BenchmarkResult:
        """Benchmark memory usage under stress"""
        async with self.benchmark_context("Memory Stress Test") as tracker:
            
            end_time = time.time() + duration_seconds
            data_structures = []
            
            operation_count = 0
            while time.time() < end_time:
                start_time = time.time()
                
                try:
                    # Simulate memory-intensive operations
                    operation_type = operation_count % 3
                    
                    if operation_type == 0:
                        # Create large data structure
                        large_data = {
                            f'key_{i}': np.random.rand(1000) for i in range(100)
                        }
                        data_structures.append(large_data)
                    
                    elif operation_type == 1:
                        # Process existing data
                        if data_structures:
                            data = data_structures[operation_count % len(data_structures)]
                            processed = {k: np.sum(v) for k, v in data.items()}
                    
                    else:
                        # Cleanup some data
                        if data_structures and len(data_structures) > 20:
                            data_structures.pop(0)
                            gc.collect()
                    
                    duration = time.time() - start_time
                    tracker.record_success(duration)
                    operation_count += 1
                    
                except Exception as e:
                    logger.error(f"Memory stress test error in operation {operation_count}", error=str(e))
                    tracker.record_error()
                
                # Brief pause
                await asyncio.sleep(0.01)
        
        return self.results[-1]
    
    async def run_comprehensive_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        benchmarks = {
            'ml_inference': self.run_ml_inference_benchmark,
            'database_performance': self.run_database_performance_benchmark,
            'concurrent_optimization': self.run_concurrent_optimization_benchmark,
            'memory_stress': self.run_memory_stress_test
        }
        
        suite_results = {}
        
        for name, benchmark_func in benchmarks.items():
            logger.info(f"Running benchmark: {name}")
            try:
                result = await benchmark_func()
                suite_results[name] = result
                logger.info(f"Benchmark '{name}' completed successfully")
            except Exception as e:
                logger.error(f"Benchmark '{name}' failed", error=str(e))
        
        logger.info("Comprehensive benchmark suite completed", 
                   total_benchmarks=len(suite_results))
        
        return suite_results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            return {'error': 'No benchmark results available'}
        
        report = {
            'summary': {
                'total_benchmarks': len(self.results),
                'total_operations': sum(r.total_operations for r in self.results),
                'overall_success_rate': sum(r.success_count for r in self.results) / 
                                      sum(r.total_operations for r in self.results) * 100
            },
            'benchmarks': {},
            'recommendations': self._generate_recommendations(),
            'system_info': self._get_system_info(),
            'generated_at': datetime.now().isoformat()
        }
        
        for result in self.results:
            report['benchmarks'][result.test_name] = {
                'duration_seconds': result.duration_seconds,
                'operations_per_second': result.operations_per_second,
                'success_rate': result.success_rate,
                'memory_usage_mb': result.memory_usage_mb,
                'cpu_usage_percent': result.cpu_usage_percent,
                'percentiles': result.percentiles,
                'performance_profile': result.metrics.get('performance_profile', {})
            }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Analyze results and generate recommendations
        avg_ops_per_sec = statistics.mean(r.operations_per_second for r in self.results)
        avg_memory_usage = statistics.mean(r.memory_usage_mb for r in self.results)
        avg_success_rate = statistics.mean(r.success_rate for r in self.results)
        
        if avg_ops_per_sec < 100:
            recommendations.append("Consider optimizing critical path operations to improve throughput")
        
        if avg_memory_usage > 500:
            recommendations.append("High memory usage detected - consider implementing memory pooling or caching strategies")
        
        if avg_success_rate < 95:
            recommendations.append("Success rate below 95% - investigate error handling and retry mechanisms")
        
        # Check for specific performance issues
        for result in self.results:
            if 'ML Inference' in result.test_name and result.operations_per_second < 50:
                recommendations.append("ML inference performance is low - consider model optimization or GPU acceleration")
            
            if 'Database' in result.test_name and result.operations_per_second < 200:
                recommendations.append("Database performance bottleneck detected - consider connection pooling or query optimization")
            
            if 'Concurrent' in result.test_name and result.success_rate < 90:
                recommendations.append("Concurrency issues detected - review threading and async patterns")
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def export_results(self, file_path: str):
        """Export benchmark results to JSON file"""
        report = self.generate_performance_report()
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Benchmark results exported to {file_path}")


async def main():
    """Example usage of the benchmarking system"""
    # Initialize database manager (would be configured properly in real usage)
    db_manager = DatabaseManager('sqlite:///benchmark_test.db')
    
    # Create benchmark suite
    suite = BenchmarkSuite(db_manager)
    
    try:
        # Run comprehensive benchmarks
        results = await suite.run_comprehensive_benchmark_suite()
        
        # Generate and export report
        report = suite.generate_performance_report()
        suite.export_results('benchmark_results.json')
        
        # Print summary
        print("\n" + "="*50)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*50)
        
        for name, result in results.items():
            print(f"\n{name.upper()}:")
            print(f"  Operations/sec: {result.operations_per_second:.2f}")
            print(f"  Success rate: {result.success_rate:.1f}%")
            print(f"  Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  Duration: {result.duration_seconds:.2f}s")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
    except Exception as e:
        logger.error("Benchmark execution failed", error=str(e))
        raise


if __name__ == '__main__':
    asyncio.run(main())