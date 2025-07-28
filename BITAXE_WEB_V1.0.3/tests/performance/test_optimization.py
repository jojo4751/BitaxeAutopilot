"""
Unit tests for the performance optimization framework
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from performance.optimization import (
    PerformanceOptimizer, OptimizationResult, BottleneckAnalysis,
    AdaptiveParameterTuner
)
from performance.benchmarking import BenchmarkSuite, BenchmarkResult


class TestOptimizationResult:
    """Test OptimizationResult dataclass"""
    
    def test_optimization_result_creation(self):
        """Test creating an optimization result"""
        result = OptimizationResult(
            optimization_type="cpu_optimization",
            parameter_name="worker_threads",
            original_value=8,
            optimized_value=12,
            performance_improvement=15.5,
            confidence_score=0.85,
            applied_at=datetime.now(),
            description="Optimized worker threads from 8 to 12"
        )
        
        assert result.optimization_type == "cpu_optimization"
        assert result.parameter_name == "worker_threads"
        assert result.original_value == 8
        assert result.optimized_value == 12
        assert result.performance_improvement == 15.5
        assert result.confidence_score == 0.85


class TestBottleneckAnalysis:
    """Test BottleneckAnalysis dataclass"""
    
    def test_bottleneck_analysis_creation(self):
        """Test creating a bottleneck analysis"""
        analysis = BottleneckAnalysis(
            component="cpu",
            severity="high",
            description="High CPU usage detected",
            impact_score=85.0,
            recommendations=["Reduce worker threads", "Optimize algorithms"],
            metrics={"cpu_usage_percent": 85.0}
        )
        
        assert analysis.component == "cpu"
        assert analysis.severity == "high"
        assert analysis.impact_score == 85.0
        assert len(analysis.recommendations) == 2
        assert "cpu_usage_percent" in analysis.metrics


class TestAdaptiveParameterTuner:
    """Test AdaptiveParameterTuner class"""
    
    def test_parameter_tuner_init(self):
        """Test parameter tuner initialization"""
        tuner = AdaptiveParameterTuner()
        
        assert tuner.learning_rate == 0.1
        assert tuner.exploration_factor == 0.2
        assert len(tuner.parameter_history) == 0
    
    def test_suggest_parameter_value_initial(self):
        """Test parameter suggestion with no history"""
        tuner = AdaptiveParameterTuner()
        
        # For numeric parameters with no history, should return random value in range
        suggested = tuner.suggest_parameter_value("test_param", 10, (5, 15))
        assert 5 <= suggested <= 15
    
    def test_suggest_parameter_value_with_history(self):
        """Test parameter suggestion with history"""
        tuner = AdaptiveParameterTuner()
        
        # Add some history
        tuner.record_result("test_param", 8, 0.7)
        tuner.record_result("test_param", 10, 0.8)
        tuner.record_result("test_param", 12, 0.9)
        
        # Should suggest value based on history
        suggested = tuner.suggest_parameter_value("test_param", 10, (5, 20))
        assert 5 <= suggested <= 20
    
    def test_record_result(self):
        """Test recording parameter results"""
        tuner = AdaptiveParameterTuner()
        
        tuner.record_result("test_param", 10, 0.8)
        tuner.record_result("test_param", 12, 0.9)
        
        assert len(tuner.parameter_history["test_param"]) == 2
        assert tuner.parameter_history["test_param"][0] == (10, 0.8)
        assert tuner.parameter_history["test_param"][1] == (12, 0.9)
    
    def test_record_result_history_limit(self):
        """Test history limit is enforced"""
        tuner = AdaptiveParameterTuner()
        
        # Add more than 50 results
        for i in range(60):
            tuner.record_result("test_param", i, i * 0.01)
        
        # Should keep only last 50
        assert len(tuner.parameter_history["test_param"]) == 50
        assert tuner.parameter_history["test_param"][0] == (10, 0.1)  # First kept result
        assert tuner.parameter_history["test_param"][-1] == (59, 0.59)  # Last result


@pytest.mark.asyncio
class TestPerformanceOptimizer:
    """Test PerformanceOptimizer class"""
    
    @pytest.fixture
    def mock_benchmark_suite(self):
        """Create mock benchmark suite"""
        return Mock(spec=BenchmarkSuite)
    
    @pytest.fixture
    def optimizer(self, mock_benchmark_suite):
        """Create optimizer with mocked dependencies"""
        return PerformanceOptimizer(mock_benchmark_suite)
    
    def test_optimizer_init(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.benchmark_suite is not None
        assert optimizer.parameter_tuner is not None
        assert optimizer.optimization_history == []
        assert optimizer.current_config is not None
        assert "ml_batch_size" in optimizer.optimization_parameters
    
    def test_load_default_config(self, optimizer):
        """Test loading default configuration"""
        config = optimizer._load_default_config()
        
        expected_keys = [
            'ml_batch_size', 'db_connection_pool_size', 'async_worker_threads',
            'cache_size_mb', 'gc_threshold', 'optimization_interval'
        ]
        
        for key in expected_keys:
            assert key in config
        
        assert isinstance(config['ml_batch_size'], int)
        assert isinstance(config['db_connection_pool_size'], int)
    
    async def test_establish_performance_baseline(self, optimizer):
        """Test establishing performance baseline"""
        # Mock benchmark results
        mock_results = {
            'ml_inference': Mock(spec=BenchmarkResult),
            'database_performance': Mock(spec=BenchmarkResult)
        }
        optimizer.benchmark_suite.run_comprehensive_benchmark_suite = AsyncMock(return_value=mock_results)
        
        baseline = await optimizer.establish_performance_baseline()
        
        assert baseline == mock_results
        assert optimizer.performance_baseline == mock_results
    
    @patch('performance.optimization.psutil')
    async def test_analyze_system_resource_bottlenecks(self, mock_psutil, optimizer):
        """Test system resource bottleneck analysis"""
        # Mock high CPU usage
        mock_psutil.cpu_percent.return_value = 90.0
        mock_psutil.virtual_memory.return_value.percent = 85.0
        mock_psutil.disk_io_counters.return_value = None
        
        bottlenecks = await optimizer._analyze_system_resource_bottlenecks()
        
        # Should detect CPU and memory bottlenecks
        cpu_bottleneck = next((b for b in bottlenecks if b.component == 'cpu'), None)
        memory_bottleneck = next((b for b in bottlenecks if b.component == 'memory'), None)
        
        assert cpu_bottleneck is not None
        assert cpu_bottleneck.severity == 'medium'  # 90% is medium severity
        assert memory_bottleneck is not None
        assert memory_bottleneck.severity == 'medium'  # 85% is medium severity
    
    async def test_analyze_bottlenecks_no_baseline(self, optimizer):
        """Test bottleneck analysis without baseline"""
        optimizer.performance_baseline = None
        
        # Mock current results
        mock_results = {'test': Mock(spec=BenchmarkResult)}
        optimizer.benchmark_suite.run_comprehensive_benchmark_suite = AsyncMock(return_value=mock_results)
        
        with patch.object(optimizer, '_analyze_system_resource_bottlenecks', new_callable=AsyncMock) as mock_system:
            mock_system.return_value = []
            
            bottlenecks = await optimizer.analyze_bottlenecks()
            
            # Should only have system bottlenecks since no baseline
            assert isinstance(bottlenecks, list)
    
    async def test_analyze_bottlenecks_with_degradation(self, optimizer):
        """Test bottleneck analysis with performance degradation"""
        # Set up baseline
        baseline_result = Mock(spec=BenchmarkResult)
        baseline_result.operations_per_second = 100.0
        optimizer.performance_baseline = {'test_benchmark': baseline_result}
        
        # Current result with degraded performance
        current_result = Mock(spec=BenchmarkResult)
        current_result.operations_per_second = 60.0  # 40% degradation
        current_result.memory_usage_mb = 512.0
        current_result.cpu_usage_percent = 75.0
        current_result.success_rate = 95.0
        
        mock_results = {'test_benchmark': current_result}
        optimizer.benchmark_suite.run_comprehensive_benchmark_suite = AsyncMock(return_value=mock_results)
        
        with patch.object(optimizer, '_analyze_system_resource_bottlenecks', new_callable=AsyncMock) as mock_system:
            mock_system.return_value = []
            
            bottlenecks = await optimizer.analyze_bottlenecks()
            
            # Should detect performance degradation
            assert len(bottlenecks) == 1
            bottleneck = bottlenecks[0]
            assert bottleneck.component == 'test_benchmark'
            assert bottleneck.severity == 'high'  # 40% degradation is high severity
            assert bottleneck.impact_score == 40.0
    
    def test_generate_bottleneck_recommendations(self, optimizer):
        """Test generating bottleneck recommendations"""
        mock_result = Mock(spec=BenchmarkResult)
        mock_result.operations_per_second = 30.0
        mock_result.memory_usage_mb = 1500.0
        mock_result.cpu_usage_percent = 95.0
        mock_result.success_rate = 85.0
        
        # Test ML inference recommendations
        recommendations = optimizer._generate_bottleneck_recommendations('ml_inference', mock_result)
        assert any('ML model inference' in rec for rec in recommendations)
        assert any('memory usage' in rec for rec in recommendations)
        assert any('CPU-intensive' in rec for rec in recommendations)
        
        # Test database recommendations
        recommendations = optimizer._generate_bottleneck_recommendations('database_performance', mock_result)
        assert any('database' in rec.lower() for rec in recommendations)
    
    async def test_optimize_cpu_usage(self, optimizer):
        """Test CPU usage optimization"""
        original_value = optimizer.current_config['async_worker_threads']
        
        with patch.object(optimizer, '_measure_cpu_performance', new_callable=AsyncMock) as mock_measure:
            # Mock performance improvement
            mock_measure.side_effect = [70.0, 80.0]  # Before and after performance
            
            # Mock parameter tuner to suggest a different value
            optimizer.parameter_tuner.suggest_parameter_value = Mock(return_value=original_value + 2)
            
            result = await optimizer._optimize_cpu_usage()
            
            assert result is not None
            assert result.optimization_type == 'cpu_optimization'
            assert result.parameter_name == 'async_worker_threads'
            assert result.original_value == original_value
            assert result.optimized_value == original_value + 2
            assert result.performance_improvement > 0
    
    async def test_optimize_cpu_usage_no_improvement(self, optimizer):
        """Test CPU optimization with no improvement"""
        original_value = optimizer.current_config['async_worker_threads']
        
        with patch.object(optimizer, '_measure_cpu_performance', new_callable=AsyncMock) as mock_measure:
            # Mock no performance improvement
            mock_measure.side_effect = [70.0, 70.5]  # Minimal improvement
            
            optimizer.parameter_tuner.suggest_parameter_value = Mock(return_value=original_value + 2)
            
            result = await optimizer._optimize_cpu_usage()
            
            # Should revert change and return None
            assert result is None
            assert optimizer.current_config['async_worker_threads'] == original_value
    
    async def test_optimize_memory_usage(self, optimizer):
        """Test memory usage optimization"""
        original_value = optimizer.current_config['cache_size_mb']
        
        with patch('performance.optimization.psutil') as mock_psutil:
            # Mock normal memory usage
            mock_psutil.virtual_memory.return_value.percent = 70.0
            
            with patch.object(optimizer, '_measure_memory_efficiency', new_callable=AsyncMock) as mock_measure:
                mock_measure.return_value = 85.0
                
                optimizer.parameter_tuner.suggest_parameter_value = Mock(return_value=original_value + 100)
                
                result = await optimizer._optimize_memory_usage()
                
                assert result is not None
                assert result.optimization_type == 'memory_optimization'
                assert result.parameter_name == 'cache_size_mb'
    
    async def test_optimize_ml_performance(self, optimizer):
        """Test ML performance optimization"""
        # Set up baseline
        baseline_result = Mock(spec=BenchmarkResult)
        baseline_result.operations_per_second = 100.0
        optimizer.performance_baseline = {'ml_inference': baseline_result}
        
        # Mock improved ML benchmark result
        improved_result = Mock(spec=BenchmarkResult)
        improved_result.operations_per_second = 110.0  # 10% improvement
        optimizer.benchmark_suite.run_ml_inference_benchmark = AsyncMock(return_value=improved_result)
        
        original_value = optimizer.current_config['ml_batch_size']
        optimizer.parameter_tuner.suggest_parameter_value = Mock(return_value=original_value + 5)
        
        result = await optimizer._optimize_ml_performance()
        
        assert result is not None
        assert result.optimization_type == 'ml_optimization'
        assert result.parameter_name == 'ml_batch_size'
        assert result.performance_improvement == 10.0
    
    async def test_optimize_system_performance(self, optimizer):
        """Test complete system performance optimization"""
        # Set up baseline
        await optimizer.establish_performance_baseline()
        
        # Mock bottleneck analysis
        mock_bottleneck = BottleneckAnalysis(
            component='cpu',
            severity='high',
            description='High CPU usage',
            impact_score=80.0,
            recommendations=['Optimize CPU usage'],
            metrics={}
        )
        
        with patch.object(optimizer, 'analyze_bottlenecks', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = [mock_bottleneck]
            
            with patch.object(optimizer, '_optimize_for_bottleneck', new_callable=AsyncMock) as mock_optimize:
                mock_optimization = OptimizationResult(
                    optimization_type='cpu_optimization',
                    parameter_name='async_worker_threads',
                    original_value=8,
                    optimized_value=10,
                    performance_improvement=5.0,
                    confidence_score=0.8,
                    applied_at=datetime.now(),
                    description='Test optimization'
                )
                mock_optimize.return_value = [mock_optimization]
                
                results = await optimizer.optimize_system_performance(max_iterations=2)
                
                assert isinstance(results, list)
                assert len(results) > 0
    
    def test_generate_optimization_report_no_history(self, optimizer):
        """Test generating report with no optimization history"""
        report = optimizer.generate_optimization_report()
        
        assert 'summary' in report
        assert report['summary'] == 'No optimizations applied yet'
        assert 'recommendations' in report
    
    def test_generate_optimization_report_with_history(self, optimizer):
        """Test generating report with optimization history"""
        # Add mock optimization to history
        optimization = OptimizationResult(
            optimization_type='cpu_optimization',
            parameter_name='async_worker_threads',
            original_value=8,
            optimized_value=10,
            performance_improvement=15.0,
            confidence_score=0.85,
            applied_at=datetime.now(),
            description='Test optimization'
        )
        optimizer.optimization_history.append(optimization)
        
        report = optimizer.generate_optimization_report()
        
        assert 'summary' in report
        assert report['summary']['total_optimizations'] == 1
        assert report['summary']['average_improvement'] == '15.00%'
        assert 'optimizations_by_type' in report
        assert 'cpu_optimization' in report['optimizations_by_type']
    
    def test_generate_future_recommendations(self, optimizer):
        """Test generating future recommendations"""
        recommendations = optimizer._generate_future_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('optimization cycles' in rec for rec in recommendations)
    
    def test_export_optimization_results(self, optimizer, tmp_path):
        """Test exporting optimization results"""
        # Add mock result
        optimization = OptimizationResult(
            optimization_type='test_optimization',
            parameter_name='test_param',
            original_value=10,
            optimized_value=15,
            performance_improvement=20.0,
            confidence_score=0.9,
            applied_at=datetime.now(),
            description='Test optimization'
        )
        optimizer.optimization_history.append(optimization)
        
        # Export to temporary file
        export_file = tmp_path / "optimization_results.json"
        optimizer.export_optimization_results(str(export_file))
        
        assert export_file.exists()
        
        # Verify file contents
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert 'summary' in data
        assert 'optimizations_by_type' in data


@pytest.mark.asyncio 
class TestOptimizationIntegration:
    """Integration tests for optimization components"""
    
    @pytest.fixture
    def mock_benchmark_suite(self):
        """Create mock benchmark suite for integration tests"""
        suite = Mock(spec=BenchmarkSuite)
        
        # Mock benchmark results
        mock_result = Mock(spec=BenchmarkResult)
        mock_result.operations_per_second = 100.0
        mock_result.memory_usage_mb = 256.0
        mock_result.cpu_usage_percent = 50.0
        mock_result.success_rate = 95.0
        
        suite.run_comprehensive_benchmark_suite = AsyncMock(return_value={
            'ml_inference': mock_result,
            'database_performance': mock_result,
            'concurrent_optimization': mock_result,
            'memory_stress': mock_result
        })
        
        suite.run_ml_inference_benchmark = AsyncMock(return_value=mock_result)
        suite.run_database_performance_benchmark = AsyncMock(return_value=mock_result)
        suite.run_concurrent_optimization_benchmark = AsyncMock(return_value=mock_result)
        
        return suite
    
    async def test_full_optimization_cycle(self, mock_benchmark_suite):
        """Test complete optimization cycle"""
        optimizer = PerformanceOptimizer(mock_benchmark_suite)
        
        # Establish baseline
        baseline = await optimizer.establish_performance_baseline()
        assert baseline is not None
        
        # Run optimization with mocked system resources
        with patch('performance.optimization.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 85.0  # High CPU to trigger optimization
            mock_psutil.virtual_memory.return_value.percent = 75.0
            mock_psutil.disk_io_counters.return_value = None
            
            results = await optimizer.optimize_system_performance(max_iterations=2)
            
            # Should have attempted some optimizations
            assert isinstance(results, list)
    
    def test_adaptive_parameter_tuning_convergence(self):
        """Test that parameter tuning converges to optimal values"""
        tuner = AdaptiveParameterTuner()
        
        # Simulate optimization where optimal value is 15
        optimal_value = 15
        value_range = (5, 25)
        
        # Run multiple iterations
        for iteration in range(20):
            current_value = 10 + iteration % 10  # Vary starting point
            suggested_value = tuner.suggest_parameter_value('test_param', current_value, value_range)
            
            # Simulate performance function (peaks at optimal_value)
            performance = 1.0 - abs(suggested_value - optimal_value) / 20.0
            tuner.record_result('test_param', suggested_value, performance)
        
        # After many iterations, should suggest values closer to optimal
        final_suggestion = tuner.suggest_parameter_value('test_param', 10, value_range)
        
        # Should be reasonably close to optimal (within 50% of range)
        assert abs(final_suggestion - optimal_value) < (value_range[1] - value_range[0]) * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])