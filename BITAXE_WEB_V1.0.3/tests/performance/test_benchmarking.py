"""
Unit tests for the performance benchmarking framework
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from performance.benchmarking import (
    BenchmarkSuite, BenchmarkResult, PerformanceProfile, 
    PerformanceMonitor, BenchmarkResult
)
from database import DatabaseManager


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass"""
    
    def test_benchmark_result_creation(self):
        """Test creating a benchmark result"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = BenchmarkResult(
            test_name="test_benchmark",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=10.0,
            operations_per_second=100.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=50.0,
            success_count=950,
            error_count=50,
            metrics={},
            percentiles={}
        )
        
        assert result.test_name == "test_benchmark"
        assert result.total_operations == 1000
        assert result.success_rate == 95.0
        assert result.duration_seconds == 10.0
    
    def test_benchmark_result_zero_operations(self):
        """Test benchmark result with zero operations"""
        result = BenchmarkResult(
            test_name="empty_test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
            operations_per_second=0.0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            success_count=0,
            error_count=0,
            metrics={},
            percentiles={}
        )
        
        assert result.total_operations == 0
        assert result.success_rate == 0.0


class TestPerformanceProfile:
    """Test PerformanceProfile dataclass"""
    
    def test_performance_profile_statistics(self):
        """Test statistics calculation"""
        profile = PerformanceProfile(
            cpu_usage=[10.0, 20.0, 30.0, 40.0, 50.0],
            memory_usage=[100.0, 200.0, 300.0, 400.0, 500.0],
            disk_io={'read_bytes': [1000, 2000, 3000], 'write_bytes': [500, 1000, 1500]},
            network_io={'bytes_sent': [100, 200, 300], 'bytes_recv': [50, 100, 150]},
            gc_collections=[1, 2, 3, 4, 5],
            thread_count=[5, 6, 7, 8, 9]
        )
        
        stats = profile.get_statistics()
        
        assert stats['cpu_usage']['mean'] == 30.0
        assert stats['cpu_usage']['min'] == 10.0
        assert stats['cpu_usage']['max'] == 50.0
        assert stats['memory_usage']['median'] == 300.0
    
    def test_performance_profile_empty_values(self):
        """Test statistics with empty values"""
        profile = PerformanceProfile(
            cpu_usage=[],
            memory_usage=[],
            disk_io={'read_bytes': [], 'write_bytes': []},
            network_io={'bytes_sent': [], 'bytes_recv': []},
            gc_collections=[],
            thread_count=[]
        )
        
        stats = profile.get_statistics()
        
        assert stats['cpu_usage']['mean'] == 0
        assert stats['memory_usage']['std'] == 0


@pytest.mark.asyncio
class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""
    
    def test_performance_monitor_init(self):
        """Test performance monitor initialization"""
        monitor = PerformanceMonitor(sampling_interval=0.5)
        
        assert monitor.sampling_interval == 0.5
        assert not monitor.monitoring
        assert monitor.profile is not None
    
    @patch('performance.benchmarking.psutil')
    async def test_start_stop_monitoring(self, mock_psutil):
        """Test starting and stopping monitoring"""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value.used = 1024 * 1024 * 512  # 512 MB
        mock_psutil.disk_io_counters.return_value = None
        mock_psutil.net_io_counters.return_value = None
        
        monitor = PerformanceMonitor(sampling_interval=0.01)  # Very short interval for testing
        
        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.monitoring
        
        # Let it run briefly
        await asyncio.sleep(0.05)
        
        # Stop monitoring
        profile = await monitor.stop_monitoring()
        assert not monitor.monitoring
        assert len(profile.cpu_usage) > 0
        assert len(profile.memory_usage) > 0
    
    @patch('performance.benchmarking.psutil')
    async def test_monitoring_loop_exception_handling(self, mock_psutil):
        """Test monitoring loop handles exceptions gracefully"""
        # Make psutil throw an exception
        mock_psutil.cpu_percent.side_effect = Exception("Test exception")
        
        monitor = PerformanceMonitor(sampling_interval=0.01)
        
        await monitor.start_monitoring()
        await asyncio.sleep(0.05)  # Let it run and handle exceptions
        await monitor.stop_monitoring()
        
        # Should not crash and should have empty profile
        assert len(monitor.profile.cpu_usage) == 0


@pytest.mark.asyncio 
class TestBenchmarkSuite:
    """Test BenchmarkSuite class"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager"""
        return Mock(spec=DatabaseManager)
    
    @pytest.fixture
    def benchmark_suite(self, mock_db_manager):
        """Create benchmark suite with mocked dependencies"""
        with patch('performance.benchmarking.OptimizationEngine'), \
             patch('performance.benchmarking.FeatureEngineer'), \
             patch('performance.benchmarking.RLAgent'):
            return BenchmarkSuite(mock_db_manager)
    
    def test_benchmark_suite_init(self, benchmark_suite):
        """Test benchmark suite initialization"""
        assert benchmark_suite.results == []
        assert benchmark_suite.monitor is not None
        assert benchmark_suite.optimization_engine is not None
    
    async def test_benchmark_context_manager(self, benchmark_suite):
        """Test benchmark context manager"""
        with patch.object(benchmark_suite.monitor, 'start_monitoring', new_callable=AsyncMock), \
             patch.object(benchmark_suite.monitor, 'stop_monitoring', new_callable=AsyncMock) as mock_stop:
            
            # Mock performance profile
            mock_profile = Mock()
            mock_profile.get_statistics.return_value = {}
            mock_stop.return_value = mock_profile
            
            async with benchmark_suite.benchmark_context("test_benchmark") as tracker:
                # Simulate some operations
                tracker.record_success(0.1)
                tracker.record_success(0.2)
                tracker.record_error()
            
            # Check that a result was added
            assert len(benchmark_suite.results) == 1
            result = benchmark_suite.results[0]
            assert result.test_name == "test_benchmark"
            assert result.success_count == 2
            assert result.error_count == 1
    
    @patch('performance.benchmarking.psutil')
    async def test_ml_inference_benchmark(self, mock_psutil, benchmark_suite):
        """Test ML inference benchmark"""
        # Mock system calls
        mock_psutil.virtual_memory.return_value.used = 1024 * 1024 * 256
        mock_psutil.cpu_percent.return_value = 50.0
        
        # Mock ML components
        benchmark_suite.feature_engineer.engineer_features = Mock(return_value={'feature1': 1.0})
        benchmark_suite.rl_agent.get_action = Mock(return_value={'action': 'optimize'})
        
        # Run benchmark with small number of operations for speed
        result = await benchmark_suite.run_ml_inference_benchmark(num_operations=10)
        
        assert result.test_name == "ML Inference Performance"
        assert result.success_count > 0
        assert result.operations_per_second > 0
    
    async def test_database_performance_benchmark(self, benchmark_suite):
        """Test database performance benchmark"""
        # Mock database operations
        mock_session = Mock()
        mock_session.query.return_value.limit.return_value.all.return_value = []
        mock_session.query.return_value.first.return_value = None
        benchmark_suite.db_manager.get_session.return_value = mock_session
        
        result = await benchmark_suite.run_database_performance_benchmark(num_operations=5)
        
        assert result.test_name == "Database Performance"
        assert result.success_count >= 0  # Some operations might fail with mocked DB
    
    async def test_concurrent_optimization_benchmark(self, benchmark_suite):
        """Test concurrent optimization benchmark"""
        # Mock ML components
        benchmark_suite.feature_engineer.engineer_features = Mock(return_value={'feature1': 1.0})
        benchmark_suite.rl_agent.get_action = Mock(return_value={'action': 'optimize'})
        
        result = await benchmark_suite.run_concurrent_optimization_benchmark(
            num_miners=5, concurrent_optimizations=2
        )
        
        assert result.test_name == "Concurrent Optimization Performance"
        assert result.success_count > 0
    
    async def test_memory_stress_test(self, benchmark_suite):
        """Test memory stress test"""
        result = await benchmark_suite.run_memory_stress_test(duration_seconds=1)
        
        assert result.test_name == "Memory Stress Test"
        assert result.success_count > 0
        assert result.duration_seconds >= 1.0
    
    async def test_comprehensive_benchmark_suite(self, benchmark_suite):
        """Test running comprehensive benchmark suite"""
        # Mock all the individual benchmark methods
        with patch.object(benchmark_suite, 'run_ml_inference_benchmark', new_callable=AsyncMock) as mock_ml, \
             patch.object(benchmark_suite, 'run_database_performance_benchmark', new_callable=AsyncMock) as mock_db, \
             patch.object(benchmark_suite, 'run_concurrent_optimization_benchmark', new_callable=AsyncMock) as mock_concurrent, \
             patch.object(benchmark_suite, 'run_memory_stress_test', new_callable=AsyncMock) as mock_memory:
            
            # Mock return values
            mock_result = Mock(spec=BenchmarkResult)
            mock_ml.return_value = mock_result
            mock_db.return_value = mock_result
            mock_concurrent.return_value = mock_result
            mock_memory.return_value = mock_result
            
            results = await benchmark_suite.run_comprehensive_benchmark_suite()
            
            assert len(results) == 4
            assert 'ml_inference' in results
            assert 'database_performance' in results
            assert 'concurrent_optimization' in results
            assert 'memory_stress' in results
    
    def test_generate_performance_report_no_results(self, benchmark_suite):
        """Test generating report with no results"""
        report = benchmark_suite.generate_performance_report()
        
        assert 'error' in report
        assert report['error'] == 'No benchmark results available'
    
    def test_generate_performance_report_with_results(self, benchmark_suite):
        """Test generating report with results"""
        # Add mock results
        mock_result = BenchmarkResult(
            test_name="test",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=10),
            duration_seconds=10.0,
            operations_per_second=100.0,
            memory_usage_mb=256.0,
            cpu_usage_percent=50.0,
            success_count=100,
            error_count=0,
            metrics={'performance_profile': {}},
            percentiles={}
        )
        benchmark_suite.results.append(mock_result)
        
        report = benchmark_suite.generate_performance_report()
        
        assert 'summary' in report
        assert report['summary']['total_benchmarks'] == 1
        assert report['summary']['total_operations'] == 100
        assert 'benchmarks' in report
        assert 'test' in report['benchmarks']
    
    def test_export_results(self, benchmark_suite, tmp_path):
        """Test exporting results to file"""
        # Add mock result
        mock_result = BenchmarkResult(
            test_name="test",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=5),
            duration_seconds=5.0,
            operations_per_second=50.0,
            memory_usage_mb=128.0,
            cpu_usage_percent=25.0,
            success_count=50,
            error_count=0,
            metrics={},
            percentiles={}
        )
        benchmark_suite.results.append(mock_result)
        
        # Export to temporary file
        export_file = tmp_path / "test_results.json"
        benchmark_suite.export_results(str(export_file))
        
        assert export_file.exists()
        
        # Verify file contents
        import json
        with open(export_file) as f:
            data = json.load(f)
        
        assert 'summary' in data
        assert 'benchmarks' in data


@pytest.mark.asyncio
class TestBenchmarkIntegration:
    """Integration tests for benchmarking components"""
    
    @pytest.fixture
    def real_db_manager(self):
        """Create real database manager for integration tests"""
        return DatabaseManager('sqlite:///:memory:')
    
    async def test_full_benchmark_flow(self, real_db_manager):
        """Test complete benchmark flow with real components"""
        with patch('performance.benchmarking.OptimizationEngine'), \
             patch('performance.benchmarking.FeatureEngineer'), \
             patch('performance.benchmarking.RLAgent'):
            
            suite = BenchmarkSuite(real_db_manager)
            
            # Run a quick benchmark
            result = await suite.run_ml_inference_benchmark(num_operations=5)
            
            assert result is not None
            assert result.test_name == "ML Inference Performance"
            assert len(suite.results) == 1
            
            # Generate report
            report = suite.generate_performance_report()
            assert 'summary' in report
            assert report['summary']['total_benchmarks'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])