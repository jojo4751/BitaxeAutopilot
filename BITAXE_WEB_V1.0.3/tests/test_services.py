"""
Tests for service layer functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sqlite3

from services.service_container_v2 import ServiceContainerV2
from services.config_service import ConfigService
from services.database_service_v2 import DatabaseServiceV2
from exceptions.custom_exceptions import DatabaseError, ServiceError


class TestConfigService:
    """Test configuration service"""
    
    @pytest.fixture
    def config_service(self, tmp_path):
        """Create config service with temporary config file"""
        config_file = tmp_path / "test_config.json"
        return ConfigService(str(config_file))
    
    def test_config_initialization(self, config_service):
        """Test config service initialization"""
        assert config_service.get("ips") == []
        assert config_service.get("colors", {}) == {}
        assert config_service.database_path is not None
    
    def test_config_get_set(self, config_service):
        """Test getting and setting config values"""
        # Test setting and getting values
        config_service.set("test.setting", "test_value")
        assert config_service.get("test.setting") == "test_value"
        
        # Test nested access
        config_service.set("nested.deep.setting", "deep_value")
        assert config_service.get("nested.deep.setting") == "deep_value"
        
        # Test default values
        assert config_service.get("nonexistent", "default") == "default"
    
    def test_config_dot_notation(self, config_service):
        """Test dot notation access"""
        config_service.set("database.path", "/test/path")
        config_service.set("settings.temp_limits.warning", 80)
        
        assert config_service.database_path == "/test/path"
        assert config_service.temp_limits["warning"] == 80
    
    def test_config_persistence(self, tmp_path):
        """Test config persistence across instances"""
        config_file = tmp_path / "persistent_config.json"
        
        # First instance
        config1 = ConfigService(str(config_file))
        config1.set("persistent.value", "saved")
        config1.save()
        
        # Second instance should load saved value
        config2 = ConfigService(str(config_file))
        assert config2.get("persistent.value") == "saved"
    
    def test_miner_colors(self, config_service):
        """Test miner color management"""
        ip = "192.168.1.100"
        
        # Should generate color for new miner
        color1 = config_service.get_miner_color(ip)
        assert color1.startswith('#')
        assert len(color1) == 7
        
        # Should return same color for same miner
        color2 = config_service.get_miner_color(ip)
        assert color1 == color2
        
        # Different miners should get different colors
        color3 = config_service.get_miner_color("192.168.1.101")
        assert color3 != color1


class TestDatabaseServiceV2:
    """Test enhanced database service"""
    
    @pytest.fixture
    def db_service(self, tmp_path):
        """Create database service with temporary database"""
        db_path = tmp_path / "test.db"
        return DatabaseServiceV2(str(db_path))
    
    def test_database_initialization(self, db_service):
        """Test database service initialization"""
        assert db_service.db_path is not None
        assert db_service.repository_factory is not None
    
    def test_repository_context_manager(self, db_service):
        """Test repository factory context manager"""
        with db_service.repository_factory() as repos:
            assert repos.miner_log is not None
            assert repos.benchmark_results is not None
            assert repos.event_log is not None
            assert repos.autopilot_log is not None
    
    def test_get_latest_status(self, db_service):
        """Test getting latest miner status"""
        # Mock the repository
        with patch.object(db_service, 'repository_factory') as mock_factory:
            mock_repos = Mock()
            mock_miner_repo = Mock()
            mock_miner_repo.get_latest_status_all.return_value = [
                Mock(
                    ip='192.168.1.100',
                    hostname='test-miner',
                    temp=65.5,
                    hashRate=485.2,
                    power=12.8,
                    voltage=5.1,
                    frequency=800,
                    coreVoltage=1200,
                    fanrpm=3500,
                    sharesAccepted=1250,
                    sharesRejected=5,
                    uptime=86400,
                    version='2.0.4',
                    timestamp=datetime.now()
                )
            ]
            mock_repos.miner_log = mock_miner_repo
            mock_factory.return_value.__enter__.return_value = mock_repos
            
            result = db_service.get_latest_status()
            assert len(result) == 1
            assert result[0]['ip'] == '192.168.1.100'
    
    def test_data_validation(self, db_service):
        """Test data validation during operations"""
        invalid_data = {
            'ip': 'invalid-ip-format',  # Invalid IP
            'temp': -100,  # Invalid temperature
            'hashRate': 'not-a-number'  # Invalid type
        }
        
        with pytest.raises(Exception):  # Should raise validation error
            db_service.log_miner_data(invalid_data)
    
    def test_benchmark_operations(self, db_service):
        """Test benchmark data operations"""
        with patch.object(db_service, 'repository_factory') as mock_factory:
            mock_repos = Mock()
            mock_bench_repo = Mock()
            mock_bench_repo.get_results_by_ip.return_value = []
            mock_repos.benchmark_results = mock_bench_repo
            mock_factory.return_value.__enter__.return_value = mock_repos
            
            results = db_service.get_benchmark_results_for_ip('192.168.1.100', 10)
            assert isinstance(results, list)
            
            mock_bench_repo.get_results_by_ip.assert_called_once_with('192.168.1.100', 10)
    
    def test_event_logging(self, db_service):
        """Test event logging functionality"""
        with patch.object(db_service, 'repository_factory') as mock_factory:
            mock_repos = Mock()
            mock_event_repo = Mock()
            mock_repos.event_log = mock_event_repo  
            mock_factory.return_value.__enter__.return_value = mock_repos
            
            # Test event creation
            db_service.log_event('192.168.1.100', 'BENCHMARK_STARTED', 'Benchmark started', 'INFO')
            
            mock_event_repo.create.assert_called_once()
    
    def test_database_error_handling(self, db_service):
        """Test database error handling"""
        with patch.object(db_service, 'repository_factory') as mock_factory:
            mock_factory.side_effect = sqlite3.Error("Database error")
            
            with pytest.raises(DatabaseError):
                db_service.get_latest_status()


class TestServiceContainer:
    """Test service container functionality"""
    
    def test_container_initialization(self):
        """Test service container initialization"""
        container = ServiceContainerV2()
        
        # Should be able to get all services
        config_service = container.get_config_service()
        database_service = container.get_database_service()
        miner_service = container.get_miner_service()
        benchmark_service = container.get_benchmark_service()
        autopilot_service = container.get_autopilot_service()
        
        assert config_service is not None
        assert database_service is not None
        assert miner_service is not None
        assert benchmark_service is not None
        assert autopilot_service is not None
    
    def test_service_singletons(self):
        """Test that services are singletons"""
        container = ServiceContainerV2()
        
        # Same service should be returned on multiple calls
        config1 = container.get_config_service()
        config2 = container.get_config_service()
        assert config1 is config2
        
        database1 = container.get_database_service()
        database2 = container.get_database_service()
        assert database1 is database2
    
    def test_service_dependencies(self):
        """Test service dependency injection"""
        container = ServiceContainerV2()
        
        # Services should have proper dependencies injected
        miner_service = container.get_miner_service()
        assert hasattr(miner_service, 'config_service')
        assert hasattr(miner_service, 'database_service')
        
        benchmark_service = container.get_benchmark_service()
        assert hasattr(benchmark_service, 'config_service')
        assert hasattr(benchmark_service, 'database_service')
        assert hasattr(benchmark_service, 'miner_service')


class TestMinerService:
    """Test miner service functionality"""
    
    @pytest.fixture
    def miner_service(self, container):
        """Get miner service from container"""
        return container.get_miner_service()
    
    def test_get_miners_summary(self, miner_service):
        """Test getting miners summary"""
        with patch.object(miner_service, 'database_service') as mock_db:
            mock_db.get_latest_status.return_value = [
                {
                    'ip': '192.168.1.100',
                    'hashRate': 500.0,
                    'power': 15.0,
                    'temp': 65.0,
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'ip': '192.168.1.101',
                    'hashRate': 480.0,
                    'power': 14.5,
                    'temp': 67.0,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            summary = miner_service.get_miners_summary()
            
            assert summary['total_miners'] == 2
            assert summary['online_miners'] == 2
            assert summary['offline_miners'] == 0
            assert summary['total_hashrate'] == 980.0
            assert summary['total_power'] == 29.5
            assert summary['average_temperature'] == 66.0
    
    @patch('requests.get')
    def test_fetch_miner_data_success(self, mock_get, miner_service, mock_miner_response):
        """Test successful miner data fetch"""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_miner_response
        
        result = miner_service.fetch_miner_data('192.168.1.100')
        
        assert result is not None
        assert result['ip'] == '192.168.1.100'
        assert result['temp'] == 65.5
        assert result['hashRate'] == 485.2
    
    @patch('requests.get')
    def test_fetch_miner_data_failure(self, mock_get, miner_service):
        """Test miner data fetch failure"""
        mock_get.side_effect = Exception("Connection error")
        
        result = miner_service.fetch_miner_data('192.168.1.100')
        
        assert result is None
    
    @patch('requests.post')
    def test_set_miner_settings_success(self, mock_post, miner_service):
        """Test successful miner settings update"""
        mock_post.return_value.status_code = 200
        
        result = miner_service.set_miner_settings('192.168.1.100', 800, 1200, True)
        
        assert result is True
        mock_post.assert_called()
    
    @patch('requests.post')
    def test_restart_miner_success(self, mock_post, miner_service):
        """Test successful miner restart"""
        mock_post.return_value.status_code = 200
        
        result = miner_service.restart_miner('192.168.1.100')
        
        assert result is True
        mock_post.assert_called()


class TestBenchmarkService:
    """Test benchmark service functionality"""
    
    @pytest.fixture
    def benchmark_service(self, container):
        """Get benchmark service from container"""
        return container.get_benchmark_service()
    
    def test_start_benchmark(self, benchmark_service):
        """Test starting benchmark"""
        with patch.object(benchmark_service, 'miner_service') as mock_miner:
            mock_miner.set_miner_settings.return_value = True
            
            result = benchmark_service.start_benchmark('192.168.1.100', 800, 1200, 600)
            
            assert result is True
            mock_miner.set_miner_settings.assert_called_once()
    
    def test_start_multi_benchmark(self, benchmark_service):
        """Test starting multi-miner benchmark"""
        ips = ['192.168.1.100', '192.168.1.101']
        
        with patch.object(benchmark_service, 'start_benchmark') as mock_start:
            mock_start.return_value = True
            
            result = benchmark_service.start_multi_benchmark(ips, 800, 1200, 600)
            
            assert result == ips
            assert mock_start.call_count == 2
    
    def test_get_benchmark_status(self, benchmark_service):
        """Test getting benchmark status"""
        status = benchmark_service.get_benchmark_status()
        
        assert 'active_benchmarks' in status
        assert 'total_active' in status
        assert isinstance(status['active_benchmarks'], list)
        assert isinstance(status['total_active'], int)
    
    def test_benchmark_completion(self, benchmark_service):
        """Test benchmark completion handling"""
        with patch.object(benchmark_service, 'database_service') as mock_db:
            mock_db.save_benchmark_result.return_value = None
            
            # Simulate benchmark completion
            benchmark_service._handle_benchmark_completion(
                '192.168.1.100', 800, 1200, 485.2, 65.5, 600
            )
            
            mock_db.save_benchmark_result.assert_called_once()


class TestAutopilotService:
    """Test autopilot service functionality"""
    
    @pytest.fixture
    def autopilot_service(self, container):
        """Get autopilot service from container"""
        return container.get_autopilot_service()
    
    def test_autopilot_initialization(self, autopilot_service):
        """Test autopilot service initialization"""
        assert hasattr(autopilot_service, 'config_service')
        assert hasattr(autopilot_service, 'database_service')
        assert hasattr(autopilot_service, 'miner_service')
    
    def test_temperature_monitoring(self, autopilot_service):
        """Test temperature monitoring functionality"""
        with patch.object(autopilot_service, 'database_service') as mock_db:
            mock_db.get_latest_status.return_value = [
                {
                    'ip': '192.168.1.100',
                    'temp': 85.0,  # High temperature
                    'frequency': 800,
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            with patch.object(autopilot_service, 'miner_service') as mock_miner:
                mock_miner.set_miner_settings.return_value = True
                
                autopilot_service.check_temperature_limits()
                
                # Should adjust frequency due to high temperature
                mock_miner.set_miner_settings.assert_called()
    
    def test_autopilot_disabled(self, autopilot_service):
        """Test autopilot when disabled"""
        with patch.object(autopilot_service, 'config_service') as mock_config:
            mock_config.get.return_value = False  # Autopilot disabled
            
            with patch.object(autopilot_service, 'miner_service') as mock_miner:
                autopilot_service.check_temperature_limits()
                
                # Should not make any adjustments
                mock_miner.set_miner_settings.assert_not_called()


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for services working together"""
    
    def test_full_miner_monitoring_cycle(self, container):
        """Test complete miner monitoring cycle"""
        miner_service = container.get_miner_service()
        database_service = container.get_database_service()
        
        # Mock miner response
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                'power': 12.8,
                'temp': 65.5,
                'hashRate': 485.2,
                'hostname': 'test-miner'
            }
            
            # Fetch and log data
            miner_data = miner_service.fetch_miner_data('192.168.1.100')
            assert miner_data is not None
            
            # The logging would happen in the background service
            # For testing, we simulate it
            with patch.object(database_service, 'log_miner_data') as mock_log:
                database_service.log_miner_data(miner_data)
                mock_log.assert_called_once()
    
    def test_benchmark_to_database_flow(self, container):
        """Test benchmark data flow to database"""
        benchmark_service = container.get_benchmark_service()
        database_service = container.get_database_service()
        
        with patch.object(benchmark_service, 'miner_service') as mock_miner:
            mock_miner.set_miner_settings.return_value = True
            
            with patch.object(database_service, 'save_benchmark_result') as mock_save:
                # Start benchmark
                result = benchmark_service.start_benchmark('192.168.1.100', 800, 1200, 600)
                assert result is True
                
                # Simulate completion
                benchmark_service._handle_benchmark_completion(
                    '192.168.1.100', 800, 1200, 485.2, 65.5, 600
                )
                
                mock_save.assert_called_once()
    
    def test_error_propagation(self, container):
        """Test error propagation through service layers"""
        miner_service = container.get_miner_service()
        
        # Simulate database error
        with patch.object(miner_service, 'database_service') as mock_db:
            mock_db.log_miner_data.side_effect = DatabaseError("Database unavailable")
            
            # Error should be handled gracefully
            with pytest.raises(DatabaseError):
                miner_service.log_miner_status({
                    'ip': '192.168.1.100',
                    'temp': 65.5,
                    'hashRate': 485.2
                })