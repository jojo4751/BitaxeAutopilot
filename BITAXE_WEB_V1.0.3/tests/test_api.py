"""
Tests for API endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
from datetime import datetime

from tests.conftest import assert_json_response, assert_api_success, assert_error_response


class TestMinersAPI:
    """Test miner management API endpoints"""
    
    def test_get_miners_list(self, client, auth_headers, sample_miner_data):
        """Test getting list of miners"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_latest_status.return_value = [sample_miner_data]
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/miners', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert 'data' in data
            assert 'pagination' in data
            
            miners = data['data']
            assert len(miners) == 1
            
            miner = miners[0]
            assert miner['ip'] == sample_miner_data['ip']
            assert miner['hostname'] == sample_miner_data['hostname']
            assert miner['hash_rate'] == sample_miner_data['hashRate']
    
    def test_get_miners_pagination(self, client, auth_headers):
        """Test miners list pagination"""
        # Create multiple miner records
        miners_data = []
        for i in range(75):  # More than default page size
            miners_data.append({
                'ip': f'192.168.1.{i+100}',
                'hostname': f'test-miner-{i:03d}',
                'hashRate': 485.2,
                'power': 12.8,
                'timestamp': datetime.now().isoformat()
            })
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_latest_status.return_value = miners_data
            mock_container.return_value.get_database_service.return_value = mock_db
            
            # Test first page
            response = client.get('/api/v1/miners?page=1&page_size=25', headers=auth_headers)
            data = assert_json_response(response, 200)
            
            assert len(data['data']) == 25
            pagination = data['pagination']
            assert pagination['page'] == 1
            assert pagination['page_size'] == 25
            assert pagination['total_count'] == 75
            assert pagination['total_pages'] == 3
            assert pagination['has_next'] is True
            assert pagination['has_previous'] is False
    
    def test_get_miner_detail(self, client, auth_headers, sample_miner_data):
        """Test getting specific miner details"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_latest_status_by_ip.return_value = sample_miner_data
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/miners/192.168.1.100', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            miner = data['data']
            assert miner['ip'] == sample_miner_data['ip']
            assert miner['hostname'] == sample_miner_data['hostname']
    
    def test_get_miner_not_found(self, client, auth_headers):
        """Test getting non-existent miner"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_latest_status_by_ip.return_value = None
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/miners/192.168.1.999', headers=auth_headers)
            
            data = assert_json_response(response, 404)
            assert_error_response(data, 'MINER_NOT_FOUND')
    
    def test_update_miner_settings(self, client, auth_headers):
        """Test updating miner settings"""
        settings_data = {
            'frequency': 800,
            'core_voltage': 1200,
            'autofanspeed': True
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_miner_service = Mock()
            mock_miner_service.set_miner_settings.return_value = True
            mock_container.return_value.get_miner_service.return_value = mock_miner_service
            
            response = client.put('/api/v1/miners/192.168.1.100/settings',
                                headers=auth_headers,
                                json=settings_data)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['frequency'] == settings_data['frequency']
            assert data['data']['core_voltage'] == settings_data['core_voltage']
            
            # Verify service was called correctly
            mock_miner_service.set_miner_settings.assert_called_once_with(
                '192.168.1.100', 800, 1200, True
            )
    
    def test_update_miner_settings_failure(self, client, auth_headers):
        """Test miner settings update failure"""
        settings_data = {
            'frequency': 800,
            'core_voltage': 1200,
            'autofanspeed': True
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_miner_service = Mock()
            mock_miner_service.set_miner_settings.return_value = False
            mock_container.return_value.get_miner_service.return_value = mock_miner_service
            
            response = client.put('/api/v1/miners/192.168.1.100/settings',
                                headers=auth_headers,
                                json=settings_data)
            
            data = assert_json_response(response, 500)
            assert_error_response(data, 'SETTINGS_UPDATE_FAILED')
    
    def test_restart_miner(self, client, auth_headers):
        """Test restarting miner"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_miner_service = Mock()
            mock_miner_service.restart_miner.return_value = True
            mock_container.return_value.get_miner_service.return_value = mock_miner_service
            
            response = client.post('/api/v1/miners/192.168.1.100/restart',
                                 headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['ip'] == '192.168.1.100'
            assert data['data']['action'] == 'restart'
            
            mock_miner_service.restart_miner.assert_called_once_with('192.168.1.100')
    
    def test_get_miners_summary(self, client, auth_headers):
        """Test getting miners summary"""
        summary_data = {
            'total_miners': 5,
            'online_miners': 4,
            'offline_miners': 1,
            'total_hashrate': 1942.8,
            'total_power': 64.2,
            'total_efficiency': 30.3,
            'average_temperature': 67.2,
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_miner_service = Mock()
            mock_miner_service.get_miners_summary.return_value = summary_data
            mock_container.return_value.get_miner_service.return_value = mock_miner_service
            
            response = client.get('/api/v1/miners/summary', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            summary = data['data']
            assert summary['total_miners'] == 5
            assert summary['online_miners'] == 4
            assert summary['total_hashrate'] == 1942.8


class TestBenchmarksAPI:
    """Test benchmark API endpoints"""
    
    def test_start_benchmark(self, client, auth_headers, sample_benchmark_data):
        """Test starting single benchmark"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_benchmark_service = Mock()
            mock_benchmark_service.start_benchmark.return_value = True
            mock_container.return_value.get_benchmark_service.return_value = mock_benchmark_service
            
            response = client.post('/api/v1/benchmarks',
                                 headers=auth_headers,
                                 json=sample_benchmark_data)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['ip'] == sample_benchmark_data['ip']
            assert data['data']['status'] == 'started'
            
            mock_benchmark_service.start_benchmark.assert_called_once_with(
                '192.168.1.100', 800, 1200, 600
            )
    
    def test_start_multi_benchmark(self, client, auth_headers):
        """Test starting multi-miner benchmark"""
        benchmark_data = {
            'ips': ['192.168.1.100', '192.168.1.101'],
            'frequency': 800,
            'core_voltage': 1200,
            'duration': 600
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_benchmark_service = Mock()
            mock_benchmark_service.start_multi_benchmark.return_value = ['192.168.1.100', '192.168.1.101']
            mock_container.return_value.get_benchmark_service.return_value = mock_benchmark_service
            
            response = client.post('/api/v1/benchmarks/multi',
                                 headers=auth_headers,
                                 json=benchmark_data)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['started_ips'] == ['192.168.1.100', '192.168.1.101']
            assert data['data']['status'] == 'started'
    
    def test_get_benchmark_status(self, client, auth_headers):
        """Test getting benchmark status"""
        status_data = {
            'active_benchmarks': ['192.168.1.100', '192.168.1.101'],
            'total_active': 2
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_benchmark_service = Mock()
            mock_benchmark_service.get_benchmark_status.return_value = status_data
            mock_container.return_value.get_benchmark_service.return_value = mock_benchmark_service
            
            response = client.get('/api/v1/benchmarks/status', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            status = data['data']
            assert status['active_benchmarks'] == ['192.168.1.100', '192.168.1.101']
            assert status['total_active'] == 2
    
    def test_get_benchmark_results(self, client, auth_headers):
        """Test getting benchmark results"""
        results_data = [
            ['192.168.1.100', 800, 1200, 485.2, 65.5, 26.3, datetime.now().isoformat(), 600]
        ]
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_benchmark_results.return_value = results_data
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/benchmarks/results', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            results = data['data']
            assert len(results) == 1
            
            result = results[0]
            assert result['ip'] == '192.168.1.100'
            assert result['frequency'] == 800
            assert result['core_voltage'] == 1200


class TestEventsAPI:
    """Test events API endpoints"""
    
    def test_get_events(self, client, auth_headers):
        """Test getting events"""
        events_data = [
            {
                'timestamp': datetime.now().isoformat(),
                'ip': '192.168.1.100',
                'event_type': 'BENCHMARK_COMPLETED',
                'message': 'Benchmark completed successfully',
                'severity': 'INFO'
            }
        ]
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_event_log.return_value = events_data
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/events', headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            events = data['data']
            assert len(events) == 1
            
            event = events[0]
            assert event['ip'] == '192.168.1.100'
            assert event['event_type'] == 'BENCHMARK_COMPLETED'
    
    def test_get_events_with_filters(self, client, auth_headers):
        """Test getting events with filters"""
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_db = Mock()
            mock_db.get_event_log.return_value = []
            mock_container.return_value.get_database_service.return_value = mock_db
            
            response = client.get('/api/v1/events?ip=192.168.1.100&event_type=BENCHMARK&severity=INFO', 
                                headers=auth_headers)
            
            data = assert_json_response(response, 200)
            
            # Check that filters were passed to service
            mock_db.get_event_log.assert_called_once_with(
                limit=100,
                ip_filter='192.168.1.100',
                event_type_filter='BENCHMARK'
            )


class TestHealthAPI:
    """Test health monitoring API endpoints"""
    
    def test_get_system_health(self, client):
        """Test getting system health (no auth required)"""
        health_data = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'total_checks': 8,
            'status_counts': {
                'healthy': 7,
                'degraded': 1,
                'unhealthy': 0,
                'unknown': 0
            },
            'checks': {
                'database': {
                    'component': 'database',
                    'status': 'healthy',
                    'message': 'Database is accessible',
                    'details': {},
                    'timestamp': datetime.now().isoformat(),
                    'duration_ms': 15.3
                }
            }
        }
        
        with patch('health.health_checks.get_health_manager') as mock_health:
            mock_health.return_value.get_overall_health.return_value = health_data
            
            response = client.get('/api/v1/health')
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            health = data['data']
            assert health['overall_status'] == 'healthy'
            assert health['total_checks'] == 8
    
    def test_get_component_health(self, client):
        """Test getting specific component health"""
        component_data = {
            'component': 'database',
            'status': 'healthy',
            'message': 'Database is accessible',
            'details': {'connection_info': {'pool_size': 10}},
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('health.health_checks.get_health_manager') as mock_health:
            mock_health.return_value.get_health_status.return_value = component_data
            
            response = client.get('/api/v1/health/database')
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            health = data['data']
            assert health['component'] == 'database'
            assert health['status'] == 'healthy'


class TestConfigAPI:
    """Test configuration API endpoints"""
    
    def test_update_config(self, client, auth_headers):
        """Test updating configuration"""
        config_data = {
            'key': 'settings.benchmark_interval_sec',
            'value': '3600'
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            mock_config_service = Mock()
            mock_container.return_value.get_config_service.return_value = mock_config_service
            
            response = client.put('/api/v1/config',
                                headers=auth_headers,
                                json=config_data)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['key'] == config_data['key']
            assert data['data']['value'] == config_data['value']
            
            mock_config_service.set.assert_called_once_with(
                'settings.benchmark_interval_sec', '3600'
            )


class TestAPIValidation:
    """Test API request validation"""
    
    def test_invalid_json(self, client, auth_headers):
        """Test invalid JSON request"""
        response = client.post('/api/v1/benchmarks',
                             headers=auth_headers,
                             data='invalid json',
                             content_type='application/json')
        
        data = assert_json_response(response, 400)
        assert_error_response(data)
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields"""
        # Missing frequency and core_voltage
        benchmark_data = {'ip': '192.168.1.100'}
        
        response = client.post('/api/v1/benchmarks',
                             headers=auth_headers,
                             json=benchmark_data)
        
        data = assert_json_response(response, 400)
        assert_error_response(data)
    
    def test_invalid_ip_format(self, client, auth_headers):
        """Test invalid IP address format"""
        benchmark_data = {
            'ip': 'invalid-ip',
            'frequency': 800,
            'core_voltage': 1200
        }
        
        response = client.post('/api/v1/benchmarks',
                             headers=auth_headers,
                             json=benchmark_data)
        
        data = assert_json_response(response, 400)
        assert_error_response(data)
    
    def test_out_of_range_values(self, client, auth_headers):
        """Test out of range values"""
        settings_data = {
            'frequency': 5000,  # Too high
            'core_voltage': 100  # Too low
        }
        
        response = client.put('/api/v1/miners/192.168.1.100/settings',
                            headers=auth_headers,
                            json=settings_data)
        
        data = assert_json_response(response, 400)
        assert_error_response(data)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    def test_complete_benchmark_workflow(self, client, auth_headers):
        """Test complete benchmark workflow"""
        benchmark_data = {
            'ip': '192.168.1.100',
            'frequency': 800,
            'core_voltage': 1200,
            'duration': 600
        }
        
        with patch('services.service_container_v2.get_container_v2') as mock_container:
            # Setup mocks
            mock_benchmark_service = Mock()
            mock_benchmark_service.start_benchmark.return_value = True
            mock_benchmark_service.get_benchmark_status.return_value = {
                'active_benchmarks': ['192.168.1.100'],
                'total_active': 1
            }
            
            mock_db = Mock()
            mock_db.get_benchmark_results.return_value = [
                ['192.168.1.100', 800, 1200, 485.2, 65.5, 26.3, datetime.now().isoformat(), 600]
            ]
            
            mock_container.return_value.get_benchmark_service.return_value = mock_benchmark_service
            mock_container.return_value.get_database_service.return_value = mock_db
            
            # 1. Start benchmark
            start_response = client.post('/api/v1/benchmarks',
                                       headers=auth_headers,
                                       json=benchmark_data)
            
            start_data = assert_json_response(start_response, 200)
            assert start_data['data']['status'] == 'started'
            
            # 2. Check status
            status_response = client.get('/api/v1/benchmarks/status',
                                       headers=auth_headers)
            
            status_data = assert_json_response(status_response, 200)
            assert '192.168.1.100' in status_data['data']['active_benchmarks']
            
            # 3. Get results
            results_response = client.get('/api/v1/benchmarks/results',
                                        headers=auth_headers)
            
            results_data = assert_json_response(results_response, 200)
            assert len(results_data['data']) == 1
            assert results_data['data'][0]['ip'] == '192.168.1.100'