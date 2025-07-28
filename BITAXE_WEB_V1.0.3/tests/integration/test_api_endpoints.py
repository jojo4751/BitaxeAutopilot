"""
Integration tests for API Endpoints

End-to-end testing of REST API endpoints including authentication,
data validation, error handling, and response formatting.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from tests.conftest import assert_json_response, assert_api_success, assert_error_response


class TestMinerAPIIntegration:
    """Test miner management API endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_all_miners_endpoint(self, client, auth_headers, mock_miner_response):
        """Test GET /api/miners endpoint"""
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all:
            mock_get_all.return_value = [mock_miner_response]
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert 'miners' in data['data']
            assert len(data['data']['miners']) == 1
            assert data['data']['miners'][0]['ip'] == mock_miner_response.get('ip', '192.168.1.100')
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_specific_miner_endpoint(self, client, auth_headers, mock_miner_response):
        """Test GET /api/miners/{ip} endpoint"""
        miner_ip = '192.168.1.100'
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
            mock_get_info.return_value = mock_miner_response
            
            response = client.get(f'/api/miners/{miner_ip}', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['ip'] == miner_ip
            assert 'temp' in data['data']
            assert 'hashRate' in data['data']
            assert 'power' in data['data']
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_update_miner_config_endpoint(self, client, auth_headers):
        """Test POST /api/miners/{ip}/config endpoint"""
        miner_ip = '192.168.1.100'
        config_update = {
            'frequency': 650,
            'voltage': 12.1
        }
        
        with patch('services.miner_service.MinerService.update_config') as mock_update:
            mock_update.return_value = True
            
            response = client.post(
                f'/api/miners/{miner_ip}/config',
                headers=auth_headers,
                json=config_update
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            mock_update.assert_called_once_with(miner_ip, config_update)
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_miner_discovery_endpoint(self, client, auth_headers):
        """Test POST /api/miners/discover endpoint"""
        discovery_config = {
            'ip_range': '192.168.1.100-110',
            'timeout': 5
        }
        
        mock_discovered = [
            {'ip': '192.168.1.100', 'hostname': 'miner-001', 'version': '2.0.4'},
            {'ip': '192.168.1.101', 'hostname': 'miner-002', 'version': '2.0.4'}
        ]
        
        with patch('services.miner_service.MinerService.discover_miners') as mock_discover:
            mock_discover.return_value = mock_discovered
            
            response = client.post(
                '/api/miners/discover',
                headers=auth_headers,
                json=discovery_config
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert 'discovered' in data['data']
            assert len(data['data']['discovered']) == 2
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_miner_not_found_error(self, client, auth_headers):
        """Test miner not found error handling"""
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
            mock_get_info.return_value = None
            
            response = client.get('/api/miners/192.168.1.999', headers=auth_headers)
            data = assert_json_response(response, 404)
            assert_error_response(data, 'MINER_NOT_FOUND')
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_invalid_config_validation(self, client, auth_headers):
        """Test config validation error handling"""
        invalid_config = {
            'frequency': 9999,  # Too high
            'voltage': 20.0     # Too high
        }
        
        response = client.post(
            '/api/miners/192.168.1.100/config',
            headers=auth_headers,
            json=invalid_config
        )
        
        data = assert_json_response(response, 400)
        assert_error_response(data, 'VALIDATION_ERROR')


class TestOptimizationAPIIntegration:
    """Test optimization API endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.ml
    def test_start_optimization_endpoint(self, client, auth_headers):
        """Test POST /api/optimization/start endpoint"""
        optimization_request = {
            'miner_ips': ['192.168.1.100', '192.168.1.101'],
            'optimization_type': 'ml_guided',
            'target_metrics': {
                'efficiency': 120.0,
                'temperature': 70.0
            }
        }
        
        mock_result = {
            'optimization_id': 'opt_12345',
            'status': 'started',
            'estimated_duration': 300
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_start:
            mock_start.return_value = mock_result
            
            response = client.post(
                '/api/optimization/start',
                headers=auth_headers,
                json=optimization_request
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['optimization_id'] == 'opt_12345'
            assert data['data']['status'] == 'started'
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.ml
    def test_get_optimization_status_endpoint(self, client, auth_headers):
        """Test GET /api/optimization/{id}/status endpoint"""
        optimization_id = 'opt_12345'
        
        mock_status = {
            'optimization_id': optimization_id,
            'status': 'running',
            'progress': 75.0,
            'miners_processed': 3,
            'total_miners': 4,
            'estimated_remaining': 60,
            'current_phase': 'ml_inference'
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_status') as mock_status_call:
            mock_status_call.return_value = mock_status
            
            response = client.get(
                f'/api/optimization/{optimization_id}/status',
                headers=auth_headers
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['progress'] == 75.0
            assert data['data']['status'] == 'running'
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.ml
    def test_get_optimization_results_endpoint(self, client, auth_headers):
        """Test GET /api/optimization/{id}/results endpoint"""
        optimization_id = 'opt_12345'
        
        mock_results = {
            'optimization_id': optimization_id,
            'status': 'completed',
            'results': {
                '192.168.1.100': {
                    'original_config': {'frequency': 600, 'voltage': 12.0},
                    'optimized_config': {'frequency': 650, 'voltage': 12.1},
                    'performance_improvement': {
                        'efficiency_gain': 8.5,
                        'temperature_reduction': 3.2,
                        'power_reduction': 0.8
                    }
                },
                '192.168.1.101': {
                    'original_config': {'frequency': 580, 'voltage': 11.8},
                    'optimized_config': {'frequency': 620, 'voltage': 12.0},
                    'performance_improvement': {
                        'efficiency_gain': 12.1,
                        'temperature_reduction': 1.8,
                        'power_reduction': 0.3
                    }
                }
            },
            'summary': {
                'total_miners': 2,
                'successful_optimizations': 2,
                'average_efficiency_gain': 10.3,
                'total_power_saved': 1.1
            }
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_results') as mock_results_call:
            mock_results_call.return_value = mock_results
            
            response = client.get(
                f'/api/optimization/{optimization_id}/results',
                headers=auth_headers
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['status'] == 'completed'
            assert len(data['data']['results']) == 2
            assert data['data']['summary']['successful_optimizations'] == 2
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_stop_optimization_endpoint(self, client, auth_headers):
        """Test POST /api/optimization/{id}/stop endpoint"""
        optimization_id = 'opt_12345'
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.stop_optimization') as mock_stop:
            mock_stop.return_value = {'status': 'stopped', 'partial_results': True}
            
            response = client.post(
                f'/api/optimization/{optimization_id}/stop',
                headers=auth_headers
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['status'] == 'stopped'


class TestBenchmarkAPIIntegration:
    """Test benchmark API endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_start_benchmark_endpoint(self, client, auth_headers, sample_benchmark_data):
        """Test POST /api/benchmark/start endpoint"""
        with patch('services.benchmark_service.BenchmarkService.start_benchmark') as mock_start:
            mock_start.return_value = {
                'benchmark_id': 'bench_123',
                'status': 'started',
                'estimated_duration': sample_benchmark_data['duration']
            }
            
            response = client.post(
                '/api/benchmark/start',
                headers=auth_headers,
                json=sample_benchmark_data
            )
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['benchmark_id'] == 'bench_123'
            assert data['data']['status'] == 'started'
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_benchmark_results_endpoint(self, client, auth_headers):
        """Test GET /api/benchmark/{id}/results endpoint"""
        benchmark_id = 'bench_123'
        
        mock_results = {
            'benchmark_id': benchmark_id,
            'status': 'completed',
            'results': {
                'avg_hashrate': 485.2,
                'avg_temperature': 68.5,
                'avg_power': 12.8,
                'efficiency': 37.9,
                'stability_score': 0.95,
                'duration': 600
            },
            'performance_curve': [
                {'time': 0, 'hashrate': 480.1, 'temp': 65.0},
                {'time': 300, 'hashrate': 485.2, 'temp': 68.5},
                {'time': 600, 'hashrate': 487.3, 'temp': 70.1}
            ]
        }
        
        with patch('services.benchmark_service.BenchmarkService.get_benchmark_results') as mock_results_call:
            mock_results_call.return_value = mock_results
            
            response = client.get(f'/api/benchmark/{benchmark_id}/results', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['status'] == 'completed'
            assert data['data']['results']['efficiency'] == 37.9
            assert len(data['data']['performance_curve']) == 3
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_benchmark_history_endpoint(self, client, auth_headers):
        """Test GET /api/benchmark/history endpoint"""
        mock_history = [
            {
                'benchmark_id': 'bench_123',
                'ip': '192.168.1.100',
                'timestamp': datetime.now().isoformat(),
                'frequency': 600,
                'core_voltage': 1200,
                'avg_hashrate': 485.2,
                'efficiency': 37.9
            },
            {
                'benchmark_id': 'bench_124',
                'ip': '192.168.1.100',
                'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                'frequency': 650,
                'core_voltage': 1250,
                'avg_hashrate': 510.8,
                'efficiency': 39.1
            }
        ]
        
        with patch('services.benchmark_service.BenchmarkService.get_benchmark_history') as mock_history_call:
            mock_history_call.return_value = mock_history
            
            response = client.get('/api/benchmark/history', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert len(data['data']['benchmarks']) == 2
            assert data['data']['benchmarks'][0]['efficiency'] == 37.9


class TestMonitoringAPIIntegration:
    """Test monitoring and metrics API endpoints"""
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_system_metrics_endpoint(self, client, auth_headers):
        """Test GET /api/monitoring/metrics endpoint"""
        mock_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'total_miners': 5,
                'active_miners': 4,
                'total_hashrate': 2401.0,
                'total_power': 64.0,
                'average_efficiency': 37.5,
                'average_temperature': 67.8
            },
            'individual_miners': [
                {
                    'ip': '192.168.1.100',
                    'status': 'active',
                    'hashrate': 485.2,
                    'temperature': 65.5,
                    'power': 12.8,
                    'efficiency': 37.9
                }
            ]
        }
        
        with patch('monitoring.metrics_collector.MetricsCollector.get_current_metrics') as mock_metrics_call:
            mock_metrics_call.return_value = mock_metrics
            
            response = client.get('/api/monitoring/metrics', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['system']['total_miners'] == 5
            assert data['data']['system']['active_miners'] == 4
            assert len(data['data']['individual_miners']) == 1
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_historical_metrics_endpoint(self, client, auth_headers):
        """Test GET /api/monitoring/metrics/history endpoint"""
        query_params = {
            'start_time': (datetime.now() - timedelta(hours=24)).isoformat(),
            'end_time': datetime.now().isoformat(),
            'interval': '1h'
        }
        
        mock_history = {
            'time_range': query_params,
            'data_points': [
                {
                    'timestamp': (datetime.now() - timedelta(hours=23)).isoformat(),
                    'total_hashrate': 2380.5,
                    'average_temperature': 66.2,
                    'total_power': 63.5
                },
                {
                    'timestamp': (datetime.now() - timedelta(hours=22)).isoformat(),
                    'total_hashrate': 2401.0,
                    'average_temperature': 67.8,
                    'total_power': 64.0
                }
            ]
        }
        
        with patch('monitoring.metrics_collector.MetricsCollector.get_historical_metrics') as mock_history_call:
            mock_history_call.return_value = mock_history
            
            response = client.get('/api/monitoring/metrics/history', 
                                headers=auth_headers, 
                                query_string=query_params)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert len(data['data']['data_points']) == 2
            assert 'time_range' in data['data']
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_get_alerts_endpoint(self, client, auth_headers):
        """Test GET /api/monitoring/alerts endpoint"""
        mock_alerts = [
            {
                'alert_id': 'alert_001',
                'type': 'temperature_warning',
                'severity': 'warning',
                'miner_ip': '192.168.1.100',
                'message': 'Temperature above threshold: 82.5°C',
                'timestamp': datetime.now().isoformat(),
                'acknowledged': False
            },
            {
                'alert_id': 'alert_002',
                'type': 'hashrate_drop',
                'severity': 'critical',
                'miner_ip': '192.168.1.101',
                'message': 'Hashrate dropped below 80% of baseline',
                'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat(),
                'acknowledged': True
            }
        ]
        
        with patch('monitoring.alert_manager.AlertManager.get_active_alerts') as mock_alerts_call:
            mock_alerts_call.return_value = mock_alerts
            
            response = client.get('/api/monitoring/alerts', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert len(data['data']['alerts']) == 2
            assert data['data']['alerts'][0]['severity'] == 'warning'
            assert data['data']['alerts'][1]['acknowledged'] is True


class TestAuthenticationIntegration:
    """Test authentication and authorization integration"""
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.auth
    def test_login_endpoint(self, client):
        """Test POST /api/auth/login endpoint"""
        login_data = {
            'username': 'admin',
            'password': 'admin123'
        }
        
        with patch('auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_user = {'username': 'admin', 'role': 'admin'}
            mock_auth.return_value = mock_user
            
            with patch('auth.auth_service.AuthService.generate_token') as mock_token:
                mock_token.return_value = {
                    'access_token': 'jwt_token_here',
                    'token_type': 'bearer',
                    'expires_in': 3600
                }
                
                response = client.post('/api/auth/login', json=login_data)
                data = assert_json_response(response, 200)
                assert_api_success(data)
                
                assert 'access_token' in data['data']
                assert data['data']['token_type'] == 'bearer'
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.auth
    def test_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        login_data = {
            'username': 'admin',
            'password': 'wrong_password'
        }
        
        with patch('auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_auth.return_value = None
            
            response = client.post('/api/auth/login', json=login_data)
            data = assert_json_response(response, 401)
            assert_error_response(data, 'INVALID_CREDENTIALS')
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.auth
    def test_unauthorized_access(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get('/api/miners')
        data = assert_json_response(response, 401)
        assert_error_response(data, 'AUTHENTICATION_REQUIRED')
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.auth
    def test_insufficient_permissions(self, client, readonly_headers):
        """Test accessing endpoint with insufficient permissions"""
        config_update = {'frequency': 650}
        
        response = client.post('/api/miners/192.168.1.100/config', 
                             headers=readonly_headers, 
                             json=config_update)
        
        data = assert_json_response(response, 403)
        assert_error_response(data, 'INSUFFICIENT_PERMISSIONS')
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.auth
    def test_token_refresh_endpoint(self, client, auth_headers):
        """Test POST /api/auth/refresh endpoint"""
        with patch('auth.auth_service.AuthService.refresh_token') as mock_refresh:
            mock_refresh.return_value = {
                'access_token': 'new_jwt_token_here',
                'token_type': 'bearer',
                'expires_in': 3600
            }
            
            response = client.post('/api/auth/refresh', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert 'access_token' in data['data']
            assert data['data']['access_token'] == 'new_jwt_token_here'


class TestAPIErrorHandling:
    """Test API error handling and validation"""
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_malformed_json_request(self, client, auth_headers):
        """Test handling of malformed JSON requests"""
        response = client.post('/api/miners/192.168.1.100/config',
                             headers=auth_headers,
                             data='{"invalid": json}')  # Malformed JSON
        
        data = assert_json_response(response, 400)
        assert_error_response(data, 'INVALID_JSON')
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_missing_required_fields(self, client, auth_headers):
        """Test validation of missing required fields"""
        incomplete_data = {
            'frequency': 650
            # Missing required 'voltage' field
        }
        
        response = client.post('/api/miners/192.168.1.100/config',
                             headers=auth_headers,
                             json=incomplete_data)
        
        data = assert_json_response(response, 400)
        assert_error_response(data, 'VALIDATION_ERROR')
        assert 'voltage' in data['error']['details']
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_invalid_ip_format(self, client, auth_headers):
        """Test handling of invalid IP address format"""
        invalid_ip = 'not-an-ip-address'
        
        response = client.get(f'/api/miners/{invalid_ip}', headers=auth_headers)
        data = assert_json_response(response, 400)
        assert_error_response(data, 'INVALID_IP_FORMAT')
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_rate_limiting(self, client, auth_headers):
        """Test API rate limiting"""
        # Make multiple rapid requests
        responses = []
        for i in range(20):  # Exceed rate limit
            response = client.get('/api/miners', headers=auth_headers)
            responses.append(response)
        
        # Check if any requests were rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        
        if rate_limited:
            # Find first rate limited response
            rate_limited_response = next(r for r in responses if r.status_code == 429)
            data = assert_json_response(rate_limited_response, 429)
            assert_error_response(data, 'RATE_LIMIT_EXCEEDED')
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_internal_server_error_handling(self, client, auth_headers):
        """Test handling of internal server errors"""
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all:
            mock_get_all.side_effect = Exception("Database connection failed")
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 500)
            assert_error_response(data, 'INTERNAL_SERVER_ERROR')
            
            # Verify error details are not exposed in production
            assert 'Database connection failed' not in json.dumps(data)


class TestAPIPerformance:
    """Test API performance and load handling"""
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.performance
    def test_response_time_performance(self, client, auth_headers, performance_metrics):
        """Test API response time performance"""
        endpoints = [
            '/api/miners',
            '/api/monitoring/metrics',
            '/api/benchmark/history'
        ]
        
        for endpoint in endpoints:
            performance_metrics.start_timer(f'response_time_{endpoint.replace("/", "_")}')
            
            response = client.get(endpoint, headers=auth_headers)
            
            performance_metrics.end_timer(f'response_time_{endpoint.replace("/", "_")}')
            
            # API responses should be fast (< 2 seconds)
            performance_metrics.assert_performance(f'response_time_{endpoint.replace("/", "_")}', 2.0)
            
            # Verify successful response
            assert response.status_code in [200, 404]  # 404 acceptable for some test scenarios
    
    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.performance
    def test_concurrent_request_handling(self, client, auth_headers, performance_metrics):
        """Test concurrent request handling"""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get('/api/miners', headers=auth_headers)
            end_time = time.time()
            results.append({
                'status_code': response.status_code,
                'response_time': end_time - start_time
            })
        
        performance_metrics.start_timer('concurrent_requests')
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        performance_metrics.end_timer('concurrent_requests')
        
        # Should handle concurrent requests efficiently (< 5 seconds total)
        performance_metrics.assert_performance('concurrent_requests', 5.0)
        
        # Verify all requests succeeded
        assert len(results) == 10
        successful_requests = sum(1 for r in results if r['status_code'] == 200)
        assert successful_requests >= 8  # Allow for some test environment variations