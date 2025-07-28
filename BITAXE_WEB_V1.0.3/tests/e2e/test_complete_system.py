"""
End-to-End System Tests

Complete system testing including full ML optimization workflows,
user scenarios, and real-world system behavior validation.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json
import tempfile
import os

from tests.conftest import assert_json_response, assert_api_success


class TestCompleteOptimizationWorkflow:
    """Test complete ML optimization workflow end-to-end"""
    
    @pytest.mark.e2e
    @pytest.mark.ml
    async def test_full_ml_optimization_cycle(self, client, auth_headers, sample_miner_telemetry, sample_weather_data):
        """Test complete ML optimization cycle from discovery to results"""
        
        # Step 1: Discover miners
        with patch('services.miner_service.MinerService.discover_miners') as mock_discover:
            mock_discover.return_value = [
                {'ip': '192.168.1.100', 'hostname': 'miner-001', 'version': '2.0.4'},
                {'ip': '192.168.1.101', 'hostname': 'miner-002', 'version': '2.0.4'}
            ]
            
            response = client.post('/api/miners/discover', 
                                 headers=auth_headers,
                                 json={'ip_range': '192.168.1.100-110', 'timeout': 5})
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            assert len(data['data']['discovered']) == 2
        
        # Step 2: Get initial miner status
        mock_initial_status = {
            'ip': '192.168.1.100',
            'hostname': 'miner-001',
            'temp': 75.0,  # Running hot
            'hashRate': 450.0,  # Low hashrate
            'power': 15.0,  # High power consumption
            'voltage': 12.5,
            'frequency': 700,
            'efficiency': 30.0,  # Poor efficiency
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
            mock_get_info.return_value = mock_initial_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            initial_efficiency = data['data']['efficiency']
            initial_temp = data['data']['temp']
            assert initial_efficiency == 30.0
            assert initial_temp == 75.0
        
        # Step 3: Start ML-guided optimization
        optimization_request = {
            'miner_ips': ['192.168.1.100', '192.168.1.101'],
            'optimization_type': 'ml_guided',
            'target_metrics': {
                'efficiency': 40.0,  # Target 40 TH/s per kW
                'temperature': 70.0  # Target 70°C
            },
            'safety_limits': {
                'max_temperature': 85.0,
                'min_efficiency': 25.0
            }
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_start_opt:
            mock_start_opt.return_value = {
                'optimization_id': 'opt_e2e_test_001',
                'status': 'started',
                'estimated_duration': 300,
                'miners_count': 2
            }
            
            response = client.post('/api/optimization/start',
                                 headers=auth_headers,
                                 json=optimization_request)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            optimization_id = data['data']['optimization_id']
            assert optimization_id == 'opt_e2e_test_001'
        
        # Step 4: Monitor optimization progress
        progress_updates = [
            {'status': 'running', 'progress': 25.0, 'current_phase': 'data_collection'},
            {'status': 'running', 'progress': 50.0, 'current_phase': 'feature_engineering'},
            {'status': 'running', 'progress': 75.0, 'current_phase': 'ml_inference'},
            {'status': 'completed', 'progress': 100.0, 'current_phase': 'optimization_complete'}
        ]
        
        for i, progress in enumerate(progress_updates):
            with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_status') as mock_status:
                mock_status.return_value = {
                    'optimization_id': optimization_id,
                    'status': progress['status'],
                    'progress': progress['progress'],
                    'current_phase': progress['current_phase'],
                    'miners_processed': min(i + 1, 2),
                    'total_miners': 2,
                    'estimated_remaining': max(0, 60 - (i * 20))
                }
                
                response = client.get(f'/api/optimization/{optimization_id}/status',
                                    headers=auth_headers)
                
                data = assert_json_response(response, 200)
                assert_api_success(data)
                
                assert data['data']['progress'] == progress['progress']
                assert data['data']['current_phase'] == progress['current_phase']
        
        # Step 5: Get optimization results
        mock_optimization_results = {
            'optimization_id': optimization_id,
            'status': 'completed',
            'completion_time': datetime.now().isoformat(),
            'results': {
                '192.168.1.100': {
                    'original_config': {
                        'frequency': 700,
                        'voltage': 12.5
                    },
                    'optimized_config': {
                        'frequency': 620,  # Reduced for better efficiency
                        'voltage': 12.0    # Reduced for lower power
                    },
                    'performance_improvement': {
                        'efficiency_gain': 15.5,  # 30.0 -> 45.5 TH/s/kW
                        'temperature_reduction': 8.0,  # 75.0 -> 67.0°C
                        'power_reduction': 2.5,  # 15.0 -> 12.5W
                        'hashrate_change': -20.0  # Slight reduction for efficiency
                    },
                    'confidence_score': 0.87,
                    'safety_validated': True
                },
                '192.168.1.101': {
                    'original_config': {
                        'frequency': 650,
                        'voltage': 12.0
                    },
                    'optimized_config': {
                        'frequency': 680,
                        'voltage': 12.2
                    },
                    'performance_improvement': {
                        'efficiency_gain': 12.0,
                        'temperature_reduction': 5.0,
                        'power_reduction': 1.0,
                        'hashrate_change': 30.0
                    },
                    'confidence_score': 0.92,
                    'safety_validated': True
                }
            },
            'summary': {
                'total_miners': 2,
                'successful_optimizations': 2,
                'failed_optimizations': 0,
                'average_efficiency_gain': 13.75,
                'total_power_saved': 3.5,
                'total_temp_reduction': 6.5,
                'optimization_duration': 285
            }
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_results') as mock_results:
            mock_results.return_value = mock_optimization_results
            
            response = client.get(f'/api/optimization/{optimization_id}/results',
                                headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['status'] == 'completed'
            assert data['data']['summary']['successful_optimizations'] == 2
            assert data['data']['summary']['average_efficiency_gain'] > 10.0
        
        # Step 6: Apply optimizations to miners
        for miner_ip, result in mock_optimization_results['results'].items():
            optimized_config = result['optimized_config']
            
            with patch('services.miner_service.MinerService.update_config') as mock_update:
                mock_update.return_value = True
                
                response = client.post(f'/api/miners/{miner_ip}/config',
                                     headers=auth_headers,
                                     json=optimized_config)
                
                data = assert_json_response(response, 200)
                assert_api_success(data)
        
        # Step 7: Verify post-optimization status
        mock_optimized_status = {
            'ip': '192.168.1.100',
            'hostname': 'miner-001',
            'temp': 67.0,  # Reduced temperature
            'hashRate': 430.0,  # Slightly lower but more efficient
            'power': 12.5,  # Reduced power consumption
            'voltage': 12.0,  # Optimized voltage
            'frequency': 620,  # Optimized frequency
            'efficiency': 34.4,  # Improved efficiency (430/12.5)
            'timestamp': datetime.now().isoformat()
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info_final:
            mock_get_info_final.return_value = mock_optimized_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            final_efficiency = data['data']['efficiency']
            final_temp = data['data']['temp']
            
            # Verify improvements
            assert final_efficiency > initial_efficiency
            assert final_temp < initial_temp
            assert final_temp <= 70.0  # Met temperature target


class TestSystemReliabilityScenarios:
    """Test system reliability and error handling scenarios"""
    
    @pytest.mark.e2e
    async def test_network_failure_recovery(self, client, auth_headers):
        """Test system behavior during network failures"""
        
        # Step 1: Normal operation
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all:
            mock_get_all.return_value = [
                {'ip': '192.168.1.100', 'status': 'active', 'temp': 65.0}
            ]
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
        
        # Step 2: Simulate network failure
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all_fail:
            mock_get_all_fail.side_effect = ConnectionError("Network unreachable")
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 503)  # Service unavailable
            assert data['success'] is False
            assert 'network' in data['error']['message'].lower()
        
        # Step 3: Recovery
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all_recovery:
            mock_get_all_recovery.return_value = [
                {'ip': '192.168.1.100', 'status': 'active', 'temp': 65.0}
            ]
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
    
    @pytest.mark.e2e
    async def test_high_temperature_emergency_response(self, client, auth_headers):
        """Test emergency response to high temperature conditions"""
        
        # Step 1: Normal operation
        normal_status = {
            'ip': '192.168.1.100',
            'temp': 70.0,
            'hashRate': 485.0,
            'power': 12.0,
            'frequency': 600,
            'voltage': 12.0
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
            mock_get_info.return_value = normal_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
        
        # Step 2: Critical temperature detected
        critical_status = {
            'ip': '192.168.1.100',
            'temp': 95.0,  # Critical temperature
            'hashRate': 520.0,
            'power': 18.0,
            'frequency': 750,
            'voltage': 13.0
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_critical:
            mock_get_critical.return_value = critical_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            # Verify critical temperature is reported
            assert data['data']['temp'] == 95.0
        
        # Step 3: Emergency optimization triggered
        emergency_optimization = {
            'miner_ips': ['192.168.1.100'],
            'optimization_type': 'emergency_cooling',
            'priority': 'critical'
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_emergency:
            mock_emergency.return_value = {
                'optimization_id': 'emergency_001',
                'status': 'started',
                'priority': 'critical',
                'estimated_duration': 30  # Fast emergency optimization
            }
            
            response = client.post('/api/optimization/start',
                                 headers=auth_headers,
                                 json=emergency_optimization)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
        
        # Step 4: Emergency configuration applied
        emergency_config = {
            'frequency': 500,  # Drastically reduced
            'voltage': 11.5    # Reduced voltage
        }
        
        with patch('services.miner_service.MinerService.update_config') as mock_emergency_config:
            mock_emergency_config.return_value = True
            
            response = client.post('/api/miners/192.168.1.100/config',
                                 headers=auth_headers,
                                 json=emergency_config)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
        
        # Step 5: Verify temperature reduction
        recovered_status = {
            'ip': '192.168.1.100',
            'temp': 72.0,  # Temperature reduced
            'hashRate': 380.0,  # Lower hashrate but safe
            'power': 10.5,  # Reduced power
            'frequency': 500,
            'voltage': 11.5
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_recovered:
            mock_get_recovered.return_value = recovered_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            # Verify temperature is back to safe levels
            assert data['data']['temp'] < 85.0
    
    @pytest.mark.e2e
    async def test_multiple_concurrent_optimizations(self, client, auth_headers):
        """Test handling multiple concurrent optimization requests"""
        
        # Step 1: Start multiple optimizations simultaneously
        optimization_requests = [
            {
                'miner_ips': ['192.168.1.100'],
                'optimization_type': 'efficiency',
                'request_id': 'opt_req_1'
            },
            {
                'miner_ips': ['192.168.1.101'],
                'optimization_type': 'temperature',
                'request_id': 'opt_req_2'
            },
            {
                'miner_ips': ['192.168.1.102'],
                'optimization_type': 'power',
                'request_id': 'opt_req_3'
            }
        ]
        
        optimization_ids = []
        
        for i, request in enumerate(optimization_requests):
            with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_start:
                mock_start.return_value = {
                    'optimization_id': f'concurrent_opt_{i+1}',
                    'status': 'queued',  # May be queued due to resource limits
                    'estimated_duration': 120,
                    'queue_position': i + 1 if i > 0 else None
                }
                
                response = client.post('/api/optimization/start',
                                     headers=auth_headers,
                                     json=request)
                
                data = assert_json_response(response, 200)
                assert_api_success(data)
                optimization_ids.append(data['data']['optimization_id'])
        
        # Step 2: Monitor all optimizations
        for opt_id in optimization_ids:
            with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_status') as mock_status:
                mock_status.return_value = {
                    'optimization_id': opt_id,
                    'status': 'running',
                    'progress': 50.0
                }
                
                response = client.get(f'/api/optimization/{opt_id}/status',
                                    headers=auth_headers)
                
                data = assert_json_response(response, 200)
                assert_api_success(data)
                assert data['data']['optimization_id'] == opt_id


class TestUserWorkflows:
    """Test complete user workflows and scenarios"""
    
    @pytest.mark.e2e
    def test_new_user_onboarding_workflow(self, client):
        """Test complete new user onboarding workflow"""
        
        # Step 1: User login
        login_data = {'username': 'newuser', 'password': 'newpass123'}
        
        with patch('auth.auth_service.AuthService.authenticate_user') as mock_auth:
            mock_user = {'username': 'newuser', 'role': 'operator'}
            mock_auth.return_value = mock_user
            
            with patch('auth.auth_service.AuthService.generate_token') as mock_token:
                mock_token.return_value = {
                    'access_token': 'new_user_token',
                    'token_type': 'bearer',
                    'expires_in': 3600
                }
                
                response = client.post('/api/auth/login', json=login_data)
                data = assert_json_response(response, 200)
                assert_api_success(data)
                
                auth_headers = {'Authorization': f'Bearer {data["data"]["access_token"]}'}
        
        # Step 2: System overview (first thing new users typically check)
        with patch('monitoring.metrics_collector.MetricsCollector.get_current_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'system': {
                    'total_miners': 0,  # No miners initially
                    'active_miners': 0,
                    'total_hashrate': 0.0
                }
            }
            
            response = client.get('/api/monitoring/metrics', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            assert data['data']['system']['total_miners'] == 0
        
        # Step 3: Miner discovery
        with patch('services.miner_service.MinerService.discover_miners') as mock_discover:
            mock_discover.return_value = [
                {'ip': '192.168.1.100', 'hostname': 'bitaxe-new', 'version': '2.0.4'}
            ]
            
            response = client.post('/api/miners/discover',
                                 headers=auth_headers,
                                 json={'ip_range': '192.168.1.1-254'})
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            assert len(data['data']['discovered']) == 1
        
        # Step 4: First miner status check
        mock_miner_status = {
            'ip': '192.168.1.100',
            'temp': 68.5,
            'hashRate': 485.2,
            'power': 12.8,
            'status': 'active'
        }
        
        with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
            mock_get_info.return_value = mock_miner_status
            
            response = client.get('/api/miners/192.168.1.100', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            assert data['data']['status'] == 'active'
        
        # Step 5: First optimization attempt
        optimization_request = {
            'miner_ips': ['192.168.1.100'],
            'optimization_type': 'balanced'
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_opt:
            mock_opt.return_value = {
                'optimization_id': 'newuser_first_opt',
                'status': 'started'
            }
            
            response = client.post('/api/optimization/start',
                                 headers=auth_headers,
                                 json=optimization_request)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
    
    @pytest.mark.e2e
    def test_power_user_advanced_workflow(self, client, auth_headers):
        """Test advanced power user workflow with complex optimizations"""
        
        # Step 1: Bulk miner management
        miners = [f'192.168.1.{100+i}' for i in range(10)]
        
        for miner_ip in miners:
            mock_status = {
                'ip': miner_ip,
                'temp': 65.0 + (hash(miner_ip) % 20),  # Varied temperatures
                'hashRate': 480.0 + (hash(miner_ip) % 40),
                'power': 12.0 + (hash(miner_ip) % 3),
                'efficiency': 35.0 + (hash(miner_ip) % 10)
            }
            
            with patch('services.miner_service.MinerService.get_miner_info') as mock_get_info:
                mock_get_info.return_value = mock_status
                
                response = client.get(f'/api/miners/{miner_ip}', headers=auth_headers)
                data = assert_json_response(response, 200)
                assert_api_success(data)
        
        # Step 2: Advanced optimization with custom parameters
        advanced_optimization = {
            'miner_ips': miners,
            'optimization_type': 'custom',
            'parameters': {
                'target_efficiency': 42.0,
                'temperature_limit': 75.0,
                'power_budget': 120.0,  # Total power budget for all miners
                'algorithm': 'genetic_algorithm',
                'generations': 50,
                'population_size': 100
            },
            'constraints': {
                'min_hashrate_retention': 0.9,  # Retain at least 90% of hashrate
                'max_individual_power': 15.0,
                'coordination_enabled': True
            }
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_advanced:
            mock_advanced.return_value = {
                'optimization_id': 'advanced_opt_001',
                'status': 'started',
                'estimated_duration': 600,  # 10 minutes for complex optimization
                'algorithm': 'genetic_algorithm'
            }
            
            response = client.post('/api/optimization/start',
                                 headers=auth_headers,
                                 json=advanced_optimization)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
        
        # Step 3: Monitor detailed progress
        detailed_progress = {
            'optimization_id': 'advanced_opt_001',
            'status': 'running',
            'progress': 75.0,
            'detailed_metrics': {
                'current_generation': 38,
                'best_fitness': 0.87,
                'population_diversity': 0.65,
                'convergence_rate': 0.15
            },
            'intermediate_results': {
                'miners_optimized': 7,
                'average_improvement': 12.5,
                'power_savings': 8.5
            }
        }
        
        with patch('ml_engine.core.optimization_engine.OptimizationEngine.get_optimization_status') as mock_detailed:
            mock_detailed.return_value = detailed_progress
            
            response = client.get('/api/optimization/advanced_opt_001/status',
                                headers=auth_headers)
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert 'detailed_metrics' in data['data']
            assert data['data']['detailed_metrics']['current_generation'] == 38
    
    @pytest.mark.e2e
    def test_monitoring_dashboard_workflow(self, client, auth_headers):
        """Test monitoring dashboard data collection workflow"""
        
        # Step 1: Real-time metrics
        current_metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'total_miners': 5,
                'active_miners': 5,
                'total_hashrate': 2425.0,
                'total_power': 64.0,
                'average_efficiency': 37.9,
                'average_temperature': 68.2
            },
            'individual_miners': [
                {
                    'ip': f'192.168.1.{100+i}',
                    'hashrate': 485.0,
                    'temperature': 68.0 + i,
                    'power': 12.8,
                    'efficiency': 37.9,
                    'status': 'active'
                } for i in range(5)
            ],
            'alerts': [
                {
                    'type': 'info',
                    'message': 'System optimization completed',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
        
        with patch('monitoring.metrics_collector.MetricsCollector.get_current_metrics') as mock_current:
            mock_current.return_value = current_metrics
            
            response = client.get('/api/monitoring/metrics', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['system']['total_miners'] == 5
            assert len(data['data']['individual_miners']) == 5
        
        # Step 2: Historical data
        historical_data = {
            'time_range': {
                'start': (datetime.now() - timedelta(hours=24)).isoformat(),
                'end': datetime.now().isoformat()
            },
            'data_points': [
                {
                    'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                    'total_hashrate': 2425.0 - (i * 10),
                    'average_temperature': 68.2 + (i * 0.1),
                    'total_power': 64.0 + (i * 0.5)
                } for i in range(24)
            ]
        }
        
        with patch('monitoring.metrics_collector.MetricsCollector.get_historical_metrics') as mock_historical:
            mock_historical.return_value = historical_data
            
            response = client.get('/api/monitoring/metrics/history',
                                headers=auth_headers,
                                query_string={
                                    'start_time': (datetime.now() - timedelta(hours=24)).isoformat(),
                                    'end_time': datetime.now().isoformat(),
                                    'interval': '1h'
                                })
            
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert len(data['data']['data_points']) == 24
        
        # Step 3: Performance trends analysis
        trends = {
            'efficiency_trend': {
                'direction': 'improving',
                'rate': 0.5,  # 0.5% per day
                'confidence': 0.85
            },
            'temperature_trend': {
                'direction': 'stable',
                'rate': 0.1,
                'confidence': 0.92
            },
            'power_trend': {
                'direction': 'decreasing',
                'rate': -0.2,
                'confidence': 0.78
            }
        }
        
        with patch('monitoring.trend_analyzer.TrendAnalyzer.analyze_trends') as mock_trends:
            mock_trends.return_value = trends
            
            response = client.get('/api/monitoring/trends', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert data['data']['efficiency_trend']['direction'] == 'improving'


class TestSystemScalability:
    """Test system scalability and performance under load"""
    
    @pytest.mark.e2e
    @pytest.mark.performance
    def test_large_scale_miner_management(self, client, auth_headers, performance_metrics):
        """Test system performance with large number of miners"""
        
        # Simulate 100 miners
        num_miners = 100
        miners = [f'192.168.{1 + i//254}.{(i%254) + 1}' for i in range(num_miners)]
        
        performance_metrics.start_timer('large_scale_mining')
        
        # Test getting status of all miners
        mock_all_miners = []
        for i, miner_ip in enumerate(miners):
            mock_all_miners.append({
                'ip': miner_ip,
                'hostname': f'miner-{i:03d}',
                'temp': 65.0 + (i % 20),
                'hashRate': 480.0 + (i % 40),
                'power': 12.0 + (i % 3),
                'status': 'active'
            })
        
        with patch('services.miner_service.MinerService.get_all_miners') as mock_get_all:
            mock_get_all.return_value = mock_all_miners
            
            response = client.get('/api/miners', headers=auth_headers)
            data = assert_json_response(response, 200)
            assert_api_success(data)
            
            assert len(data['data']['miners']) == num_miners
        
        performance_metrics.end_timer('large_scale_mining')
        
        # Should handle 100 miners efficiently (< 5 seconds)
        performance_metrics.assert_performance('large_scale_mining', 5.0)
    
    @pytest.mark.e2e
    @pytest.mark.performance
    def test_concurrent_optimization_scaling(self, client, auth_headers, performance_metrics):
        """Test concurrent optimization handling"""
        
        # Test multiple simultaneous optimization requests
        num_concurrent = 10
        
        performance_metrics.start_timer('concurrent_optimizations')
        
        optimization_ids = []
        for i in range(num_concurrent):
            optimization_request = {
                'miner_ips': [f'192.168.1.{100 + i}'],
                'optimization_type': 'efficiency'
            }
            
            with patch('ml_engine.core.optimization_engine.OptimizationEngine.start_optimization') as mock_start:
                mock_start.return_value = {
                    'optimization_id': f'concurrent_{i}',
                    'status': 'queued' if i > 2 else 'started'  # Queue after 3 concurrent
                }
                
                response = client.post('/api/optimization/start',
                                     headers=auth_headers,
                                     json=optimization_request)
                
                data = assert_json_response(response, 200)
                assert_api_success(data)
                optimization_ids.append(data['data']['optimization_id'])
        
        performance_metrics.end_timer('concurrent_optimizations')
        
        # Should handle concurrent requests quickly (< 3 seconds)
        performance_metrics.assert_performance('concurrent_optimizations', 3.0)
        
        assert len(optimization_ids) == num_concurrent