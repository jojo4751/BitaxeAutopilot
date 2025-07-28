"""
Integration tests for Async Services

Testing async service integration, communication, and coordination
including service lifecycle, inter-service communication, and error handling.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from async_services.async_miner_service import AsyncMinerService
from async_services.async_optimization_service import AsyncOptimizationService
from async_services.async_monitoring_service import AsyncMonitoringService
from async_services.task_queue import TaskQueue, Task
from async_services.service_coordinator import ServiceCoordinator
from services.miner_service import MinerService
from monitoring.metrics_collector import MetricsCollector


class TestAsyncServiceIntegration:
    """Test async service integration and coordination"""
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_service_lifecycle_coordination(self, async_app):
        """Test coordinated service startup and shutdown"""
        # Mock services
        miner_service = Mock(spec=AsyncMinerService)
        miner_service.start = AsyncMock()
        miner_service.stop = AsyncMock()
        miner_service.is_running = False
        
        optimization_service = Mock(spec=AsyncOptimizationService)
        optimization_service.start = AsyncMock()
        optimization_service.stop = AsyncMock()
        optimization_service.is_running = False
        
        monitoring_service = Mock(spec=AsyncMonitoringService)
        monitoring_service.start = AsyncMock()
        monitoring_service.stop = AsyncMock()
        monitoring_service.is_running = False
        
        # Create service coordinator
        coordinator = ServiceCoordinator([
            miner_service,
            optimization_service,
            monitoring_service
        ])
        
        # Test coordinated startup
        await coordinator.start_all_services()
        
        # Verify all services were started
        miner_service.start.assert_called_once()
        optimization_service.start.assert_called_once()
        monitoring_service.start.assert_called_once()
        
        # Test coordinated shutdown
        await coordinator.stop_all_services()
        
        # Verify all services were stopped
        miner_service.stop.assert_called_once()
        optimization_service.stop.assert_called_once()
        monitoring_service.stop.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_inter_service_communication(self, sample_miner_data):
        """Test communication between async services"""
        # Mock services with communication
        miner_service = Mock(spec=AsyncMinerService)
        miner_service.get_miner_status = AsyncMock(return_value=sample_miner_data)
        miner_service.update_miner_config = AsyncMock(return_value=True)
        
        optimization_service = Mock(spec=AsyncOptimizationService)
        optimization_service.optimize_miner = AsyncMock(return_value={
            'status': 'success',
            'optimizations': [{
                'type': 'frequency',
                'adjustments': {'frequency': 650, 'voltage': 12.1}
            }]
        })
        
        monitoring_service = Mock(spec=AsyncMonitoringService)
        monitoring_service.record_optimization = AsyncMock()
        
        # Test optimization workflow
        miner_ip = '192.168.1.100'
        
        # Step 1: Get current miner status
        current_status = await miner_service.get_miner_status(miner_ip)
        assert current_status == sample_miner_data
        
        # Step 2: Request optimization
        optimization_result = await optimization_service.optimize_miner(
            miner_ip, current_status
        )
        assert optimization_result['status'] == 'success'
        
        # Step 3: Apply optimization
        new_config = optimization_result['optimizations'][0]['adjustments']
        success = await miner_service.update_miner_config(miner_ip, new_config)
        assert success is True
        
        # Step 4: Record optimization
        await monitoring_service.record_optimization(miner_ip, optimization_result)
        
        # Verify all interactions occurred
        miner_service.get_miner_status.assert_called_once_with(miner_ip)
        optimization_service.optimize_miner.assert_called_once_with(miner_ip, current_status)
        miner_service.update_miner_config.assert_called_once_with(miner_ip, new_config)
        monitoring_service.record_optimization.assert_called_once_with(miner_ip, optimization_result)
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_task_queue_integration(self):
        """Test task queue integration with services"""
        # Create task queue
        task_queue = TaskQueue(max_concurrent_tasks=5)
        await task_queue.start()
        
        # Mock task processing
        async def mock_process_optimization(miner_ip, telemetry):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                'miner_ip': miner_ip,
                'status': 'completed',
                'timestamp': datetime.now()
            }
        
        # Create tasks
        tasks = []
        for i in range(10):
            task = Task(
                task_id=f'opt_task_{i}',
                task_type='optimization',
                priority=1,
                payload={
                    'miner_ip': f'192.168.1.{100 + i}',
                    'telemetry': {'temp': 65 + i, 'power': 12}
                },
                handler=mock_process_optimization
            )
            tasks.append(task)
        
        # Submit tasks
        for task in tasks:
            await task_queue.submit_task(task)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Verify queue is empty (all tasks processed)
        assert task_queue.pending_tasks == 0
        
        await task_queue.stop()
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery in async services"""
        # Mock service with failures
        failing_service = Mock(spec=AsyncMinerService)
        failing_service.start = AsyncMock(side_effect=Exception("Service startup failed"))
        failing_service.stop = AsyncMock()
        failing_service.is_running = False
        
        healthy_service = Mock(spec=AsyncOptimizationService)
        healthy_service.start = AsyncMock()
        healthy_service.stop = AsyncMock()
        healthy_service.is_running = False
        
        coordinator = ServiceCoordinator([failing_service, healthy_service])
        
        # Test error handling during startup
        with pytest.raises(Exception, match="Service startup failed"):
            await coordinator.start_all_services()
        
        # Verify healthy service was not affected
        healthy_service.start.assert_called_once()
        
        # Test recovery mechanism
        failing_service.start = AsyncMock()  # Fix the failing service
        
        # Retry startup
        await coordinator.start_all_services()
        
        # Verify both services started successfully
        assert failing_service.start.call_count == 2  # Original fail + retry
        assert healthy_service.start.call_count == 2  # Original + retry
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_service_health_monitoring(self):
        """Test service health monitoring and alerting"""
        from async_services.health_monitor import HealthMonitor
        
        # Mock services with health status
        services = []
        for i in range(3):
            service = Mock()
            service.name = f'service_{i}'
            service.get_health_status = AsyncMock(return_value={
                'status': 'healthy',
                'uptime': 3600,
                'memory_usage': 50.5,
                'cpu_usage': 25.2,
                'last_heartbeat': datetime.now()
            })
            services.append(service)
        
        # Create health monitor
        health_monitor = HealthMonitor(services, check_interval=1.0)
        
        # Start monitoring
        await health_monitor.start()
        
        # Wait for health checks
        await asyncio.sleep(2.0)
        
        # Verify health checks were performed
        for service in services:
            service.get_health_status.assert_called()
        
        # Test unhealthy service detection
        services[1].get_health_status = AsyncMock(return_value={
            'status': 'unhealthy',
            'error': 'Memory usage too high',
            'memory_usage': 95.0
        })
        
        # Wait for another health check
        await asyncio.sleep(2.0)
        
        # Verify unhealthy service was detected
        health_status = await health_monitor.get_overall_health()
        assert health_status['overall_status'] == 'degraded'
        assert len(health_status['unhealthy_services']) == 1
        
        await health_monitor.stop()


class TestAsyncAPIIntegration:
    """Test async API endpoint integration"""
    
    @pytest.mark.integration
    @pytest.mark.async
    @pytest.mark.api
    async def test_async_miner_endpoints(self, async_app, sample_miner_data):
        """Test async miner API endpoints"""
        from quart.testing import QuartClient
        
        # Mock async client
        async with async_app.test_client() as client:
            # Mock miner service responses
            with patch.object(async_app.miner_service, 'get_all_miners') as mock_get_all:
                mock_get_all.return_value = [sample_miner_data]
                
                # Test get all miners endpoint
                response = await client.get('/api/v2/miners')
                assert response.status_code == 200
                
                data = await response.get_json()
                assert data['success'] is True
                assert len(data['data']) == 1
                assert data['data'][0]['ip'] == sample_miner_data['ip']
            
            # Test get specific miner
            with patch.object(async_app.miner_service, 'get_miner_status') as mock_get_status:
                mock_get_status.return_value = sample_miner_data
                
                response = await client.get('/api/v2/miners/192.168.1.100')
                assert response.status_code == 200
                
                data = await response.get_json()
                assert data['success'] is True
                assert data['data']['ip'] == '192.168.1.100'
    
    @pytest.mark.integration
    @pytest.mark.async
    @pytest.mark.api
    async def test_async_optimization_endpoints(self, async_app):
        """Test async optimization API endpoints"""
        async with async_app.test_client() as client:
            # Test start optimization
            with patch.object(async_app.optimization_service, 'start_optimization') as mock_start:
                mock_start.return_value = {
                    'status': 'started',
                    'optimization_id': 'opt_12345'
                }
                
                response = await client.post('/api/v2/optimization/start', json={
                    'miner_ips': ['192.168.1.100'],
                    'optimization_type': 'ml_guided'
                })
                
                assert response.status_code == 200
                data = await response.get_json()
                assert data['success'] is True
                assert data['data']['status'] == 'started'
            
            # Test get optimization status
            with patch.object(async_app.optimization_service, 'get_optimization_status') as mock_status:
                mock_status.return_value = {
                    'optimization_id': 'opt_12345',
                    'status': 'running',
                    'progress': 75.0,
                    'miners_optimized': 3,
                    'total_miners': 4
                }
                
                response = await client.get('/api/v2/optimization/opt_12345/status')
                assert response.status_code == 200
                
                data = await response.get_json()
                assert data['success'] is True
                assert data['data']['progress'] == 75.0
    
    @pytest.mark.integration
    @pytest.mark.async
    @pytest.mark.api
    async def test_websocket_integration(self, async_app):
        """Test WebSocket integration for real-time updates"""
        async with async_app.test_client() as client:
            # Test WebSocket connection
            async with client.websocket('/api/v2/ws/miners') as websocket:
                # Mock sending miner update
                test_update = {
                    'type': 'miner_update',
                    'data': {
                        'ip': '192.168.1.100',
                        'temp': 67.5,
                        'hashRate': 490.2,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                # Simulate server sending update
                await websocket.send_json(test_update)
                
                # Verify message format
                assert 'type' in test_update
                assert 'data' in test_update
                assert test_update['type'] == 'miner_update'
    
    @pytest.mark.integration
    @pytest.mark.async
    @pytest.mark.api
    async def test_concurrent_api_requests(self, async_app, performance_metrics):
        """Test concurrent API request handling"""
        async with async_app.test_client() as client:
            # Mock service responses
            with patch.object(async_app.miner_service, 'get_miner_status') as mock_get_status:
                mock_get_status.return_value = {
                    'ip': '192.168.1.100',
                    'temp': 65.5,
                    'hashRate': 485.2
                }
                
                performance_metrics.start_timer('concurrent_requests')
                
                # Create concurrent requests
                tasks = []
                for i in range(20):
                    task = asyncio.create_task(
                        client.get(f'/api/v2/miners/192.168.1.{100 + (i % 5)}')
                    )
                    tasks.append(task)
                
                # Wait for all requests
                responses = await asyncio.gather(*tasks)
                
                performance_metrics.end_timer('concurrent_requests')
                
                # Should handle all requests quickly (< 5 seconds)
                performance_metrics.assert_performance('concurrent_requests', 5.0)
                
                # Verify all responses
                for response in responses:
                    assert response.status_code == 200


class TestAsyncDataProcessing:
    """Test async data processing integration"""
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_streaming_data_processing(self, sample_miner_telemetry):
        """Test streaming data processing pipeline"""
        from async_services.stream_processor import StreamProcessor
        
        # Create stream processor
        processor = StreamProcessor(buffer_size=100, batch_size=10)
        
        # Mock processing handler
        processed_batches = []
        
        async def process_batch(batch):
            processed_batches.append(batch)
            return len(batch)
        
        processor.set_batch_handler(process_batch)
        
        # Start processor  
        await processor.start()
        
        # Stream data
        for data_point in sample_miner_telemetry:
            await processor.add_data_point(data_point)
        
        # Flush remaining data
        await processor.flush()
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Verify processing
        assert len(processed_batches) > 0
        total_processed = sum(len(batch) for batch in processed_batches)
        assert total_processed == len(sample_miner_telemetry)
        
        await processor.stop()
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_async_database_operations(self, test_database, sample_miner_data):
        """Test async database operations"""
        from async_services.async_database_service import AsyncDatabaseService
        
        # Mock async database service
        db_service = Mock(spec=AsyncDatabaseService)
        db_service.insert_miner_status = AsyncMock(return_value=True)
        db_service.get_miner_history = AsyncMock(return_value=[sample_miner_data])
        db_service.bulk_insert = AsyncMock(return_value=10)
        
        # Test single insert
        success = await db_service.insert_miner_status(sample_miner_data)
        assert success is True
        
        # Test bulk insert
        bulk_data = [sample_miner_data] * 10
        count = await db_service.bulk_insert('miner_status', bulk_data)
        assert count == 10
        
        # Test query
        history = await db_service.get_miner_history(
            '192.168.1.100',
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        assert len(history) == 1
        assert history[0] == sample_miner_data
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_async_caching_integration(self):
        """Test async caching integration"""
        from async_services.cache_service import AsyncCacheService
        
        # Mock cache service
        cache_service = Mock(spec=AsyncCacheService)
        cache_service.get = AsyncMock(return_value=None)
        cache_service.set = AsyncMock(return_value=True)
        cache_service.delete = AsyncMock(return_value=True)
        cache_service.exists = AsyncMock(return_value=False)
        
        # Test cache miss and set
        key = 'miner_status:192.168.1.100'
        cached_value = await cache_service.get(key)
        assert cached_value is None
        
        # Set value
        test_data = {'temp': 65.5, 'hashRate': 485.2}
        success = await cache_service.set(key, test_data, ttl=300)
        assert success is True
        
        # Mock cache hit
        cache_service.get = AsyncMock(return_value=test_data)
        cache_service.exists = AsyncMock(return_value=True)
        
        # Test cache hit
        cached_value = await cache_service.get(key)
        assert cached_value == test_data
        
        exists = await cache_service.exists(key)
        assert exists is True
        
        # Test delete
        success = await cache_service.delete(key)
        assert success is True


class TestAsyncErrorHandling:
    """Test async error handling and resilience"""
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_service_resilience(self):
        """Test service resilience to failures"""
        from async_services.resilient_service import ResilientService
        
        # Mock service with intermittent failures
        failure_count = 0
        
        async def flaky_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception(f"Temporary failure {failure_count}")
            return "Success"
        
        # Create resilient service wrapper
        resilient_service = ResilientService(
            operation=flaky_operation,
            max_retries=5,
            retry_delay=0.1,
            backoff_multiplier=1.5
        )
        
        # Test with retries
        result = await resilient_service.execute_with_retries()
        assert result == "Success"
        assert failure_count == 4  # 3 failures + 1 success
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker pattern integration"""
        from async_services.circuit_breaker import CircuitBreaker
        
        # Mock failing service
        call_count = 0
        
        async def failing_service():
            nonlocal call_count
            call_count += 1
            raise Exception("Service unavailable")
        
        # Create circuit breaker
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=Exception
        )
        
        # Test circuit breaker opening
        for i in range(5):
            try:
                await circuit_breaker.call(failing_service)
                assert False, "Should have raised exception"
            except Exception as e:
                if i < 3:
                    assert str(e) == "Service unavailable"
                else:
                    assert "Circuit breaker is open" in str(e)
        
        # Verify circuit opened after threshold
        assert circuit_breaker.state == 'open'
        assert call_count == 3  # Only 3 calls made before circuit opened
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Test half-open state
        try:
            await circuit_breaker.call(failing_service)
            assert False, "Should have raised exception"
        except Exception as e:
            assert str(e) == "Service unavailable"
        
        # Circuit should be open again
        assert circuit_breaker.state == 'open'
    
    @pytest.mark.integration
    @pytest.mark.async
    async def test_timeout_handling(self):
        """Test timeout handling in async operations"""
        # Mock slow operation
        async def slow_operation():
            await asyncio.sleep(2.0)
            return "Completed"
        
        # Test operation with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=1.0)
        
        # Test operation without timeout
        result = await asyncio.wait_for(slow_operation(), timeout=3.0)
        assert result == "Completed"