"""
Async Miner Service

High-performance async miner management with concurrent operations.
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import ServiceError, ErrorCode
from async_services.async_database_service import AsyncDatabaseService
from services.config_service import ConfigService

logger = get_logger("bitaxe.async_miner_service")


@dataclass
class MinerTask:
    """Represents a miner operation task"""
    id: str
    miner_ip: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AsyncMinerService:
    """
    High-performance async miner service with concurrent operations
    
    Features:
    - Concurrent HTTP operations with connection pooling
    - Background task processing with priority queues
    - Circuit breaker pattern for failing miners
    - Automatic retry logic with exponential backoff
    - Real-time metrics collection
    - Health monitoring and auto-recovery
    """
    
    def __init__(self, config_service: ConfigService, database_service: AsyncDatabaseService):
        self.config_service = config_service
        self.database_service = database_service
        
        # Async HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.session_timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        # Task processing
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, MinerTask] = {}
        self.max_concurrent_tasks = 20
        
        # Circuit breaker for failing miners
        self.miner_failures: Dict[str, List[datetime]] = {}
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_window = timedelta(minutes=5)
        self.circuit_breaker_cooldown = timedelta(minutes=2)
        self.blocked_miners: Dict[str, datetime] = {}
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'concurrent_operations': 0,
            'circuit_breaker_activations': 0,
            'tasks_completed': 0,
            'tasks_failed': 0
        }
        self.response_times: List[float] = []
        
        # Background workers
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start the async miner service"""
        if self.is_running:
            return
        
        logger.info("Starting async miner service")
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=10,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.session_timeout,
            headers={'User-Agent': 'BitAxe-AsyncService/1.0'}
        )
        
        # Start background workers
        self.is_running = True
        
        # Task processor workers
        for i in range(min(self.max_concurrent_tasks // 4, 5)):
            worker = asyncio.create_task(self._task_worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Metrics collector
        metrics_worker = asyncio.create_task(self._metrics_collector())
        self.workers.append(metrics_worker)
        
        # Circuit breaker maintenance
        cb_worker = asyncio.create_task(self._circuit_breaker_maintenance())
        self.workers.append(cb_worker)
        
        # Periodic health checks
        health_worker = asyncio.create_task(self._health_checker())
        self.workers.append(health_worker)
        
        logger.info(f"Started {len(self.workers)} background workers")
    
    async def stop(self):
        """Stop the async miner service"""
        if not self.is_running:
            return
        
        logger.info("Stopping async miner service")
        self.is_running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.workers.clear()
        self.running_tasks.clear()
        
        logger.info("Async miner service stopped")
    
    def _is_circuit_breaker_open(self, miner_ip: str) -> bool:
        """Check if circuit breaker is open for a miner"""
        if miner_ip in self.blocked_miners:
            block_time = self.blocked_miners[miner_ip]
            if datetime.now() - block_time < self.circuit_breaker_cooldown:
                return True
            else:
                # Cooldown expired, remove from blocked list
                del self.blocked_miners[miner_ip]
        
        return False
    
    def _record_failure(self, miner_ip: str):
        """Record a failure for circuit breaker"""
        now = datetime.now()
        
        if miner_ip not in self.miner_failures:
            self.miner_failures[miner_ip] = []
        
        # Add current failure
        self.miner_failures[miner_ip].append(now)
        
        # Remove old failures outside window
        cutoff = now - self.circuit_breaker_window
        self.miner_failures[miner_ip] = [
            failure for failure in self.miner_failures[miner_ip]
            if failure > cutoff
        ]
        
        # Check if threshold exceeded
        if len(self.miner_failures[miner_ip]) >= self.circuit_breaker_threshold:
            self.blocked_miners[miner_ip] = now
            self.metrics['circuit_breaker_activations'] += 1
            logger.warning(f"Circuit breaker activated for miner {miner_ip}",
                         failure_count=len(self.miner_failures[miner_ip]))
    
    def _record_success(self, miner_ip: str):
        """Record a success for circuit breaker"""
        # Clear failure history on success
        if miner_ip in self.miner_failures:
            self.miner_failures[miner_ip].clear()
        
        # Remove from blocked list if present
        if miner_ip in self.blocked_miners:
            del self.blocked_miners[miner_ip]
            logger.info(f"Circuit breaker cleared for miner {miner_ip}")
    
    async def _make_http_request(self, miner_ip: str, endpoint: str, 
                                method: str = 'GET', data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to miner with circuit breaker and metrics"""
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(miner_ip):
            logger.debug(f"Circuit breaker open for {miner_ip}, skipping request")
            return None
        
        if not self.session:
            raise ServiceError("HTTP session not initialized", "async_miner_service")
        
        url = f"http://{miner_ip}{endpoint}"
        start_time = time.time()
        
        try:
            self.metrics['total_requests'] += 1
            self.metrics['concurrent_operations'] += 1
            
            async with self.session.request(method, url, json=data) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                if response.status == 200:
                    result = await response.json()
                    self.metrics['successful_requests'] += 1
                    self._record_success(miner_ip)
                    
                    logger.debug(f"HTTP request successful to {miner_ip}",
                               endpoint=endpoint, response_time=response_time)
                    
                    return result
                else:
                    logger.warning(f"HTTP request failed to {miner_ip}",
                                 endpoint=endpoint, status=response.status)
                    self._record_failure(miner_ip)
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"HTTP timeout to {miner_ip}", endpoint=endpoint)
            self._record_failure(miner_ip)
            return None
        except Exception as e:
            logger.error(f"HTTP error to {miner_ip}", endpoint=endpoint, error=str(e))
            self._record_failure(miner_ip)
            return None
        finally:
            self.metrics['concurrent_operations'] -= 1
            if not self.response_times:
                self.metrics['failed_requests'] += 1
    
    async def fetch_miner_data_async(self, miner_ip: str) -> Optional[Dict[str, Any]]:
        """Async fetch miner data"""
        data = await self._make_http_request(miner_ip, '/api/system/info')
        
        if data:
            # Enhance data with additional info
            data['ip'] = miner_ip
            data['timestamp'] = datetime.now().isoformat()
            
            # Calculate efficiency if possible
            if data.get('hashRate') and data.get('power'):
                data['efficiency'] = data['hashRate'] / data['power']
            
            return data
        
        return None
    
    async def set_miner_settings_async(self, miner_ip: str, frequency: int, 
                                     core_voltage: int, autofanspeed: bool = True) -> bool:
        """Async set miner settings"""
        settings_data = {
            'frequency': frequency,
            'coreVoltage': core_voltage,
            'autofanspeed': autofanspeed
        }
        
        result = await self._make_http_request(miner_ip, '/api/system/settings', 'POST', settings_data)
        return result is not None
    
    async def restart_miner_async(self, miner_ip: str) -> bool:
        """Async restart miner"""
        result = await self._make_http_request(miner_ip, '/api/system/restart', 'POST')
        return result is not None
    
    async def fetch_all_miners_concurrent(self, miner_ips: List[str]) -> List[Dict[str, Any]]:
        """Fetch data from all miners concurrently"""
        logger.info(f"Fetching data from {len(miner_ips)} miners concurrently")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        async def fetch_with_semaphore(ip):
            async with semaphore:
                return await self.fetch_miner_data_async(ip)
        
        # Execute all requests concurrently
        tasks = [fetch_with_semaphore(ip) for ip in miner_ips]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data from {miner_ips[i]}", error=str(result))
                failed_count += 1
            elif result is not None:
                successful_results.append(result)
            else:
                failed_count += 1
        
        logger.info(f"Fetched data from {len(successful_results)} miners, {failed_count} failures")
        return successful_results
    
    async def update_miners_settings_batch(self, settings_updates: List[Dict[str, Any]]) -> List[bool]:
        """Update multiple miners' settings concurrently"""
        logger.info(f"Updating settings for {len(settings_updates)} miners")
        
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks // 2)  # More conservative for settings
        
        async def update_with_semaphore(update):
            async with semaphore:
                ip = update['ip']
                frequency = update['frequency']
                core_voltage = update['core_voltage']
                autofanspeed = update.get('autofanspeed', True)
                
                return await self.set_miner_settings_async(ip, frequency, core_voltage, autofanspeed)
        
        tasks = [update_with_semaphore(update) for update in settings_updates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to False
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                clean_results.append(False)
            else:
                clean_results.append(result)
        
        successful_count = sum(clean_results)
        logger.info(f"Updated settings for {successful_count}/{len(settings_updates)} miners")
        
        return clean_results
    
    async def submit_task(self, task: MinerTask) -> str:
        """Submit a task to the processing queue"""
        await self.task_queue.put((task.priority, task))
        logger.debug(f"Task {task.id} submitted to queue", task_type=task.task_type, miner_ip=task.miner_ip)
        return task.id
    
    async def get_task_status(self, task_id: str) -> Optional[MinerTask]:
        """Get task status"""
        # Check running tasks
        if task_id in self.running_tasks:
            # Find the corresponding MinerTask
            for task in self.completed_tasks.values():
                if task.id == task_id:
                    return task
        
        # Check completed tasks
        return self.completed_tasks.get(task_id)
    
    async def _task_worker(self, worker_name: str):
        """Background task worker"""
        logger.info(f"Task worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                logger.debug(f"Worker {worker_name} processing task {task.id}")
                
                # Mark task as running
                task.status = "running"
                task.started_at = datetime.now()
                
                # Create asyncio task for the operation
                operation_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task.id] = operation_task
                
                try:
                    # Wait for task completion
                    result = await operation_task
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.now()
                    self.metrics['tasks_completed'] += 1
                    
                    logger.debug(f"Task {task.id} completed successfully")
                    
                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                    task.completed_at = datetime.now()
                    self.metrics['tasks_failed'] += 1
                    
                    logger.error(f"Task {task.id} failed", error=str(e))
                    
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = "pending"
                        task.started_at = None
                        task.completed_at = None
                        task.error = None
                        
                        # Re-queue with delay
                        await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                        await self.task_queue.put((task.priority, task))
                        
                        logger.info(f"Task {task.id} re-queued for retry {task.retry_count}")
                        continue
                
                finally:
                    # Remove from running tasks
                    if task.id in self.running_tasks:
                        del self.running_tasks[task.id]
                
                # Store completed task
                self.completed_tasks[task.id] = task
                
                # Mark queue task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Task worker {worker_name} error", error=str(e))
                await asyncio.sleep(1)
        
        logger.info(f"Task worker {worker_name} stopped")
    
    async def _execute_task(self, task: MinerTask) -> Any:
        """Execute a specific task"""
        task_type = task.task_type
        miner_ip = task.miner_ip
        payload = task.payload
        
        if task_type == "fetch_data":
            return await self.fetch_miner_data_async(miner_ip)
        
        elif task_type == "update_settings":
            frequency = payload['frequency']
            core_voltage = payload['core_voltage']
            autofanspeed = payload.get('autofanspeed', True)
            return await self.set_miner_settings_async(miner_ip, frequency, core_voltage, autofanspeed)
        
        elif task_type == "restart":
            return await self.restart_miner_async(miner_ip)
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _metrics_collector(self):
        """Collect and update metrics periodically"""
        logger.info("Metrics collector started")
        
        while self.is_running:
            try:
                # Update average response time
                if self.response_times:
                    self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)
                    
                    # Keep only recent response times (last 1000)
                    if len(self.response_times) > 1000:
                        self.response_times = self.response_times[-1000:]
                
                # Log metrics periodically
                logger.info("Service metrics",
                           total_requests=self.metrics['total_requests'],
                           success_rate=self.metrics['successful_requests'] / max(1, self.metrics['total_requests']) * 100,
                           avg_response_time=self.metrics['avg_response_time'],
                           concurrent_ops=self.metrics['concurrent_operations'],
                           circuit_breaker_activations=self.metrics['circuit_breaker_activations'],
                           tasks_completed=self.metrics['tasks_completed'],
                           tasks_failed=self.metrics['tasks_failed'],
                           running_tasks=len(self.running_tasks),
                           queue_size=self.task_queue.qsize())
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error("Metrics collector error", error=str(e))
                await asyncio.sleep(5)
    
    async def _circuit_breaker_maintenance(self):
        """Maintain circuit breaker state"""
        logger.info("Circuit breaker maintenance started")
        
        while self.is_running:
            try:
                now = datetime.now()
                
                # Clean up old failure records
                for miner_ip in list(self.miner_failures.keys()):
                    cutoff = now - self.circuit_breaker_window
                    self.miner_failures[miner_ip] = [
                        failure for failure in self.miner_failures[miner_ip]
                        if failure > cutoff
                    ]
                    
                    # Remove empty entries
                    if not self.miner_failures[miner_ip]:
                        del self.miner_failures[miner_ip]
                
                # Check for expired cooldowns
                expired_blocks = []
                for miner_ip, block_time in self.blocked_miners.items():
                    if now - block_time >= self.circuit_breaker_cooldown:
                        expired_blocks.append(miner_ip)
                
                for miner_ip in expired_blocks:
                    del self.blocked_miners[miner_ip]
                    logger.info(f"Circuit breaker cooldown expired for {miner_ip}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Circuit breaker maintenance error", error=str(e))
                await asyncio.sleep(5)
    
    async def _health_checker(self):
        """Periodic health checks for miners"""
        logger.info("Health checker started")
        
        while self.is_running:
            try:
                # Get all configured miners
                miner_ips = self.config_service.ips
                
                if miner_ips:
                    # Perform health checks
                    health_results = await self.fetch_all_miners_concurrent(miner_ips)
                    
                    # Log health data to database
                    for result in health_results:
                        if result:
                            await self.database_service.log_miner_data_async(result)
                    
                    logger.debug(f"Health check completed for {len(health_results)}/{len(miner_ips)} miners")
                
                # Wait for next check
                health_check_interval = self.config_service.get('settings.health_check_interval_sec', 30)
                await asyncio.sleep(health_check_interval)
                
            except Exception as e:
                logger.error("Health checker error", error=str(e))
                await asyncio.sleep(5)
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get current service metrics"""
        return {
            **self.metrics,
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'blocked_miners': len(self.blocked_miners),
            'miner_failures': {ip: len(failures) for ip, failures in self.miner_failures.items()},
            'worker_count': len(self.workers),
            'is_running': self.is_running
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'blocked_miners': list(self.blocked_miners.keys()),
            'miner_failure_counts': {ip: len(failures) for ip, failures in self.miner_failures.items()},
            'threshold': self.circuit_breaker_threshold,
            'window_minutes': self.circuit_breaker_window.total_seconds() / 60,
            'cooldown_minutes': self.circuit_breaker_cooldown.total_seconds() / 60
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()