"""
Async Service Manager

Centralized management of all async services with lifecycle management and monitoring.
"""

import asyncio
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import json

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import ServiceError, ErrorCode
from async_services.async_miner_service import AsyncMinerService
from async_services.async_database_service import AsyncDatabaseService
from background_tasks.task_scheduler import TaskScheduler
from monitoring.metrics_collector import MetricsCollector, get_metrics_collector
from services.config_service import ConfigService

logger = get_logger("bitaxe.async_service_manager")


@dataclass
class ServiceInfo:
    """Information about a managed service"""
    name: str
    service: Any
    status: str  # starting, running, stopping, stopped, error
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    error: Optional[str] = None
    restart_count: int = 0
    health_check_interval: float = 30.0
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, degraded, unhealthy, unknown


class AsyncServiceManager:
    """
    Centralized async service manager
    
    Features:
    - Service lifecycle management (start, stop, restart)
    - Health monitoring and auto-recovery
    - Graceful shutdown handling
    - Service dependency management
    - Metrics collection and reporting
    - Error handling and logging
    """
    
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        
        # Service registry
        self.services: Dict[str, ServiceInfo] = {}
        self.service_dependencies: Dict[str, List[str]] = {}
        
        # Core services
        self.database_service: Optional[AsyncDatabaseService] = None
        self.miner_service: Optional[AsyncMinerService] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        
        # Management state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.health_check_task: Optional[asyncio.Task] = None
        self.auto_recovery_enabled = True
        self.max_restart_attempts = 3
        self.restart_cooldown = 30  # seconds
        
        # Metrics
        self.service_metrics = {
            'total_services': 0,
            'running_services': 0,
            'failed_services': 0,
            'total_restarts': 0,
            'uptime': 0.0
        }
        self.start_time: Optional[datetime] = None
    
    async def initialize(self):
        """Initialize all services"""
        if self.is_running:
            return
        
        logger.info("Initializing async service manager")
        
        try:
            # Initialize metrics collector first
            self.metrics_collector = get_metrics_collector()
            await self._register_service('metrics_collector', self.metrics_collector)
            
            # Initialize database service
            db_path = self.config_service.database_path
            max_connections = self.config_service.get('database.max_connections', 10)
            self.database_service = AsyncDatabaseService(db_path, max_connections)
            await self._register_service('database_service', self.database_service)
            
            # Initialize task scheduler
            max_workers = self.config_service.get('scheduler.max_workers', 10)
            max_concurrent = self.config_service.get('scheduler.max_concurrent_tasks', 50)
            self.task_scheduler = TaskScheduler(max_workers, max_concurrent)
            await self._register_service('task_scheduler', self.task_scheduler)
            
            # Initialize miner service
            self.miner_service = AsyncMinerService(self.config_service, self.database_service)
            await self._register_service('miner_service', self.miner_service)
            
            # Set up service dependencies
            self.service_dependencies = {
                'miner_service': ['database_service', 'metrics_collector'],
                'task_scheduler': ['metrics_collector'],
                'database_service': ['metrics_collector']
            }
            
            # Register built-in tasks
            await self._register_built_in_tasks()
            
            logger.info(f"Initialized {len(self.services)} services")
            
        except Exception as e:
            logger.error("Service manager initialization failed", error=str(e))
            raise ServiceError(f"Failed to initialize service manager: {e}")
    
    async def start_all(self):
        """Start all services in dependency order"""
        if self.is_running:
            return
        
        logger.info("Starting all async services")
        self.is_running = True
        self.start_time = datetime.now()
        self.shutdown_event.clear()
        
        try:
            # Start services in dependency order
            start_order = self._get_service_start_order()
            
            for service_name in start_order:
                await self._start_service(service_name)
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor())
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Record metrics
            self.metrics_collector.set_gauge('service_manager_services_total', len(self.services))
            self.metrics_collector.increment_counter('service_manager_starts_total')
            
            logger.info("All async services started successfully")
            
        except Exception as e:
            logger.error("Failed to start services", error=str(e))
            await self.stop_all()
            raise ServiceError(f"Failed to start services: {e}")
    
    async def stop_all(self):
        """Stop all services in reverse dependency order"""
        if not self.is_running:
            return
        
        logger.info("Stopping all async services")
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Cancel health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop services in reverse dependency order
            stop_order = self._get_service_stop_order()
            
            for service_name in stop_order:
                await self._stop_service(service_name)
            
            # Calculate uptime
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.service_metrics['uptime'] = uptime
                
                if self.metrics_collector:
                    self.metrics_collector.set_gauge('service_manager_uptime_seconds', uptime)
            
            logger.info("All async services stopped")
            
        except Exception as e:
            logger.error("Error during service shutdown", error=str(e))
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        logger.info(f"Restarting service: {service_name}")
        
        try:
            await self._stop_service(service_name)
            await asyncio.sleep(1)  # Brief pause between stop and start
            await self._start_service(service_name)
            
            service_info = self.services[service_name]
            service_info.restart_count += 1
            self.service_metrics['total_restarts'] += 1
            
            if self.metrics_collector:
                self.metrics_collector.increment_counter('service_restarts_total', 
                                                       tags={'service': service_name})
            
            logger.info(f"Service {service_name} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}", error=str(e))
            return False
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            'manager': {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'total_services': len(self.services),
                'auto_recovery_enabled': self.auto_recovery_enabled
            },
            'services': {},
            'metrics': self.service_metrics
        }
        
        for name, service_info in self.services.items():
            status['services'][name] = {
                'status': service_info.status,
                'start_time': service_info.start_time.isoformat() if service_info.start_time else None,
                'stop_time': service_info.stop_time.isoformat() if service_info.stop_time else None,
                'restart_count': service_info.restart_count,
                'health_status': service_info.health_status,
                'last_health_check': service_info.last_health_check.isoformat() if service_info.last_health_check else None,
                'error': service_info.error
            }
            
            # Add service-specific metrics if available
            if hasattr(service_info.service, 'get_service_metrics'):
                try:
                    service_metrics = service_info.service.get_service_metrics()
                    status['services'][name]['metrics'] = service_metrics
                except Exception as e:
                    logger.debug(f"Could not get metrics for {name}", error=str(e))
        
        return status
    
    async def get_service_health(self) -> Dict[str, str]:
        """Get health status of all services"""
        health = {}
        
        for name, service_info in self.services.items():
            health[name] = service_info.health_status
        
        return health
    
    async def _register_service(self, name: str, service: Any):
        """Register a service with the manager"""
        service_info = ServiceInfo(
            name=name,
            service=service,
            status='stopped'
        )
        
        self.services[name] = service_info
        logger.debug(f"Registered service: {name}")
    
    async def _start_service(self, service_name: str):
        """Start a specific service"""
        if service_name not in self.services:
            raise ServiceError(f"Service {service_name} not registered")
        
        service_info = self.services[service_name]
        
        if service_info.status == 'running':
            logger.debug(f"Service {service_name} already running")
            return
        
        logger.info(f"Starting service: {service_name}")
        
        try:
            service_info.status = 'starting'
            service_info.error = None
            
            # Check dependencies
            dependencies = self.service_dependencies.get(service_name, [])
            for dep_name in dependencies:
                if dep_name in self.services and self.services[dep_name].status != 'running':
                    raise ServiceError(f"Dependency {dep_name} not running for service {service_name}")
            
            # Start the service
            if hasattr(service_info.service, 'start'):
                await service_info.service.start()
            
            service_info.status = 'running'
            service_info.start_time = datetime.now()
            service_info.health_status = 'healthy'
            
            logger.info(f"Service {service_name} started successfully")
            
        except Exception as e:
            service_info.status = 'error'
            service_info.error = str(e)
            logger.error(f"Failed to start service {service_name}", error=str(e))
            raise
    
    async def _stop_service(self, service_name: str):
        """Stop a specific service"""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} not found for stopping")
            return
        
        service_info = self.services[service_name]
        
        if service_info.status in ['stopped', 'stopping']:
            logger.debug(f"Service {service_name} already stopped/stopping")
            return
        
        logger.info(f"Stopping service: {service_name}")
        
        try:
            service_info.status = 'stopping'
            
            # Stop the service
            if hasattr(service_info.service, 'stop'):
                await service_info.service.stop()
            
            service_info.status = 'stopped'
            service_info.stop_time = datetime.now()
            service_info.health_status = 'unknown'
            
            logger.info(f"Service {service_name} stopped successfully")
            
        except Exception as e:
            service_info.status = 'error'
            service_info.error = str(e)
            logger.error(f"Error stopping service {service_name}", error=str(e))
    
    def _get_service_start_order(self) -> List[str]:
        """Get service start order based on dependencies"""
        # Simple topological sort
        order = []
        visited = set()
        
        def visit(service_name):
            if service_name in visited:
                return
            
            visited.add(service_name)
            dependencies = self.service_dependencies.get(service_name, [])
            
            for dep in dependencies:
                if dep in self.services:
                    visit(dep)
            
            order.append(service_name)
        
        for service_name in self.services:
            visit(service_name)
        
        return order
    
    def _get_service_stop_order(self) -> List[str]:
        """Get service stop order (reverse of start order)"""
        return list(reversed(self._get_service_start_order()))
    
    async def _health_monitor(self):
        """Monitor service health and perform auto-recovery"""
        logger.info("Service health monitor started")
        
        while self.is_running:
            try:
                for service_name, service_info in self.services.items():
                    if service_info.status != 'running':
                        continue
                    
                    # Check if health check is due
                    now = datetime.now()
                    if (service_info.last_health_check is None or 
                        (now - service_info.last_health_check).total_seconds() >= service_info.health_check_interval):
                        
                        health_status = await self._check_service_health(service_name)
                        service_info.health_status = health_status
                        service_info.last_health_check = now
                        
                        # Record health metrics
                        if self.metrics_collector:
                            self.metrics_collector.set_gauge('service_health_status', 
                                                            1 if health_status == 'healthy' else 0,
                                                            tags={'service': service_name})
                        
                        # Auto-recovery for unhealthy services
                        if (health_status == 'unhealthy' and 
                            self.auto_recovery_enabled and 
                            service_info.restart_count < self.max_restart_attempts):
                            
                            logger.warning(f"Service {service_name} is unhealthy, attempting restart")
                            
                            # Wait for cooldown period
                            await asyncio.sleep(self.restart_cooldown)
                            
                            # Restart the service
                            restart_task = asyncio.create_task(self.restart_service(service_name))
                            # Don't await here to avoid blocking health checks
                
                # Update service metrics
                self._update_service_metrics()
                
                await asyncio.sleep(10)  # Health check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(5)
        
        logger.info("Service health monitor stopped")
    
    async def _check_service_health(self, service_name: str) -> str:
        """Check health of a specific service"""
        service_info = self.services.get(service_name)
        if not service_info:
            return 'unknown'
        
        try:
            # Check if service has custom health check
            if hasattr(service_info.service, 'get_health_status'):
                return await service_info.service.get_health_status()
            
            # Basic health check - service is running
            if service_info.status == 'running':
                return 'healthy'
            elif service_info.status == 'error':
                return 'unhealthy'
            else:
                return 'degraded'
                
        except Exception as e:
            logger.error(f"Health check failed for {service_name}", error=str(e))
            return 'unhealthy'
    
    def _update_service_metrics(self):
        """Update service manager metrics"""
        running_count = sum(1 for s in self.services.values() if s.status == 'running')
        failed_count = sum(1 for s in self.services.values() if s.status == 'error')
        
        self.service_metrics.update({
            'total_services': len(self.services),
            'running_services': running_count,
            'failed_services': failed_count
        })
        
        if self.metrics_collector:
            self.metrics_collector.set_gauge('service_manager_running_services', running_count)
            self.metrics_collector.set_gauge('service_manager_failed_services', failed_count)
    
    async def _register_built_in_tasks(self):
        """Register built-in scheduled tasks"""
        if not self.task_scheduler:
            return
        
        # Register task functions
        self.task_scheduler.register_task('collect_miner_data', self._collect_all_miner_data)
        self.task_scheduler.register_task('cleanup_old_data', self._cleanup_old_data)
        self.task_scheduler.register_task('health_check_miners', self._health_check_all_miners)
        self.task_scheduler.register_task('update_system_metrics', self._update_system_metrics)
        
        # Schedule recurring tasks
        from datetime import timedelta
        
        # Collect miner data every 30 seconds
        collection_interval = self.config_service.get('miner.collection_interval_sec', 30)
        self.task_scheduler.schedule_recurring_function(
            self._collect_all_miner_data,
            timedelta(seconds=collection_interval),
            name='collect_miner_data'
        )
        
        # Cleanup old data daily
        self.task_scheduler.schedule_recurring_function(
            self._cleanup_old_data,
            timedelta(days=1),
            name='cleanup_old_data'
        )
        
        # Health check miners every 5 minutes
        self.task_scheduler.schedule_recurring_function(
            self._health_check_all_miners,
            timedelta(minutes=5),
            name='health_check_miners'
        )
        
        # Update system metrics every minute
        self.task_scheduler.schedule_recurring_function(
            self._update_system_metrics,
            timedelta(minutes=1),
            name='update_system_metrics'
        )
        
        logger.info("Registered built-in scheduled tasks")
    
    async def _collect_all_miner_data(self):
        """Task to collect data from all miners"""
        if not self.miner_service:
            return
        
        try:
            miner_ips = self.config_service.ips
            if miner_ips:
                miner_data = await self.miner_service.fetch_all_miners_concurrent(miner_ips)
                
                # Log successful collection
                self.metrics_collector.set_gauge('miner_data_collection_count', len(miner_data))
                self.metrics_collector.increment_counter('miner_data_collections_total')
                
                logger.debug(f"Collected data from {len(miner_data)} miners")
        
        except Exception as e:
            logger.error("Failed to collect miner data", error=str(e))
            self.metrics_collector.increment_counter('miner_data_collection_errors_total')
    
    async def _cleanup_old_data(self):
        """Task to cleanup old database data"""
        if not self.database_service:
            return
        
        try:
            # This would typically be implemented in the database service
            logger.info("Running database cleanup task")
            self.metrics_collector.increment_counter('database_cleanup_runs_total')
        
        except Exception as e:
            logger.error("Database cleanup failed", error=str(e))
            self.metrics_collector.increment_counter('database_cleanup_errors_total')
    
    async def _health_check_all_miners(self):
        """Task to perform health checks on all miners"""
        if not self.miner_service:
            return
        
        try:
            miner_ips = self.config_service.ips
            if miner_ips:
                # Get circuit breaker status
                cb_status = self.miner_service.get_circuit_breaker_status()
                blocked_miners = cb_status.get('blocked_miners', [])
                
                self.metrics_collector.set_gauge('miners_blocked_count', len(blocked_miners))
                
                logger.debug(f"Health checked {len(miner_ips)} miners, {len(blocked_miners)} blocked")
        
        except Exception as e:
            logger.error("Miner health check failed", error=str(e))
    
    async def _update_system_metrics(self):
        """Task to update system-level metrics"""
        try:
            # Update service manager metrics
            self._update_service_metrics()
            
            # Record task scheduler metrics
            if self.task_scheduler:
                scheduler_metrics = self.task_scheduler.get_metrics()
                for metric_name, value in scheduler_metrics.items():
                    if isinstance(value, (int, float)):
                        self.metrics_collector.set_gauge(f'task_scheduler_{metric_name}', value)
            
            logger.debug("Updated system metrics")
        
        except Exception as e:
            logger.error("System metrics update failed", error=str(e))
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown")
                asyncio.create_task(self.stop_all())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
        except Exception as e:
            logger.warning("Could not setup signal handlers", error=str(e))
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        await self.start_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop_all()


# Global service manager instance
_service_manager: Optional[AsyncServiceManager] = None


def get_service_manager(config_service: ConfigService = None) -> AsyncServiceManager:
    """Get global service manager instance"""
    global _service_manager
    if _service_manager is None:
        if config_service is None:
            raise ValueError("ConfigService required for first service manager initialization")
        _service_manager = AsyncServiceManager(config_service)
    return _service_manager