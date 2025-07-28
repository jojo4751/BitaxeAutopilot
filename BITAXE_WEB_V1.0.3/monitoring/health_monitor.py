"""
Comprehensive Health Monitoring System

Advanced health monitoring with service health checks, dependency validation,
performance monitoring, and automated recovery actions.
"""

import asyncio
import time
import psutil
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

from bitaxe_logging.structured_logger import get_logger

logger = get_logger("bitaxe.health_monitor")


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks"""
    HTTP = "http"
    TCP = "tcp"
    DATABASE = "database"
    REDIS = "redis"
    CUSTOM = "custom"
    SYSTEM = "system"
    SERVICE = "service"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'response_time_ms': self.response_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'error': self.error
        }


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_type: HealthCheckType
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10
    critical: bool = False  # If true, failure affects overall system health
    
    # HTTP-specific
    url: Optional[str] = None
    expected_status: int = 200
    expected_content: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    # TCP-specific
    host: Optional[str] = None
    port: Optional[int] = None
    
    # Database-specific
    connection_string: Optional[str] = None
    query: Optional[str] = None
    
    # Custom check function
    custom_check: Optional[Callable[[], Any]] = None
    
    # Thresholds
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    
    # Retry configuration
    retry_count: int = 3
    retry_delay: float = 1.0
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check"""
        start_time = time.time()
        
        try:
            if self.check_type == HealthCheckType.HTTP:
                result = await self._check_http()
            elif self.check_type == HealthCheckType.TCP:
                result = await self._check_tcp()
            elif self.check_type == HealthCheckType.DATABASE:
                result = await self._check_database()
            elif self.check_type == HealthCheckType.REDIS:
                result = await self._check_redis()
            elif self.check_type == HealthCheckType.SYSTEM:
                result = await self._check_system()
            elif self.check_type == HealthCheckType.CUSTOM:
                result = await self._check_custom()
            else:
                result = HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Unknown check type: {self.check_type.value}"
                )
            
            result.response_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def _check_http(self) -> HealthCheckResult:
        """Execute HTTP health check"""
        if not self.url:
            raise ValueError("URL required for HTTP health check")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
            async with session.get(self.url, headers=self.headers) as response:
                status = HealthStatus.HEALTHY
                message = f"HTTP {response.status}"
                
                # Check status code
                if response.status != self.expected_status:
                    status = HealthStatus.UNHEALTHY
                    message = f"Expected status {self.expected_status}, got {response.status}"
                
                # Check content if specified
                if self.expected_content and status == HealthStatus.HEALTHY:
                    content = await response.text()
                    if self.expected_content not in content:
                        status = HealthStatus.UNHEALTHY
                        message = f"Expected content '{self.expected_content}' not found"
                
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=message,
                    details={
                        'status_code': response.status,
                        'content_length': len(await response.text()) if status == HealthStatus.HEALTHY else 0
                    }
                )
    
    async def _check_tcp(self) -> HealthCheckResult:
        """Execute TCP health check"""
        if not self.host or not self.port:
            raise ValueError("Host and port required for TCP health check")
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout_seconds
            )
            writer.close()
            await writer.wait_closed()
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"TCP connection to {self.host}:{self.port} successful"
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"TCP connection to {self.host}:{self.port} timed out"
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"TCP connection to {self.host}:{self.port} failed: {str(e)}"
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Execute database health check"""
        # This would need specific database implementations
        # For now, just check if connection string is provided
        if not self.connection_string:
            raise ValueError("Connection string required for database health check")
        
        # Placeholder implementation
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Database check placeholder - implement specific DB logic"
        )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Execute Redis health check"""
        # Placeholder - would need Redis client implementation
        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Redis check placeholder - implement Redis client logic"
        )
    
    async def _check_system(self) -> HealthCheckResult:
        """Execute system health check"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            messages = []
            
            # Check CPU
            if self.critical_threshold and cpu_percent > self.critical_threshold:
                status = HealthStatus.UNHEALTHY
                messages.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif self.warning_threshold and cpu_percent > self.warning_threshold:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                messages.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            # Check Memory
            memory_threshold_critical = 90.0
            memory_threshold_warning = 80.0
            
            if memory.percent > memory_threshold_critical:
                status = HealthStatus.UNHEALTHY
                messages.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > memory_threshold_warning:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                messages.append(f"Memory usage high: {memory.percent:.1f}%")
            
            # Check Disk
            disk_threshold_critical = 95.0
            disk_threshold_warning = 85.0
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > disk_threshold_critical:
                status = HealthStatus.UNHEALTHY
                messages.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > disk_threshold_warning:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                messages.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources within normal limits"
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_free_gb': disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"System check failed: {str(e)}",
                error=str(e)
            )
    
    async def _check_custom(self) -> HealthCheckResult:
        """Execute custom health check"""
        if not self.custom_check:
            raise ValueError("Custom check function required for custom health check")
        
        try:
            if asyncio.iscoroutinefunction(self.custom_check):
                result = await self.custom_check()
            else:
                result = self.custom_check()
            
            # Custom check should return dict with status and message
            if isinstance(result, dict):
                status_str = result.get('status', 'healthy')
                status = HealthStatus(status_str) if status_str in [s.value for s in HealthStatus] else HealthStatus.UNKNOWN
                
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=result.get('message', 'Custom check completed'),
                    details=result.get('details', {})
                )
            else:
                # Simple boolean result
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                return HealthCheckResult(
                    name=self.name,
                    status=status,
                    message=f"Custom check result: {result}"
                )
                
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check failed: {str(e)}",
                error=str(e)
            )


class HealthMonitor:
    """
    Comprehensive Health Monitoring System
    
    Features:
    - Multiple health check types (HTTP, TCP, database, system, custom)
    - Configurable intervals and thresholds
    - Retry logic and error handling
    - Health status aggregation
    - Historical health tracking
    - Alert integration
    - Automated recovery actions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Results storage
        self.latest_results: Dict[str, HealthCheckResult] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Overall system health
        self.system_health_status = HealthStatus.UNKNOWN
        self.system_health_message = "Health monitoring not started"
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Callbacks
        self.status_change_callbacks: List[Callable[[str, HealthCheckResult], None]] = []
        self.recovery_actions: Dict[str, Callable[[HealthCheckResult], None]] = {}
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'healthy_checks': 0,
            'unhealthy_checks': 0,
            'degraded_checks': 0,
            'last_full_scan': None
        }
        
        logger.info("Health Monitor initialized")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name} ({health_check.check_type.value})")
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.latest_results:
                del self.latest_results[name]
            logger.info(f"Removed health check: {name}")
    
    def add_status_change_callback(self, callback: Callable[[str, HealthCheckResult], None]):
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)
    
    def add_recovery_action(self, check_name: str, action: Callable[[HealthCheckResult], None]):
        """Add recovery action for specific health check"""
        self.recovery_actions[check_name] = action
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        healthy_count = len([r for r in self.latest_results.values() if r.status == HealthStatus.HEALTHY])
        degraded_count = len([r for r in self.latest_results.values() if r.status == HealthStatus.DEGRADED])
        unhealthy_count = len([r for r in self.latest_results.values() if r.status == HealthStatus.UNHEALTHY])
        total_count = len(self.latest_results)
        
        # Determine overall status
        if unhealthy_count > 0:
            # Check if any critical services are unhealthy
            critical_unhealthy = any(
                r.status == HealthStatus.UNHEALTHY and self.health_checks[r.name].critical
                for r in self.latest_results.values()
            )
            overall_status = HealthStatus.UNHEALTHY if critical_unhealthy else HealthStatus.DEGRADED
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            'status': overall_status.value,
            'message': self.system_health_message,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_count,
                'healthy': healthy_count,
                'degraded': degraded_count,
                'unhealthy': unhealthy_count
            },
            'checks': {name: result.to_dict() for name, result in self.latest_results.items()},
            'statistics': self.stats
        }
    
    def get_health_check_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get latest result for specific health check"""
        return self.latest_results.get(name)
    
    def get_health_history(self, name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health history for specific check"""
        if name not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            result.to_dict() for result in self.health_history[name]
            if result.timestamp >= cutoff_time
        ]
    
    async def run_health_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check"""
        if name not in self.health_checks:
            logger.warning(f"Health check not found: {name}")
            return None
        
        health_check = self.health_checks[name]
        if not health_check.enabled:
            return None
        
        try:
            # Execute with retries
            last_exception = None
            for attempt in range(health_check.retry_count + 1):
                try:
                    result = await health_check.execute()
                    
                    # Store result
                    previous_result = self.latest_results.get(name)
                    self.latest_results[name] = result
                    self.health_history[name].append(result)
                    
                    # Update statistics
                    self.stats['total_checks'] += 1
                    if result.status == HealthStatus.HEALTHY:
                        self.stats['healthy_checks'] += 1
                    elif result.status == HealthStatus.DEGRADED:
                        self.stats['degraded_checks'] += 1
                    elif result.status == HealthStatus.UNHEALTHY:
                        self.stats['unhealthy_checks'] += 1
                    
                    # Check for status changes
                    if previous_result and previous_result.status != result.status:
                        logger.info(f"Health check status changed: {name} {previous_result.status.value} -> {result.status.value}")
                        
                        # Notify callbacks
                        for callback in self.status_change_callbacks:
                            try:
                                callback(name, result)
                            except Exception as e:
                                logger.error(f"Status change callback error for {name}", error=str(e))
                        
                        # Execute recovery action if available
                        if result.status == HealthStatus.UNHEALTHY and name in self.recovery_actions:
                            try:
                                recovery_action = self.recovery_actions[name]
                                if asyncio.iscoroutinefunction(recovery_action):
                                    await recovery_action(result)
                                else:
                                    recovery_action(result)
                                logger.info(f"Recovery action executed for {name}")
                            except Exception as e:
                                logger.error(f"Recovery action failed for {name}", error=str(e))
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    if attempt < health_check.retry_count:
                        logger.warning(f"Health check {name} failed (attempt {attempt + 1}), retrying: {str(e)}")
                        await asyncio.sleep(health_check.retry_delay)
                    else:
                        logger.error(f"Health check {name} failed after {health_check.retry_count + 1} attempts", error=str(e))
            
            # All retries failed
            if last_exception:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed after retries: {str(last_exception)}",
                    error=str(last_exception)
                )
                self.latest_results[name] = error_result
                self.health_history[name].append(error_result)
                return error_result
                
        except Exception as e:
            logger.error(f"Unexpected error in health check {name}", error=str(e))
            error_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Unexpected error: {str(e)}",
                error=str(e)
            )
            self.latest_results[name] = error_result
            return error_result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all enabled health checks"""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name, health_check in self.health_checks.items():
            if health_check.enabled:
                tasks.append((name, self.run_health_check(name)))
        
        if tasks:
            completed_tasks = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for (name, _), result in zip(tasks, completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"Health check {name} raised exception", error=str(result))
                    results[name] = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Exception: {str(result)}",
                        error=str(result)
                    )
                elif result:
                    results[name] = result
        
        self.stats['last_full_scan'] = datetime.now().isoformat()
        return results
    
    async def start(self):
        """Start the health monitor"""
        if self.is_running:
            return
        
        logger.info("Starting Health Monitor")
        self.is_running = True
        
        # Start background monitoring
        self.monitor_task = asyncio.create_task(self._monitor_worker())
        
        logger.info("Health Monitor started")
    
    async def stop(self):
        """Stop the health monitor"""
        if not self.is_running:
            return
        
        logger.info("Stopping Health Monitor")
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health Monitor stopped")
    
    async def _monitor_worker(self):
        """Background worker for continuous health monitoring"""
        logger.debug("Health monitor worker started")
        
        while self.is_running:
            try:
                # Run all health checks
                await self.run_all_health_checks()
                
                # Update system health status
                system_health = self.get_system_health()
                self.system_health_status = HealthStatus(system_health['status'])
                self.system_health_message = f"System health: {system_health['summary']['healthy']}/{system_health['summary']['total_checks']} checks healthy"
                
                # Wait for next iteration (use minimum interval from all checks)
                min_interval = min((check.interval_seconds for check in self.health_checks.values() if check.enabled), default=30)
                await asyncio.sleep(min_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor worker error", error=str(e))
                await asyncio.sleep(30)
        
        logger.debug("Health monitor worker stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# Predefined health checks
def create_default_health_checks() -> List[HealthCheck]:
    """Create default health checks for BitAxe system"""
    return [
        # System resource check
        HealthCheck(
            name="system_resources",
            check_type=HealthCheckType.SYSTEM,
            critical=True,
            interval_seconds=30,
            warning_threshold=80.0,  # CPU warning at 80%
            critical_threshold=95.0   # CPU critical at 95%
        ),
        
        # Application HTTP health check
        HealthCheck(
            name="app_health_endpoint",
            check_type=HealthCheckType.HTTP,
            url="http://localhost:5000/health",
            expected_status=200,
            critical=True,
            interval_seconds=15
        ),
        
        # Database connectivity
        HealthCheck(
            name="database_connectivity",
            check_type=HealthCheckType.TCP,
            host="localhost",
            port=5432,
            critical=True,
            interval_seconds=30
        ),
        
        # Redis connectivity
        HealthCheck(
            name="redis_connectivity",
            check_type=HealthCheckType.TCP,
            host="localhost",
            port=6379,
            critical=False,
            interval_seconds=30
        )
    ]


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        
        # Add default health checks
        for health_check in create_default_health_checks():
            _health_monitor.add_health_check(health_check)
    
    return _health_monitor