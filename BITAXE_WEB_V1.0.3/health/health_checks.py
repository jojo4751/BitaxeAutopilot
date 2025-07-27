import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import requests
from dataclasses import dataclass

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import HealthCheckError, ServiceUnavailableError

logger = get_logger("bitaxe.health")


class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


class HealthCheck(ABC):
    """Abstract base class for health checks"""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Perform the health check"""
        pass
    
    def _timed_check(self) -> HealthCheckResult:
        """Execute check with timing"""
        start_time = time.time()
        try:
            result = self.check()
            duration = (time.time() - start_time) * 1000
            result.duration_ms = duration
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Health check failed for {self.name}",
                        component=self.name,
                        duration_ms=duration,
                        error=str(e))
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=datetime.now(),
                duration_ms=duration
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity"""
    
    def __init__(self, database_manager, timeout_seconds: float = 5.0):
        super().__init__("database", timeout_seconds)
        self.database_manager = database_manager
    
    def check(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # Test basic connectivity
            is_healthy = self.database_manager.health_check()
            
            if is_healthy:
                # Get connection pool stats
                conn_info = self.database_manager.get_connection_info()
                
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database is accessible",
                    details={
                        "connection_info": conn_info,
                        "pool_stats": {
                            "checked_out": conn_info.get("checked_out", 0),
                            "checked_in": conn_info.get("checked_in", 0),
                            "pool_size": conn_info.get("pool_size", 0)
                        }
                    },
                    timestamp=datetime.now(),
                    duration_ms=0
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Database is not accessible",
                    details={},
                    timestamp=datetime.now(),
                    duration_ms=0
                )
                
        except Exception as e:
            raise HealthCheckError(self.name, str(e))


class MinerHealthCheck(HealthCheck):
    """Health check for individual miner connectivity"""
    
    def __init__(self, ip: str, timeout_seconds: float = 3.0):
        super().__init__(f"miner_{ip}", timeout_seconds)
        self.ip = ip
    
    def check(self) -> HealthCheckResult:
        """Check miner connectivity"""
        try:
            response = requests.get(
                f"http://{self.ip}/api/system/info",
                timeout=self.timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                temp = data.get('temp', 0)
                hashrate = data.get('hashRate', 0)
                
                # Determine health based on metrics
                status = HealthStatus.HEALTHY
                warnings = []
                
                if temp > 80:
                    status = HealthStatus.DEGRADED
                    warnings.append(f"High temperature: {temp}°C")
                
                if hashrate <= 0:
                    status = HealthStatus.DEGRADED
                    warnings.append("No hash rate detected")
                
                return HealthCheckResult(
                    component=self.name,
                    status=status,
                    message="Miner is responding" + (f" (warnings: {', '.join(warnings)})" if warnings else ""),
                    details={
                        "ip": self.ip,
                        "temperature": temp,
                        "hashrate": hashrate,
                        "power": data.get('power', 0),
                        "hostname": data.get('hostname'),
                        "warnings": warnings
                    },
                    timestamp=datetime.now(),
                    duration_ms=0
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Miner returned HTTP {response.status_code}",
                    details={"ip": self.ip, "status_code": response.status_code},
                    timestamp=datetime.now(),
                    duration_ms=0
                )
                
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Miner request timed out",
                details={"ip": self.ip, "timeout_seconds": self.timeout_seconds},
                timestamp=datetime.now(),
                duration_ms=0
            )
        
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Cannot connect to miner",
                details={"ip": self.ip},
                timestamp=datetime.now(),
                duration_ms=0
            )
        
        except Exception as e:
            raise HealthCheckError(self.name, str(e))


class ServiceHealthCheck(HealthCheck):
    """Health check for internal services"""
    
    def __init__(self, service_name: str, check_function: Callable[[], bool],
                 timeout_seconds: float = 5.0):
        super().__init__(f"service_{service_name}", timeout_seconds)
        self.service_name = service_name
        self.check_function = check_function
    
    def check(self) -> HealthCheckResult:
        """Check service health using provided function"""
        try:
            is_healthy = self.check_function()
            
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message=f"Service {self.service_name} is {'healthy' if is_healthy else 'unhealthy'}",
                details={"service_name": self.service_name},
                timestamp=datetime.now(),
                duration_ms=0
            )
            
        except Exception as e:
            raise HealthCheckError(self.name, str(e))


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, memory, disk)"""
    
    def __init__(self, timeout_seconds: float = 2.0):
        super().__init__("system_resources", timeout_seconds)
    
    def check(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            import psutil
            
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine health based on thresholds
            status = HealthStatus.HEALTHY
            warnings = []
            
            if cpu_percent > 90:
                status = HealthStatus.DEGRADED
                warnings.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                status = HealthStatus.DEGRADED
                warnings.append(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                status = HealthStatus.DEGRADED
                warnings.append(f"High disk usage: {disk.percent}%")
            
            return HealthCheckResult(
                component=self.name,
                status=status,
                message="System resources checked" + (f" (warnings: {', '.join(warnings)})" if warnings else ""),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "warnings": warnings
                },
                timestamp=datetime.now(),
                duration_ms=0
            )
            
        except ImportError:
            return HealthCheckResult(
                component=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for system monitoring",
                details={},
                timestamp=datetime.now(),
                duration_ms=0
            )
        
        except Exception as e:
            raise HealthCheckError(self.name, str(e))


class HealthCheckManager:
    """Manager for running and coordinating health checks"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_intervals: Dict[str, float] = {}
        self.running = False
        self.check_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
    
    def add_health_check(self, health_check: HealthCheck, interval_seconds: float = 60.0):
        """Add a health check to be monitored"""
        with self.lock:
            self.health_checks[health_check.name] = health_check
            self.check_intervals[health_check.name] = interval_seconds
            
        logger.info(f"Added health check: {health_check.name}",
                   component=health_check.name,
                   interval_seconds=interval_seconds)
    
    def remove_health_check(self, name: str):
        """Remove a health check"""
        with self.lock:
            self.health_checks.pop(name, None)
            self.check_intervals.pop(name, None)
            self.last_results.pop(name, None)
        
        logger.info(f"Removed health check: {name}",
                   component=name)
    
    def run_single_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a single health check by name"""
        health_check = self.health_checks.get(name)
        if not health_check:
            logger.warning(f"Health check not found: {name}")
            return None
        
        try:
            result = health_check._timed_check()
            with self.lock:
                self.last_results[name] = result
            
            logger.debug(f"Health check completed",
                        component=name,
                        status=result.status.value,
                        duration_ms=result.duration_ms)
            
            return result
        
        except Exception as e:
            logger.error(f"Health check execution failed",
                        component=name,
                        error=str(e))
            return None
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results"""
        results = {}
        
        for name in self.health_checks.keys():
            result = self.run_single_check(name)
            if result:
                results[name] = result
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        with self.lock:
            results = self.last_results.copy()
        
        if not results:
            # Run checks if no recent results
            results = self.run_all_checks()
        
        # Calculate overall status
        statuses = [result.status for result in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        # Count status types
        status_counts = {}
        for status in HealthStatus:
            status_counts[status.value] = sum(1 for s in statuses if s == status)
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "total_checks": len(results),
            "status_counts": status_counts,
            "checks": {name: result.to_dict() for name, result in results.items()}
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.running:
            logger.warning("Health monitoring already running")
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health monitoring started",
                   total_checks=len(self.health_checks))
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_check_times = {name: datetime.min for name in self.health_checks.keys()}
        
        while self.running:
            try:
                now = datetime.now()
                
                # Check which health checks need to run
                for name, health_check in self.health_checks.items():
                    interval = self.check_intervals[name]
                    last_check = last_check_times.get(name, datetime.min)
                    
                    if (now - last_check).total_seconds() >= interval:
                        self.run_single_check(name)
                        last_check_times[name] = now
                
                # Sleep for a short interval
                time.sleep(5.0)
                
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                time.sleep(10.0)  # Sleep longer on errors
    
    def get_health_status(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for specific component or all components"""
        if component:
            with self.lock:
                result = self.last_results.get(component)
            
            if result:
                return {
                    "component": component,
                    "status": result.status.value,
                    "timestamp": result.timestamp.isoformat(),
                    "details": result.to_dict()
                }
            else:
                return {
                    "component": component,
                    "status": "unknown",
                    "message": "No recent health check data"
                }
        else:
            return self.get_overall_health()


# Global health check manager instance
_health_manager: Optional[HealthCheckManager] = None


def get_health_manager() -> HealthCheckManager:
    """Get global health check manager"""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthCheckManager()
    return _health_manager


def initialize_health_checks(container):
    """Initialize health checks for all system components"""
    manager = get_health_manager()
    
    # Database health check
    db_check = DatabaseHealthCheck(container.get_database_service().db_manager)
    manager.add_health_check(db_check, interval_seconds=30.0)
    
    # Miner health checks
    config_service = container.get_config_service()
    for ip in config_service.ips:
        miner_check = MinerHealthCheck(ip)
        manager.add_health_check(miner_check, interval_seconds=60.0)
    
    # Service health checks
    autopilot_check = ServiceHealthCheck(
        "autopilot",
        lambda: container.get_autopilot_service().is_running
    )
    manager.add_health_check(autopilot_check, interval_seconds=30.0)
    
    benchmark_check = ServiceHealthCheck(
        "benchmark",
        lambda: len(container.get_benchmark_service().get_benchmarking_ips()) >= 0
    )
    manager.add_health_check(benchmark_check, interval_seconds=60.0)
    
    # System resources check
    system_check = SystemResourceHealthCheck()
    manager.add_health_check(system_check, interval_seconds=120.0)
    
    # Start monitoring
    manager.start_monitoring()
    
    logger.info("Health checks initialized",
               total_checks=len(manager.health_checks))
    
    return manager