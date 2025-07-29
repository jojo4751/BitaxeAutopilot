from typing import Dict, Any, Optional, Type, TypeVar
import os

from .config_service import ConfigService
from .database_service import DatabaseService
from .miner_service import MinerService
from .benchmark_service import BenchmarkService
from .autopilot_service import AutopilotService

T = TypeVar('T')


class ServiceContainer:
    """Dependency injection container for managing services"""
    
    def __init__(self, config_path: Optional[str] = None):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self.config_path = config_path or os.path.join("config", "config.json")
        
        # Initialize core services
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all services with proper dependency injection"""
        
        # Initialize ConfigService first (no dependencies)
        config_service = ConfigService(self.config_path)
        self._register_singleton('config_service', config_service)
        
        # Initialize DatabaseService (depends on ConfigService)
        database_service = DatabaseService(config_service)
        database_service.initialize_tables()  # Ensure tables exist
        self._register_singleton('database_service', database_service)
        
        # Initialize MinerService (depends on ConfigService and DatabaseService)
        miner_service = MinerService(config_service, database_service)
        self._register_singleton('miner_service', miner_service)
        
        # Initialize BenchmarkService (depends on ConfigService, DatabaseService, MinerService)
        benchmark_service = BenchmarkService(config_service, database_service, miner_service)
        self._register_singleton('benchmark_service', benchmark_service)
        
        # Initialize AutopilotService (depends on all other services)
        autopilot_service = AutopilotService(
            config_service, database_service, miner_service, benchmark_service
        )
        self._register_singleton('autopilot_service', autopilot_service)

    def _register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton service instance"""
        self._singletons[name] = instance

    def get(self, service_name: str) -> Any:
        """Get a service by name"""
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        raise ValueError(f"Service '{service_name}' not found")

    def get_config_service(self) -> ConfigService:
        """Get the ConfigService instance"""
        return self.get('config_service')

    def get_database_service(self) -> DatabaseService:
        """Get the DatabaseService instance"""
        return self.get('database_service')

    def get_miner_service(self) -> MinerService:
        """Get the MinerService instance"""
        return self.get('miner_service')

    def get_benchmark_service(self) -> BenchmarkService:
        """Get the BenchmarkService instance"""
        return self.get('benchmark_service')

    def get_autopilot_service(self) -> AutopilotService:
        """Get the AutopilotService instance"""
        return self.get('autopilot_service')

    def reload_config(self) -> None:
        """Reload configuration across all services"""
        config_service = self.get_config_service()
        config_service.reload_config()
        
        # Log the reload
        database_service = self.get_database_service()
        database_service.log_event("SYSTEM", "CONFIG_RELOADED", "Configuration reloaded")

    def shutdown(self) -> None:
        """Shutdown all services gracefully"""
        try:
            # Shutdown autopilot first
            autopilot_service = self.get_autopilot_service()
            autopilot_service.stop_autopilot()
            
            # Shutdown benchmark service
            benchmark_service = self.get_benchmark_service()
            benchmark_service.shutdown()
            
            # Log shutdown
            database_service = self.get_database_service()
            database_service.log_event("SYSTEM", "SERVICES_SHUTDOWN", "All services shut down")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health_status = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": None
        }
        
        try:
            from datetime import datetime
            health_status["timestamp"] = datetime.now().isoformat()
            
            # Check ConfigService
            config_service = self.get_config_service()
            try:
                config_service.get_config()
                health_status["services"]["config"] = "healthy"
            except Exception as e:
                health_status["services"]["config"] = f"error: {e}"
                health_status["overall_status"] = "degraded"
            
            # Check DatabaseService
            database_service = self.get_database_service()
            try:
                with database_service.get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                health_status["services"]["database"] = "healthy"
            except Exception as e:
                health_status["services"]["database"] = f"error: {e}"
                health_status["overall_status"] = "degraded"
            
            # Check MinerService
            miner_service = self.get_miner_service()
            try:
                online_miners = 0
                total_miners = len(config_service.ips)
                for ip in config_service.ips:
                    if miner_service.is_miner_online(ip):
                        online_miners += 1
                
                health_status["services"]["miners"] = f"{online_miners}/{total_miners} online"
                if online_miners == 0 and total_miners > 0:
                    health_status["overall_status"] = "degraded"
            except Exception as e:
                health_status["services"]["miners"] = f"error: {e}"
                health_status["overall_status"] = "degraded"
            
            # Check BenchmarkService
            benchmark_service = self.get_benchmark_service()
            try:
                status = benchmark_service.get_benchmark_status()
                health_status["services"]["benchmarks"] = f"active: {status['total_active']}"
            except Exception as e:
                health_status["services"]["benchmarks"] = f"error: {e}"
                health_status["overall_status"] = "degraded"
            
            # Check AutopilotService
            autopilot_service = self.get_autopilot_service()
            try:
                status = autopilot_service.get_autopilot_status()
                health_status["services"]["autopilot"] = "running" if status["is_running"] else "stopped"
            except Exception as e:
                health_status["services"]["autopilot"] = f"error: {e}"
                health_status["overall_status"] = "degraded"
                
        except Exception as e:
            health_status["overall_status"] = "error"
            health_status["error"] = str(e)
        
        return health_status

    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics from all services"""
        stats = {}
        
        try:
            # Config stats
            config_service = self.get_config_service()
            stats["config"] = {
                "total_miners": len(config_service.ips),
                "log_interval": config_service.log_interval,
                "benchmark_interval": config_service.benchmark_interval
            }
            
            # Database stats
            database_service = self.get_database_service()
            latest_status = database_service.get_latest_status()
            stats["miners"] = {
                "total_configured": len(config_service.ips),
                "last_seen": len(latest_status),
                "latest_data": latest_status
            }
            
            # Benchmark stats
            benchmark_service = self.get_benchmark_service()
            benchmark_status = benchmark_service.get_benchmark_status()
            stats["benchmarks"] = benchmark_status
            
            # Autopilot stats
            autopilot_service = self.get_autopilot_service()
            autopilot_status = autopilot_service.get_autopilot_status()
            stats["autopilot"] = autopilot_status
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats


# Global service container instance
_container: Optional[ServiceContainer] = None


def get_container(config_path: Optional[str] = None) -> ServiceContainer:
    """Get the global service container instance"""
    global _container
    if _container is None:
        _container = ServiceContainer(config_path)
    return _container


def initialize_services(config_path: Optional[str] = None) -> ServiceContainer:
    """Initialize and return the service container"""
    global _container
    _container = ServiceContainer(config_path)
    return _container


def shutdown_services() -> None:
    """Shutdown all services"""
    global _container
    if _container:
        _container.shutdown()
        _container = None