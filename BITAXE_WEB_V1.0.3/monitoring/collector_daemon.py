"""
Monitoring Collector Daemon

Standalone daemon process for comprehensive system monitoring, metrics collection,
alerting, and health monitoring. Designed to run as a background service.
"""

import asyncio
import signal
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitaxe_logging.structured_logger import get_logger
from monitoring.metrics_collector import MetricsCollector, get_metrics_collector
from monitoring.alert_manager import AlertManager, AlertRule, AlertSeverity
from monitoring.health_monitor import HealthMonitor, get_health_monitor
from monitoring.dashboard import get_dashboard

logger = get_logger("bitaxe.monitoring_daemon")


class MonitoringDaemon:
    """
    Comprehensive Monitoring Daemon
    
    Orchestrates all monitoring components:
    - Metrics collection and aggregation
    - Health monitoring and checks
    - Alert management and notifications
    - Dashboard data provisioning
    - Historical data management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.metrics_collector = get_metrics_collector()
        self.health_monitor = get_health_monitor()
        self.alert_manager = AlertManager(self.config.get('alerting', {}))
        self.dashboard = get_dashboard()
        
        # State management
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.start_time = None
        self.last_metrics_export = time.time()
        self.metrics_export_interval = self.config.get('metrics_export_interval', 60)
        
        logger.info("Monitoring Daemon initialized", config_keys=list(self.config.keys()))
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'metrics_collection': {
                'interval': 10,
                'retention_hours': 24,
                'export_format': 'prometheus'
            },
            'health_monitoring': {
                'interval': 30,
                'enable_recovery_actions': True
            },
            'alerting': {
                'evaluation_interval': 30,
                'notification_channels': {
                    'console': {
                        'type': 'webhook',
                        'url': 'http://localhost/alerts',
                        'enabled': False
                    }
                },
                'correlation_rules': [
                    {
                        'name_pattern': 'high_',
                        'group_by': ['name'],
                        'suppress_duplicates': True
                    }
                ]
            },
            'dashboard': {
                'update_interval': 5,
                'history_retention_hours': 168  # 7 days
            },
            'system': {
                'pid_file': '/tmp/bitaxe_monitoring.pid',
                'log_level': 'INFO',
                'prometheus_port': 9091
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge configs (file config overwrites defaults)
                    self._deep_merge(default_config, file_config)
                    logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}", error=str(e))
        
        return default_config
    
    def _deep_merge(self, base: Dict, overlay: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create PID file
        pid_file = self.config['system']['pid_file']
        try:
            with open(pid_file, 'w') as f:
                f.write(str(os.getpid()))
            logger.info(f"PID file created: {pid_file}")
        except Exception as e:
            logger.warning(f"Failed to create PID file: {e}")
    
    def _setup_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                condition="metrics.get('system', {}).get('cpu_percent', 0) > 80",
                severity=AlertSeverity.HIGH,
                message="CPU usage is above 80%",
                cooldown_seconds=300,
                tags={'category': 'system', 'resource': 'cpu'}
            ),
            AlertRule(
                name="High Memory Usage",
                condition="metrics.get('system', {}).get('memory_percent', 0) > 85",
                severity=AlertSeverity.HIGH,
                message="Memory usage is above 85%",
                cooldown_seconds=300,
                tags={'category': 'system', 'resource': 'memory'}
            ),
            AlertRule(
                name="High Disk Usage",
                condition="metrics.get('system', {}).get('disk_percent', 0) > 90",
                severity=AlertSeverity.CRITICAL,
                message="Disk usage is above 90%",
                cooldown_seconds=600,
                tags={'category': 'system', 'resource': 'disk'}
            ),
            AlertRule(
                name="High Error Rate",
                condition="metrics.get('application', {}).get('error_rate', 0) > 5",
                severity=AlertSeverity.CRITICAL,
                message="Application error rate is above 5%",
                cooldown_seconds=180,
                tags={'category': 'application', 'type': 'errors'}
            ),
            AlertRule(
                name="High Response Time",
                condition="metrics.get('application', {}).get('response_time_percentiles', {}).get('p95', 0) > 5000",
                severity=AlertSeverity.MEDIUM,
                message="95th percentile response time is above 5 seconds",
                cooldown_seconds=300,
                tags={'category': 'application', 'type': 'performance'}
            ),
            AlertRule(
                name="Service Health Degraded",
                condition="any(status != 'healthy' for status in metrics.get('services', {}).values())",
                severity=AlertSeverity.HIGH,
                message="One or more services are not healthy",
                cooldown_seconds=120,
                tags={'category': 'services', 'type': 'health'}
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
        
        logger.info(f"Added {len(default_rules)} default alert rules")
    
    async def start(self):
        """Start the monitoring daemon"""
        if self.is_running:
            logger.warning("Monitoring daemon is already running")
            return
        
        logger.info("Starting Monitoring Daemon")
        self.start_time = datetime.now()
        self.is_running = True
        
        try:
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Setup alert rules
            self._setup_alert_rules()
            
            # Set metrics callback for alert manager
            self.alert_manager.set_metrics_callback(
                lambda: self.metrics_collector.get_latest_metrics()
            )
            
            # Start all components
            await self.metrics_collector.start()
            await self.health_monitor.start()
            await self.alert_manager.start()
            await self.dashboard.start()
            
            # Start background tasks
            self.tasks = [
                asyncio.create_task(self._metrics_export_worker()),
                asyncio.create_task(self._health_integration_worker()),
                asyncio.create_task(self._status_reporter_worker()),
                asyncio.create_task(self._cleanup_worker())
            ]
            
            logger.info("Monitoring Daemon started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error("Error starting monitoring daemon", error=str(e))
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitoring daemon"""
        if not self.is_running:
            return
        
        logger.info("Stopping Monitoring Daemon")
        self.is_running = False
        
        try:
            # Cancel background tasks
            for task in self.tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Stop all components
            await self.dashboard.stop()
            await self.alert_manager.stop()
            await self.health_monitor.stop()
            await self.metrics_collector.stop()
            
            # Cleanup PID file
            pid_file = self.config['system']['pid_file']
            try:
                if os.path.exists(pid_file):
                    os.unlink(pid_file)
                    logger.info(f"PID file removed: {pid_file}")
            except Exception as e:
                logger.warning(f"Failed to remove PID file: {e}")
            
            uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
            logger.info(f"Monitoring Daemon stopped (uptime: {uptime})")
            
        except Exception as e:
            logger.error("Error stopping monitoring daemon", error=str(e))
    
    async def _metrics_export_worker(self):
        """Background worker for metrics export"""
        logger.debug("Metrics export worker started")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_metrics_export >= self.metrics_export_interval:
                    # Export metrics in configured format
                    export_format = self.config['metrics_collection'].get('export_format', 'json')
                    metrics_data = self.metrics_collector.export_metrics(export_format)
                    
                    # Write to file or send to external system
                    # For now, just log the export
                    logger.debug(f"Metrics exported ({len(metrics_data)} bytes, format: {export_format})")
                    
                    self.last_metrics_export = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics export worker error", error=str(e))
                await asyncio.sleep(30)
        
        logger.debug("Metrics export worker stopped")
    
    async def _health_integration_worker(self):
        """Background worker for health monitor integration"""
        logger.debug("Health integration worker started")
        
        while self.is_running:
            try:
                # Get system health
                system_health = self.health_monitor.get_system_health()
                
                # Record health metrics
                self.metrics_collector.set_gauge('system_health_status', 
                                                1 if system_health['status'] == 'healthy' else 0,
                                                tags={'status': system_health['status']})
                
                self.metrics_collector.set_gauge('health_checks_total', 
                                                system_health['summary']['total_checks'])
                
                self.metrics_collector.set_gauge('health_checks_healthy', 
                                                system_health['summary']['healthy'])
                
                self.metrics_collector.set_gauge('health_checks_unhealthy', 
                                                system_health['summary']['unhealthy'])
                
                # Create alerts for unhealthy services
                if system_health['status'] in ['unhealthy', 'degraded']:
                    unhealthy_checks = [
                        name for name, check in system_health['checks'].items()
                        if check['status'] == 'unhealthy'
                    ]
                    
                    if unhealthy_checks:
                        self.alert_manager.create_manual_alert(
                            name="Health Check Failures",
                            message=f"Unhealthy checks: {', '.join(unhealthy_checks)}",
                            severity=AlertSeverity.HIGH,
                            source="health_monitor",
                            tags={'category': 'health'},
                            details={'unhealthy_checks': unhealthy_checks}
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health integration worker error", error=str(e))
                await asyncio.sleep(60)
        
        logger.debug("Health integration worker stopped")
    
    async def _status_reporter_worker(self):
        """Background worker for status reporting"""
        logger.debug("Status reporter worker started")
        
        while self.is_running:
            try:
                # Log daemon status periodically
                uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
                
                status = {
                    'daemon_uptime_seconds': uptime.total_seconds(),
                    'metrics_collector_running': self.metrics_collector.is_running,
                    'health_monitor_running': self.health_monitor.is_running,
                    'alert_manager_running': self.alert_manager.is_running,
                    'dashboard_running': self.dashboard.is_running,
                    'active_tasks': len([t for t in self.tasks if not t.done()]),
                    'memory_usage_mb': self._get_memory_usage()
                }
                
                # Record daemon metrics
                for key, value in status.items():
                    if isinstance(value, (int, float)):
                        self.metrics_collector.set_gauge(f'daemon_{key}', value)
                
                logger.info("Daemon status", **status)
                
                await asyncio.sleep(300)  # Report every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Status reporter worker error", error=str(e))
                await asyncio.sleep(300)
        
        logger.debug("Status reporter worker stopped")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        logger.debug("Cleanup worker started")
        
        while self.is_running:
            try:
                # Cleanup historical data based on retention settings
                retention_hours = self.config['dashboard'].get('history_retention_hours', 168)
                cutoff_time = datetime.now() - timedelta(hours=retention_hours)
                
                # This would clean up database records, log files, etc.
                # For now, just log the cleanup activity
                logger.debug(f"Cleanup maintenance (retention: {retention_hours}h)")
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup worker error", error=str(e))
                await asyncio.sleep(3600)
        
        logger.debug("Cleanup worker stopped")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def get_daemon_status(self) -> Dict[str, Any]:
        """Get comprehensive daemon status"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'daemon': {
                'running': self.is_running,
                'uptime_seconds': uptime.total_seconds(),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'pid': os.getpid(),
                'memory_usage_mb': self._get_memory_usage()
            },
            'components': {
                'metrics_collector': {
                    'running': self.metrics_collector.is_running,
                    'collection_interval': self.metrics_collector.collection_interval
                },
                'health_monitor': {
                    'running': self.health_monitor.is_running,
                    'system_status': self.health_monitor.system_health_status.value
                },
                'alert_manager': {
                    'running': self.alert_manager.is_running,
                    'active_alerts': len(self.alert_manager.active_alerts),
                    'total_rules': len(self.alert_manager.alert_rules)
                },
                'dashboard': {
                    'running': self.dashboard.is_running,
                    'websocket_clients': len(self.dashboard.websocket_clients)
                }
            },
            'tasks': {
                'total': len(self.tasks),
                'running': len([t for t in self.tasks if not t.done()]),
                'completed': len([t for t in self.tasks if t.done() and not t.cancelled()]),
                'cancelled': len([t for t in self.tasks if t.cancelled()]),
                'failed': len([t for t in self.tasks if t.done() and t.exception()])
            },
            'config': {
                'metrics_export_interval': self.metrics_export_interval,
                'log_level': self.config['system']['log_level']
            }
        }


async def main():
    """Main entry point for the monitoring daemon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BitAxe Monitoring Daemon')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon (detach from terminal)')
    parser.add_argument('--pid-file', type=str, help='PID file path (overrides config)')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level')
    
    args = parser.parse_args()
    
    # Initialize daemon
    daemon = MonitoringDaemon(config_path=args.config)
    
    # Override config with command line arguments
    if args.pid_file:
        daemon.config['system']['pid_file'] = args.pid_file
    if args.log_level:
        daemon.config['system']['log_level'] = args.log_level
    
    try:
        if args.daemon:
            # Detach from terminal (basic daemonization)
            if os.fork() > 0:
                sys.exit(0)  # Parent exits
            
            # Child continues as daemon
            os.setsid()
            os.chdir('/')
            
            # Redirect stdin/stdout/stderr
            with open('/dev/null', 'r') as f:
                os.dup2(f.fileno(), sys.stdin.fileno())
            with open('/dev/null', 'w') as f:
                os.dup2(f.fileno(), sys.stdout.fileno())
                os.dup2(f.fileno(), sys.stderr.fileno())
        
        # Start the daemon
        await daemon.start()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Daemon failed", error=str(e))
        sys.exit(1)


if __name__ == '__main__':
    # Run the daemon
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)