"""
Real-time Monitoring Dashboard

Interactive monitoring dashboard with real-time metrics, charts, and system status.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import statistics

from quart import Blueprint, render_template, jsonify, websocket, request
from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from async_services.async_service_manager import get_service_manager

logger = get_logger("bitaxe.monitoring_dashboard")

# Create monitoring blueprint
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')


@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure"""
    timestamp: str
    system_metrics: Dict[str, Any]
    miner_metrics: Dict[str, Any]
    service_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class MonitoringDashboard:
    """
    Real-time monitoring dashboard
    
    Features:
    - Real-time metrics display
    - Interactive charts and graphs
    - System health monitoring
    - Alert management
    - Performance analytics
    - Historical data visualization
    """
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.websocket_clients: List = []
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'error_rate': 5.0,
            'response_time_p95': 5000.0,  # 5 seconds
            'miner_failure_rate': 10.0
        }
        
        # Historical data storage (in-memory for demo)
        self.historical_data: List[DashboardMetrics] = []
        self.max_history_points = 1440  # 24 hours at 1-minute intervals
        
        self.is_running = False
        self.broadcast_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the monitoring dashboard"""
        if self.is_running:
            return
        
        logger.info("Starting monitoring dashboard")
        self.is_running = True
        
        # Start background data broadcasting
        self.broadcast_task = asyncio.create_task(self._broadcast_metrics())
        
        logger.info("Monitoring dashboard started")
    
    async def stop(self):
        """Stop the monitoring dashboard"""
        if not self.is_running:
            return
        
        logger.info("Stopping monitoring dashboard")
        self.is_running = False
        
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring dashboard stopped")
    
    async def get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics"""
        try:
            # Get metrics from collector
            metrics = self.metrics_collector.get_latest_metrics()
            
            # Get service manager status
            service_manager = get_service_manager()
            service_status = await service_manager.get_service_status()
            
            # Extract system metrics
            system_metrics = metrics.get('system', {})
            
            # Calculate miner metrics
            miner_metrics = await self._calculate_miner_metrics(service_manager)
            
            # Extract service metrics
            service_metrics = {
                'total_services': service_status['manager']['total_services'],
                'running_services': len([s for s in service_status['services'].values() if s['status'] == 'running']),
                'failed_services': len([s for s in service_status['services'].values() if s['status'] == 'error']),
                'service_health': {name: info['health_status'] for name, info in service_status['services'].items()}
            }
            
            # Calculate performance metrics
            app_metrics = metrics.get('application', {})
            performance_metrics = {
                'request_count': app_metrics.get('request_count', 0),
                'error_count': app_metrics.get('error_count', 0),
                'error_rate': app_metrics.get('error_rate', 0),
                'avg_response_time': app_metrics.get('avg_response_time', 0),
                'response_time_percentiles': app_metrics.get('response_time_percentiles', {}),
                'throughput': self._calculate_throughput(app_metrics)
            }
            
            # Generate alerts
            alerts = self._generate_alerts(system_metrics, performance_metrics, service_metrics)
            
            dashboard_metrics = DashboardMetrics(
                timestamp=datetime.now().isoformat(),
                system_metrics=system_metrics,
                miner_metrics=miner_metrics,
                service_metrics=service_metrics,
                performance_metrics=performance_metrics,
                alerts=alerts
            )
            
            # Store historical data
            self._store_historical_data(dashboard_metrics)
            
            return dashboard_metrics
            
        except Exception as e:
            logger.error("Error getting dashboard metrics", error=str(e))
            return DashboardMetrics(
                timestamp=datetime.now().isoformat(),
                system_metrics={},
                miner_metrics={},
                service_metrics={},
                performance_metrics={},
                alerts=[{'type': 'error', 'message': f'Failed to collect metrics: {e}'}]
            )
    
    async def _calculate_miner_metrics(self, service_manager) -> Dict[str, Any]:
        """Calculate miner-specific metrics"""
        try:
            if not service_manager.miner_service:
                return {}
            
            miner_service_metrics = service_manager.miner_service.get_service_metrics()
            cb_status = service_manager.miner_service.get_circuit_breaker_status()
            
            # Get latest miner data
            if service_manager.database_service:
                latest_miners = await service_manager.database_service.get_latest_status_async()
            else:
                latest_miners = []
            
            # Calculate aggregated metrics
            total_miners = len(latest_miners)
            online_miners = len([m for m in latest_miners if m.get('hashRate', 0) > 0])
            total_hashrate = sum(m.get('hashRate', 0) for m in latest_miners)
            total_power = sum(m.get('power', 0) for m in latest_miners)
            avg_temperature = statistics.mean([m.get('temp', 0) for m in latest_miners if m.get('temp')]) if latest_miners else 0
            
            return {
                'total_miners': total_miners,
                'online_miners': online_miners,
                'offline_miners': total_miners - online_miners,
                'total_hashrate': total_hashrate,
                'total_power': total_power,
                'avg_temperature': avg_temperature,
                'efficiency': total_hashrate / total_power if total_power > 0 else 0,
                'blocked_miners': len(cb_status.get('blocked_miners', [])),
                'success_rate': miner_service_metrics.get('successful_requests', 0) / max(1, miner_service_metrics.get('total_requests', 1)) * 100,
                'circuit_breaker_activations': miner_service_metrics.get('circuit_breaker_activations', 0),
                'concurrent_operations': miner_service_metrics.get('concurrent_operations', 0)
            }
            
        except Exception as e:
            logger.error("Error calculating miner metrics", error=str(e))
            return {}
    
    def _calculate_throughput(self, app_metrics: Dict[str, Any]) -> float:
        """Calculate request throughput"""
        # This is a simplified calculation
        # In production, you'd track requests over time windows
        request_count = app_metrics.get('request_count', 0)
        
        # Estimate throughput based on recent requests
        # This would be more accurate with proper time-series data
        return request_count / 3600.0  # Requests per second (rough estimate)
    
    def _generate_alerts(self, system_metrics: Dict[str, Any], 
                        performance_metrics: Dict[str, Any],
                        service_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on thresholds"""
        alerts = []
        
        try:
            # System alerts
            cpu_percent = system_metrics.get('cpu_percent', 0)
            if cpu_percent > self.alert_thresholds['cpu_percent']:
                alerts.append({
                    'type': 'warning',
                    'category': 'system',
                    'message': f'High CPU usage: {cpu_percent:.1f}%',
                    'threshold': self.alert_thresholds['cpu_percent'],
                    'current_value': cpu_percent
                })
            
            memory_percent = system_metrics.get('memory_percent', 0)
            if memory_percent > self.alert_thresholds['memory_percent']:
                alerts.append({
                    'type': 'warning',
                    'category': 'system',
                    'message': f'High memory usage: {memory_percent:.1f}%',
                    'threshold': self.alert_thresholds['memory_percent'],
                    'current_value': memory_percent
                })
            
            disk_percent = system_metrics.get('disk_percent', 0)
            if disk_percent > self.alert_thresholds['disk_percent']:
                alerts.append({
                    'type': 'critical',
                    'category': 'system',
                    'message': f'High disk usage: {disk_percent:.1f}%',
                    'threshold': self.alert_thresholds['disk_percent'],
                    'current_value': disk_percent
                })
            
            # Performance alerts
            error_rate = performance_metrics.get('error_rate', 0)
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': 'critical',
                    'category': 'performance',
                    'message': f'High error rate: {error_rate:.1f}%',
                    'threshold': self.alert_thresholds['error_rate'],
                    'current_value': error_rate
                })
            
            p95_response_time = performance_metrics.get('response_time_percentiles', {}).get('p95', 0)
            if p95_response_time > self.alert_thresholds['response_time_p95']:
                alerts.append({
                    'type': 'warning',
                    'category': 'performance',
                    'message': f'High response time (P95): {p95_response_time:.0f}ms',
                    'threshold': self.alert_thresholds['response_time_p95'],
                    'current_value': p95_response_time
                })
            
            # Service alerts
            failed_services = service_metrics.get('failed_services', 0)
            if failed_services > 0:
                alerts.append({
                    'type': 'critical',
                    'category': 'services',
                    'message': f'{failed_services} service(s) failed',
                    'current_value': failed_services
                })
            
            # Health check alerts
            unhealthy_services = [name for name, health in service_metrics.get('service_health', {}).items() 
                                if health == 'unhealthy']
            if unhealthy_services:
                alerts.append({
                    'type': 'critical',
                    'category': 'health',
                    'message': f'Unhealthy services: {", ".join(unhealthy_services)}',
                    'services': unhealthy_services
                })
            
        except Exception as e:
            logger.error("Error generating alerts", error=str(e))
            alerts.append({
                'type': 'error',
                'category': 'system',
                'message': f'Alert generation failed: {e}'
            })
        
        return alerts
    
    def _store_historical_data(self, metrics: DashboardMetrics):
        """Store metrics for historical analysis"""
        self.historical_data.append(metrics)
        
        # Keep only recent data
        if len(self.historical_data) > self.max_history_points:
            self.historical_data = self.historical_data[-self.max_history_points:]
    
    def get_historical_data(self, hours: int = 24) -> List[DashboardMetrics]:
        """Get historical data for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.historical_data
            if datetime.fromisoformat(metrics.timestamp) >= cutoff_time
        ]
    
    async def _broadcast_metrics(self):
        """Background task to broadcast metrics to WebSocket clients"""
        logger.info("Metrics broadcast task started")
        
        while self.is_running:
            try:
                if self.websocket_clients:
                    metrics = await self.get_current_metrics()
                    metrics_data = asdict(metrics)
                    
                    # Broadcast to all connected clients
                    disconnected_clients = []
                    for client in self.websocket_clients:
                        try:
                            await client.send(json.dumps({
                                'type': 'metrics_update',
                                'data': metrics_data
                            }))
                        except Exception:
                            disconnected_clients.append(client)
                    
                    # Remove disconnected clients
                    for client in disconnected_clients:
                        self.websocket_clients.remove(client)
                
                await asyncio.sleep(5)  # Broadcast every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics broadcast error", error=str(e))
                await asyncio.sleep(5)
        
        logger.info("Metrics broadcast task stopped")
    
    def add_websocket_client(self, client):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.append(client)
        logger.debug(f"WebSocket client added, total: {len(self.websocket_clients)}")
    
    def remove_websocket_client(self, client):
        """Remove WebSocket client"""
        if client in self.websocket_clients:
            self.websocket_clients.remove(client)
            logger.debug(f"WebSocket client removed, total: {len(self.websocket_clients)}")


# Global dashboard instance
_dashboard: Optional[MonitoringDashboard] = None


def get_dashboard() -> MonitoringDashboard:
    """Get global dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = MonitoringDashboard()
    return _dashboard


# Blueprint routes
@monitoring_bp.route('/')
async def dashboard_home():
    """Main monitoring dashboard page"""
    try:
        dashboard = get_dashboard()
        initial_metrics = await dashboard.get_current_metrics()
        
        return await render_template('monitoring/dashboard.html', 
                                   initial_metrics=asdict(initial_metrics))
    except Exception as e:
        logger.error("Dashboard home error", error=str(e))
        return await render_template('error.html', error='Failed to load monitoring dashboard')


@monitoring_bp.route('/api/metrics')
async def api_current_metrics():
    """API endpoint for current metrics"""
    try:
        dashboard = get_dashboard()
        metrics = await dashboard.get_current_metrics()
        
        return jsonify(asdict(metrics))
    except Exception as e:
        logger.error("API metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500


@monitoring_bp.route('/api/historical')
async def api_historical_metrics():
    """API endpoint for historical metrics"""
    try:
        hours = request.args.get('hours', 24, type=int)
        dashboard = get_dashboard()
        historical_data = dashboard.get_historical_data(hours)
        
        return jsonify([asdict(metrics) for metrics in historical_data])
    except Exception as e:
        logger.error("API historical metrics error", error=str(e))
        return jsonify({'error': str(e)}), 500


@monitoring_bp.route('/api/alerts')
async def api_current_alerts():
    """API endpoint for current alerts"""
    try:
        dashboard = get_dashboard()
        metrics = await dashboard.get_current_metrics()
        
        return jsonify({'alerts': metrics.alerts})
    except Exception as e:
        logger.error("API alerts error", error=str(e))
        return jsonify({'error': str(e)}), 500


@monitoring_bp.websocket('/ws/metrics')
async def websocket_metrics():
    """WebSocket endpoint for real-time metrics"""
    dashboard = get_dashboard()
    
    try:
        logger.info("WebSocket connection established for monitoring")
        dashboard.add_websocket_client(websocket._get_current_object())
        
        # Send initial metrics
        initial_metrics = await dashboard.get_current_metrics()
        await websocket.send(json.dumps({
            'type': 'initial_metrics',
            'data': asdict(initial_metrics)
        }))
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            try:
                message = await websocket.receive()
                if message == 'ping':
                    await websocket.send('pong')
            except Exception:
                break
                
    except Exception as e:
        logger.error("WebSocket metrics error", error=str(e))
    finally:
        dashboard.remove_websocket_client(websocket._get_current_object())
        logger.info("WebSocket connection closed for monitoring")


def register_monitoring_blueprint(app):
    """Register monitoring blueprint with Flask app"""
    app.register_blueprint(monitoring_bp)
    
    logger.info("Monitoring dashboard blueprint registered",
               prefix="/monitoring")