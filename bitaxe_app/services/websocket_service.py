"""
BitAxe WebSocket Service
Real-time data streaming service for live dashboard updates
"""

import json
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms
from flask import request

logger = logging.getLogger(__name__)


class WebSocketService:
    """WebSocket service for real-time data streaming."""
    
    def __init__(self, app, config_service, database_manager, analytics_service):
        self.app = app
        self.config_service = config_service
        self.database_manager = database_manager
        self.analytics_service = analytics_service
        
        # Initialize SocketIO
        self.socketio = SocketIO(
            app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False
        )
        
        # Tracking
        self.connected_clients = {}
        self.client_subscriptions = {}
        self.broadcast_thread = None
        self.broadcast_running = False
        
        # Configuration
        self.broadcast_interval = 5  # seconds
        self.max_clients = 100
        
        # Register event handlers
        self._register_handlers()
        
        logger.info("WebSocket service initialized")
    
    def _register_handlers(self):
        """Register WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect(auth):
            """Handle client connection."""
            client_id = request.sid
            client_info = {
                'id': client_id,
                'connected_at': datetime.now(),
                'ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }
            
            self.connected_clients[client_id] = client_info
            self.client_subscriptions[client_id] = {
                'miners': [],
                'alerts': True,
                'system_stats': True,
                'profitability': True
            }
            
            logger.info(f"Client connected: {client_id} from {client_info['ip']}")
            
            # Send initial data
            self._send_initial_data(client_id)
            
            # Start broadcast thread if not running
            if not self.broadcast_running:
                self._start_broadcast_thread()
            
            return {'status': 'connected', 'client_id': client_id}
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            
            if client_id in self.connected_clients:
                client_info = self.connected_clients[client_id]
                logger.info(f"Client disconnected: {client_id} from {client_info['ip']}")
                
                del self.connected_clients[client_id]
                del self.client_subscriptions[client_id]
            
            # Stop broadcast thread if no clients
            if not self.connected_clients and self.broadcast_running:
                self._stop_broadcast_thread()
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription requests."""
            client_id = request.sid
            
            if client_id not in self.client_subscriptions:
                return {'status': 'error', 'message': 'Client not found'}
            
            subscription_type = data.get('type')
            subscription_data = data.get('data', {})
            
            if subscription_type == 'miners':
                # Subscribe to specific miners
                miner_ips = subscription_data.get('ips', [])
                self.client_subscriptions[client_id]['miners'] = miner_ips
                
                # Join rooms for each miner
                for ip in miner_ips:
                    join_room(f'miner_{ip}')
                
                logger.debug(f"Client {client_id} subscribed to miners: {miner_ips}")
                
            elif subscription_type == 'alerts':
                self.client_subscriptions[client_id]['alerts'] = subscription_data.get('enabled', True)
                if subscription_data.get('enabled', True):
                    join_room('alerts')
                else:
                    leave_room('alerts')
            
            elif subscription_type == 'system_stats':
                self.client_subscriptions[client_id]['system_stats'] = subscription_data.get('enabled', True)
                if subscription_data.get('enabled', True):
                    join_room('system_stats')
                else:
                    leave_room('system_stats')
            
            return {'status': 'subscribed', 'type': subscription_type}
        
        @self.socketio.on('get_historical_data')
        def handle_historical_data_request(data):
            """Handle requests for historical data."""
            try:
                ip = data.get('ip')
                hours = data.get('hours', 6)
                
                if not ip:
                    return {'status': 'error', 'message': 'IP address required'}
                
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
                
                if ip in historical_data:
                    # Format data for charts
                    chart_data = self._format_chart_data(historical_data[ip])
                    return {
                        'status': 'success',
                        'ip': ip,
                        'data': chart_data,
                        'timeframe': f'{hours}h'
                    }
                else:
                    return {'status': 'error', 'message': 'No data found'}
                    
            except Exception as e:
                logger.error(f"Error handling historical data request: {e}")
                return {'status': 'error', 'message': str(e)}
        
        @self.socketio.on('acknowledge_alert')
        def handle_acknowledge_alert(data):
            """Handle alert acknowledgment."""
            try:
                alert_id = data.get('alert_id')
                user = data.get('user', 'web_user')
                
                if not alert_id:
                    return {'status': 'error', 'message': 'Alert ID required'}
                
                # Update alert in database (would need to implement this method)
                # self.database_manager.acknowledge_alert(alert_id, user)
                
                # Broadcast acknowledgment to all clients
                self.socketio.emit('alert_acknowledged', {
                    'alert_id': alert_id,
                    'acknowledged_by': user,
                    'acknowledged_at': datetime.now().isoformat()
                }, room='alerts')
                
                return {'status': 'success', 'alert_id': alert_id}
                
            except Exception as e:
                logger.error(f"Error acknowledging alert: {e}")
                return {'status': 'error', 'message': str(e)}
    
    def _send_initial_data(self, client_id: str):
        """Send initial data to newly connected client."""
        try:
            # Get latest miner status
            latest_status = self.database_manager.get_latest_status()
            
            # Get active alerts
            active_alerts = self.database_manager.get_active_alerts()
            
            # Get system stats
            system_stats = self._get_system_stats()
            
            # Send initial data
            self.socketio.emit('initial_data', {
                'miners': latest_status,
                'alerts': active_alerts,
                'system_stats': system_stats,
                'timestamp': datetime.now().isoformat()
            }, room=client_id)
            
        except Exception as e:
            logger.error(f"Error sending initial data to {client_id}: {e}")
    
    def _start_broadcast_thread(self):
        """Start the background broadcast thread."""
        if self.broadcast_thread is None or not self.broadcast_thread.is_alive():
            self.broadcast_running = True
            self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
            self.broadcast_thread.start()
            logger.info("WebSocket broadcast thread started")
    
    def _stop_broadcast_thread(self):
        """Stop the background broadcast thread."""
        self.broadcast_running = False
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=5)
        logger.info("WebSocket broadcast thread stopped")
    
    def _broadcast_loop(self):
        """Main broadcast loop for real-time updates."""
        logger.info(f"Starting WebSocket broadcast loop (interval: {self.broadcast_interval}s)")
        
        last_data = {}
        
        while self.broadcast_running and self.connected_clients:
            try:
                # Get current data
                current_data = self._get_broadcast_data()
                
                # Only broadcast if data has changed significantly
                if self._should_broadcast(current_data, last_data):
                    self._broadcast_updates(current_data)
                    last_data = current_data
                
                time.sleep(self.broadcast_interval)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                time.sleep(self.broadcast_interval)
        
        logger.info("WebSocket broadcast loop ended")
    
    def _get_broadcast_data(self) -> Dict[str, Any]:
        """Get data for broadcasting to clients."""
        try:
            # Get latest miner status
            miners_data = self.database_manager.get_latest_status()
            
            # Get recent alerts (last 5 minutes)
            recent_time = datetime.now() - timedelta(minutes=5)
            recent_alerts = []
            try:
                all_alerts = self.database_manager.get_active_alerts()
                recent_alerts = [
                    alert for alert in all_alerts 
                    if datetime.fromisoformat(alert.get('timestamp', '1970-01-01')) >= recent_time
                ]
            except Exception as e:
                logger.warning(f"Error getting recent alerts: {e}")
            
            # Get system stats
            system_stats = self._get_system_stats()
            
            # Get profitability data
            profitability_data = self._get_fleet_profitability()
            
            return {
                'miners': miners_data,
                'alerts': recent_alerts,
                'system_stats': system_stats,
                'profitability': profitability_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting broadcast data: {e}")
            return {
                'miners': [],
                'alerts': [],
                'system_stats': {},
                'profitability': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_broadcast(self, current_data: Dict[str, Any], last_data: Dict[str, Any]) -> bool:
        """Determine if data has changed enough to warrant broadcasting."""
        if not last_data:
            return True
        
        # Check if miners data has changed
        current_miners = {m.get('ip'): m for m in current_data.get('miners', [])}
        last_miners = {m.get('ip'): m for m in last_data.get('miners', [])}
        
        # Check for new/removed miners
        if set(current_miners.keys()) != set(last_miners.keys()):
            return True
        
        # Check for significant changes in miner data
        for ip, current_miner in current_miners.items():
            if ip not in last_miners:
                continue
                
            last_miner = last_miners[ip]
            
            # Check key metrics for significant changes
            for metric in ['hashRate', 'temp', 'power', 'efficiency']:
                current_val = current_miner.get(metric, 0)
                last_val = last_miner.get(metric, 0)
                
                # Consider 5% change as significant for hashrate/power, 1°C for temperature
                if metric == 'temp':
                    if abs(current_val - last_val) >= 1.0:
                        return True
                elif last_val > 0 and abs(current_val - last_val) / last_val >= 0.05:
                    return True
        
        # Check for new alerts
        if len(current_data.get('alerts', [])) != len(last_data.get('alerts', [])):
            return True
        
        return False
    
    def _broadcast_updates(self, data: Dict[str, Any]):
        """Broadcast updates to connected clients."""
        try:
            # Broadcast to all clients
            self.socketio.emit('live_update', data)
            
            # Broadcast miner-specific data to subscribed rooms
            for miner in data.get('miners', []):
                ip = miner.get('ip')
                if ip:
                    self.socketio.emit('miner_update', miner, room=f'miner_{ip}')
            
            # Broadcast alerts if any
            if data.get('alerts'):
                self.socketio.emit('new_alerts', data['alerts'], room='alerts')
            
            logger.debug(f"Broadcasted updates to {len(self.connected_clients)} clients")
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get database stats
            db_stats = self.database_manager.get_database_stats()
            
            return {
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_used_gb': disk.used / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                },
                'database': {
                    'size_mb': db_stats.get('file_size_mb', 0),
                    'total_logs': db_stats.get('logs_count', 0),
                    'total_alerts': db_stats.get('performance_alerts_count', 0)
                },
                'fleet': {
                    'total_miners': len(self.config_service.ips),
                    'connected_clients': len(self.connected_clients)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def _get_fleet_profitability(self) -> Dict[str, Any]:
        """Get fleet-wide profitability metrics."""
        try:
            # Get recent profitability data
            recent_time = datetime.now() - timedelta(hours=1)
            profitability_data = self.database_manager.get_profitability_data(start_time=recent_time)
            
            if not profitability_data:
                return {}
            
            # Calculate fleet totals
            total_hashrate = sum(d.get('hashrate', 0) for d in profitability_data)
            total_power = sum(d.get('power_consumption', 0) for d in profitability_data)
            total_daily_profit = sum(d.get('estimated_daily_profit_usd', 0) for d in profitability_data)
            total_daily_revenue = sum(d.get('estimated_daily_usd', 0) for d in profitability_data)
            total_daily_cost = sum(d.get('daily_power_cost_usd', 0) for d in profitability_data)
            
            # Calculate averages
            avg_efficiency = total_hashrate / total_power if total_power > 0 else 0
            profit_margin = (total_daily_profit / total_daily_revenue * 100) if total_daily_revenue > 0 else 0
            
            return {
                'total_hashrate_th': total_hashrate,
                'total_power_kw': total_power,
                'total_daily_profit_usd': total_daily_profit,
                'total_daily_revenue_usd': total_daily_revenue,
                'total_daily_cost_usd': total_daily_cost,
                'avg_efficiency_thw': avg_efficiency,
                'profit_margin_percent': profit_margin,
                'active_miners': len(set(d.get('ip') for d in profitability_data))
            }
            
        except Exception as e:
            logger.error(f"Error getting fleet profitability: {e}")
            return {}
    
    def _format_chart_data(self, raw_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format raw data for chart visualization."""
        try:
            if not raw_data:
                return {}
            
            # Sort by timestamp
            sorted_data = sorted(raw_data, key=lambda x: x.get('timestamp', ''))
            
            # Extract time series data
            timestamps = []
            hashrates = []
            temperatures = []
            powers = []
            efficiencies = []
            
            for point in sorted_data:
                timestamp = point.get('timestamp')
                if timestamp:
                    timestamps.append(timestamp)
                    hashrates.append(point.get('hashRate', 0))
                    temperatures.append(point.get('temp', 0))
                    powers.append(point.get('power', 0))
                    
                    # Calculate efficiency
                    hashrate = point.get('hashRate', 0)
                    power = point.get('power', 0)
                    efficiency = hashrate / power if power > 0 else 0
                    efficiencies.append(efficiency)
            
            return {
                'timestamps': timestamps,
                'hashrate': hashrates,
                'temperature': temperatures,
                'power': powers,
                'efficiency': efficiencies,
                'data_points': len(timestamps)
            }
            
        except Exception as e:
            logger.error(f"Error formatting chart data: {e}")
            return {}
    
    def get_connected_clients_info(self) -> List[Dict[str, Any]]:
        """Get information about connected clients."""
        return [
            {
                'id': client_id,
                'connected_at': info['connected_at'].isoformat(),
                'ip': info['ip'],
                'user_agent': info['user_agent'],
                'subscriptions': self.client_subscriptions.get(client_id, {})
            }
            for client_id, info in self.connected_clients.items()
        ]
    
    def broadcast_custom_event(self, event_name: str, data: Any, room: Optional[str] = None):
        """Broadcast a custom event to clients."""
        try:
            if room:
                self.socketio.emit(event_name, data, room=room)
            else:
                self.socketio.emit(event_name, data)
            
            logger.debug(f"Broadcasted custom event '{event_name}' to room '{room or 'all'}'")
            
        except Exception as e:
            logger.error(f"Error broadcasting custom event '{event_name}': {e}")
    
    def shutdown(self):
        """Shutdown the WebSocket service."""
        logger.info("Shutting down WebSocket service...")
        self._stop_broadcast_thread()
        
        # Disconnect all clients
        for client_id in list(self.connected_clients.keys()):
            self.socketio.disconnect(client_id)
        
        logger.info("WebSocket service shutdown complete")