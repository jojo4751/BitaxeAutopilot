"""
Async BitAxe Application

Main async application with comprehensive async/await patterns, background tasks, and monitoring.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime
from quart import Quart, render_template, request, redirect, url_for, jsonify
from quart_cors import cors
import hypercorn.asyncio
from hypercorn.config import Config

from logging.structured_logger import get_logger
from services.config_service import ConfigService
from async_services.async_service_manager import get_service_manager
from monitoring.metrics_collector import get_metrics_collector
from utils.rate_limiter import init_rate_limiting
from api.swagger_config import setup_api_documentation
from api.v1.blueprints import register_api_blueprint

logger = get_logger("bitaxe.async_app")


class AsyncBitAxeApp:
    """
    Async BitAxe Application
    
    Features:
    - Full async/await support with Quart
    - Background task management
    - Real-time metrics and monitoring
    - Graceful shutdown handling
    - Health checks and status endpoints
    - WebSocket support for real-time updates
    """
    
    def __init__(self):
        # Create Quart app
        self.app = Quart(__name__)
        self.app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback-secret-key")
        
        # Enable CORS
        self.app = cors(self.app, allow_origin="*")
        
        # Services
        self.config_service: Optional[ConfigService] = None
        self.service_manager = None
        self.metrics_collector = None
        
        # Application state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self._setup_routes()
        self._setup_error_handlers()
    
    async def initialize(self):
        """Initialize the application"""
        logger.info("Initializing async BitAxe application")
        
        try:
            # Initialize config service
            config_path = os.environ.get("BITAXE_CONFIG_PATH", "config.json")
            self.config_service = ConfigService(config_path)
            
            # Initialize metrics collector
            self.metrics_collector = get_metrics_collector()
            await self.metrics_collector.start()
            
            # Initialize service manager
            self.service_manager = get_service_manager(self.config_service)
            await self.service_manager.initialize()
            
            # Setup API documentation and endpoints
            setup_api_documentation(self.app)
            register_api_blueprint(self.app)
            
            # Initialize rate limiting
            init_rate_limiting(self.app)
            
            logger.info("Async BitAxe application initialized successfully")
            
        except Exception as e:
            logger.error("Application initialization failed", error=str(e))
            raise
    
    async def start(self):
        """Start the application and all services"""
        if self.is_running:
            return
        
        logger.info("Starting async BitAxe application")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start all services
            await self.service_manager.start_all()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Record startup metrics
            self.metrics_collector.increment_counter('app_starts_total')
            self.metrics_collector.set_gauge('app_status', 1)
            
            logger.info("Async BitAxe application started successfully")
            
        except Exception as e:
            logger.error("Application startup failed", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the application and all services"""
        if not self.is_running:
            return
        
        logger.info("Stopping async BitAxe application")
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop all services
            if self.service_manager:
                await self.service_manager.stop_all()
            
            # Stop metrics collector
            if self.metrics_collector:
                await self.metrics_collector.stop()
            
            # Record shutdown metrics
            if self.start_time and self.metrics_collector:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.metrics_collector.set_gauge('app_uptime_seconds', uptime)
                self.metrics_collector.set_gauge('app_status', 0)
            
            logger.info("Async BitAxe application stopped")
            
        except Exception as e:
            logger.error("Error during application shutdown", error=str(e))
    
    def _setup_routes(self):
        """Setup application routes"""
        
        @self.app.route("/")
        @self.app.route("/status")
        async def status():
            """Main status page"""
            try:
                if not self.service_manager or not self.service_manager.database_service:
                    miners = []
                else:
                    miners = await self.service_manager.database_service.get_latest_status_async()
                
                return await render_template("status.html", miners=miners)
            except Exception as e:
                logger.error("Status page error", error=str(e))
                return await render_template("error.html", error="Failed to load status")
        
        @self.app.route("/dashboard")
        async def dashboard():
            """Main dashboard page"""
            try:
                if not self.service_manager or not self.service_manager.database_service:
                    miners = []
                    plot_data = {}
                else:
                    miners = await self.service_manager.database_service.get_latest_status_async()
                    # TODO: Implement async history data retrieval
                    plot_data = {}
                
                return await render_template("dashboard.html", miners=miners, plot_data=plot_data)
            except Exception as e:
                logger.error("Dashboard page error", error=str(e))
                return await render_template("error.html", error="Failed to load dashboard")
        
        @self.app.route("/dashboard/<ip>")
        async def miner_dashboard(ip):
            """Individual miner dashboard"""
            try:
                miners = self.config_service.ips if self.config_service else []
                color = self.config_service.get_miner_color(ip) if self.config_service else "#000000"
                
                if self.service_manager and self.service_manager.database_service:
                    status = await self.service_manager.database_service.get_latest_status_by_ip_async(ip)
                    benchmarks = await self.service_manager.database_service.get_benchmark_results_by_ip_async(ip, 10)
                else:
                    status = None
                    benchmarks = []
                
                # TODO: Implement async history data
                plot_data = {"traces": []}
                
                return await render_template("miner_dashboard.html",
                                           miner=status,
                                           ip=ip,
                                           color=color,
                                           plot_data=plot_data,
                                           benchmarks=benchmarks)
            except Exception as e:
                logger.error("Miner dashboard error", ip=ip, error=str(e))
                return await render_template("error.html", error=f"Failed to load miner dashboard for {ip}")
        
        @self.app.route("/metrics")
        async def metrics_endpoint():
            """Prometheus-style metrics endpoint"""
            try:
                if not self.metrics_collector:
                    return "Metrics collector not available", 503
                
                metrics_data = self.metrics_collector.export_metrics('prometheus')
                return metrics_data, 200, {'Content-Type': 'text/plain; charset=utf-8'}
            except Exception as e:
                logger.error("Metrics endpoint error", error=str(e))
                return "Error collecting metrics", 500
        
        @self.app.route("/health")
        async def health_check():
            """Application health check endpoint"""
            try:
                health_status = {
                    'status': 'healthy' if self.is_running else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                    'services': {}
                }
                
                if self.service_manager:
                    service_health = await self.service_manager.get_service_health()
                    health_status['services'] = service_health
                    
                    # Overall health based on services
                    if all(status == 'healthy' for status in service_health.values()):
                        health_status['status'] = 'healthy'
                    elif any(status == 'unhealthy' for status in service_health.values()):
                        health_status['status'] = 'unhealthy'
                    else:
                        health_status['status'] = 'degraded'
                
                status_code = 200 if health_status['status'] == 'healthy' else 503
                return jsonify(health_status), status_code
                
            except Exception as e:
                logger.error("Health check error", error=str(e))
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 503
        
        @self.app.route("/admin/status")
        async def admin_status():
            """Detailed admin status page"""
            try:
                status_data = {
                    'app': {
                        'is_running': self.is_running,
                        'start_time': self.start_time.isoformat() if self.start_time else None,
                        'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                    },
                    'services': {},
                    'metrics': {}
                }
                
                if self.service_manager:
                    status_data['services'] = await self.service_manager.get_service_status()
                
                if self.metrics_collector:
                    status_data['metrics'] = self.metrics_collector.get_latest_metrics()
                
                return jsonify(status_data)
                
            except Exception as e:
                logger.error("Admin status error", error=str(e))
                return jsonify({'error': str(e)}), 500
        
        @self.app.websocket('/ws/live-data')
        async def websocket_live_data():
            """WebSocket endpoint for live data updates"""
            try:
                logger.info("WebSocket connection established")
                
                while True:
                    # Get latest miner data
                    if self.service_manager and self.service_manager.database_service:
                        miners = await self.service_manager.database_service.get_latest_status_async()
                        
                        # Send data to client
                        await websocket.send_json({
                            'type': 'miner_data',
                            'data': miners,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Send metrics data
                    if self.metrics_collector:
                        metrics = self.metrics_collector.get_latest_metrics()
                        await websocket.send_json({
                            'type': 'metrics',
                            'data': metrics,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Wait before next update
                    await asyncio.sleep(5)
                    
            except Exception as e:
                logger.error("WebSocket error", error=str(e))
            finally:
                logger.info("WebSocket connection closed")
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        async def not_found(error):
            return await render_template('error.html', error='Page not found'), 404
        
        @self.app.errorhandler(500)
        async def internal_error(error):
            logger.error("Internal server error", error=str(error))
            return await render_template('error.html', error='Internal server error'), 500
        
        @self.app.before_request
        async def before_request():
            """Record request metrics"""
            request.start_time = asyncio.get_event_loop().time()
        
        @self.app.after_request
        async def after_request(response):
            """Record response metrics"""
            try:
                if hasattr(request, 'start_time') and self.metrics_collector:
                    duration = asyncio.get_event_loop().time() - request.start_time
                    endpoint = request.endpoint or 'unknown'
                    
                    self.metrics_collector.record_request(
                        duration=duration,
                        status_code=response.status_code,
                        endpoint=endpoint
                    )
                
                return response
            except Exception as e:
                logger.error("After request error", error=str(e))
                return response
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        # Real-time data broadcaster
        broadcast_task = asyncio.create_task(self._data_broadcaster())
        self.background_tasks.append(broadcast_task)
        
        # Metrics reporter
        metrics_task = asyncio.create_task(self._metrics_reporter())
        self.background_tasks.append(metrics_task)
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _stop_background_tasks(self):
        """Stop background tasks"""
        logger.info("Stopping background tasks")
        
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
    
    async def _data_broadcaster(self):
        """Background task to broadcast real-time data"""
        logger.info("Data broadcaster started")
        
        while self.is_running:
            try:
                # This would broadcast data to WebSocket clients
                # Implementation depends on WebSocket management system
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Data broadcaster error", error=str(e))
                await asyncio.sleep(5)
        
        logger.info("Data broadcaster stopped")
    
    async def _metrics_reporter(self):
        """Background task to report metrics"""
        logger.info("Metrics reporter started")
        
        while self.is_running:
            try:
                if self.metrics_collector:
                    # Record app-specific metrics
                    self.metrics_collector.set_gauge('app_background_tasks', len(self.background_tasks))
                    
                    # Log metrics summary
                    metrics = self.metrics_collector.get_latest_metrics()
                    logger.info("Application metrics summary",
                               uptime_seconds=metrics.get('uptime_seconds', 0),
                               request_count=metrics.get('application', {}).get('request_count', 0),
                               error_rate=metrics.get('application', {}).get('error_rate', 0))
                
                await asyncio.sleep(60)  # Report every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics reporter error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("Metrics reporter stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


async def main():
    """Main application entry point"""
    logger.info("Starting BitAxe Async Application")
    
    app_instance = AsyncBitAxeApp()
    
    try:
        # Initialize and start the application
        async with app_instance:
            # Setup graceful shutdown
            shutdown_event = asyncio.Event()
            
            def signal_handler():
                logger.info("Shutdown signal received")
                shutdown_event.set()
            
            # Register signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
            
            # Configure Hypercorn
            config = Config()
            config.bind = [f"0.0.0.0:{os.environ.get('PORT', 5000)}"]
            config.use_reloader = False
            config.accesslog = "-"
            config.errorlog = "-"
            
            logger.info(f"Starting server on {config.bind[0]}")
            
            # Start the web server
            server_task = asyncio.create_task(
                hypercorn.asyncio.serve(app_instance.app, config, shutdown_trigger=shutdown_event.wait)
            )
            
            # Wait for shutdown signal
            await shutdown_event.wait()
            
            logger.info("Graceful shutdown initiated")
            
            # Wait for server to stop
            await server_task
            
    except Exception as e:
        logger.error("Application error", error=str(e))
        sys.exit(1)
    
    logger.info("BitAxe Async Application stopped")


if __name__ == "__main__":
    # Run the async application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Fatal application error", error=str(e))
        sys.exit(1)