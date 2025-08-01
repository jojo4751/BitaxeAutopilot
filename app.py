"""
BitAxe Web Management System V2.0.0
Main Flask application entry point

A comprehensive web interface for managing BitAxe ASIC miners
with real-time monitoring, control, and optimization features.

This application provides:
- Real-time miner monitoring and control
- Automated benchmarking and optimization
- Historical data analysis and visualization
- REST API for external integrations
- Comprehensive logging and error handling
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from flask import Flask

# Add current directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitaxe_app.core import ConfigManager, DatabaseManager
from bitaxe_app.core.exceptions import BitAxeException, ConfigurationError, DatabaseError
from bitaxe_app.routes import register_routes
from bitaxe_app.api import register_api
from bitaxe_app.services import ServiceContainer
from bitaxe_app.services.websocket_service import WebSocketService
from bitaxe_app.utils.logging import setup_logging


# Configure logging
logger = logging.getLogger(__name__)


def create_app(config_path: Optional[str] = None) -> Flask:
    """Application factory pattern for Flask app creation.
    
    This function creates and configures a Flask application instance with
    all necessary components including configuration management, database
    connections, service container, and route registration.
    
    Args:
        config_path: Optional path to configuration file. If None, uses default.
        
    Returns:
        Configured Flask application instance
        
    Raises:
        ConfigurationError: If configuration cannot be loaded
        DatabaseError: If database initialization fails
        BitAxeException: If any other initialization error occurs
    """
    try:
        # Initialize Flask app with proper configuration
        app = Flask(
            __name__, 
            template_folder='bitaxe_app/templates',
            static_folder='bitaxe_app/static'
        )
        
        # Setup logging first
        setup_logging(app)
        logger.info("Starting BitAxe Web Management System V2.0.0")
        
        # Initialize configuration manager
        config_manager = ConfigManager(config_path)
        app.config_manager = config_manager
        
        # Configure Flask settings
        app.secret_key = os.environ.get(
            "FLASK_SECRET_KEY", 
            "bitaxe-v2-secret-key-change-in-production"
        )
        
        # Set Flask configuration from BitAxe config
        app.config.update({
            'DEBUG': os.environ.get('FLASK_ENV') != 'production',
            'TESTING': False,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': True
        })
        
        # Initialize database manager
        database_manager = DatabaseManager(config_manager)
        app.database_manager = database_manager
        
        # Initialize service container
        container = ServiceContainer()
        app.container = container
        
        # Initialize WebSocket service for real-time updates
        websocket_service = WebSocketService(
            app, 
            container.get_config_service(),
            container.get_database_service(),
            container.get_analytics_service()
        )
        app.websocket_service = websocket_service
        app.socketio = websocket_service.socketio
        
        # Register error handlers
        register_error_handlers(app)
        
        # Register routes and API endpoints
        register_routes(app)
        register_api(app)
        
        # Register monitoring routes if available
        try:
            from web.monitoring_routes import monitoring_bp
            app.register_blueprint(monitoring_bp)
            logger.info("Monitoring routes registered successfully")
        except ImportError:
            logger.warning("Monitoring routes not available")
        
        # Perform health check
        health_status = container.health_check()
        if health_status['overall_status'] != 'healthy':
            logger.warning(f"Application health check shows issues: {health_status}")
        else:
            logger.info("Application health check passed")
        
        logger.info("BitAxe application initialized successfully")
        return app
        
    except (ConfigurationError, DatabaseError) as e:
        logger.error(f"Critical initialization error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected initialization error: {e}")
        raise BitAxeException(f"Application initialization failed: {e}")


def register_error_handlers(app: Flask) -> None:
    """Register comprehensive error handlers for the application.
    
    Args:
        app: Flask application instance
    """
    from flask import jsonify, request, render_template
    
    @app.errorhandler(ConfigurationError)
    def handle_config_error(error):
        """Handle configuration errors."""
        logger.error(f"Configuration error: {error}")
        
        if request.path.startswith('/api/'):
            return jsonify(error.to_dict()), 500
        
        return render_template('error.html', 
                             error_title="Configuration Error",
                             error_message=str(error),
                             error_details=error.context), 500
    
    @app.errorhandler(DatabaseError)
    def handle_database_error(error):
        """Handle database errors."""
        logger.error(f"Database error: {error}")
        
        if request.path.startswith('/api/'):
            return jsonify(error.to_dict()), 500
        
        return render_template('error.html',
                             error_title="Database Error", 
                             error_message=str(error),
                             error_details=error.context), 500
    
    @app.errorhandler(BitAxeException)
    def handle_bitaxe_error(error):
        """Handle general BitAxe errors."""
        logger.error(f"BitAxe error: {error}")
        
        if request.path.startswith('/api/'):
            return jsonify(error.to_dict()), 500
        
        return render_template('error.html',
                             error_title="System Error",
                             error_message=str(error),
                             error_details=error.context), 500
    
    @app.errorhandler(404)
    def handle_not_found_error(error):
        """Handle 404 errors."""
        if request.path.startswith('/api/'):
            return jsonify({
                'error_type': 'NotFound',
                'error_code': 'NOT_FOUND',
                'message': 'The requested resource was not found',
                'path': request.path
            }), 404
        
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle internal server errors."""
        logger.error(f"Internal server error: {error}")
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error_type': 'InternalServerError',
                'error_code': 'INTERNAL_ERROR',
                'message': 'An internal server error occurred'
            }), 500
        
        return render_template('error.html',
                             error_title="Internal Server Error",
                             error_message="An unexpected error occurred",
                             error_details={}), 500


def main() -> None:
    """Main entry point for development server.
    
    Starts the Flask development server with proper configuration
    and graceful shutdown handling.
    """
    try:
        # Create application instance
        app = create_app()
        
        # Get server configuration
        debug_mode = os.environ.get("FLASK_ENV") != "production"
        port = int(os.environ.get("PORT", 5000))
        host = os.environ.get("HOST", "0.0.0.0")
        
        logger.info(f"Starting development server on {host}:{port} (debug={debug_mode})")
        
        # Start the development server with SocketIO support
        app.socketio.run(
            app,
            host=host,
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        raise
    finally:
        # Ensure graceful cleanup
        try:
            if 'app' in locals():
                if hasattr(app, 'websocket_service'):
                    logger.info("Shutting down WebSocket service...")
                    app.websocket_service.shutdown()
                
                if hasattr(app, 'container'):
                    logger.info("Shutting down services...")
                    app.container.shutdown()
                    logger.info("Services shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    main()