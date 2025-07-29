"""
BitAxe Web Management System V2.0.0
Main Flask application entry point

A comprehensive web interface for managing BitAxe ASIC miners
with real-time monitoring, control, and optimization features.
"""

import os
import sys
from flask import Flask

# Add current directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitaxe_app.routes import register_routes
from bitaxe_app.api import register_api
from bitaxe_app.services import ServiceContainer


def create_app(config_path=None):
    """Application factory pattern for Flask app creation"""
    
    # Initialize Flask app
    app = Flask(__name__, 
                template_folder='bitaxe_app/templates',
                static_folder='bitaxe_app/static')
    
    # Configuration
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "bitaxe-v2-secret-key")
    
    # Initialize service container
    config_path = config_path or os.path.join("config", "config.json")
    container = ServiceContainer(config_path)
    
    # Make services available to the app
    app.container = container
    
    # Register routes and API endpoints
    register_routes(app)
    register_api(app)
    
    return app


def main():
    """Main entry point for development server"""
    app = create_app()
    
    # Production settings
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug_mode
        )
    finally:
        # Cleanup services on shutdown
        if hasattr(app, 'container'):
            app.container.shutdown()


if __name__ == "__main__":
    main()