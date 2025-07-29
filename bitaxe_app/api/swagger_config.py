"""
Swagger/OpenAPI configuration and initialization
"""

from flask import Flask
from flask_restx import Api, apidoc

from .openapi_spec import api
from .restx_blueprints import register_restx_api


def configure_swagger(app: Flask) -> None:
    """Configure Swagger/OpenAPI documentation"""
    
    # Update Flask-RESTX configuration
    app.config.update({
        'RESTX_MASK_SWAGGER': False,  # Disable field masking in Swagger UI
        'RESTX_VALIDATE': True,       # Enable request validation
        'RESTX_JSON': {
            'indent': 2,              # Pretty print JSON responses
            'separators': (',', ': ')
        },
        'SWAGGER_UI_DOC_EXPANSION': 'list',     # Expand operation lists by default
        'SWAGGER_UI_REQUEST_DURATION': True,    # Show request duration
        'SWAGGER_UI_OAUTH_REALM': 'BitAxe API', # OAuth realm for Swagger UI
        'SWAGGER_UI_OAUTH_CLIENT_ID': 'swagger-ui',
        'SWAGGER_SUPPORTED_SUBMIT_METHODS': ['get', 'post', 'put', 'delete', 'patch']
    })
    
    # Register the API with enhanced documentation
    register_restx_api(app)
    
    # Add custom CSS for Swagger UI styling
    @app.route('/swagger-ui-custom.css')
    def swagger_ui_css():
        """Custom CSS for Swagger UI"""
        custom_css = """
        .swagger-ui .topbar { 
            background-color: #2c3e50; 
        }
        .swagger-ui .topbar .download-url-wrapper .select-label {
            color: #ecf0f1;
        }
        .swagger-ui .info .title {
            color: #2c3e50;
        }
        .swagger-ui .scheme-container {
            background: #ecf0f1;
            box-shadow: 0 1px 2px 0 rgba(0,0,0,.15);
        }
        .swagger-ui .opblock.opblock-post {
            border-color: #27ae60;
            background: rgba(39, 174, 96, .1);
        }
        .swagger-ui .opblock.opblock-get {
            border-color: #3498db;
            background: rgba(52, 152, 219, .1);
        }
        .swagger-ui .opblock.opblock-put {
            border-color: #f39c12;
            background: rgba(243, 156, 18, .1);
        }
        .swagger-ui .opblock.opblock-delete {
            border-color: #e74c3c;
            background: rgba(231, 76, 60, .1);
        }
        """
        return custom_css, 200, {'Content-Type': 'text/css'}


def get_api_info() -> dict:
    """Get API information for documentation"""
    return {
        'title': 'BitAxe Web Management API',
        'version': '1.0.0',
        'description': '''
        # BitAxe Web Management API
        
        RESTful API for managing BitAxe ASIC Bitcoin mining devices.
        
        ## Features
        
        - **Miner Management**: Monitor and control individual miners
        - **Benchmark System**: Execute performance benchmarks
        - **Health Monitoring**: System-wide health checks
        - **Event Logging**: Comprehensive event tracking
        - **Configuration**: Dynamic configuration management
        - **Authentication**: JWT-based security with role-based access
        
        ## Getting Started
        
        1. **Authentication**: Use `/api/v1/auth/login` to obtain a JWT token
        2. **Authorization**: Include the token in the `Authorization` header as `Bearer <token>`
        3. **API Calls**: Make requests to the documented endpoints
        
        ## Default Users
        
        - **admin**: Full access (admin:admin123)
        - **operator**: Control operations (operator:operator123)  
        - **readonly**: Read-only access (readonly:readonly123)
        
        ## Rate Limits
        
        - Admin: 1000 requests/hour
        - Operator: 500 requests/hour
        - Readonly: 200 requests/hour
        
        ## Error Handling
        
        All API responses follow a consistent format:
        ```json
        {
            "success": boolean,
            "message": "string",
            "timestamp": "ISO8601",
            "data": object,
            "error": {
                "code": "ERROR_CODE",
                "message": "Error description"
            }
        }
        ```
        
        ## Changelog
        
        - **v1.0.0**: Initial API release with comprehensive miner management
        ''',
        'contact': {
            'name': 'BitAxe API Support',
            'url': 'https://github.com/BitAxe/web-management',
            'email': 'support@bitaxe.org'
        },
        'license': {
            'name': 'MIT License',
            'url': 'https://opensource.org/licenses/MIT'
        },
        'termsOfService': 'https://bitaxe.org/terms'
    }


def add_swagger_routes(app: Flask) -> None:
    """Add additional Swagger-related routes"""
    
    @app.route('/api/docs')
    def swagger_docs():
        """Redirect to Swagger UI"""
        return apidoc.ui_for(api)
    
    @app.route('/api/swagger.json')
    def swagger_json():
        """Return OpenAPI specification as JSON"""
        return api.__schema__
    
    @app.route('/api/redoc')
    def redoc():
        """Alternative API documentation using ReDoc"""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>BitAxe API Documentation</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            <redoc spec-url="/api/swagger.json"></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        '''


def setup_api_documentation(app: Flask) -> None:
    """Setup complete API documentation"""
    configure_swagger(app)
    add_swagger_routes(app)