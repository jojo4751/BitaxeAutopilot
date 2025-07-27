"""
OpenAPI specification for BitAxe Web Management API
"""

from typing import Dict, Any
from flask_restx import Api, Namespace, Resource, fields
from api.models import (
    APIResponse, ErrorResponse, PaginationParams, PaginatedResponse,
    MinerStatus, MinerSettings, MinersListResponse, MinersSummary,
    BenchmarkRequest, MultiBenchmarkRequest, BenchmarkResult, BenchmarkStatus,
    Event, EventsQuery, ComponentHealth, SystemHealth,
    ConfigUpdate, LoginRequest, TokenResponse, UserInfo
)

# Create API instance with OpenAPI documentation
api = Api(
    version='1.0',
    title='BitAxe Web Management API',
    description='''
    RESTful API for managing BitAxe ASIC Bitcoin mining devices.
    
    This API provides comprehensive management capabilities including:
    - Miner monitoring and control
    - Benchmark execution and analysis
    - System health monitoring
    - Event logging and tracking
    - Configuration management
    - User authentication and authorization
    
    ## Authentication
    Most endpoints require Bearer token authentication. Use the `/auth/login` endpoint 
    to obtain an access token, then include it in the Authorization header:
    ```
    Authorization: Bearer <your-token>
    ```
    
    ## Rate Limiting
    API requests are rate-limited per user. Limits are enforced based on user roles.
    
    ## Pagination
    List endpoints support pagination with `page` and `page_size` query parameters.
    Default page size is 50, maximum is 1000.
    
    ## Error Handling
    All errors follow a consistent format with HTTP status codes and structured error responses.
    ''',
    doc='/api/docs/',
    prefix='/api/v1',
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT Bearer token. Format: Bearer <token>'
        }
    },
    security='Bearer'
)

# Define common models for documentation
error_model = api.model('ErrorResponse', {
    'success': fields.Boolean(required=True, description='Always false for errors', example=False),
    'message': fields.String(description='Human-readable error message'),
    'timestamp': fields.DateTime(description='Error timestamp'),
    'error': fields.Raw(description='Error details', example={
        'code': 'VALIDATION_ERROR',
        'details': {'field': 'frequency', 'message': 'Value must be between 400 and 1200'}
    })
})

pagination_model = api.model('Pagination', {
    'page': fields.Integer(description='Current page number', example=1),
    'page_size': fields.Integer(description='Items per page', example=50),
    'total_count': fields.Integer(description='Total number of items', example=125),
    'total_pages': fields.Integer(description='Total number of pages', example=3),
    'has_next': fields.Boolean(description='Whether there is a next page', example=True),
    'has_previous': fields.Boolean(description='Whether there is a previous page', example=False)
})

# Miner models
miner_status_model = api.model('MinerStatus', {
    'ip': fields.String(required=True, description='Miner IP address', example='192.168.1.100'),
    'hostname': fields.String(description='Miner hostname', example='bitaxe-001'),
    'temperature': fields.Float(description='Temperature in Celsius', example=65.5),
    'hash_rate': fields.Float(description='Hash rate in GH/s', example=485.2),
    'power': fields.Float(description='Power consumption in watts', example=12.8),
    'voltage': fields.Float(description='Input voltage', example=5.1),
    'frequency': fields.Integer(description='Frequency in MHz', example=800),
    'core_voltage': fields.Integer(description='Core voltage in mV', example=1200),
    'fan_rpm': fields.Integer(description='Fan RPM', example=3500),
    'shares_accepted': fields.Integer(description='Accepted shares', example=1250),
    'shares_rejected': fields.Integer(description='Rejected shares', example=5),
    'uptime': fields.Integer(description='Uptime in seconds', example=86400),
    'version': fields.String(description='Firmware version', example='2.0.4'),
    'efficiency': fields.Float(description='Efficiency in GH/W', example=37.9),
    'last_seen': fields.DateTime(required=True, description='Last data update timestamp')
})

miner_settings_model = api.model('MinerSettings', {
    'frequency': fields.Integer(required=True, description='Frequency in MHz (400-1200)', 
                               min=400, max=1200, example=800),
    'core_voltage': fields.Integer(required=True, description='Core voltage in mV (800-1500)', 
                                  min=800, max=1500, example=1200),
    'autofanspeed': fields.Boolean(required=True, description='Enable automatic fan speed control', example=True),
    'fanspeed': fields.Integer(description='Manual fan speed percentage (0-100)', 
                              min=0, max=100, example=75)
})

miners_summary_model = api.model('MinersSummary', {
    'total_miners': fields.Integer(required=True, description='Total configured miners', example=5),
    'online_miners': fields.Integer(required=True, description='Number of online miners', example=4),
    'offline_miners': fields.Integer(required=True, description='Number of offline miners', example=1),
    'total_hashrate': fields.Float(required=True, description='Combined hash rate in GH/s', example=1942.8),
    'total_power': fields.Float(required=True, description='Combined power consumption in watts', example=64.2),
    'total_efficiency': fields.Float(required=True, description='Overall efficiency in GH/W', example=30.3),
    'average_temperature': fields.Float(required=True, description='Average temperature across all miners', example=67.2),
    'timestamp': fields.DateTime(required=True, description='Summary timestamp')
})

# Benchmark models
benchmark_request_model = api.model('BenchmarkRequest', {
    'ip': fields.String(required=True, description='Miner IP address', 
                       pattern=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', example='192.168.1.100'),
    'frequency': fields.Integer(required=True, description='Frequency in MHz (400-1200)', 
                               min=400, max=1200, example=800),
    'core_voltage': fields.Integer(required=True, description='Core voltage in mV (800-1500)', 
                                  min=800, max=1500, example=1200),
    'duration': fields.Integer(description='Benchmark duration in seconds (60-7200)', 
                              min=60, max=7200, default=600, example=600)
})

multi_benchmark_request_model = api.model('MultiBenchmarkRequest', {
    'ips': fields.List(fields.String, required=True, description='List of miner IP addresses',
                      example=['192.168.1.100', '192.168.1.101']),
    'frequency': fields.Integer(required=True, description='Frequency in MHz (400-1200)', 
                               min=400, max=1200, example=800),
    'core_voltage': fields.Integer(required=True, description='Core voltage in mV (800-1500)', 
                                  min=800, max=1500, example=1200),
    'duration': fields.Integer(description='Benchmark duration in seconds (60-7200)', 
                              min=60, max=7200, default=600, example=600)
})

benchmark_result_model = api.model('BenchmarkResult', {
    'id': fields.Integer(required=True, description='Benchmark result ID', example=123),
    'ip': fields.String(required=True, description='Miner IP address', example='192.168.1.100'),
    'frequency': fields.Integer(required=True, description='Frequency in MHz', example=800),
    'core_voltage': fields.Integer(required=True, description='Core voltage in mV', example=1200),
    'average_hashrate': fields.Float(description='Average hash rate in GH/s', example=485.2),
    'average_temperature': fields.Float(description='Average temperature in Celsius', example=65.5),
    'efficiency_jth': fields.Float(description='Efficiency in J/TH', example=26.3),
    'average_vr_temp': fields.Float(description='Average VR temperature', example=58.2),
    'duration': fields.Integer(required=True, description='Actual benchmark duration', example=600),
    'samples_count': fields.Integer(description='Number of samples collected', example=40),
    'aborted': fields.Boolean(description='Whether benchmark was aborted', example=False),
    'abort_reason': fields.String(description='Reason for abort if applicable'),
    'timestamp': fields.DateTime(required=True, description='Benchmark completion timestamp')
})

benchmark_status_model = api.model('BenchmarkStatus', {
    'active_benchmarks': fields.List(fields.String, required=True, 
                                   description='List of IPs with active benchmarks',
                                   example=['192.168.1.100', '192.168.1.101']),
    'total_active': fields.Integer(required=True, description='Number of active benchmarks', example=2)
})

# Event models
event_model = api.model('Event', {
    'id': fields.Integer(required=True, description='Event ID', example=456),
    'timestamp': fields.DateTime(required=True, description='Event timestamp'),
    'ip': fields.String(required=True, description='Related IP address or SYSTEM', example='192.168.1.100'),
    'event_type': fields.String(required=True, description='Event type', example='BENCHMARK_COMPLETED'),
    'message': fields.String(required=True, description='Event message', example='Benchmark completed successfully'),
    'severity': fields.String(required=True, description='Event severity', 
                             enum=['INFO', 'WARNING', 'ERROR', 'CRITICAL'], example='INFO')
})

# Health models
component_health_model = api.model('ComponentHealth', {
    'component': fields.String(required=True, description='Component name', example='database'),
    'status': fields.String(required=True, description='Health status', 
                           enum=['healthy', 'degraded', 'unhealthy', 'unknown'], example='healthy'),
    'message': fields.String(required=True, description='Health message', example='Database is accessible'),
    'details': fields.Raw(description='Additional health details', example={
        'connection_info': {'pool_size': 10, 'checked_out': 2}
    }),
    'timestamp': fields.DateTime(required=True, description='Health check timestamp'),
    'duration_ms': fields.Float(required=True, description='Health check duration in milliseconds', example=15.3)
})

system_health_model = api.model('SystemHealth', {
    'overall_status': fields.String(required=True, description='Overall system status', 
                                   enum=['healthy', 'degraded', 'unhealthy', 'unknown'], example='healthy'),
    'timestamp': fields.DateTime(required=True, description='Health check timestamp'),
    'total_checks': fields.Integer(required=True, description='Total number of health checks', example=8),
    'status_counts': fields.Raw(required=True, description='Count of each status type', example={
        'healthy': 7, 'degraded': 1, 'unhealthy': 0, 'unknown': 0
    }),
    'checks': fields.Raw(required=True, description='Individual component health')
})

# Authentication models
login_request_model = api.model('LoginRequest', {
    'username': fields.String(required=True, description='Username', example='admin'),
    'password': fields.String(required=True, description='Password', example='password123')
})

token_response_model = api.model('TokenData', {
    'access_token': fields.String(required=True, description='JWT access token'),
    'token_type': fields.String(required=True, description='Token type', example='bearer'),
    'expires_in': fields.Integer(required=True, description='Token expiry in seconds', example=3600),
    'expires_at': fields.String(required=True, description='Token expiry timestamp')
})

user_info_model = api.model('UserInfo', {
    'username': fields.String(required=True, description='Username', example='admin'),
    'roles': fields.List(fields.String, required=True, description='User roles', 
                        example=['admin', 'operator']),
    'permissions': fields.List(fields.String, required=True, description='User permissions', 
                              example=['read', 'write', 'control']),
    'is_active': fields.Boolean(required=True, description='Whether user is active', example=True),
    'created_at': fields.String(description='User creation timestamp'),
    'last_login': fields.String(description='Last login timestamp')
})

# Configuration models
config_update_model = api.model('ConfigUpdate', {
    'key': fields.String(required=True, description='Configuration key', 
                        example='settings.benchmark_interval_sec'),
    'value': fields.String(required=True, description='Configuration value', example='3600')
})

# Success response wrappers
api_response_model = api.model('APIResponse', {
    'success': fields.Boolean(required=True, description='Whether the request was successful', example=True),
    'message': fields.String(description='Human-readable message'),
    'timestamp': fields.DateTime(description='Response timestamp'),
    'request_id': fields.String(description='Unique request identifier')
})

paginated_miners_response = api.model('PaginatedMinersResponse', {
    'success': fields.Boolean(required=True, example=True),
    'message': fields.String(),
    'timestamp': fields.DateTime(),
    'data': fields.List(fields.Nested(miner_status_model)),
    'pagination': fields.Nested(pagination_model)
})

miners_summary_response = api.model('MinersSummaryResponse', {
    'success': fields.Boolean(required=True, example=True),
    'data': fields.Nested(miners_summary_model)
})

benchmark_results_response = api.model('BenchmarkResultsResponse', {
    'success': fields.Boolean(required=True, example=True),
    'data': fields.List(fields.Nested(benchmark_result_model))
})

events_response = api.model('EventsResponse', {
    'success': fields.Boolean(required=True, example=True),
    'data': fields.List(fields.Nested(event_model))
})

health_response = api.model('HealthResponse', {
    'success': fields.Boolean(required=True, example=True),
    'data': fields.Nested(system_health_model)
})

token_response_wrapper = api.model('TokenResponse', {
    'success': fields.Boolean(required=True, example=True),
    'message': fields.String(example='Login successful'),
    'data': fields.Nested(token_response_model)
})

user_info_response = api.model('UserInfoResponse', {
    'success': fields.Boolean(required=True, example=True),
    'data': fields.Nested(user_info_model)
})

# Export models for use in blueprints
__all__ = [
    'api',
    'error_model',
    'miner_status_model',
    'miner_settings_model',
    'miners_summary_model',
    'benchmark_request_model',
    'multi_benchmark_request_model',
    'benchmark_result_model',
    'benchmark_status_model',
    'event_model',
    'component_health_model',
    'system_health_model',
    'login_request_model',
    'token_response_model',
    'user_info_model',
    'config_update_model',
    'paginated_miners_response',
    'miners_summary_response',
    'benchmark_results_response',
    'events_response',
    'health_response',
    'token_response_wrapper',
    'user_info_response'
]