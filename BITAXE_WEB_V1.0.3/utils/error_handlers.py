import functools
import traceback
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Tuple, Union
from flask import request, jsonify, render_template
from werkzeug.exceptions import HTTPException
from pydantic import ValidationError

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import (
    BitaxeException, ErrorCode, ValidationError as CustomValidationError,
    MinerError, DatabaseError, BenchmarkError, TemperatureError,
    AutopilotError, ServiceError, HealthCheckError
)

logger = get_logger("bitaxe.error_handler")


class ErrorResponse:
    """Standardized error response format"""
    
    def __init__(self, error_code: str, message: str, details: Optional[Dict[str, Any]] = None,
                 status_code: int = 500, request_id: Optional[str] = None):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        self.request_id = request_id
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        result = {
            "error": {
                "code": self.error_code,
                "message": self.message,
                "timestamp": self.timestamp
            }
        }
        
        if self.details:
            result["error"]["details"] = self.details
        
        if self.request_id:
            result["error"]["request_id"] = self.request_id
        
        return result
    
    def to_json_response(self):
        """Convert to Flask JSON response"""
        return jsonify(self.to_dict()), self.status_code


def get_request_context() -> Dict[str, Any]:
    """Extract relevant request context for logging"""
    context = {}
    
    try:
        if request:
            context.update({
                "method": request.method,
                "url": request.url,
                "endpoint": request.endpoint,
                "remote_addr": request.remote_addr,
                "user_agent": request.headers.get('User-Agent', ''),
            })
            
            # Add form data for POST requests (excluding sensitive data)
            if request.method == "POST" and request.form:
                form_data = {}
                for key, value in request.form.items():
                    # Don't log sensitive fields
                    if key.lower() not in ['password', 'token', 'secret', 'key']:
                        form_data[key] = value
                context["form_data"] = form_data
    except RuntimeError:
        # Outside request context
        pass
    
    return context


def handle_bitaxe_exception(error: BitaxeException) -> ErrorResponse:
    """Handle custom BITAXE exceptions"""
    # Map error codes to HTTP status codes
    status_code_mapping = {
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.CONFIGURATION_ERROR: 400,
        ErrorCode.MINER_NOT_FOUND: 404,
        ErrorCode.MINER_CONNECTION_ERROR: 503,
        ErrorCode.MINER_TIMEOUT_ERROR: 504,
        ErrorCode.MINER_API_ERROR: 502,
        ErrorCode.DATABASE_CONNECTION_ERROR: 503,
        ErrorCode.DATABASE_TIMEOUT_ERROR: 504,
        ErrorCode.BENCHMARK_ALREADY_RUNNING: 409,
        ErrorCode.BENCHMARK_INVALID_SETTINGS: 400,
        ErrorCode.AUTOPILOT_ALREADY_RUNNING: 409,
        ErrorCode.AUTOPILOT_NOT_RUNNING: 409,
        ErrorCode.SERVICE_UNAVAILABLE: 503,
        ErrorCode.HEALTH_CHECK_FAILED: 503,
    }
    
    status_code = status_code_mapping.get(error.error_code, 500)
    
    return ErrorResponse(
        error_code=error.error_code.value,
        message=error.message,
        details=error.context,
        status_code=status_code
    )


def handle_validation_error(error: ValidationError) -> ErrorResponse:
    """Handle Pydantic validation errors"""
    details = {
        "validation_errors": []
    }
    
    for err in error.errors():
        details["validation_errors"].append({
            "field": ".".join(str(x) for x in err["loc"]),
            "message": err["msg"],
            "type": err["type"]
        })
    
    return ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="Data validation failed",
        details=details,
        status_code=400
    )


def handle_http_exception(error: HTTPException) -> ErrorResponse:
    """Handle standard HTTP exceptions"""
    return ErrorResponse(
        error_code=f"HTTP_{error.code}",
        message=error.description or f"HTTP {error.code} error",
        status_code=error.code or 500
    )


def handle_generic_exception(error: Exception) -> ErrorResponse:
    """Handle unexpected exceptions"""
    return ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        details={"error_type": type(error).__name__},
        status_code=500
    )


def route_error_boundary(return_json: bool = False, 
                        fallback_template: Optional[str] = None):
    """
    Decorator that provides error boundary for Flask routes
    
    Args:
        return_json: If True, always return JSON responses
        fallback_template: Template to render for HTML responses on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_context = get_request_context()
            request_id = f"req_{int(datetime.now().timestamp() * 1000)}"
            
            try:
                # Log request start
                logger.info("Request started",
                           function=func.__name__,
                           request_id=request_id,
                           **request_context)
                
                # Execute the route function
                start_time = datetime.now()
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log successful request
                logger.info("Request completed successfully",
                           function=func.__name__,
                           request_id=request_id,
                           duration_ms=duration,
                           **request_context)
                
                return result
                
            except BitaxeException as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log the error with context
                logger.error("Request failed with BITAXE exception",
                           function=func.__name__,
                           request_id=request_id,
                           duration_ms=duration,
                           error_code=e.error_code.value,
                           error_message=e.message,
                           error_context=e.context,
                           **request_context)
                
                error_response = handle_bitaxe_exception(e)
                error_response.request_id = request_id
                
                return _format_error_response(error_response, return_json, fallback_template)
                
            except ValidationError as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.error("Request failed with validation error",
                           function=func.__name__,
                           request_id=request_id,
                           duration_ms=duration,
                           validation_errors=e.errors(),
                           **request_context)
                
                error_response = handle_validation_error(e)
                error_response.request_id = request_id
                
                return _format_error_response(error_response, return_json, fallback_template)
                
            except HTTPException as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                logger.warning("Request failed with HTTP exception",
                             function=func.__name__,
                             request_id=request_id,
                             duration_ms=duration,
                             http_status=e.code,
                             **request_context)
                
                error_response = handle_http_exception(e)
                error_response.request_id = request_id
                
                return _format_error_response(error_response, return_json, fallback_template)
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log unexpected errors with full traceback
                logger.exception("Request failed with unexpected exception",
                               function=func.__name__,
                               request_id=request_id,
                               duration_ms=duration,
                               error_type=type(e).__name__,
                               **request_context)
                
                error_response = handle_generic_exception(e)
                error_response.request_id = request_id
                
                return _format_error_response(error_response, return_json, fallback_template)
        
        return wrapper
    return decorator


def _format_error_response(error_response: ErrorResponse, return_json: bool, 
                          fallback_template: Optional[str]) -> Union[Tuple, str]:
    """Format error response based on content type preference"""
    
    # Always return JSON for API routes or if explicitly requested
    if return_json or _wants_json():
        return error_response.to_json_response()
    
    # Return HTML template if specified and request accepts HTML
    if fallback_template and _wants_html():
        try:
            return render_template(
                fallback_template,
                error=error_response.to_dict()["error"],
                status_code=error_response.status_code
            ), error_response.status_code
        except Exception as template_error:
            logger.error("Failed to render error template",
                        template=fallback_template,
                        error=str(template_error))
            # Fall back to JSON
            return error_response.to_json_response()
    
    # Default to JSON response
    return error_response.to_json_response()


def _wants_json() -> bool:
    """Check if client prefers JSON response"""
    try:
        return (
            request.path.startswith('/api/') or
            request.headers.get('Content-Type', '').startswith('application/json') or
            'application/json' in request.headers.get('Accept', '')
        )
    except RuntimeError:
        return True  # Default to JSON outside request context


def _wants_html() -> bool:
    """Check if client prefers HTML response"""
    try:
        return 'text/html' in request.headers.get('Accept', '')
    except RuntimeError:
        return False


def api_error_boundary(func: Callable) -> Callable:
    """Convenience decorator for API routes that always return JSON"""
    return route_error_boundary(return_json=True)(func)


def web_error_boundary(fallback_template: str = "error.html"):
    """Convenience decorator for web routes with HTML fallback"""
    return route_error_boundary(return_json=False, fallback_template=fallback_template)


def register_flask_error_handlers(app):
    """Register global error handlers for Flask app"""
    
    @app.errorhandler(BitaxeException)
    def handle_bitaxe_error(error):
        """Global handler for BITAXE exceptions"""
        request_context = get_request_context()
        
        logger.error("Unhandled BITAXE exception",
                    error_code=error.error_code.value,
                    error_message=error.message,
                    error_context=error.context,
                    **request_context)
        
        error_response = handle_bitaxe_exception(error)
        return _format_error_response(error_response, _wants_json(), "error.html")
    
    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(error):
        """Global handler for Pydantic validation errors"""
        request_context = get_request_context()
        
        logger.error("Unhandled validation error",
                    validation_errors=error.errors(),
                    **request_context)
        
        error_response = handle_validation_error(error)
        return _format_error_response(error_response, _wants_json(), "error.html")
    
    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Global handler for internal server errors"""
        request_context = get_request_context()
        
        logger.exception("Internal server error",
                        **request_context)
        
        error_response = ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            status_code=500
        )
        
        return _format_error_response(error_response, _wants_json(), "error.html")
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Global handler for 404 errors"""
        error_response = ErrorResponse(
            error_code="NOT_FOUND",
            message="The requested resource was not found",
            status_code=404
        )
        
        return _format_error_response(error_response, _wants_json(), "404.html")
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Global handler for 400 errors"""
        error_response = ErrorResponse(
            error_code="BAD_REQUEST",
            message="The request was malformed or invalid",
            status_code=400
        )
        
        return _format_error_response(error_response, _wants_json(), "error.html")


# Context manager for operation-level error handling
class operation_error_boundary:
    """Context manager for handling errors in operations"""
    
    def __init__(self, operation_name: str, logger_instance=None):
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation_name}",
                         operation=self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds() * 1000
        
        if exc_type is None:
            # Success
            self.logger.debug(f"Operation completed: {self.operation_name}",
                            operation=self.operation_name,
                            duration_ms=duration)
            return False
        
        if issubclass(exc_type, BitaxeException):
            # Known exception - log and let it propagate
            self.logger.error(f"Operation failed: {self.operation_name}",
                            operation=self.operation_name,
                            duration_ms=duration,
                            error_code=exc_val.error_code.value,
                            error_message=exc_val.message,
                            error_context=exc_val.context)
            return False
        
        # Unexpected exception - log with traceback and let it propagate
        self.logger.exception(f"Operation failed with unexpected error: {self.operation_name}",
                            operation=self.operation_name,
                            duration_ms=duration,
                            error_type=exc_type.__name__)
        return False