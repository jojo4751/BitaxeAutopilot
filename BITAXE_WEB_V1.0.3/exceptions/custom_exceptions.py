from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Standardized error codes for the application"""
    
    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    
    # Database errors
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_OPERATION_ERROR = "DATABASE_OPERATION_ERROR"
    DATABASE_INTEGRITY_ERROR = "DATABASE_INTEGRITY_ERROR"
    DATABASE_TIMEOUT_ERROR = "DATABASE_TIMEOUT_ERROR"
    
    # Miner communication errors
    MINER_CONNECTION_ERROR = "MINER_CONNECTION_ERROR"
    MINER_TIMEOUT_ERROR = "MINER_TIMEOUT_ERROR"
    MINER_API_ERROR = "MINER_API_ERROR"
    MINER_INVALID_RESPONSE = "MINER_INVALID_RESPONSE"
    MINER_NOT_FOUND = "MINER_NOT_FOUND"
    MINER_OFFLINE = "MINER_OFFLINE"
    
    # Benchmark errors
    BENCHMARK_ALREADY_RUNNING = "BENCHMARK_ALREADY_RUNNING"
    BENCHMARK_INVALID_SETTINGS = "BENCHMARK_INVALID_SETTINGS"
    BENCHMARK_TIMEOUT = "BENCHMARK_TIMEOUT"
    BENCHMARK_ABORTED = "BENCHMARK_ABORTED"
    BENCHMARK_FAILED = "BENCHMARK_FAILED"
    
    # Autopilot errors
    AUTOPILOT_NOT_RUNNING = "AUTOPILOT_NOT_RUNNING"
    AUTOPILOT_ALREADY_RUNNING = "AUTOPILOT_ALREADY_RUNNING"
    AUTOPILOT_CONFIGURATION_ERROR = "AUTOPILOT_CONFIGURATION_ERROR"
    
    # Temperature and safety errors
    TEMPERATURE_OVERHEAT = "TEMPERATURE_OVERHEAT"
    TEMPERATURE_LIMIT_EXCEEDED = "TEMPERATURE_LIMIT_EXCEEDED"
    SAFETY_LIMIT_EXCEEDED = "SAFETY_LIMIT_EXCEEDED"
    
    # Service errors
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    SERVICE_TIMEOUT = "SERVICE_TIMEOUT"
    SERVICE_CONFIGURATION_ERROR = "SERVICE_CONFIGURATION_ERROR"
    
    # Health check errors
    HEALTH_CHECK_FAILED = "HEALTH_CHECK_FAILED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


class BitaxeException(Exception):
    """Base exception class for all BITAXE-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 context: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses"""
        result = {
            "error_code": self.error_code.value,
            "message": self.message,
            "context": self.context
        }
        
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }
        
        return result
    
    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"


class ConfigurationError(BitaxeException):
    """Raised when there's a configuration-related error"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 expected_type: Optional[str] = None, actual_value: Optional[Any] = None):
        context = {}
        if config_key:
            context["config_key"] = config_key
        if expected_type:
            context["expected_type"] = expected_type
        if actual_value is not None:
            context["actual_value"] = str(actual_value)
        
        super().__init__(message, ErrorCode.CONFIGURATION_ERROR, context)


class ValidationError(BitaxeException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 value: Optional[Any] = None, constraints: Optional[Dict[str, Any]] = None):
        context = {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        if constraints:
            context["constraints"] = constraints
        
        super().__init__(message, ErrorCode.VALIDATION_ERROR, context)


class DatabaseError(BitaxeException):
    """Base class for database-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATABASE_OPERATION_ERROR,
                 operation: Optional[str] = None, table: Optional[str] = None, cause: Optional[Exception] = None):
        context = {}
        if operation:
            context["operation"] = operation
        if table:
            context["table"] = table
        
        super().__init__(message, error_code, context, cause)


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    
    def __init__(self, message: str = "Failed to connect to database", database_url: Optional[str] = None, cause: Optional[Exception] = None):
        context = {}
        if database_url:
            # Don't log sensitive connection details
            context["database_type"] = database_url.split("://")[0] if "://" in database_url else "unknown"
        
        super().__init__(message, ErrorCode.DATABASE_CONNECTION_ERROR, context=context, cause=cause)


class DatabaseTimeoutError(DatabaseError):
    """Raised when database operation times out"""
    
    def __init__(self, message: str = "Database operation timed out", timeout_seconds: Optional[float] = None, cause: Optional[Exception] = None):
        context = {}
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds
        
        super().__init__(message, ErrorCode.DATABASE_TIMEOUT_ERROR, context=context, cause=cause)


class MinerError(BitaxeException):
    """Base class for miner-related errors"""
    
    def __init__(self, message: str, ip: str, error_code: ErrorCode = ErrorCode.MINER_CONNECTION_ERROR,
                 endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        context = {"ip": ip}
        if endpoint:
            context["endpoint"] = endpoint
        
        super().__init__(message, error_code, context, cause)


class MinerConnectionError(MinerError):
    """Raised when unable to connect to miner"""
    
    def __init__(self, ip: str, message: str = None, endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        if not message:
            message = f"Failed to connect to miner at {ip}"
        super().__init__(message, ip, ErrorCode.MINER_CONNECTION_ERROR, endpoint, cause)


class MinerTimeoutError(MinerError):
    """Raised when miner communication times out"""
    
    def __init__(self, ip: str, timeout_seconds: float, endpoint: Optional[str] = None, cause: Optional[Exception] = None):
        message = f"Miner communication timed out after {timeout_seconds}s"
        context = {"timeout_seconds": timeout_seconds}
        super().__init__(message, ip, ErrorCode.MINER_TIMEOUT_ERROR, endpoint, cause)
        self.context.update(context)


class MinerAPIError(MinerError):
    """Raised when miner API returns an error"""
    
    def __init__(self, ip: str, status_code: int, response_text: str = None, endpoint: Optional[str] = None):
        message = f"Miner API error: HTTP {status_code}"
        context = {"status_code": status_code}
        if response_text:
            context["response_text"] = response_text[:500]  # Limit response text length
        
        super().__init__(message, ip, ErrorCode.MINER_API_ERROR, endpoint)
        self.context.update(context)


class MinerOfflineError(MinerError):
    """Raised when miner is offline or not responding"""
    
    def __init__(self, ip: str, last_seen: Optional[str] = None):
        message = f"Miner {ip} is offline"
        context = {}
        if last_seen:
            context["last_seen"] = last_seen
        
        super().__init__(message, ip, ErrorCode.MINER_OFFLINE)
        self.context.update(context)


class BenchmarkError(BitaxeException):
    """Base class for benchmark-related errors"""
    
    def __init__(self, message: str, ip: str, error_code: ErrorCode = ErrorCode.BENCHMARK_FAILED,
                 frequency: Optional[int] = None, voltage: Optional[int] = None, cause: Optional[Exception] = None):
        context = {"ip": ip}
        if frequency:
            context["frequency"] = frequency
        if voltage:
            context["voltage"] = voltage
        
        super().__init__(message, error_code, context, cause)


class BenchmarkAlreadyRunningError(BenchmarkError):
    """Raised when trying to start benchmark while one is already running"""
    
    def __init__(self, ip: str):
        message = f"Benchmark already running for miner {ip}"
        super().__init__(message, ip, ErrorCode.BENCHMARK_ALREADY_RUNNING)


class BenchmarkInvalidSettingsError(BenchmarkError):
    """Raised when benchmark settings are invalid"""
    
    def __init__(self, ip: str, frequency: int, voltage: int, reason: str = None):
        message = f"Invalid benchmark settings: {frequency}MHz @ {voltage}mV"
        if reason:
            message += f" - {reason}"
        
        super().__init__(message, ip, ErrorCode.BENCHMARK_INVALID_SETTINGS, frequency, voltage)


class BenchmarkAbortedError(BenchmarkError):
    """Raised when benchmark is aborted due to safety concerns"""
    
    def __init__(self, ip: str, reason: str, temperature: Optional[float] = None):
        message = f"Benchmark aborted: {reason}"
        context = {"abort_reason": reason}
        if temperature:
            context["temperature"] = temperature
        
        super().__init__(message, ip, ErrorCode.BENCHMARK_ABORTED)
        self.context.update(context)


class TemperatureError(BitaxeException):
    """Base class for temperature-related errors"""
    
    def __init__(self, message: str, ip: str, temperature: float, 
                 error_code: ErrorCode = ErrorCode.TEMPERATURE_LIMIT_EXCEEDED):
        context = {"ip": ip, "temperature": temperature}
        super().__init__(message, error_code, context)


class OverheatError(TemperatureError):
    """Raised when miner temperature exceeds overheat threshold"""
    
    def __init__(self, ip: str, temperature: float, threshold: float):
        message = f"Miner overheat detected: {temperature}°C > {threshold}°C"
        context = {"threshold": threshold}
        super().__init__(message, ip, temperature, ErrorCode.TEMPERATURE_OVERHEAT)
        self.context.update(context)


class AutopilotError(BitaxeException):
    """Base class for autopilot-related errors"""
    
    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.AUTOPILOT_CONFIGURATION_ERROR, cause: Optional[Exception] = None):
        super().__init__(message, error_code, cause=cause)


class AutopilotNotRunningError(AutopilotError):
    """Raised when trying to perform autopilot operation while not running"""
    
    def __init__(self, operation: str):
        message = f"Cannot perform '{operation}': autopilot not running"
        super().__init__(message, ErrorCode.AUTOPILOT_NOT_RUNNING)


class AutopilotAlreadyRunningError(AutopilotError):
    """Raised when trying to start autopilot while already running"""
    
    def __init__(self):
        message = "Autopilot is already running"
        super().__init__(message, ErrorCode.AUTOPILOT_ALREADY_RUNNING)


class ServiceError(BitaxeException):
    """Base class for service-related errors"""
    
    def __init__(self, message: str, service_name: str, error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE, cause: Optional[Exception] = None):
        context = {"service_name": service_name}
        super().__init__(message, error_code, context, cause)


class ServiceUnavailableError(ServiceError):
    """Raised when a required service is unavailable"""
    
    def __init__(self, service_name: str, reason: Optional[str] = None):
        message = f"Service '{service_name}' is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, service_name, ErrorCode.SERVICE_UNAVAILABLE)


class HealthCheckError(BitaxeException):
    """Raised when health check fails"""
    
    def __init__(self, component: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Health check failed for {component}: {reason}"
        context = {"component": component, "reason": reason}
        if details:
            context["details"] = details
        
        super().__init__(message, ErrorCode.HEALTH_CHECK_FAILED, context)


# Utility functions for exception handling
def wrap_database_error(func):
    """Decorator to wrap database exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, BitaxeException):
                raise
            raise DatabaseError(f"Database operation failed: {str(e)}", cause=e)
    return wrapper


def wrap_miner_error(ip: str):
    """Decorator factory to wrap miner communication exceptions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, BitaxeException):
                    raise
                raise MinerConnectionError(ip, cause=e)
        return wrapper
    return decorator


def handle_validation_error(func):
    """Decorator to handle validation errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            raise ValidationError(str(e), cause=e)
        except TypeError as e:
            raise ValidationError(f"Type error: {str(e)}", cause=e)
    return wrapper