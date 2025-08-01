"""
BitAxe V2.0.0 - Custom Exception Classes
Comprehensive exception hierarchy for better error handling and debugging
"""

from typing import Optional, Dict, Any


class BitAxeException(Exception):
    """Base exception class for all BitAxe-related errors.
    
    This serves as the parent class for all custom exceptions in the BitAxe system,
    providing consistent error handling and context information.
    
    Attributes:
        message: Human-readable error message
        error_code: Unique error code for programmatic handling
        context: Additional context information for debugging
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize BitAxe exception.
        
        Args:
            message: Human-readable error description
            error_code: Unique identifier for this error type
            context: Additional debugging information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context
        }
    
    def __str__(self) -> str:
        """String representation of the exception."""
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(BitAxeException):
    """Exception raised for configuration-related errors.
    
    This includes invalid configuration files, missing required settings,
    or configuration validation failures.
    """
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        config_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize configuration error.
        
        Args:
            message: Error description
            config_path: Path to the configuration file
            config_key: Specific configuration key that caused the error
            **kwargs: Additional context information
        """
        context = kwargs
        if config_path:
            context['config_path'] = config_path
        if config_key:
            context['config_key'] = config_key
            
        super().__init__(message, 'CONFIG_ERROR', context)


class DatabaseError(BitAxeException):
    """Exception raised for database-related errors.
    
    This includes connection failures, query errors, schema issues,
    and data integrity problems.
    """
    
    def __init__(
        self,
        message: str,
        database_path: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize database error.
        
        Args:
            message: Error description
            database_path: Path to the database file
            query: SQL query that caused the error
            **kwargs: Additional context information
        """
        context = kwargs
        if database_path:
            context['database_path'] = database_path
        if query:
            context['query'] = query
            
        super().__init__(message, 'DATABASE_ERROR', context)


class MinerError(BitAxeException):
    """Exception raised for miner communication and control errors.
    
    This includes network connectivity issues, API communication failures,
    and miner hardware problems.
    """
    
    def __init__(
        self,
        message: str,
        miner_ip: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize miner error.
        
        Args:
            message: Error description
            miner_ip: IP address of the affected miner
            api_endpoint: API endpoint that failed
            status_code: HTTP status code (if applicable)
            **kwargs: Additional context information
        """
        context = kwargs
        if miner_ip:
            context['miner_ip'] = miner_ip
        if api_endpoint:
            context['api_endpoint'] = api_endpoint
        if status_code:
            context['status_code'] = status_code
            
        super().__init__(message, 'MINER_ERROR', context)


class BenchmarkError(BitAxeException):
    """Exception raised for benchmark-related errors.
    
    This includes benchmark execution failures, invalid benchmark parameters,
    and benchmark result processing errors.
    """
    
    def __init__(
        self,
        message: str,
        miner_ip: Optional[str] = None,
        frequency: Optional[int] = None,
        voltage: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize benchmark error.
        
        Args:
            message: Error description
            miner_ip: IP address of the miner being benchmarked
            frequency: Frequency setting for the benchmark
            voltage: Voltage setting for the benchmark
            **kwargs: Additional context information
        """
        context = kwargs
        if miner_ip:
            context['miner_ip'] = miner_ip
        if frequency:
            context['frequency'] = frequency
        if voltage:
            context['voltage'] = voltage
            
        super().__init__(message, 'BENCHMARK_ERROR', context)


class AutopilotError(BitAxeException):
    """Exception raised for autopilot system errors.
    
    This includes autopilot control failures, optimization algorithm errors,
    and safety mechanism triggers.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize autopilot error.
        
        Args:
            message: Error description
            operation: Autopilot operation that failed
            **kwargs: Additional context information
        """
        context = kwargs
        if operation:
            context['operation'] = operation
            
        super().__init__(message, 'AUTOPILOT_ERROR', context)


class ValidationError(BitAxeException):
    """Exception raised for data validation errors.
    
    This includes invalid input parameters, schema validation failures,
    and data format errors.
    """
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize validation error.
        
        Args:
            message: Error description
            field_name: Name of the field that failed validation
            field_value: Value that failed validation
            expected_type: Expected data type or format
            **kwargs: Additional context information
        """
        context = kwargs
        if field_name:
            context['field_name'] = field_name
        if field_value is not None:
            context['field_value'] = field_value
        if expected_type:
            context['expected_type'] = expected_type
            
        super().__init__(message, 'VALIDATION_ERROR', context)


class SecurityError(BitAxeException):
    """Exception raised for security-related errors.
    
    This includes authentication failures, authorization errors,
    and security policy violations.
    """
    
    def __init__(
        self,
        message: str,
        security_context: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize security error.
        
        Args:
            message: Error description
            security_context: Context where the security error occurred
            **kwargs: Additional context information
        """
        context = kwargs
        if security_context:
            context['security_context'] = security_context
            
        super().__init__(message, 'SECURITY_ERROR', context)


class APIError(BitAxeException):
    """Exception raised for API-related errors.
    
    This includes HTTP client errors, API response parsing failures,
    and external service communication issues.
    """
    
    def __init__(
        self,
        message: str,
        api_url: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        response_data: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize API error.
        
        Args:
            message: Error description
            api_url: URL that caused the error
            method: HTTP method used
            status_code: HTTP status code returned
            response_data: Response data from the API
            **kwargs: Additional context information
        """
        context = kwargs
        if api_url:
            context['api_url'] = api_url
        if method:
            context['method'] = method
        if status_code:
            context['status_code'] = status_code
        if response_data:
            context['response_data'] = response_data
            
        super().__init__(message, 'API_ERROR', context)