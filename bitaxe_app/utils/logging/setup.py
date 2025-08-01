"""
BitAxe V2.0.0 - Logging Setup and Configuration
Centralized logging configuration for consistent logging across the application
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from flask import Flask

from .structured_logger import StructuredLogger


def setup_logging(
    app: Optional[Flask] = None,
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    structured_format: bool = True
) -> None:
    """Setup comprehensive logging configuration for the BitAxe application.
    
    This function configures logging with the following features:
    - Console and file output with different levels
    - Rotating file handler to prevent large log files
    - Structured JSON logging for production environments
    - Integration with Flask's logging system
    - Proper formatting and error handling
    
    Args:
        app: Flask application instance (optional)
        log_level: Default log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (if None, creates default path)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output to console
        structured_format: Whether to use structured JSON logging
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory if needed
    if log_file is None:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'bitaxe.log'
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    
    # Setup formatters
    if structured_format:
        # Structured JSON formatter for production
        console_formatter = StructuredFormatter(include_extra=False)
        file_formatter = StructuredFormatter(include_extra=True)
    else:
        # Simple text formatter for development
        console_format = '%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
        file_format = '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d: %(message)s'
        
        console_formatter = logging.Formatter(console_format)
        file_formatter = logging.Formatter(file_format)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Configure Flask logging if app is provided
    if app:
        app.logger.setLevel(numeric_level)
        
        # Disable Flask's default logging in production
        if not app.config.get('DEBUG', False):
            logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Configure third-party loggers
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Set specific loggers to appropriate levels
    logging.getLogger('bitaxe_app').setLevel(numeric_level)
    logging.getLogger('bitaxe_app.core').setLevel(numeric_level)
    logging.getLogger('bitaxe_app.services').setLevel(numeric_level)
    
    # Log the setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}, structured={structured_format}")


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging.
    
    This formatter creates JSON-structured log entries with consistent
    fields for better log parsing and analysis in production environments.
    """
    
    def __init__(self, include_extra: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_extra: Whether to include extra fields in log records
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON string representation of the log record
        """
        import json
        from datetime import datetime
        
        # Base log entry structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add process information
        log_entry['process'] = {
            'pid': os.getpid(),
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        try:
            return json.dumps(log_entry, ensure_ascii=False)
        except Exception:
            # Fallback to simple format if JSON serialization fails
            return f"{record.levelname}: {record.getMessage()}"


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
    
    return logger


def log_performance(func):
    """Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with performance logging
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(
                f"Function {func.__name__} completed",
                extra={
                    'function': func.__name__,
                    'execution_time_ms': round(execution_time * 1000, 2),
                    'module': func.__module__
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Function {func.__name__} failed: {e}",
                extra={
                    'function': func.__name__,
                    'execution_time_ms': round(execution_time * 1000, 2),
                    'module': func.__module__,
                    'error': str(e)
                },
                exc_info=True
            )
            raise
    
    return wrapper


def log_database_operation(operation_type: str):
    """Decorator to log database operations.
    
    Args:
        operation_type: Type of database operation (SELECT, INSERT, UPDATE, DELETE)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.debug(
                    f"Database {operation_type} operation completed",
                    extra={
                        'operation_type': operation_type,
                        'function': func.__name__,
                        'execution_time_ms': round(execution_time * 1000, 2)
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    f"Database {operation_type} operation failed: {e}",
                    extra={
                        'operation_type': operation_type,
                        'function': func.__name__,
                        'execution_time_ms': round(execution_time * 1000, 2),
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator