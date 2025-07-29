import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def __init__(self, service_name: str = "bitaxe", version: str = "1.0.3"):
        super().__init__()
        self.service_name = service_name
        self.version = version
        self.hostname = self._get_hostname()
        
    def _get_hostname(self) -> str:
        """Get system hostname"""
        try:
            import socket
            return socket.gethostname()
        except:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "version": self.version,
            "hostname": self.hostname,
            "thread": threading.current_thread().name,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and not record.exc_info:
            log_data["stack_trace"] = traceback.format_stack()
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""
    
    def __init__(self, name: str, service_name: str = "bitaxe", 
                 log_level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.service_name = service_name
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create JSON formatter
        json_formatter = JSONFormatter(service_name)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with additional context"""
        extra_fields = {}
        
        # Extract context fields
        for key, value in kwargs.items():
            if key not in ['exc_info']:
                extra_fields[key] = value
        
        # Create log record with extra fields
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), 
            kwargs.get('exc_info'), extra={"extra_fields": extra_fields}
        )
        
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def log_request(self, method: str, url: str, status_code: int, 
                   duration_ms: float, **kwargs):
        """Log HTTP request"""
        self.info("HTTP request completed", 
                 request_method=method,
                 request_url=url,
                 status_code=status_code,
                 duration_ms=duration_ms,
                 **kwargs)
    
    def log_database_operation(self, operation: str, table: str, 
                              duration_ms: float, records_affected: int = None, **kwargs):
        """Log database operation"""
        self.info("Database operation completed",
                 db_operation=operation,
                 db_table=table,
                 duration_ms=duration_ms,
                 records_affected=records_affected,
                 **kwargs)
    
    def log_miner_event(self, ip: str, event_type: str, message: str, 
                       **kwargs):
        """Log miner-specific event"""
        self.info("Miner event",
                 miner_ip=ip,
                 event_type=event_type,
                 event_message=message,
                 **kwargs)
    
    def log_benchmark_event(self, ip: str, frequency: int, voltage: int,
                           event_type: str, **kwargs):
        """Log benchmark-specific event"""
        self.info("Benchmark event",
                 miner_ip=ip,
                 frequency=frequency,
                 voltage=voltage,
                 benchmark_event=event_type,
                 **kwargs)
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str, **kwargs):
        """Log performance metric"""
        self.info("Performance metric",
                 metric_name=metric_name,
                 metric_value=value,
                 metric_unit=unit,
                 **kwargs)
    
    @contextmanager
    def log_execution_time(self, operation_name: str, **context):
        """Context manager to log execution time"""
        start_time = datetime.now()
        try:
            yield
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self.info(f"Operation completed: {operation_name}",
                     operation=operation_name,
                     duration_ms=duration,
                     status="success",
                     **context)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self.error(f"Operation failed: {operation_name}",
                      operation=operation_name,
                      duration_ms=duration,
                      status="failed",
                      error=str(e),
                      **context)
            raise


class LoggerFactory:
    """Factory for creating structured loggers"""
    
    _loggers: Dict[str, StructuredLogger] = {}
    _default_config = {
        "service_name": "bitaxe",
        "log_level": "INFO",
        "log_file": None
    }
    
    @classmethod
    def configure(cls, service_name: str = "bitaxe", log_level: str = "INFO",
                 log_file: Optional[str] = None):
        """Configure default logger settings"""
        cls._default_config.update({
            "service_name": service_name,
            "log_level": log_level,
            "log_file": log_file
        })
    
    @classmethod
    def get_logger(cls, name: str, **override_config) -> StructuredLogger:
        """Get or create a structured logger"""
        if name not in cls._loggers:
            config = cls._default_config.copy()
            config.update(override_config)
            
            cls._loggers[name] = StructuredLogger(
                name=name,
                service_name=config["service_name"],
                log_level=config["log_level"],
                log_file=config["log_file"]
            )
        
        return cls._loggers[name]
    
    @classmethod
    def clear_loggers(cls):
        """Clear all cached loggers (useful for testing)"""
        cls._loggers.clear()


# Convenience functions
def get_logger(name: str, **config) -> StructuredLogger:
    """Get a structured logger instance"""
    return LoggerFactory.get_logger(name, **config)


def configure_logging(service_name: str = "bitaxe", log_level: str = "INFO",
                     log_file: Optional[str] = None):
    """Configure global logging settings"""
    LoggerFactory.configure(service_name, log_level, log_file)


# Global logger instances
app_logger = get_logger("bitaxe.app")
database_logger = get_logger("bitaxe.database")
miner_logger = get_logger("bitaxe.miner")
benchmark_logger = get_logger("bitaxe.benchmark")
autopilot_logger = get_logger("bitaxe.autopilot")
health_logger = get_logger("bitaxe.health")