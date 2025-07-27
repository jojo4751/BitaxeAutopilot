import time
import random
import functools
from typing import Callable, Type, Union, Tuple, Optional, Any
from datetime import datetime, timedelta
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import (
    MinerConnectionError, MinerTimeoutError, MinerAPIError, 
    DatabaseConnectionError, DatabaseTimeoutError
)

logger = get_logger("bitaxe.retry")


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 backoff_strategy: str = "exponential"):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.backoff_strategy = backoff_strategy
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        if self.backoff_strategy == "linear":
            delay = self.base_delay * attempt
        elif self.backoff_strategy == "exponential":
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.backoff_strategy == "fixed":
            delay = self.base_delay
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay


def retry_on_exception(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
                      config: Optional[RetryConfig] = None,
                      on_retry: Optional[Callable[[int, Exception], None]] = None):
    """
    Decorator that retries function execution on specified exceptions
    
    Args:
        exceptions: Exception types to retry on
        config: Retry configuration
        on_retry: Callback function called on each retry attempt
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful execution after retries
                    if attempt > 1:
                        logger.info(f"Function succeeded after {attempt} attempts",
                                  function=func.__name__,
                                  attempt=attempt,
                                  total_attempts=config.max_attempts)
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts:
                        logger.error(f"Function failed after {attempt} attempts",
                                   function=func.__name__,
                                   attempt=attempt,
                                   total_attempts=config.max_attempts,
                                   error=str(e))
                        break
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    
                    logger.warning(f"Function failed, retrying in {delay:.2f}s",
                                 function=func.__name__,
                                 attempt=attempt,
                                 total_attempts=config.max_attempts,
                                 delay_seconds=delay,
                                 error=str(e))
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, e)
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"Function failed with unexpected exception",
                               function=func.__name__,
                               attempt=attempt,
                               error=str(e))
                    raise
            
            # Re-raise the last exception if all retries failed
            raise last_exception
        
        return wrapper
    return decorator


def retry_http_request(config: Optional[RetryConfig] = None,
                      status_codes: Optional[Tuple[int, ...]] = None):
    """
    Decorator specifically for HTTP requests with smart retry logic
    
    Args:
        config: Retry configuration
        status_codes: HTTP status codes to retry on (defaults to 5xx and 429)
    """
    if config is None:
        config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0)
    
    if status_codes is None:
        # Retry on server errors and rate limiting
        status_codes = (429, 500, 502, 503, 504)
    
    # Exceptions to retry on
    retry_exceptions = (ConnectionError, Timeout, HTTPError)
    
    def should_retry_response(response):
        """Check if response should be retried"""
        return response.status_code in status_codes
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            last_response = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    response = func(*args, **kwargs)
                    
                    # Check if response indicates retry is needed
                    if hasattr(response, 'status_code') and should_retry_response(response):
                        last_response = response
                        
                        # Don't retry on last attempt
                        if attempt == config.max_attempts:
                            logger.error(f"HTTP request failed after {attempt} attempts",
                                       function=func.__name__,
                                       attempt=attempt,
                                       status_code=response.status_code,
                                       url=getattr(response, 'url', 'unknown'))
                            return response
                        
                        # Calculate delay and wait
                        delay = config.calculate_delay(attempt)
                        
                        # Check for Retry-After header
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                retry_delay = float(retry_after)
                                delay = min(retry_delay, config.max_delay)
                            except ValueError:
                                pass  # Use calculated delay
                        
                        logger.warning(f"HTTP request failed, retrying in {delay:.2f}s",
                                     function=func.__name__,
                                     attempt=attempt,
                                     status_code=response.status_code,
                                     delay_seconds=delay,
                                     url=getattr(response, 'url', 'unknown'))
                        
                        time.sleep(delay)
                        continue
                    
                    # Successful response
                    if attempt > 1:
                        logger.info(f"HTTP request succeeded after {attempt} attempts",
                                  function=func.__name__,
                                  attempt=attempt,
                                  status_code=getattr(response, 'status_code', 'unknown'),
                                  url=getattr(response, 'url', 'unknown'))
                    
                    return response
                    
                except retry_exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts:
                        logger.error(f"HTTP request failed after {attempt} attempts",
                                   function=func.__name__,
                                   attempt=attempt,
                                   error=str(e))
                        break
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    
                    logger.warning(f"HTTP request failed, retrying in {delay:.2f}s",
                                 function=func.__name__,
                                 attempt=attempt,
                                 delay_seconds=delay,
                                 error=str(e))
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"HTTP request failed with unexpected exception",
                               function=func.__name__,
                               attempt=attempt,
                               error=str(e))
                    raise
            
            # Return last response or raise last exception
            if last_response is not None:
                return last_response
            if last_exception is not None:
                raise last_exception
        
        return wrapper
    return decorator


def retry_miner_request(ip: str, config: Optional[RetryConfig] = None):
    """
    Decorator specifically for miner API requests with proper exception handling
    
    Args:
        ip: Miner IP address for context
        config: Retry configuration
    """
    if config is None:
        config = RetryConfig(max_attempts=3, base_delay=0.5, max_delay=10.0)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful request
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    if attempt > 1:
                        logger.info(f"Miner request succeeded after {attempt} attempts",
                                  miner_ip=ip,
                                  function=func.__name__,
                                  attempt=attempt,
                                  duration_ms=duration)
                    
                    return result
                    
                except (ConnectionError, Timeout) as e:
                    # Don't retry on last attempt
                    if attempt == config.max_attempts:
                        duration = (datetime.now() - start_time).total_seconds() * 1000
                        logger.error(f"Miner request failed after {attempt} attempts",
                                   miner_ip=ip,
                                   function=func.__name__,
                                   attempt=attempt,
                                   duration_ms=duration,
                                   error=str(e))
                        
                        if isinstance(e, Timeout):
                            raise MinerTimeoutError(ip, config.max_delay, cause=e)
                        else:
                            raise MinerConnectionError(ip, cause=e)
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    
                    logger.warning(f"Miner request failed, retrying in {delay:.2f}s",
                                 miner_ip=ip,
                                 function=func.__name__,
                                 attempt=attempt,
                                 delay_seconds=delay,
                                 error=str(e))
                    
                    time.sleep(delay)
                
                except HTTPError as e:
                    # Don't retry on client errors (4xx), but retry on server errors (5xx)
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if 400 <= status_code < 500:
                            # Client error - don't retry
                            raise MinerAPIError(ip, status_code, str(e))
                        elif status_code >= 500 and attempt < config.max_attempts:
                            # Server error - retry
                            delay = config.calculate_delay(attempt)
                            logger.warning(f"Miner API server error, retrying in {delay:.2f}s",
                                         miner_ip=ip,
                                         function=func.__name__,
                                         attempt=attempt,
                                         status_code=status_code,
                                         delay_seconds=delay)
                            time.sleep(delay)
                            continue
                    
                    raise MinerAPIError(ip, getattr(e.response, 'status_code', 0), str(e))
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    logger.error(f"Miner request failed with unexpected exception",
                               miner_ip=ip,
                               function=func.__name__,
                               attempt=attempt,
                               duration_ms=duration,
                               error=str(e))
                    raise MinerConnectionError(ip, cause=e)
        
        return wrapper
    return decorator


def retry_database_operation(config: Optional[RetryConfig] = None):
    """
    Decorator for database operations with retry logic
    
    Args:
        config: Retry configuration
    """
    if config is None:
        config = RetryConfig(max_attempts=3, base_delay=0.1, max_delay=5.0)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from sqlalchemy.exc import OperationalError, DisconnectionError, TimeoutError
            
            start_time = datetime.now()
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful operation
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    if attempt > 1:
                        logger.info(f"Database operation succeeded after {attempt} attempts",
                                  function=func.__name__,
                                  attempt=attempt,
                                  duration_ms=duration)
                    
                    return result
                    
                except (OperationalError, DisconnectionError, TimeoutError) as e:
                    # Don't retry on last attempt
                    if attempt == config.max_attempts:
                        duration = (datetime.now() - start_time).total_seconds() * 1000
                        logger.error(f"Database operation failed after {attempt} attempts",
                                   function=func.__name__,
                                   attempt=attempt,
                                   duration_ms=duration,
                                   error=str(e))
                        
                        if isinstance(e, TimeoutError):
                            raise DatabaseTimeoutError(cause=e)
                        else:
                            raise DatabaseConnectionError(cause=e)
                    
                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    
                    logger.warning(f"Database operation failed, retrying in {delay:.2f}s",
                                 function=func.__name__,
                                 attempt=attempt,
                                 delay_seconds=delay,
                                 error=str(e))
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    duration = (datetime.now() - start_time).total_seconds() * 1000
                    logger.error(f"Database operation failed with unexpected exception",
                               function=func.__name__,
                               attempt=attempt,
                               duration_ms=duration,
                               error=str(e))
                    raise
        
        return wrapper
    return decorator


# Predefined retry configurations
FAST_RETRY = RetryConfig(max_attempts=2, base_delay=0.1, max_delay=1.0)
STANDARD_RETRY = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)
SLOW_RETRY = RetryConfig(max_attempts=5, base_delay=2.0, max_delay=60.0)
NETWORK_RETRY = RetryConfig(max_attempts=3, base_delay=0.5, max_delay=30.0, exponential_base=2.0)


# Convenience decorators with predefined configurations
def fast_retry(func):
    """Quick retry with minimal delay"""
    return retry_on_exception(config=FAST_RETRY)(func)


def standard_retry(func):
    """Standard retry configuration"""
    return retry_on_exception(config=STANDARD_RETRY)(func)


def network_retry(func):
    """Retry configuration optimized for network operations"""
    return retry_http_request(config=NETWORK_RETRY)(func)