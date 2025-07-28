"""
Rate limiting utilities for API endpoints
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from functools import wraps
from flask import request, jsonify, g
from collections import defaultdict, deque

from bitaxe_logging.structured_logger import get_logger

logger = get_logger("bitaxe.rate_limiter")


class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, max_requests: int, window_seconds: int, burst_size: Optional[int] = None):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
            burst_size: Maximum burst size (defaults to max_requests)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_size = burst_size or max_requests
        
        # Store request timestamps per client
        self.clients: Dict[str, deque] = defaultdict(lambda: deque())
        
        # Token bucket per client
        self.tokens: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'tokens': self.burst_size,
            'last_refill': time.time()
        })
    
    def _get_client_id(self) -> str:
        """Get client identifier from request"""
        # Try to get user from JWT token first
        if hasattr(request, 'current_user') and request.current_user:
            return f"user:{request.current_user.get('username', 'unknown')}"
        
        # Fall back to IP address
        if request.environ.get('HTTP_X_FORWARDED_FOR'):
            ip = request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
        else:
            ip = request.environ.get('REMOTE_ADDR', 'unknown')
        
        return f"ip:{ip}"
    
    def _refill_tokens(self, client_id: str) -> None:
        """Refill tokens for a client based on elapsed time"""
        bucket = self.tokens[client_id]
        now = time.time()
        
        # Calculate tokens to add based on elapsed time
        elapsed = now - bucket['last_refill']
        tokens_to_add = elapsed * (self.max_requests / self.window_seconds)
        
        # Update token count (capped at burst size)
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
    
    def is_allowed(self) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed
        
        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        client_id = self._get_client_id()
        now = time.time()
        
        # Refill tokens
        self._refill_tokens(client_id)
        
        bucket = self.tokens[client_id]
        
        # Check if we have tokens available
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            
            # Add request timestamp for window-based tracking
            requests = self.clients[client_id]
            requests.append(now)
            
            # Clean old requests outside window
            cutoff = now - self.window_seconds
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            return True, {
                'allowed': True,
                'tokens_remaining': int(bucket['tokens']),
                'requests_in_window': len(requests),
                'window_resets_at': now + self.window_seconds
            }
        else:
            # Calculate reset time
            reset_time = bucket['last_refill'] + (1 / (self.max_requests / self.window_seconds))
            
            return False, {
                'allowed': False,
                'tokens_remaining': 0,
                'requests_in_window': len(self.clients[client_id]),
                'retry_after': max(1, int(reset_time - now)),
                'window_resets_at': reset_time
            }
    
    def cleanup_old_clients(self, max_age_seconds: int = 3600) -> int:
        """Clean up old client data to prevent memory leaks"""
        now = time.time()
        cutoff = now - max_age_seconds
        
        clients_to_remove = []
        
        for client_id, requests in self.clients.items():
            # Remove old requests
            while requests and requests[0] < cutoff:
                requests.popleft()
            
            # If no recent requests, mark for removal
            if not requests:
                clients_to_remove.append(client_id)
        
        # Remove inactive clients
        for client_id in clients_to_remove:
            del self.clients[client_id]
            if client_id in self.tokens:
                del self.tokens[client_id]
        
        return len(clients_to_remove)


class RateLimitManager:
    """Manages multiple rate limiters for different user roles/endpoints"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        
        # Default rate limits by role
        self._setup_default_limits()
    
    def _setup_default_limits(self):
        """Setup default rate limits"""
        # Admin users: 1000 requests/hour (generous limit)
        self.limiters['admin'] = RateLimiter(
            max_requests=1000,
            window_seconds=3600,
            burst_size=100
        )
        
        # Operator users: 500 requests/hour
        self.limiters['operator'] = RateLimiter(
            max_requests=500,
            window_seconds=3600,
            burst_size=50
        )
        
        # Readonly users: 200 requests/hour
        self.limiters['readonly'] = RateLimiter(
            max_requests=200,
            window_seconds=3600,
            burst_size=20
        )
        
        # Anonymous/unauthenticated: 100 requests/hour (very restrictive)
        self.limiters['anonymous'] = RateLimiter(
            max_requests=100,
            window_seconds=3600,
            burst_size=10
        )
        
        # Strict limits for sensitive endpoints
        self.limiters['auth'] = RateLimiter(
            max_requests=10,
            window_seconds=300,  # 5 minutes
            burst_size=5
        )
        
        # Control operations (restart, settings changes)
        self.limiters['control'] = RateLimiter(
            max_requests=60,
            window_seconds=300,  # 5 minutes
            burst_size=10
        )
    
    def _get_user_role(self) -> str:
        """Get user role for rate limiting"""
        if hasattr(request, 'current_user') and request.current_user:
            user_roles = request.current_user.get('roles', [])
            
            # Use highest privilege role for rate limiting
            if 'admin' in user_roles:
                return 'admin'
            elif 'operator' in user_roles:
                return 'operator'
            elif 'viewer' in user_roles:
                return 'readonly'
            else:
                return 'anonymous'
        
        return 'anonymous'
    
    def check_rate_limit(self, limiter_key: Optional[str] = None) -> Tuple[bool, Dict[str, any]]:
        """
        Check rate limit for current request
        
        Args:
            limiter_key: Optional specific limiter to use, defaults to user role
        
        Returns:
            Tuple of (allowed: bool, info: dict)
        """
        if limiter_key and limiter_key in self.limiters:
            limiter = self.limiters[limiter_key]
        else:
            user_role = self._get_user_role()
            limiter = self.limiters.get(user_role, self.limiters['anonymous'])
        
        return limiter.is_allowed()
    
    def cleanup_old_data(self) -> int:
        """Clean up old data from all limiters"""
        total_cleaned = 0
        for limiter in self.limiters.values():
            total_cleaned += limiter.cleanup_old_clients()
        return total_cleaned


# Global rate limit manager instance
_rate_limit_manager: Optional[RateLimitManager] = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get global rate limit manager instance"""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager


def rate_limit(limiter_key: Optional[str] = None, per_user: bool = True):
    """
    Decorator to apply rate limiting to Flask routes
    
    Args:
        limiter_key: Optional specific limiter to use
        per_user: If True, apply per-user limits; if False, global limits
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            manager = get_rate_limit_manager()
            
            # Check rate limit
            allowed, info = manager.check_rate_limit(limiter_key)
            
            if not allowed:
                logger.warning("Rate limit exceeded",
                             endpoint=request.endpoint,
                             client_id=manager.limiters[limiter_key or 'anonymous']._get_client_id() if limiter_key else 'unknown',
                             retry_after=info.get('retry_after', 60))
                
                response = jsonify({
                    'success': False,
                    'error': {
                        'code': 'RATE_LIMIT_EXCEEDED',
                        'message': 'Rate limit exceeded. Please try again later.',
                        'retry_after': info.get('retry_after', 60)
                    }
                })
                
                # Add rate limit headers
                response.headers['X-RateLimit-Limit'] = str(manager.limiters[limiter_key or 'anonymous'].max_requests)
                response.headers['X-RateLimit-Remaining'] = str(info.get('tokens_remaining', 0))
                response.headers['X-RateLimit-Reset'] = str(int(info.get('window_resets_at', time.time() + 3600)))
                response.headers['Retry-After'] = str(info.get('retry_after', 60))
                
                return response, 429
            
            # Add rate limit info to response headers
            g.rate_limit_info = info
            
            # Execute the original function
            response = f(*args, **kwargs)
            
            # Add rate limit headers to successful responses
            if hasattr(response, 'headers'):
                limiter = manager.limiters.get(limiter_key or manager._get_user_role(), manager.limiters['anonymous'])
                response.headers['X-RateLimit-Limit'] = str(limiter.max_requests)
                response.headers['X-RateLimit-Remaining'] = str(info.get('tokens_remaining', 0))
                response.headers['X-RateLimit-Reset'] = str(int(info.get('window_resets_at', time.time() + 3600)))
            
            logger.debug("Request allowed",
                        endpoint=request.endpoint,
                        tokens_remaining=info.get('tokens_remaining', 0),
                        requests_in_window=info.get('requests_in_window', 0))
            
            return response
        
        return decorated_function
    return decorator


def init_rate_limiting(app):
    """Initialize rate limiting for Flask app"""
    
    @app.before_request
    def setup_rate_limiting():
        """Setup rate limiting context for request"""
        # This runs before each request
        pass
    
    @app.after_request  
    def cleanup_rate_limiting(response):
        """Cleanup rate limiting context after request"""
        # Periodically clean up old rate limit data
        if hasattr(g, 'rate_limit_info'):
            manager = get_rate_limit_manager()
            
            # Clean up every 1000th request (approximately)
            import random
            if random.randint(1, 1000) == 1:
                cleaned = manager.cleanup_old_data()
                if cleaned > 0:
                    logger.info("Rate limiter cleanup completed", 
                               clients_cleaned=cleaned)
        
        return response
    
    logger.info("Rate limiting initialized",
               default_limits={
                   'admin': '1000/hour',
                   'operator': '500/hour', 
                   'readonly': '200/hour',
                   'anonymous': '100/hour'
               })