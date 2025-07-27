"""
BitAxe API Client SDK

A Python client for interacting with the BitAxe Web Management API.
"""

import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import logging

logger = logging.getLogger(__name__)


class BitAxeAPIError(Exception):
    """Base exception for BitAxe API errors"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, error_code: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class AuthenticationError(BitAxeAPIError):
    """Authentication failed"""
    pass


class AuthorizationError(BitAxeAPIError):
    """Insufficient permissions"""
    pass


class RateLimitError(BitAxeAPIError):
    """Rate limit exceeded"""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, 429, 'RATE_LIMIT_EXCEEDED')
        self.retry_after = retry_after


class ValidationError(BitAxeAPIError):
    """Request validation failed"""
    pass


class BitAxeClient:
    """
    BitAxe API Client
    
    Provides a convenient interface for interacting with the BitAxe Web Management API.
    
    Example:
        client = BitAxeClient('http://localhost:5000')
        client.login('admin', 'admin123')
        
        miners = client.get_miners()
        for miner in miners:
            print(f"Miner {miner['ip']}: {miner['hash_rate']} GH/s")
    """
    
    def __init__(self, base_url: str, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize BitAxe API client
        
        Args:
            base_url: Base URL of the BitAxe API (e.g., 'http://localhost:5000')
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.session.timeout = timeout
        
        # Authentication
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without /api/v1 prefix)
            **kwargs: Additional arguments for requests
        
        Returns:
            Response JSON data
        
        Raises:
            BitAxeAPIError: On API errors
        """
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = urljoin(self.api_base, endpoint.lstrip('/'))
        
        # Add authentication header if available
        headers = kwargs.pop('headers', {})
        if self.access_token and not endpoint.startswith('/auth/login'):
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        # Set content type for JSON requests
        if 'json' in kwargs:
            headers['Content-Type'] = 'application/json'
        
        kwargs['headers'] = headers
        
        logger.debug(f"Making {method} request to {url}")
        
        try:
            response = self.session.request(method, url, **kwargs)
            self.last_request_time = time.time()
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise RateLimitError("Rate limit exceeded", retry_after)
            
            # Parse JSON response
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
            else:
                data = {"message": response.text}
            
            # Handle API errors
            if not response.ok:
                error_code = data.get('error', {}).get('code', 'UNKNOWN_ERROR')
                error_message = data.get('error', {}).get('message', data.get('message', 'Unknown error'))
                
                if response.status_code == 401:
                    raise AuthenticationError(error_message, response.status_code, error_code)
                elif response.status_code == 403:
                    raise AuthorizationError(error_message, response.status_code, error_code)
                elif response.status_code == 400:
                    raise ValidationError(error_message, response.status_code, error_code)
                else:
                    raise BitAxeAPIError(error_message, response.status_code, error_code)
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise BitAxeAPIError(f"Request failed: {e}")
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make POST request"""
        return self._make_request('POST', endpoint, json=data)
    
    def _put(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make PUT request"""
        return self._make_request('PUT', endpoint, json=data)
    
    def _delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return self._make_request('DELETE', endpoint)
    
    # Authentication methods
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate with the API
        
        Args:
            username: Username
            password: Password
        
        Returns:
            Token information
        """
        data = {'username': username, 'password': password}
        response = self._post('/auth/login', data)
        
        token_data = response['data']
        self.access_token = token_data['access_token']
        
        # Parse expiry time
        if 'expires_at' in token_data:
            self.token_expires_at = datetime.fromisoformat(token_data['expires_at'].replace('Z', '+00:00'))
        
        logger.info(f"Successfully authenticated as {username}")
        return token_data
    
    def logout(self) -> None:
        """Logout and revoke token"""
        if self.access_token:
            try:
                self._post('/auth/logout')
            except BitAxeAPIError:
                pass  # Ignore errors during logout
            finally:
                self.access_token = None
                self.token_expires_at = None
                logger.info("Logged out successfully")
    
    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information"""
        response = self._get('/auth/user')
        return response['data']
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        if not self.access_token:
            return False
        
        if self.token_expires_at and datetime.now() >= self.token_expires_at:
            self.access_token = None
            self.token_expires_at = None
            return False
        
        return True
    
    # Miner management methods
    def get_miners(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """
        Get list of miners
        
        Args:
            page: Page number (1-based)
            page_size: Items per page
        
        Returns:
            Paginated list of miners
        """
        params = {'page': page, 'page_size': page_size}
        return self._get('/miners', params)
    
    def get_all_miners(self) -> List[Dict[str, Any]]:
        """Get all miners (handles pagination automatically)"""
        all_miners = []
        page = 1
        
        while True:
            response = self.get_miners(page=page, page_size=100)
            miners = response['data']
            all_miners.extend(miners)
            
            if not response['pagination']['has_next']:
                break
            
            page += 1
        
        return all_miners
    
    def get_miner(self, ip: str) -> Dict[str, Any]:
        """
        Get specific miner details
        
        Args:
            ip: Miner IP address
        
        Returns:
            Miner details
        """
        response = self._get(f'/miners/{ip}')
        return response['data']
    
    def update_miner_settings(self, ip: str, frequency: int, core_voltage: int, 
                            autofanspeed: bool = True, fanspeed: Optional[int] = None) -> Dict[str, Any]:
        """
        Update miner settings
        
        Args:
            ip: Miner IP address
            frequency: Frequency in MHz
            core_voltage: Core voltage in mV
            autofanspeed: Enable automatic fan speed control
            fanspeed: Manual fan speed percentage (required if autofanspeed=False)
        
        Returns:
            Update confirmation
        """
        data = {
            'frequency': frequency,
            'core_voltage': core_voltage,
            'autofanspeed': autofanspeed
        }
        
        if not autofanspeed and fanspeed is not None:
            data['fanspeed'] = fanspeed
        
        response = self._put(f'/miners/{ip}/settings', data)
        return response['data']
    
    def restart_miner(self, ip: str) -> Dict[str, Any]:
        """
        Restart miner
        
        Args:
            ip: Miner IP address
        
        Returns:
            Restart confirmation
        """
        response = self._post(f'/miners/{ip}/restart')
        return response['data']
    
    def get_miners_summary(self) -> Dict[str, Any]:
        """Get miners summary statistics"""
        response = self._get('/miners/summary')
        return response['data']
    
    # Benchmark methods
    def start_benchmark(self, ip: str, frequency: int, core_voltage: int, duration: int = 600) -> Dict[str, Any]:
        """
        Start benchmark for single miner
        
        Args:
            ip: Miner IP address
            frequency: Frequency in MHz
            core_voltage: Core voltage in mV
            duration: Benchmark duration in seconds
        
        Returns:
            Benchmark start confirmation
        """
        data = {
            'ip': ip,
            'frequency': frequency,
            'core_voltage': core_voltage,
            'duration': duration
        }
        
        response = self._post('/benchmarks', data)
        return response['data']
    
    def start_multi_benchmark(self, ips: List[str], frequency: int, core_voltage: int, 
                            duration: int = 600) -> Dict[str, Any]:
        """
        Start benchmark for multiple miners
        
        Args:
            ips: List of miner IP addresses
            frequency: Frequency in MHz
            core_voltage: Core voltage in mV
            duration: Benchmark duration in seconds
        
        Returns:
            Multi-benchmark start confirmation
        """
        data = {
            'ips': ips,
            'frequency': frequency,
            'core_voltage': core_voltage,
            'duration': duration
        }
        
        response = self._post('/benchmarks/multi', data)
        return response['data']
    
    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get current benchmark status"""
        response = self._get('/benchmarks/status')
        return response['data']
    
    def get_benchmark_results(self, ip: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get benchmark results
        
        Args:
            ip: Optional IP filter
            limit: Maximum results to return
        
        Returns:
            List of benchmark results
        """
        params = {'limit': limit}
        if ip:
            params['ip'] = ip
        
        response = self._get('/benchmarks/results', params)
        return response['data']
    
    # Event methods
    def get_events(self, ip: Optional[str] = None, event_type: Optional[str] = None,
                  severity: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get system events
        
        Args:
            ip: Optional IP filter
            event_type: Optional event type filter
            severity: Optional severity filter
            limit: Maximum events to return
        
        Returns:
            List of events
        """
        params = {'limit': limit}
        if ip:
            params['ip'] = ip
        if event_type:
            params['event_type'] = event_type
        if severity:
            params['severity'] = severity
        
        response = self._get('/events', params)
        return response['data']
    
    # Health monitoring methods
    def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        response = self._get('/health')
        return response['data']
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """
        Get specific component health
        
        Args:
            component: Component name
        
        Returns:
            Component health details
        """
        response = self._get(f'/health/{component}')
        return response['data']
    
    # Configuration methods
    def update_config(self, key: str, value: str) -> Dict[str, Any]:
        """
        Update configuration setting
        
        Args:
            key: Configuration key
            value: Configuration value
        
        Returns:
            Update confirmation
        """
        data = {'key': key, 'value': value}
        response = self._put('/config', data)
        return response['data']
    
    # Utility methods
    def wait_for_benchmark_completion(self, ip: str, timeout: int = 1800, poll_interval: int = 30) -> bool:
        """
        Wait for benchmark to complete
        
        Args:
            ip: Miner IP address
            timeout: Maximum wait time in seconds
            poll_interval: Check interval in seconds
        
        Returns:
            True if completed, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_benchmark_status()
            
            if ip not in status['active_benchmarks']:
                logger.info(f"Benchmark completed for {ip}")
                return True
            
            logger.debug(f"Benchmark still running for {ip}, waiting...")
            time.sleep(poll_interval)
        
        logger.warning(f"Benchmark timeout for {ip}")
        return False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - logout on exit"""
        self.logout()


# Convenience functions
def create_client(base_url: str, username: str, password: str, **kwargs) -> BitAxeClient:
    """
    Create and authenticate BitAxe client
    
    Args:
        base_url: Base URL of the API
        username: Username for authentication
        password: Password for authentication
        **kwargs: Additional client options
    
    Returns:
        Authenticated BitAxe client
    """
    client = BitAxeClient(base_url, **kwargs)
    client.login(username, password)
    return client


# Example usage
if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python client.py <base_url> <username> <password>")
        sys.exit(1)
    
    base_url, username, password = sys.argv[1:4]
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create client and authenticate
    with create_client(base_url, username, password) as client:
        # Get user info
        user_info = client.get_user_info()
        print(f"Authenticated as: {user_info['username']}")
        print(f"Roles: {user_info['roles']}")
        
        # Get miners
        miners = client.get_all_miners()
        print(f"\nFound {len(miners)} miners:")
        
        for miner in miners:
            print(f"  {miner['ip']}: {miner.get('hash_rate', 'N/A')} GH/s, "
                  f"{miner.get('temperature', 'N/A')}°C")
        
        # Get system health
        health = client.get_health()
        print(f"\nSystem health: {health['overall_status']}")
        print(f"Total checks: {health['total_checks']}")
        
        # Get recent events
        events = client.get_events(limit=5)
        print(f"\nRecent events ({len(events)}):")
        for event in events:
            print(f"  {event['timestamp']}: {event['message']}")