"""
Async BitAxe API Client SDK

An async Python client for interacting with the BitAxe Web Management API.
"""

import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import logging

from .client import BitAxeAPIError, AuthenticationError, AuthorizationError, RateLimitError, ValidationError

logger = logging.getLogger(__name__)


class AsyncBitAxeClient:
    """
    Async BitAxe API Client
    
    Provides an async interface for interacting with the BitAxe Web Management API.
    
    Example:
        async with AsyncBitAxeClient('http://localhost:5000') as client:
            await client.login('admin', 'admin123')
            
            miners = await client.get_all_miners()
            for miner in miners:
                print(f"Miner {miner['ip']}: {miner['hash_rate']} GH/s")
    """
    
    def __init__(self, base_url: str, timeout: int = 30, verify_ssl: bool = True):
        """
        Initialize async BitAxe API client
        
        Args:
            base_url: Base URL of the BitAxe API
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.verify_ssl = verify_ssl
        
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Authentication
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.access_token:
            try:
                await self.logout()
            except:
                pass
        
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make async HTTP request to API
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional arguments
        
        Returns:
            Response JSON data
        """
        if not self.session:
            raise BitAxeAPIError("Client session not initialized. Use async context manager.")
        
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        url = urljoin(self.api_base, endpoint.lstrip('/'))
        
        # Add authentication header if available
        headers = kwargs.pop('headers', {})
        if self.access_token and not endpoint.startswith('/auth/login'):
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        # Set content type for JSON requests
        if 'json' in kwargs:
            headers['Content-Type'] = 'application/json'
        
        kwargs['headers'] = headers
        
        logger.debug(f"Making async {method} request to {url}")
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                self.last_request_time = time.time()
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    raise RateLimitError("Rate limit exceeded", retry_after)
                
                # Parse JSON response
                if response.content_type.startswith('application/json'):
                    data = await response.json()
                else:
                    text = await response.text()
                    data = {"message": text}
                
                # Handle API errors
                if response.status >= 400:
                    error_code = data.get('error', {}).get('code', 'UNKNOWN_ERROR')
                    error_message = data.get('error', {}).get('message', data.get('message', 'Unknown error'))
                    
                    if response.status == 401:
                        raise AuthenticationError(error_message, response.status, error_code)
                    elif response.status == 403:
                        raise AuthorizationError(error_message, response.status, error_code)
                    elif response.status == 400:
                        raise ValidationError(error_message, response.status, error_code)
                    else:
                        raise BitAxeAPIError(error_message, response.status, error_code)
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"Async request failed: {e}")
            raise BitAxeAPIError(f"Request failed: {e}")
    
    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async GET request"""
        return await self._make_request('GET', endpoint, params=params)
    
    async def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async POST request"""
        return await self._make_request('POST', endpoint, json=data)
    
    async def _put(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async PUT request"""
        return await self._make_request('PUT', endpoint, json=data)
    
    async def _delete(self, endpoint: str) -> Dict[str, Any]:
        """Make async DELETE request"""
        return await self._make_request('DELETE', endpoint)
    
    # Authentication methods
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate with the API
        
        Args:
            username: Username
            password: Password
        
        Returns:
            Token information
        """
        data = {'username': username, 'password': password}
        response = await self._post('/auth/login', data)
        
        token_data = response['data']
        self.access_token = token_data['access_token']
        
        if 'expires_at' in token_data:
            self.token_expires_at = datetime.fromisoformat(token_data['expires_at'].replace('Z', '+00:00'))
        
        logger.info(f"Successfully authenticated as {username}")
        return token_data
    
    async def logout(self) -> None:
        """Logout and revoke token"""
        if self.access_token:
            try:
                await self._post('/auth/logout')
            except BitAxeAPIError:
                pass
            finally:
                self.access_token = None
                self.token_expires_at = None
                logger.info("Logged out successfully")
    
    async def get_user_info(self) -> Dict[str, Any]:
        """Get current user information"""
        response = await self._get('/auth/user')
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
    async def get_miners(self, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """Get list of miners"""
        params = {'page': page, 'page_size': page_size}
        return await self._get('/miners', params)
    
    async def get_all_miners(self) -> List[Dict[str, Any]]:
        """Get all miners (handles pagination automatically)"""
        all_miners = []
        page = 1
        
        while True:
            response = await self.get_miners(page=page, page_size=100)
            miners = response['data']
            all_miners.extend(miners)
            
            if not response['pagination']['has_next']:
                break
            
            page += 1
        
        return all_miners
    
    async def get_miner(self, ip: str) -> Dict[str, Any]:
        """Get specific miner details"""
        response = await self._get(f'/miners/{ip}')
        return response['data']
    
    async def update_miner_settings(self, ip: str, frequency: int, core_voltage: int, 
                                  autofanspeed: bool = True, fanspeed: Optional[int] = None) -> Dict[str, Any]:
        """Update miner settings"""
        data = {
            'frequency': frequency,
            'core_voltage': core_voltage,
            'autofanspeed': autofanspeed
        }
        
        if not autofanspeed and fanspeed is not None:
            data['fanspeed'] = fanspeed
        
        response = await self._put(f'/miners/{ip}/settings', data)
        return response['data']
    
    async def restart_miner(self, ip: str) -> Dict[str, Any]:
        """Restart miner"""
        response = await self._post(f'/miners/{ip}/restart')
        return response['data']
    
    async def get_miners_summary(self) -> Dict[str, Any]:
        """Get miners summary statistics"""
        response = await self._get('/miners/summary')
        return response['data']
    
    # Benchmark methods
    async def start_benchmark(self, ip: str, frequency: int, core_voltage: int, duration: int = 600) -> Dict[str, Any]:
        """Start benchmark for single miner"""
        data = {
            'ip': ip,
            'frequency': frequency,
            'core_voltage': core_voltage,
            'duration': duration
        }
        
        response = await self._post('/benchmarks', data)
        return response['data']
    
    async def start_multi_benchmark(self, ips: List[str], frequency: int, core_voltage: int, 
                                  duration: int = 600) -> Dict[str, Any]:
        """Start benchmark for multiple miners"""
        data = {
            'ips': ips,
            'frequency': frequency,
            'core_voltage': core_voltage,
            'duration': duration
        }
        
        response = await self._post('/benchmarks/multi', data)
        return response['data']
    
    async def get_benchmark_status(self) -> Dict[str, Any]:
        """Get current benchmark status"""
        response = await self._get('/benchmarks/status')
        return response['data']
    
    async def get_benchmark_results(self, ip: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get benchmark results"""
        params = {'limit': limit}
        if ip:
            params['ip'] = ip
        
        response = await self._get('/benchmarks/results', params)
        return response['data']
    
    # Event methods
    async def get_events(self, ip: Optional[str] = None, event_type: Optional[str] = None,
                        severity: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get system events"""
        params = {'limit': limit}
        if ip:
            params['ip'] = ip
        if event_type:
            params['event_type'] = event_type
        if severity:
            params['severity'] = severity
        
        response = await self._get('/events', params)
        return response['data']
    
    # Health monitoring methods
    async def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        response = await self._get('/health')
        return response['data']
    
    async def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get specific component health"""
        response = await self._get(f'/health/{component}')
        return response['data']
    
    # Configuration methods
    async def update_config(self, key: str, value: str) -> Dict[str, Any]:
        """Update configuration setting"""
        data = {'key': key, 'value': value}
        response = await self._put('/config', data)
        return response['data']
    
    # Utility methods
    async def wait_for_benchmark_completion(self, ip: str, timeout: int = 1800, poll_interval: int = 30) -> bool:
        """Wait for benchmark to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_benchmark_status()
            
            if ip not in status['active_benchmarks']:
                logger.info(f"Benchmark completed for {ip}")
                return True
            
            logger.debug(f"Benchmark still running for {ip}, waiting...")
            await asyncio.sleep(poll_interval)
        
        logger.warning(f"Benchmark timeout for {ip}")
        return False
    
    # Batch operations
    async def get_miners_batch(self, ips: List[str]) -> List[Dict[str, Any]]:
        """Get multiple miners concurrently"""
        tasks = [self.get_miner(ip) for ip in ips]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        miners = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to get miner {ips[i]}: {result}")
            else:
                miners.append(result)
        
        return miners
    
    async def update_miners_settings_batch(self, settings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update multiple miners' settings concurrently
        
        Args:
            settings: List of dicts with 'ip', 'frequency', 'core_voltage', etc.
        
        Returns:
            List of update results
        """
        tasks = []
        for setting in settings:
            ip = setting.pop('ip')
            task = self.update_miner_settings(ip, **setting)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        updates = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to update miner settings: {result}")
            else:
                updates.append(result)
        
        return updates
    
    async def restart_miners_batch(self, ips: List[str]) -> List[Dict[str, Any]]:
        """Restart multiple miners concurrently"""
        tasks = [self.restart_miner(ip) for ip in ips]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        restarts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to restart miner {ips[i]}: {result}")
            else:
                restarts.append(result)
        
        return restarts


# Convenience functions
async def create_async_client(base_url: str, username: str, password: str, **kwargs) -> AsyncBitAxeClient:
    """
    Create and authenticate async BitAxe client
    
    Note: Must be used within an async context manager
    
    Args:
        base_url: Base URL of the API
        username: Username for authentication
        password: Password for authentication
        **kwargs: Additional client options
    
    Returns:
        Authenticated async BitAxe client
    """
    client = AsyncBitAxeClient(base_url, **kwargs)
    await client.login(username, password)
    return client


# Example usage
async def example_usage():
    """Example async usage"""
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    async with AsyncBitAxeClient(base_url) as client:
        # Authenticate
        await client.login(username, password)
        
        # Get user info
        user_info = await client.get_user_info()
        print(f"Authenticated as: {user_info['username']}")
        
        # Get all miners concurrently
        miners = await client.get_all_miners()
        print(f"Found {len(miners)} miners")
        
        # Get specific miners concurrently
        if len(miners) >= 2:
            ips = [miner['ip'] for miner in miners[:2]]
            miner_details = await client.get_miners_batch(ips)
            print(f"Got details for {len(miner_details)} miners")
        
        # Get system health
        health = await client.get_health()
        print(f"System health: {health['overall_status']}")
        
        # Get events concurrently with health check
        events_task = client.get_events(limit=5)
        summary_task = client.get_miners_summary()
        
        events, summary = await asyncio.gather(events_task, summary_task)
        
        print(f"Found {len(events)} recent events")
        print(f"Total hashrate: {summary['total_hashrate']} GH/s")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())