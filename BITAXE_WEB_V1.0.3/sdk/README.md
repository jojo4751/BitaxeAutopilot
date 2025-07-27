# BitAxe API Client SDK

A comprehensive Python SDK for interacting with the BitAxe Web Management API.

## Overview

The BitAxe SDK provides both synchronous and asynchronous clients for managing BitAxe ASIC Bitcoin mining devices through the web API. It includes comprehensive error handling, rate limiting, authentication, and convenient utilities for common operations.

## Features

- **Synchronous and Asynchronous Clients**: Choose the right client for your use case
- **Complete API Coverage**: Support for all BitAxe API endpoints
- **Authentication**: JWT-based authentication with automatic token management
- **Error Handling**: Comprehensive error handling with specific exception types
- **Rate Limiting**: Built-in rate limiting to respect API limits
- **Batch Operations**: Efficient batch operations for multiple miners
- **CLI Tool**: Command-line interface for quick operations
- **Type Hints**: Full type annotations for better IDE support

## Installation

```bash
# Install required dependencies
pip install requests aiohttp click
```

For development:
```bash
pip install requests aiohttp click pytest pytest-asyncio
```

## Quick Start

### Synchronous Client

```python
from sdk.client import create_client

# Create authenticated client
with create_client('http://localhost:5000', 'admin', 'admin123') as client:
    # Get all miners
    miners = client.get_all_miners()
    print(f"Found {len(miners)} miners")
    
    # Get miners summary
    summary = client.get_miners_summary()
    print(f"Total hashrate: {summary['total_hashrate']} GH/s")
    
    # Update miner settings
    result = client.update_miner_settings(
        ip='192.168.1.100',
        frequency=800,
        core_voltage=1200
    )
```

### Asynchronous Client

```python
import asyncio
from sdk.async_client import AsyncBitAxeClient

async def main():
    async with AsyncBitAxeClient('http://localhost:5000') as client:
        await client.login('admin', 'admin123')
        
        # Get miners and health concurrently
        miners_task = client.get_all_miners()
        health_task = client.get_health()
        
        miners, health = await asyncio.gather(miners_task, health_task)
        print(f"Found {len(miners)} miners, system health: {health['overall_status']}")

asyncio.run(main())
```

### CLI Usage

```bash
# Set environment variables (optional)
export BITAXE_BASE_URL=http://localhost:5000
export BITAXE_USERNAME=admin
export BITAXE_PASSWORD=admin123

# List all miners
python -m sdk.cli miners list

# Get miner details
python -m sdk.cli miners get 192.168.1.100

# Start benchmark
python -m sdk.cli benchmark start 192.168.1.100 --frequency 800 --core-voltage 1200

# Monitor system health
python -m sdk.cli health status

# Auto-benchmark all miners
python -m sdk.cli auto-benchmark --frequency 800 --core-voltage 1200 --wait
```

## API Reference

### Authentication

All clients require authentication with username and password. Tokens are automatically managed.

```python
# Manual authentication
client = BitAxeClient('http://localhost:5000')
token_data = client.login('username', 'password')

# Context manager (auto-login/logout)
with create_client(base_url, username, password) as client:
    # Client is authenticated and will auto-logout
    pass
```

### Miner Management

```python
# Get all miners with pagination
response = client.get_miners(page=1, page_size=50)
miners = response['data']
pagination = response['pagination']

# Get all miners (auto-pagination)
all_miners = client.get_all_miners()

# Get specific miner
miner = client.get_miner('192.168.1.100')

# Update miner settings
result = client.update_miner_settings(
    ip='192.168.1.100',
    frequency=800,
    core_voltage=1200,
    autofanspeed=True,
    fanspeed=None  # Only needed if autofanspeed=False
)

# Restart miner
result = client.restart_miner('192.168.1.100')

# Get miners summary
summary = client.get_miners_summary()
```

### Benchmark Operations

```python
# Start single benchmark
result = client.start_benchmark(
    ip='192.168.1.100',
    frequency=800,
    core_voltage=1200,
    duration=600
)

# Start multi-benchmark
result = client.start_multi_benchmark(
    ips=['192.168.1.100', '192.168.1.101'],
    frequency=800,
    core_voltage=1200,
    duration=600
)

# Get benchmark status
status = client.get_benchmark_status()
active_benchmarks = status['active_benchmarks']

# Get benchmark results
results = client.get_benchmark_results(ip='192.168.1.100', limit=10)

# Wait for benchmark completion
completed = client.wait_for_benchmark_completion(
    ip='192.168.1.100',
    timeout=1800,
    poll_interval=30
)
```

### Event Monitoring

```python
# Get recent events
events = client.get_events(limit=100)

# Get filtered events
events = client.get_events(
    ip='192.168.1.100',
    event_type='BENCHMARK_COMPLETED',
    severity='INFO',
    limit=50
)
```

### Health Monitoring

```python
# Get system health
health = client.get_health()
overall_status = health['overall_status']

# Get component health
db_health = client.get_component_health('database')
```

### Configuration

```python
# Update configuration
result = client.update_config('settings.temp_limit', '80')
```

## Async Client Features

The async client supports all the same operations as the sync client, plus additional batch operations:

```python
# Batch get multiple miners
miner_details = await client.get_miners_batch(['192.168.1.100', '192.168.1.101'])

# Batch update miner settings
settings_updates = [
    {'ip': '192.168.1.100', 'frequency': 800, 'core_voltage': 1200},
    {'ip': '192.168.1.101', 'frequency': 850, 'core_voltage': 1250}
]
results = await client.update_miners_settings_batch(settings_updates)

# Batch restart miners
results = await client.restart_miners_batch(['192.168.1.100', '192.168.1.101'])
```

## Error Handling

The SDK provides specific exception types for different error conditions:

```python
from sdk.client import (
    BitAxeAPIError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ValidationError
)

try:
    client.get_miner('invalid.ip')
except AuthenticationError:
    print("Authentication failed")
except AuthorizationError:
    print("Insufficient permissions")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after} seconds")
except ValidationError:
    print("Invalid request data")
except BitAxeAPIError as e:
    print(f"API error: {e.message} (code: {e.error_code})")
```

## Rate Limiting

The SDK includes built-in rate limiting to respect API limits:

- Minimum 100ms between requests
- Automatic retry-after handling for 429 responses
- Rate limit headers in responses

## Examples

See `examples.py` for comprehensive examples including:

- Basic usage patterns
- Miner management workflows
- Benchmark automation
- Batch operations
- Error handling
- Monitoring dashboard
- Async usage patterns

## CLI Commands

The CLI provides convenient access to all API functionality:

### Authentication
```bash
python -m sdk.cli auth login
```

### Miners
```bash
python -m sdk.cli miners list [--page 1] [--page-size 50]
python -m sdk.cli miners get <ip>
python -m sdk.cli miners summary
python -m sdk.cli miners update <ip> --frequency 800 --core-voltage 1200
python -m sdk.cli miners restart <ip> [--confirm]
```

### Benchmarks
```bash
python -m sdk.cli benchmark start <ip> --frequency 800 --core-voltage 1200 [--duration 600]
python -m sdk.cli benchmark multi --ips "ip1,ip2,ip3" --frequency 800 --core-voltage 1200
python -m sdk.cli benchmark status
python -m sdk.cli benchmark results [--ip <ip>] [--limit 50]
python -m sdk.cli benchmark wait <ip> [--timeout 1800]
```

### Events
```bash
python -m sdk.cli events list [--ip <ip>] [--event-type <type>] [--severity <level>]
```

### Health
```bash
python -m sdk.cli health status
python -m sdk.cli health component <component>
```

### Configuration
```bash
python -m sdk.cli config set <key> <value>
```

### Auto-benchmark
```bash
python -m sdk.cli auto-benchmark [--ips "ip1,ip2"] [--frequency 800] [--core-voltage 1200] [--wait]
```

## Environment Variables

The CLI supports environment variables for common settings:

- `BITAXE_BASE_URL`: Base URL of the API
- `BITAXE_USERNAME`: Username for authentication
- `BITAXE_PASSWORD`: Password for authentication

## Development

### Running Tests

```bash
# Run async tests
pytest tests/ -v

# Run specific test file
pytest tests/test_client.py -v
```

### Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## License

This SDK is part of the BitAxe Web Management System and follows the same license terms.