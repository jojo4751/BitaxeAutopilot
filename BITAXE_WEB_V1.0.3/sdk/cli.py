"""
BitAxe CLI Tool

Command-line interface for the BitAxe Web Management API.
"""

import click
import json
import sys
import asyncio
from datetime import datetime
from typing import Optional, List
import logging

from .client import BitAxeClient, BitAxeAPIError, create_client
from .async_client import AsyncBitAxeClient


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Config:
    """CLI configuration"""
    def __init__(self):
        self.base_url: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.verbose: bool = False
        self.format: str = 'table'


pass_config = click.make_pass_decorator(Config, ensure=True)


def handle_api_error(func):
    """Decorator to handle API errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BitAxeAPIError as e:
            if hasattr(e, 'status_code') and e.status_code:
                click.echo(f"API Error ({e.status_code}): {e.message}", err=True)
            else:
                click.echo(f"API Error: {e.message}", err=True)
            
            if hasattr(e, 'error_code') and e.error_code:
                click.echo(f"Error Code: {e.error_code}", err=True)
            
            sys.exit(1)
        except Exception as e:
            click.echo(f"Unexpected error: {e}", err=True)
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    return wrapper


def format_output(data, format_type: str = 'table'):
    """Format output based on format type"""
    if format_type == 'json':
        return json.dumps(data, indent=2, default=str)
    elif format_type == 'table':
        if isinstance(data, list) and data:
            return format_table(data)
        elif isinstance(data, dict):
            return format_dict(data)
        else:
            return str(data)
    else:
        return str(data)


def format_table(data: List[dict]) -> str:
    """Format list of dicts as table"""
    if not data:
        return "No data"
    
    # Get all unique keys
    all_keys = set()
    for item in data:
        all_keys.update(item.keys())
    
    headers = list(all_keys)
    
    # Calculate column widths
    widths = {}
    for header in headers:
        widths[header] = len(header)
        for item in data:
            value = str(item.get(header, ''))
            widths[header] = max(widths[header], len(value))
    
    # Format header
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-|-".join("-" * widths[header] for header in headers)
    
    # Format rows
    rows = []
    for item in data:
        row = " | ".join(str(item.get(header, '')).ljust(widths[header]) for header in headers)
        rows.append(row)
    
    return "\n".join([header_line, separator] + rows)


def format_dict(data: dict) -> str:
    """Format dict as key-value pairs"""
    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0
    
    lines = []
    for key, value in data.items():
        lines.append(f"{str(key).ljust(max_key_len)}: {value}")
    
    return "\n".join(lines)


@click.group()
@click.option('--base-url', envvar='BITAXE_BASE_URL', help='Base URL of BitAxe API')
@click.option('--username', envvar='BITAXE_USERNAME', help='Username for authentication')
@click.option('--password', envvar='BITAXE_PASSWORD', help='Password for authentication')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@pass_config
def cli(config, base_url, username, password, verbose, output_format):
    """BitAxe CLI - Command line interface for BitAxe Web Management API"""
    config.base_url = base_url
    config.username = username
    config.password = password
    config.verbose = verbose
    config.format = output_format
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def require_auth(func):
    """Decorator to ensure authentication parameters are provided"""
    def wrapper(config, *args, **kwargs):
        if not config.base_url:
            click.echo("Error: --base-url is required", err=True)
            sys.exit(1)
        if not config.username:
            click.echo("Error: --username is required", err=True)
            sys.exit(1)
        if not config.password:
            click.echo("Error: --password is required", err=True)
            sys.exit(1)
        return func(config, *args, **kwargs)
    return wrapper


@cli.group()
def auth():
    """Authentication commands"""
    pass


@auth.command()
@pass_config
@require_auth
@handle_api_error
def login(config):
    """Test authentication"""
    with create_client(config.base_url, config.username, config.password) as client:
        user_info = client.get_user_info()
        click.echo(format_output(user_info, config.format))


@cli.group()
def miners():
    """Miner management commands"""
    pass


@miners.command('list')
@click.option('--page', default=1, help='Page number')
@click.option('--page-size', default=50, help='Items per page')
@pass_config
@require_auth
@handle_api_error
def list_miners(config, page, page_size):
    """List all miners"""
    with create_client(config.base_url, config.username, config.password) as client:
        response = client.get_miners(page=page, page_size=page_size)
        miners = response['data']
        
        if config.format == 'json':
            click.echo(format_output(response, config.format))
        else:
            click.echo(format_output(miners, config.format))
            
            # Show pagination info
            pagination = response['pagination']
            click.echo(f"\nPage {pagination['page']} of {pagination['total_pages']} "
                      f"({pagination['total_count']} total miners)")


@miners.command('get')
@click.argument('ip')
@pass_config
@require_auth
@handle_api_error
def get_miner(config, ip):
    """Get specific miner details"""
    with create_client(config.base_url, config.username, config.password) as client:
        miner = client.get_miner(ip)
        click.echo(format_output(miner, config.format))


@miners.command('summary')
@pass_config
@require_auth
@handle_api_error
def miners_summary(config):
    """Get miners summary statistics"""
    with create_client(config.base_url, config.username, config.password) as client:
        summary = client.get_miners_summary()
        click.echo(format_output(summary, config.format))


@miners.command('update')
@click.argument('ip')
@click.option('--frequency', type=int, required=True, help='Frequency in MHz')
@click.option('--core-voltage', type=int, required=True, help='Core voltage in mV')
@click.option('--autofanspeed/--no-autofanspeed', default=True, help='Enable automatic fan speed')
@click.option('--fanspeed', type=int, help='Manual fan speed percentage (if autofanspeed disabled)')
@pass_config
@require_auth
@handle_api_error
def update_miner(config, ip, frequency, core_voltage, autofanspeed, fanspeed):
    """Update miner settings"""
    with create_client(config.base_url, config.username, config.password) as client:
        result = client.update_miner_settings(
            ip=ip,
            frequency=frequency,
            core_voltage=core_voltage,
            autofanspeed=autofanspeed,
            fanspeed=fanspeed
        )
        click.echo(format_output(result, config.format))


@miners.command('restart')
@click.argument('ip')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@pass_config
@require_auth
@handle_api_error
def restart_miner(config, ip, confirm):
    """Restart miner"""
    if not confirm:
        if not click.confirm(f'Are you sure you want to restart miner {ip}?'):
            click.echo('Aborted.')
            return
    
    with create_client(config.base_url, config.username, config.password) as client:
        result = client.restart_miner(ip)
        click.echo(format_output(result, config.format))


@cli.group()
def benchmark():
    """Benchmark commands"""
    pass


@benchmark.command('start')
@click.argument('ip')
@click.option('--frequency', type=int, required=True, help='Frequency in MHz')
@click.option('--core-voltage', type=int, required=True, help='Core voltage in mV')
@click.option('--duration', type=int, default=600, help='Benchmark duration in seconds')
@pass_config
@require_auth
@handle_api_error
def start_benchmark(config, ip, frequency, core_voltage, duration):
    """Start benchmark for single miner"""
    with create_client(config.base_url, config.username, config.password) as client:
        result = client.start_benchmark(
            ip=ip,
            frequency=frequency,
            core_voltage=core_voltage,
            duration=duration
        )
        click.echo(format_output(result, config.format))


@benchmark.command('multi')
@click.option('--ips', required=True, help='Comma-separated list of IP addresses')
@click.option('--frequency', type=int, required=True, help='Frequency in MHz')
@click.option('--core-voltage', type=int, required=True, help='Core voltage in mV')
@click.option('--duration', type=int, default=600, help='Benchmark duration in seconds')
@pass_config
@require_auth
@handle_api_error
def start_multi_benchmark(config, ips, frequency, core_voltage, duration):
    """Start benchmark for multiple miners"""
    ip_list = [ip.strip() for ip in ips.split(',')]
    
    with create_client(config.base_url, config.username, config.password) as client:
        result = client.start_multi_benchmark(
            ips=ip_list,
            frequency=frequency,
            core_voltage=core_voltage,
            duration=duration
        )
        click.echo(format_output(result, config.format))


@benchmark.command('status')
@pass_config
@require_auth
@handle_api_error
def benchmark_status(config):
    """Get benchmark status"""
    with create_client(config.base_url, config.username, config.password) as client:
        status = client.get_benchmark_status()
        click.echo(format_output(status, config.format))


@benchmark.command('results')
@click.option('--ip', help='Filter by IP address')
@click.option('--limit', type=int, default=50, help='Maximum results to return')
@pass_config
@require_auth
@handle_api_error
def benchmark_results(config, ip, limit):
    """Get benchmark results"""
    with create_client(config.base_url, config.username, config.password) as client:
        results = client.get_benchmark_results(ip=ip, limit=limit)
        click.echo(format_output(results, config.format))


@benchmark.command('wait')
@click.argument('ip')
@click.option('--timeout', type=int, default=1800, help='Timeout in seconds')
@click.option('--poll-interval', type=int, default=30, help='Poll interval in seconds')
@pass_config
@require_auth
@handle_api_error
def wait_benchmark(config, ip, timeout, poll_interval):
    """Wait for benchmark completion"""
    with create_client(config.base_url, config.username, config.password) as client:
        click.echo(f"Waiting for benchmark completion on {ip}...")
        
        completed = client.wait_for_benchmark_completion(
            ip=ip,
            timeout=timeout,
            poll_interval=poll_interval
        )
        
        if completed:
            click.echo(f"Benchmark completed for {ip}")
        else:
            click.echo(f"Benchmark timeout for {ip}")
            sys.exit(1)


@cli.group()
def events():
    """Event management commands"""
    pass


@events.command('list')
@click.option('--ip', help='Filter by IP address')
@click.option('--event-type', help='Filter by event type')
@click.option('--severity', type=click.Choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL']), 
              help='Filter by severity')
@click.option('--limit', type=int, default=100, help='Maximum events to return')
@pass_config
@require_auth
@handle_api_error
def list_events(config, ip, event_type, severity, limit):
    """List system events"""
    with create_client(config.base_url, config.username, config.password) as client:
        events_list = client.get_events(
            ip=ip,
            event_type=event_type,
            severity=severity,
            limit=limit
        )
        click.echo(format_output(events_list, config.format))


@cli.group()
def health():
    """Health monitoring commands"""
    pass


@health.command('status')
@pass_config
@require_auth
@handle_api_error
def health_status(config):
    """Get system health status"""
    with create_client(config.base_url, config.username, config.password) as client:
        health_data = client.get_health()
        click.echo(format_output(health_data, config.format))


@health.command('component')
@click.argument('component')
@pass_config
@require_auth
@handle_api_error
def component_health(config, component):
    """Get specific component health"""
    with create_client(config.base_url, config.username, config.password) as client:
        health_data = client.get_component_health(component)
        click.echo(format_output(health_data, config.format))


@cli.group()
def config():
    """Configuration commands"""
    pass


@config.command('set')
@click.argument('key')
@click.argument('value')
@pass_config
@require_auth
@handle_api_error
def set_config(config, key, value):
    """Update configuration setting"""
    with create_client(config.base_url, config.username, config.password) as client:
        result = client.update_config(key=key, value=value)
        click.echo(format_output(result, config.format))


@cli.command()
@click.option('--ips', help='Comma-separated list of IP addresses (optional)')
@click.option('--frequency', type=int, default=800, help='Benchmark frequency in MHz')
@click.option('--core-voltage', type=int, default=1200, help='Benchmark core voltage in mV')
@click.option('--duration', type=int, default=600, help='Benchmark duration in seconds')
@click.option('--wait/--no-wait', default=True, help='Wait for completion')
@pass_config
@require_auth
@handle_api_error
def auto_benchmark(config, ips, frequency, core_voltage, duration, wait):
    """
    Automatically benchmark all miners or specified miners
    """
    with create_client(config.base_url, config.username, config.password) as client:
        # Get target miners
        if ips:
            ip_list = [ip.strip() for ip in ips.split(',')]
        else:
            miners = client.get_all_miners()
            ip_list = [miner['ip'] for miner in miners]
            click.echo(f"Found {len(ip_list)} miners to benchmark")
        
        # Start benchmarks
        click.echo(f"Starting benchmarks with frequency={frequency}, core_voltage={core_voltage}, duration={duration}")
        result = client.start_multi_benchmark(
            ips=ip_list,
            frequency=frequency,
            core_voltage=core_voltage,
            duration=duration
        )
        
        started_ips = result['started_ips']
        click.echo(f"Started benchmarks for {len(started_ips)} miners")
        
        if wait and started_ips:
            click.echo("Waiting for benchmarks to complete...")
            
            for ip in started_ips:
                click.echo(f"Waiting for {ip}...")
                completed = client.wait_for_benchmark_completion(ip, timeout=duration + 300)
                
                if completed:
                    click.echo(f"✓ {ip} completed")
                else:
                    click.echo(f"✗ {ip} timed out")
            
            click.echo("All benchmarks finished!")


if __name__ == '__main__':
    cli()