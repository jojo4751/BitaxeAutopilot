"""
BitAxe SDK Examples

Examples demonstrating how to use the BitAxe API client SDK.
"""

import asyncio
import time
from datetime import datetime
import logging

from client import BitAxeClient, create_client, BitAxeAPIError
from async_client import AsyncBitAxeClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic synchronous client usage"""
    print("=== Basic Usage Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        # Create and authenticate client
        with create_client(base_url, username, password) as client:
            # Get user info
            user_info = client.get_user_info()
            print(f"Authenticated as: {user_info['username']}")
            print(f"Roles: {', '.join(user_info['roles'])}")
            
            # Get miners summary
            summary = client.get_miners_summary()
            print(f"\nMiners Summary:")
            print(f"  Total: {summary['total_miners']}")
            print(f"  Online: {summary['online_miners']}")
            print(f"  Total Hashrate: {summary['total_hashrate']:.1f} GH/s")
            print(f"  Total Power: {summary['total_power']:.1f} W")
            print(f"  Average Temp: {summary['average_temperature']:.1f}°C")
            
            # Get all miners
            miners = client.get_all_miners()
            print(f"\nFound {len(miners)} miners:")
            
            for miner in miners[:5]:  # Show first 5
                print(f"  {miner['ip']}: {miner.get('hash_rate', 'N/A')} GH/s, "
                      f"{miner.get('temperature', 'N/A')}°C")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


def example_miner_management():
    """Example of miner management operations"""
    print("\n=== Miner Management Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        with create_client(base_url, username, password) as client:
            # Get first available miner
            miners = client.get_all_miners()
            if not miners:
                print("No miners found")
                return
            
            miner_ip = miners[0]['ip']
            print(f"Working with miner: {miner_ip}")
            
            # Get detailed miner info
            miner_detail = client.get_miner(miner_ip)
            print(f"Current settings:")
            print(f"  Frequency: {miner_detail.get('frequency', 'N/A')} MHz")
            print(f"  Core Voltage: {miner_detail.get('core_voltage', 'N/A')} mV")
            print(f"  Temperature: {miner_detail.get('temperature', 'N/A')}°C")
            
            # Update miner settings (be careful with real miners!)
            print(f"\nUpdating miner settings...")
            update_result = client.update_miner_settings(
                ip=miner_ip,
                frequency=800,
                core_voltage=1200,
                autofanspeed=True
            )
            print(f"Settings updated: {update_result}")
            
            # Note: Uncomment the following to restart the miner
            # print(f"\nRestarting miner...")
            # restart_result = client.restart_miner(miner_ip)
            # print(f"Restart initiated: {restart_result}")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


def example_benchmark_workflow():
    """Example of complete benchmark workflow"""
    print("\n=== Benchmark Workflow Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        with create_client(base_url, username, password) as client:
            # Get available miners
            miners = client.get_all_miners()
            if not miners:
                print("No miners found")
                return
            
            # Select first miner for benchmark
            miner_ip = miners[0]['ip']
            print(f"Starting benchmark for miner: {miner_ip}")
            
            # Start benchmark
            benchmark_result = client.start_benchmark(
                ip=miner_ip,
                frequency=800,
                core_voltage=1200,
                duration=120  # Short duration for example
            )
            print(f"Benchmark started: {benchmark_result}")
            
            # Monitor benchmark status
            print("Monitoring benchmark progress...")
            start_time = time.time()
            
            while time.time() - start_time < 300:  # 5 minute timeout
                status = client.get_benchmark_status()
                active_benchmarks = status['active_benchmarks']
                
                if miner_ip not in active_benchmarks:
                    print("Benchmark completed!")
                    break
                
                print(f"Benchmark running... ({len(active_benchmarks)} active)")
                time.sleep(10)
            else:
                print("Benchmark timeout")
            
            # Get benchmark results
            print("\nFetching benchmark results...")
            results = client.get_benchmark_results(ip=miner_ip, limit=5)
            
            if results:
                latest = results[0]
                print(f"Latest benchmark result:")
                print(f"  Frequency: {latest['frequency']} MHz")
                print(f"  Core Voltage: {latest['core_voltage']} mV")
                print(f"  Average Hashrate: {latest.get('average_hashrate', 'N/A')} GH/s")
                print(f"  Average Temperature: {latest.get('average_temperature', 'N/A')}°C")
                print(f"  Efficiency: {latest.get('efficiency_jth', 'N/A')} J/TH")
            else:
                print("No benchmark results found")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


async def example_async_usage():
    """Example of async client usage"""
    print("\n=== Async Usage Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        async with AsyncBitAxeClient(base_url) as client:
            # Authenticate
            await client.login(username, password)
            
            # Get user info and system health concurrently
            user_task = client.get_user_info()
            health_task = client.get_health()
            summary_task = client.get_miners_summary()
            
            user_info, health, summary = await asyncio.gather(
                user_task, health_task, summary_task
            )
            
            print(f"User: {user_info['username']}")
            print(f"System Health: {health['overall_status']}")
            print(f"Total Miners: {summary['total_miners']}")
            
            # Get all miners
            miners = await client.get_all_miners()
            
            if len(miners) >= 2:
                # Get details for first 2 miners concurrently
                ips = [miner['ip'] for miner in miners[:2]]
                miner_details = await client.get_miners_batch(ips)
                
                print(f"\nGot details for {len(miner_details)} miners:")
                for detail in miner_details:
                    print(f"  {detail['ip']}: {detail.get('hash_rate', 'N/A')} GH/s")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


def example_batch_operations():
    """Example of batch operations with multiple miners"""
    print("\n=== Batch Operations Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        with create_client(base_url, username, password) as client:
            # Get all miners
            miners = client.get_all_miners()
            if len(miners) < 2:
                print("Need at least 2 miners for batch operations")
                return
            
            # Select multiple miners
            target_ips = [miner['ip'] for miner in miners[:3]]  # First 3 miners
            print(f"Working with miners: {', '.join(target_ips)}")
            
            # Start multi-benchmark
            print("Starting multi-benchmark...")
            multi_result = client.start_multi_benchmark(
                ips=target_ips,
                frequency=800,
                core_voltage=1200,
                duration=120
            )
            
            started_ips = multi_result['started_ips']
            print(f"Started benchmarks for {len(started_ips)} miners")
            
            # Wait for all benchmarks to complete
            print("Waiting for benchmarks to complete...")
            completed_count = 0
            
            for ip in started_ips:
                print(f"Waiting for {ip}...")
                completed = client.wait_for_benchmark_completion(ip, timeout=180)
                
                if completed:
                    completed_count += 1
                    print(f"✓ {ip} completed")
                else:
                    print(f"✗ {ip} timed out")
            
            print(f"Completed benchmarks: {completed_count}/{len(started_ips)}")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


async def example_async_batch_operations():
    """Example of async batch operations"""
    print("\n=== Async Batch Operations Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        async with AsyncBitAxeClient(base_url) as client:
            await client.login(username, password)
            
            # Get all miners
            miners = await client.get_all_miners()
            if len(miners) < 2:
                print("Need at least 2 miners for batch operations")
                return
            
            # Update settings for multiple miners concurrently
            settings_updates = [
                {'ip': miners[0]['ip'], 'frequency': 800, 'core_voltage': 1200},
                {'ip': miners[1]['ip'], 'frequency': 850, 'core_voltage': 1250},
            ]
            
            print("Updating miner settings concurrently...")
            update_results = await client.update_miners_settings_batch(settings_updates)
            print(f"Updated settings for {len(update_results)} miners")
            
            # Get fresh miner details
            target_ips = [miners[0]['ip'], miners[1]['ip']]
            updated_miners = await client.get_miners_batch(target_ips)
            
            print("Updated miner details:")
            for miner in updated_miners:
                print(f"  {miner['ip']}: {miner.get('frequency', 'N/A')} MHz, "
                      f"{miner.get('core_voltage', 'N/A')} mV")
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except Exception as e:
        print(f"Error: {e}")


def example_error_handling():
    """Example of error handling"""
    print("\n=== Error Handling Example ===")
    
    base_url = "http://localhost:5000"
    
    # Example 1: Authentication error
    try:
        with create_client(base_url, "invalid_user", "invalid_pass") as client:
            client.get_user_info()
    except BitAxeAPIError as e:
        print(f"Expected authentication error: {e.error_code}")
    
    # Example 2: Invalid miner IP
    try:
        with create_client(base_url, "admin", "admin123") as client:
            client.get_miner("999.999.999.999")
    except BitAxeAPIError as e:
        print(f"Expected miner not found error: {e.error_code}")
    
    # Example 3: Validation error
    try:
        with create_client(base_url, "admin", "admin123") as client:
            # Invalid frequency (too high)
            client.update_miner_settings("192.168.1.100", frequency=5000, core_voltage=1200)
    except BitAxeAPIError as e:
        print(f"Expected validation error: {e.error_code}")
    
    print("Error handling examples completed")


def example_monitoring_dashboard():
    """Example of creating a simple monitoring dashboard"""
    print("\n=== Monitoring Dashboard Example ===")
    
    base_url = "http://localhost:5000"
    username = "admin"
    password = "admin123"
    
    try:
        with create_client(base_url, username, password) as client:
            while True:
                # Clear screen (simple version)
                print("\n" * 50)
                print("=== BitAxe Monitoring Dashboard ===")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Get system health
                health = client.get_health()
                print(f"System Health: {health['overall_status'].upper()}")
                
                # Get miners summary
                summary = client.get_miners_summary()
                print(f"\nMiners Overview:")
                print(f"  Total: {summary['total_miners']} "
                      f"(Online: {summary['online_miners']}, "
                      f"Offline: {summary['offline_miners']})")
                print(f"  Total Hashrate: {summary['total_hashrate']:.1f} GH/s")
                print(f"  Total Power: {summary['total_power']:.1f} W")
                print(f"  Efficiency: {summary['total_efficiency']:.1f} GH/W")
                print(f"  Avg Temperature: {summary['average_temperature']:.1f}°C")
                
                # Get individual miners
                miners = client.get_all_miners()
                print(f"\nIndividual Miners:")
                print("IP Address      | Hash Rate | Temp | Power | Status")
                print("-" * 55)
                
                for miner in miners[:10]:  # Show first 10
                    ip = miner['ip'].ljust(15)
                    hashrate = f"{miner.get('hash_rate', 0):.1f}".rjust(9)
                    temp = f"{miner.get('temperature', 0):.1f}°C".rjust(6)
                    power = f"{miner.get('power', 0):.1f}W".rjust(7)
                    
                    # Determine status based on recent activity
                    last_seen = miner.get('last_seen', '')
                    if last_seen:
                        last_time = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        age = (datetime.now() - last_time.replace(tzinfo=None)).total_seconds()
                        status = "Online" if age < 300 else "Offline"  # 5 min threshold
                    else:
                        status = "Unknown"
                    
                    print(f"{ip} | {hashrate} | {temp} | {power} | {status}")
                
                # Get recent events
                events = client.get_events(limit=5)
                if events:
                    print(f"\nRecent Events:")
                    for event in events:
                        timestamp = event['timestamp'][:19]  # Remove microseconds
                        severity = event['severity']
                        message = event['message'][:50]  # Truncate long messages
                        print(f"  {timestamp} [{severity}] {message}")
                
                print(f"\nPress Ctrl+C to exit...")
                
                # Wait 30 seconds before refresh
                try:
                    time.sleep(30)
                except KeyboardInterrupt:
                    print("\nDashboard stopped.")
                    break
    
    except BitAxeAPIError as e:
        print(f"API Error: {e.message}")
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples"""
    print("BitAxe SDK Examples")
    print("==================")
    
    # Note: These examples assume a running BitAxe API server
    # with default credentials. Adjust as needed.
    
    # Synchronous examples
    example_basic_usage()
    example_miner_management()
    example_benchmark_workflow()
    example_batch_operations()
    example_error_handling()
    
    # Async examples
    asyncio.run(example_async_usage())
    asyncio.run(example_async_batch_operations())
    
    # Interactive example (uncomment to run)
    # example_monitoring_dashboard()


if __name__ == "__main__":
    main()