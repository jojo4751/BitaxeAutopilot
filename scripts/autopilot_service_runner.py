#!/usr/bin/env python3
"""
Autopilot service runner using the new service architecture
"""
import signal
import sys
from services.service_container import get_container


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\nShutting down autopilot...")
    container = get_container()
    container.shutdown()
    sys.exit(0)


def main():
    """Main autopilot runner"""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    container = get_container()
    autopilot_service = container.get_autopilot_service()
    config_service = container.get_config_service()
    
    print(f"Starting autopilot for {len(config_service.ips)} miners")
    print("Temperature limits:")
    print(f"  Warning: {config_service.temp_limit}°C")
    print(f"  Overheat: {config_service.temp_overheat}°C")
    print(f"Benchmark interval: {config_service.benchmark_interval}s")
    print("-" * 50)
    
    # Start the autopilot service
    success = autopilot_service.start_autopilot()
    if not success:
        print("Failed to start autopilot service")
        return 1
    
    try:
        # Keep the main thread alive
        while autopilot_service.is_running:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        print("Cleaning up...")
        container.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())