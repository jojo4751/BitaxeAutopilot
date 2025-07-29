#!/usr/bin/env python3
"""
Main data logging service using the new service architecture
"""
import time
from datetime import datetime

from services.service_container import get_container


def main():
    """Main logging loop using services"""
    container = get_container()
    config_service = container.get_config_service()
    miner_service = container.get_miner_service()
    
    log_interval = config_service.log_interval
    ips = config_service.ips
    
    print(f"Starting data logger for {len(ips)} miners with {log_interval}s interval")
    
    try:
        while True:
            for ip in ips:
                data = miner_service.fetch_miner_data(ip)
                if data:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {ip} successfully logged")
                
            print(f"Waiting {log_interval} seconds...\n")
            time.sleep(log_interval)
            
    except KeyboardInterrupt:
        print("\nShutting down data logger...")
    finally:
        container.shutdown()


if __name__ == "__main__":
    main()