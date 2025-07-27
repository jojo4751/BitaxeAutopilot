#!/usr/bin/env python3
"""
Command-line benchmark tool using the new service architecture
"""
import argparse
import sys
import time
from services.service_container import get_container


def main():
    """CLI benchmark tool"""
    parser = argparse.ArgumentParser(description="BITAXE Benchmark CLI")
    parser.add_argument("--ip", required=True, help="Miner IP address")
    parser.add_argument("--frequency", type=int, required=True, help="Frequency in MHz")
    parser.add_argument("--voltage", type=int, required=True, help="Core voltage in mV")
    parser.add_argument("--duration", type=int, default=600, help="Benchmark duration in seconds")
    parser.add_argument("--multi", nargs="*", help="Multiple IPs for multi-benchmark")
    
    args = parser.parse_args()
    
    container = get_container()
    config_service = container.get_config_service()
    benchmark_service = container.get_benchmark_service()
    miner_service = container.get_miner_service()
    
    # Validate settings
    if not miner_service.validate_settings(args.frequency, args.voltage):
        print(f"Invalid settings: {args.frequency} MHz @ {args.voltage} mV")
        print(f"Available frequencies: {config_service.freq_list}")
        print(f"Available voltages: {config_service.volt_list}")
        return 1
    
    try:
        if args.multi:
            # Multi-benchmark
            ips = args.multi
            print(f"Starting multi-benchmark for {len(ips)} miners:")
            print(f"Settings: {args.frequency} MHz @ {args.voltage} mV for {args.duration}s")
            
            started_ips = benchmark_service.start_multi_benchmark(
                ips, args.frequency, args.voltage, args.duration
            )
            
            if started_ips:
                print(f"Successfully started benchmarks for: {', '.join(started_ips)}")
                
                # Wait for completion
                print("Waiting for benchmarks to complete...")
                while any(benchmark_service.is_benchmarking(ip) for ip in started_ips):
                    time.sleep(10)
                    
                print("All benchmarks completed!")
            else:
                print("Failed to start any benchmarks")
                return 1
                
        else:
            # Single benchmark
            print(f"Starting benchmark for {args.ip}")
            print(f"Settings: {args.frequency} MHz @ {args.voltage} mV for {args.duration}s")
            
            success = benchmark_service.start_benchmark(
                args.ip, args.frequency, args.voltage, args.duration
            )
            
            if success:
                print("Benchmark started successfully")
                
                # Wait for completion
                print("Waiting for benchmark to complete...")
                while benchmark_service.is_benchmarking(args.ip):
                    time.sleep(10)
                    
                print("Benchmark completed!")
                
                # Show results
                results = benchmark_service.get_benchmark_results(args.ip, 1)
                if results:
                    result = results[0]
                    print(f"Results:")
                    print(f"  Average Hash Rate: {result[2]:.2f} GH/s")
                    print(f"  Average Temperature: {result[3]:.1f}°C")
                    print(f"  Efficiency: {result[4]:.2f} J/TH")
            else:
                print("Failed to start benchmark")
                return 1
                
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        if args.multi:
            for ip in args.multi:
                if benchmark_service.is_benchmarking(ip):
                    benchmark_service.stop_benchmark(ip)
        else:
            if benchmark_service.is_benchmarking(args.ip):
                benchmark_service.stop_benchmark(args.ip)
    finally:
        container.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())