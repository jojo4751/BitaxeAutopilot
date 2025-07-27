import time
import threading
from datetime import datetime
from typing import Set, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor


class BenchmarkService:
    def __init__(self, config_service, database_service, miner_service):
        self.config_service = config_service
        self.database_service = database_service
        self.miner_service = miner_service
        self.benchmarking_ips: Set[str] = set()
        self.benchmark_threads = {}
        self.executor = ThreadPoolExecutor(max_workers=5)

    def is_benchmarking(self, ip: str) -> bool:
        """Check if IP is currently benchmarking"""
        return ip in self.benchmarking_ips

    def get_benchmarking_ips(self) -> Set[str]:
        """Get all IPs currently benchmarking"""
        return self.benchmarking_ips.copy()

    def start_benchmark(self, ip: str, frequency: int, core_voltage: int, 
                       duration: int = 600) -> bool:
        """Start a benchmark for a specific miner"""
        if self.is_benchmarking(ip):
            self.database_service.log_event(ip, "BENCHMARK_ERROR", "Benchmark already running")
            return False

        if not self.miner_service.validate_settings(frequency, core_voltage):
            self.database_service.log_event(ip, "BENCHMARK_ERROR", "Invalid settings")
            return False

        # Submit benchmark to thread pool
        future = self.executor.submit(
            self._run_benchmark_thread, ip, frequency, core_voltage, duration
        )
        self.benchmark_threads[ip] = future
        
        return True

    def start_multi_benchmark(self, ips: List[str], frequency: int, 
                            core_voltage: int, duration: int = 600) -> List[str]:
        """Start benchmarks for multiple miners"""
        started_ips = []
        
        for ip in ips:
            if self.start_benchmark(ip, frequency, core_voltage, duration):
                started_ips.append(ip)
                self.database_service.log_event(
                    ip, "MULTI_BENCHMARK_STARTED", 
                    f"{frequency} MHz @ {core_voltage} mV für {duration}s"
                )
        
        return started_ips

    def _run_benchmark_thread(self, ip: str, frequency: int, core_voltage: int, 
                            duration: int) -> None:
        """Execute benchmark in separate thread"""
        try:
            self.benchmarking_ips.add(ip)
            self._execute_benchmark(ip, frequency, core_voltage, duration)
        finally:
            self.benchmarking_ips.discard(ip)
            if ip in self.benchmark_threads:
                del self.benchmark_threads[ip]

    def _execute_benchmark(self, ip: str, frequency: int, core_voltage: int, 
                          duration: int) -> None:
        """Execute the actual benchmark process"""
        sample_interval = 15
        total_samples = duration // sample_interval
        
        hash_rates = []
        temperatures = []
        vr_temps = []
        power_vals = []

        # Set initial settings and restart
        success = self.miner_service.set_miner_settings(ip, frequency, core_voltage)
        if not success:
            self.database_service.log_event(ip, "BENCHMARK_ERROR", "Failed to set initial settings")
            return

        # Restart miner to apply settings
        restart_success = self.miner_service.restart_miner(ip)
        if not restart_success:
            self.database_service.log_event(ip, "BENCHMARK_WARNING", "Failed to restart miner")

        self.database_service.log_event(
            ip, "BENCHMARK_STARTED", 
            f"{frequency} MHz @ {core_voltage} mV für {duration}s"
        )

        # Wait for miner to stabilize after restart
        time.sleep(90)

        # Collect samples
        for i in range(total_samples):
            try:
                data = self.miner_service.fetch_miner_data(ip)
                if not data:
                    continue

                temp = data.get("temp", 0)
                hashrate = data.get("hashRate", 0)
                power = data.get("power", 0)
                vr_temp = data.get("vrTemp", 0)

                # Check for overheating
                if temp >= self.config_service.temp_overheat:
                    self.database_service.log_event(
                        ip, "BENCHMARK_ABORTED", 
                        f"Overheating at {temp}°C"
                    )
                    # Set fallback settings
                    self.miner_service.set_fallback_settings(ip)
                    break

                # Collect valid samples
                if hashrate > 0:
                    hash_rates.append(hashrate)
                if temp > 0:
                    temperatures.append(temp)
                if power > 0:
                    power_vals.append(power)
                if vr_temp > 0:
                    vr_temps.append(vr_temp)

                time.sleep(sample_interval)

            except Exception as e:
                self.database_service.log_event(ip, "BENCHMARK_ERROR", f"Sample error: {e}")
                break

        # Process results
        if not hash_rates or not temperatures or not power_vals:
            self.database_service.log_event(ip, "BENCHMARK_ERROR", "No valid benchmark data")
            return

        # Calculate averages
        avg_hashrate = sum(hash_rates) / len(hash_rates)
        avg_temp = sum(temperatures) / len(temperatures)
        avg_power = sum(power_vals) / len(power_vals)
        avg_vr_temp = sum(vr_temps) / len(vr_temps) if vr_temps else 0

        # Calculate efficiency (J/TH)
        efficiency = (avg_power / (avg_hashrate / 1000)) if avg_hashrate > 0 else 0

        # Save benchmark results
        self.database_service.save_benchmark_result(
            ip, frequency, core_voltage, avg_hashrate, avg_temp, efficiency, duration
        )

        # Log completion
        message = f"HR={avg_hashrate:.2f} GH/s, Temp={avg_temp:.1f}°C, Eff={efficiency:.2f} J/TH"
        self.database_service.log_event(ip, "BENCHMARK_COMPLETED", message)

    def stop_benchmark(self, ip: str) -> bool:
        """Stop a running benchmark"""
        if not self.is_benchmarking(ip):
            return False

        if ip in self.benchmark_threads:
            # Note: We can't easily cancel a running thread, 
            # but we can mark it as cancelled
            self.benchmarking_ips.discard(ip)
            self.database_service.log_event(ip, "BENCHMARK_CANCELLED", "Manually stopped")
            return True

        return False

    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get current benchmark status for all miners"""
        return {
            "active_benchmarks": list(self.benchmarking_ips),
            "total_active": len(self.benchmarking_ips)
        }

    def cleanup_completed_benchmarks(self) -> None:
        """Clean up completed benchmark threads"""
        completed_ips = []
        for ip, future in self.benchmark_threads.items():
            if future.done():
                completed_ips.append(ip)

        for ip in completed_ips:
            del self.benchmark_threads[ip]
            self.benchmarking_ips.discard(ip)

    def get_benchmark_results(self, ip: Optional[str] = None, limit: int = 50) -> List[Any]:
        """Get benchmark results, optionally filtered by IP"""
        if ip:
            return self.database_service.get_benchmark_results_for_ip(ip, limit)
        else:
            return self.database_service.get_benchmark_results(limit)

    def get_best_settings_for_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get best performing settings for a specific IP"""
        results = self.database_service.get_benchmark_results_for_ip(ip, 1)
        if results:
            result = results[0]
            return {
                'frequency': result[0],
                'coreVoltage': result[1],
                'averageHashRate': result[2],
                'averageTemperature': result[3],
                'efficiencyJTH': result[4]
            }
        return None

    def schedule_routine_benchmark(self, ip: str, current_frequency: int, 
                                 current_voltage: int) -> bool:
        """Schedule a routine benchmark if interval has passed"""
        # This would typically check last benchmark time and schedule if needed
        # For now, just start a benchmark with current settings
        return self.start_benchmark(ip, current_frequency, current_voltage, 600)

    def shutdown(self) -> None:
        """Shutdown benchmark service and cleanup resources"""
        # Cancel all running benchmarks
        for ip in list(self.benchmarking_ips):
            self.stop_benchmark(ip)
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)