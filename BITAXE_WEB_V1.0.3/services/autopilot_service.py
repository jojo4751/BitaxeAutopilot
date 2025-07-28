import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set, List, Tuple
from collections import deque


class AutopilotService:
    def __init__(self, config_service, database_service, miner_service, benchmark_service):
        self.config_service = config_service
        self.database_service = database_service
        self.miner_service = miner_service
        self.benchmark_service = benchmark_service
        
        # Autopilot state
        self.is_running = False
        self.autopilot_thread = None
        self.last_benchmark_times = {}
        self.downtime_tracker = {}
        self.rolling_efficiency = {}
        self.rolling_window_size = 5
        self.efficiency_threshold = 0.90
        
        # Initialize trackers for all configured IPs
        for ip in self.config_service.ips:
            self.last_benchmark_times[ip] = datetime.now()
            self.downtime_tracker[ip] = None
            self.rolling_efficiency[ip] = deque(maxlen=self.rolling_window_size)

    def start_autopilot(self) -> bool:
        """Start the autopilot loop"""
        if self.is_running:
            return False
            
        self.is_running = True
        self.autopilot_thread = threading.Thread(target=self._autopilot_loop, daemon=True)
        self.autopilot_thread.start()
        
        self.database_service.log_event("SYSTEM", "AUTOPILOT_STARTED", "Autopilot service started")
        return True

    def stop_autopilot(self) -> bool:
        """Stop the autopilot loop"""
        if not self.is_running:
            return False
            
        self.is_running = False
        if self.autopilot_thread:
            self.autopilot_thread.join(timeout=10)
            
        self.database_service.log_event("SYSTEM", "AUTOPILOT_STOPPED", "Autopilot service stopped")
        return True

    def _autopilot_loop(self) -> None:
        """Main autopilot loop"""
        while self.is_running:
            try:
                self._process_all_miners()
                self._update_best_settings_from_benchmarks()
                
                sleep_time = self.config_service.log_interval
                time.sleep(sleep_time)
                
            except Exception as e:
                self.database_service.log_event("SYSTEM", "AUTOPILOT_ERROR", f"Loop error: {e}")
                time.sleep(30)  # Wait before retrying

    def _process_all_miners(self) -> None:
        """Process all configured miners"""
        benchmark_interval = self.config_service.benchmark_interval
        
        for ip in self.config_service.ips:
            try:
                # Skip if currently benchmarking
                if self.benchmark_service.is_benchmarking(ip):
                    print(f"[{ip}] Benchmarking in progress - skipping autopilot checks")
                    continue
                
                # Fetch current miner data
                data = self.miner_service.fetch_miner_data(ip)
                if not data:
                    continue
                
                # Extract key metrics
                now = datetime.now()
                hostname = data.get("hostname", ip)
                temp = data.get("temp", 0)
                hashrate = data.get("hashRate", 0)
                freq = data.get("frequency", 0)
                volt = data.get("coreVoltage", 0)
                power = data.get("power", 0)
                
                efficiency = hashrate / power if power > 0 else 0
                
                print(f"[{now.strftime('%H:%M:%S')}] {hostname} | "
                      f"Temp: {temp:.1f}°C | HR: {hashrate:.2f} GH/s | "
                      f"Eff: {efficiency:.2f} GH/W")
                
                # Check for hashrate zero (downtime)
                self._check_hashrate_zero(ip, hashrate)
                
                # Check for overheating
                if self._check_overheating(ip, temp):
                    continue  # Skip other checks if overheated
                
                # Apply adaptive settings for high temperature
                if temp >= self.config_service.temp_limit:
                    self._apply_adaptive_settings(ip, temp, freq, volt)
                
                # Check efficiency drift
                self._check_efficiency_drift(ip, efficiency)
                
                # Check for routine benchmark
                self._check_routine_benchmark(ip, now, benchmark_interval, freq, volt)
                
            except Exception as e:
                self.database_service.log_event(ip, "AUTOPILOT_ERROR", f"Processing error: {e}")

    def _check_hashrate_zero(self, ip: str, hashrate: float) -> None:
        """Check for zero hashrate and handle downtime"""
        if hashrate <= 0:
            if self.downtime_tracker[ip] is None:
                self.downtime_tracker[ip] = datetime.now()
            else:
                elapsed = (datetime.now() - self.downtime_tracker[ip]).total_seconds()
                if elapsed > 300:  # 5 minutes
                    self.database_service.log_event(
                        ip, "HASHRATE_ZERO_REBOOT", 
                        f"Rebooting after {elapsed:.0f}s without hashrate"
                    )
                    
                    if self.miner_service.restart_miner(ip):
                        self.downtime_tracker[ip] = None  # Reset after successful reboot
        else:
            self.downtime_tracker[ip] = None  # Reset when hashrate returns

    def _check_overheating(self, ip: str, temperature: float) -> bool:
        """Check for overheating and apply fallback if needed"""
        if temperature >= self.config_service.temp_overheat:
            self.database_service.log_event(
                ip, "TEMP_OVERHEAT", 
                f"Overheating at {temperature}°C - applying fallback"
            )
            self.miner_service.set_fallback_settings(ip)
            return True
        return False

    def _apply_adaptive_settings(self, ip: str, temp: float, current_freq: int, 
                               current_volt: int) -> None:
        """Apply adaptive settings to reduce temperature"""
        # Calculate reduced settings
        new_freq = max(current_freq - 25, 600)
        new_volt = max(current_volt - 25, 1100)
        
        # Validate settings are within acceptable ranges
        if not self.miner_service.validate_settings(new_freq, new_volt):
            # Fall back to predefined fallback settings
            self.miner_service.set_fallback_settings(ip)
            return
        
        success = self.miner_service.set_miner_settings(ip, new_freq, new_volt)
        if success:
            self.database_service.log_event(
                ip, "ADAPTIVE_SETTING", 
                f"Reduced to {new_freq} MHz @ {new_volt} mV due to temp {temp}°C"
            )

    def _check_efficiency_drift(self, ip: str, current_efficiency: float) -> None:
        """Check for efficiency drift and log warnings"""
        # Add to rolling window
        self.rolling_efficiency[ip].append(current_efficiency)
        
        if len(self.rolling_efficiency[ip]) < self.rolling_window_size:
            return  # Not enough data yet
        
        # Calculate smoothed efficiency
        smoothed_efficiency = sum(self.rolling_efficiency[ip]) / len(self.rolling_efficiency[ip])
        
        # Get best known efficiency for this IP
        best_efficiency = self.database_service.get_best_efficiency_for_ip(ip)
        if not best_efficiency:
            return  # No benchmark data yet
        
        # Check for significant drift
        if smoothed_efficiency < best_efficiency * self.efficiency_threshold:
            self.database_service.log_event(
                ip, "EFFICIENCY_DRIFT", 
                f"Current: {smoothed_efficiency:.2f} < Best: {best_efficiency:.2f} GH/W"
            )

    def _check_routine_benchmark(self, ip: str, now: datetime, benchmark_interval: int, 
                               current_freq: int, current_volt: int) -> None:
        """Check if routine benchmark is needed"""
        time_since_last = (now - self.last_benchmark_times[ip]).total_seconds()
        
        if time_since_last > benchmark_interval:
            self.database_service.log_event(
                ip, "ROUTINE_BENCHMARK", 
                f"Starting routine benchmark with {current_freq} MHz @ {current_volt} mV"
            )
            
            # Start benchmark with current settings
            success = self.benchmark_service.start_benchmark(
                ip, current_freq, current_volt, 600
            )
            
            if success:
                self.last_benchmark_times[ip] = now

    def _update_best_settings_from_benchmarks(self) -> None:
        """Update best settings based on latest benchmark results"""
        try:
            # This would typically analyze recent benchmarks and update
            # optimal settings for each miner
            pass
        except Exception as e:
            self.database_service.log_event(
                "SYSTEM", "AUTOPILOT_ERROR", 
                f"Failed to update best settings: {e}"
            )

    def get_autopilot_status(self) -> Dict[str, Any]:
        """Get current autopilot status"""
        return {
            "is_running": self.is_running,
            "monitored_ips": list(self.config_service.ips),
            "benchmarking_ips": list(self.benchmark_service.get_benchmarking_ips()),
            "downtime_trackers": {
                ip: tracker.isoformat() if tracker else None 
                for ip, tracker in self.downtime_tracker.items()
            },
            "last_benchmark_times": {
                ip: timestamp.isoformat() 
                for ip, timestamp in self.last_benchmark_times.items()
            }
        }

    def force_benchmark_all(self, frequency: int, core_voltage: int, 
                          duration: int = 600) -> List[str]:
        """Force benchmark on all configured miners"""
        if not self.miner_service.validate_settings(frequency, core_voltage):
            return []
        
        started_ips = self.benchmark_service.start_multi_benchmark(
            self.config_service.ips, frequency, core_voltage, duration
        )
        
        # Update last benchmark times for successful starts
        now = datetime.now()
        for ip in started_ips:
            self.last_benchmark_times[ip] = now
        
        return started_ips

    def set_emergency_fallback_all(self) -> List[str]:
        """Set all miners to emergency fallback settings"""
        successful_ips = []
        
        for ip in self.config_service.ips:
            if self.miner_service.set_fallback_settings(ip):
                successful_ips.append(ip)
                self.database_service.log_event(
                    ip, "EMERGENCY_FALLBACK", 
                    "Emergency fallback applied via autopilot"
                )
        
        return successful_ips

    def update_efficiency_threshold(self, threshold: float) -> None:
        """Update the efficiency drift threshold"""
        if 0.5 <= threshold <= 1.0:
            self.efficiency_threshold = threshold
            self.database_service.log_event(
                "SYSTEM", "AUTOPILOT_CONFIG", 
                f"Efficiency threshold updated to {threshold}"
            )