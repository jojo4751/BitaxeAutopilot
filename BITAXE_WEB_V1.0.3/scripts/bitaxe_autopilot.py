# scripts/bitaxe_autopilot.py

import time
import sqlite3
import requests
from datetime import datetime
from config.config_loader import load_config, reload_config
from scripts.fetcher import fetch_data
from scripts.protocol_utils import log_event, write_logfile_entry
from utils.autopilot_utils import (
    check_efficiency_drift,
    apply_adaptive_settings,
    set_fallback,
    update_best_settings_from_benchmarks
)
from scripts.benchmark_state import benchmarking_ips  # Zentraler Merker für aktiven Benchmark

cfg = load_config()

ips = cfg["config"]["ips"]
log_interval = cfg["config"]["log_interval_sec"]
temp_limit = cfg["settings"]["temp_limit"]
temp_overheat = cfg["settings"]["temp_overheat"]

db_path = cfg["paths"]["database"]

# Tracker für Routine-Benchmarks
last_benchmark_times = {ip: datetime.now() for ip in ips}

# Downtime-Tracker für Hashrate = 0
downtime_tracker = {ip: None for ip in ips}

def autopilot_loop():
    while True:
        cfg = reload_config()  # bei jedem Loop fresh laden
        benchmark_interval_sec = cfg["settings"].get("benchmark_interval_sec", 86400)

        for ip in ips:
            data = fetch_data(ip)
            if not data:
                continue

            now = datetime.now()
            hostname = data.get("hostname", ip)
            temp = data.get("temp", 0)
            hashrate = data.get("hashRate", 0)
            freq = data.get("frequency", 0)
            volt = data.get("coreVoltage", 0)
            power = data.get("power", 0)

            gh_per_watt = hashrate / power if power > 0 else 0

            print(f"[{now.strftime('%H:%M:%S')}] {hostname} | Temp: {temp:.1f}°C | HR: {hashrate:.2f} GH/s | Eff: {gh_per_watt:.2f} GH/W")

            # ✅ Benchmark-Schutz: ALLES aus während Benchmark
            if ip in benchmarking_ips:
                print(f"[{ip}] Im Benchmark → Safeguards, Adaptive & Drift AUSGESCHALTET")
                continue  # Skip alle Safeguards, Adaptive, Drift-Check

            # ✅ Hashrate Zero Safeguard mit Downtime-Timer
            if hashrate <= 0:
                if downtime_tracker[ip] is None:
                    downtime_tracker[ip] = datetime.now()
                else:
                    elapsed = (datetime.now() - downtime_tracker[ip]).total_seconds()
                    if elapsed > 300:
                        write_logfile_entry(f"[{ip}] Hashrate = 0 seit {elapsed:.0f}s → Reboot")
                        try:
                            requests.post(f"http://{ip}/api/system/restart", timeout=5)
                            log_event(ip, "HASHRATE_ZERO_REBOOT", f"Nach {elapsed:.0f}s ohne Hashrate")
                            downtime_tracker[ip] = None  # Reset nach Reboot
                        except Exception as e:
                            write_logfile_entry(f"[{ip}] Reboot-Fehler: {e}")
            else:
                downtime_tracker[ip] = None  # Reset wenn Hashrate wieder da

            # ✅ Überhitzung Fallback
            if temp >= temp_overheat:
                write_logfile_entry(f"[{ip}] Überhitzung ({temp} °C) → Fallback")
                set_fallback(ip)
                log_event(ip, "TEMP_OVERHEAT", f"{temp} °C")
                continue  # nach Fallback sofort nächsten Miner prüfen

            # ✅ Adaptive Settings nur wenn KEIN Benchmark
            if temp >= temp_limit:
                apply_adaptive_settings(ip, temp, freq, volt)

            # ✅ Effizienz-Drift nur wenn KEIN Benchmark
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            best_eff = cursor.execute(f"""
                SELECT MIN(avgEfficiency) FROM {cfg["paths"]["tuning_table"]} WHERE ip = ?
            """, (ip,)).fetchone()[0] or 1
            conn.close()

            check_efficiency_drift(ip, gh_per_watt, best_eff, benchmarking_ips)

            # ✅ Routine-Benchmark nur wenn KEIN aktiver Benchmark
            now_time = datetime.now()
            if (now_time - last_benchmark_times[ip]).total_seconds() > benchmark_interval_sec:
                write_logfile_entry(f"[{ip}] Autopilot: Routine-Benchmark gestartet")
                from scripts.benchmark_runner import run_benchmark  # Import hier lösen, kein Kreis
                benchmarking_ips.add(ip)
                try:
                    run_benchmark(ip, volt, freq, 600)
                finally:
                    benchmarking_ips.remove(ip)
                last_benchmark_times[ip] = now_time

        update_best_settings_from_benchmarks()

        print(f"Warte {log_interval} Sekunden...\n")
        time.sleep(log_interval)

if __name__ == "__main__":
    autopilot_loop()
