# scripts/benchmark_runner.py

import requests
import time
import sqlite3
from datetime import datetime
from config.config_loader import load_config
from scripts.protocol_utils import write_logfile_entry, log_event
from scripts.benchmark_state import benchmarking_ips  # Nur aus State, kein Autopilot-Import!

cfg = load_config()
db_path = cfg["paths"]["database"]
benchmark_table = cfg["paths"]["benchmark_results"]

def run_benchmark(ip, voltage, frequency, benchmark_time=600):
    """ Starte einen Benchmark-Lauf für angegebene Zeit, speichere in DB. """

    sample_interval = 15
    total_samples = benchmark_time // sample_interval

    hash_rates = []
    temperatures = []
    vr_temps = []
    power_vals = []

    payload = {"coreVoltage": voltage, "frequency": frequency}
    try:
        benchmarking_ips.add(ip)  # Markiere als aktiv
        requests.patch(f"http://{ip}/api/system", json=payload, timeout=5)
        requests.post(f"http://{ip}/api/system/restart", timeout=5)
        write_logfile_entry(f"[{ip}] Benchmark gestartet: {payload}, Dauer: {benchmark_time}s")
        log_event(ip, "BENCHMARK_STARTED", f"{frequency} MHz @ {voltage} mV für {benchmark_time}s")
    except Exception as e:
        write_logfile_entry(f"[{ip}] Benchmark-Start-Fehler: {e}")
        benchmarking_ips.discard(ip)
        return

    time.sleep(90)

    for i in range(total_samples):
        try:
            res = requests.get(f"http://{ip}/api/system/info", timeout=5).json()
            hr = res.get("hashRate", 0)
            temp = res.get("temp", 0)
            power = res.get("power", 0)
            vr_temp = res.get("vrTemp", 0)

            # ✅ Wenn zu heiß → Benchmark sauber abbrechen!
            if temp >= cfg["settings"]["temp_overheat"]:
                write_logfile_entry(f"[{ip}] Benchmark abgebrochen wegen Überhitzung bei {temp} °C")
                log_event(ip, "BENCHMARK_ABORTED", f"Überhitzung bei {temp} °C")
                break

            if hr: hash_rates.append(hr)
            if temp: temperatures.append(temp)
            if power: power_vals.append(power)
            if vr_temp: vr_temps.append(vr_temp)

            time.sleep(sample_interval)
        except Exception as e:
            write_logfile_entry(f"[{ip}] Benchmark-Loop-Fehler: {e}")
            break

    if not hash_rates or not temperatures or not power_vals:
        write_logfile_entry(f"[{ip}] Keine Benchmark-Daten.")
        benchmarking_ips.discard(ip)
        return

    avg_hr = sum(hash_rates) / len(hash_rates)
    avg_temp = sum(temperatures) / len(temperatures)
    avg_power = sum(power_vals) / len(power_vals)
    avg_vr_temp = sum(vr_temps) / len(vr_temps) if vr_temps else None

    eff = avg_power / (avg_hr / 1_000) if avg_hr > 0 else None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {benchmark_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ip TEXT,
            frequency INTEGER,
            coreVoltage INTEGER,
            averageHashRate REAL,
            averageTemperature REAL,
            efficiencyJTH REAL,
            averageVRTemp REAL,
            duration INTEGER,
            timestamp TEXT
        )
    """)
    cursor.execute(f"""
        INSERT INTO {benchmark_table} 
        (ip, frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, averageVRTemp, duration, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ip,
        frequency,
        voltage,
        avg_hr,
        avg_temp,
        eff,
        avg_vr_temp,
        benchmark_time,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

    write_logfile_entry(f"[{ip}] Benchmark fertig: HR={avg_hr:.2f} GH/s, Temp={avg_temp:.1f}°C, Eff={eff:.2f} J/TH")
    log_event(ip, "BENCHMARK_DONE", f"HR={avg_hr:.2f} GH/s, Eff={eff:.2f} J/TH")

    benchmarking_ips.discard(ip)  # Benchmark abgeschlossen!
