# autopilot_utils.py
import sqlite3
import requests
from datetime import datetime
from collections import deque
from config.config_loader import load_config
from scripts.protocol_utils import log_event, write_logfile_entry

cfg = load_config()

db_path = cfg["paths"]["database"]
tuning_table = cfg["paths"]["tuning_table"]
fallback = cfg["settings"]["fallback"]
ROLLING_WINDOW_SIZE = 5

# Rolling Cache pro IP
rolling_eff = {}

def check_efficiency_drift(ip, current_eff, best_eff, benchmarking_ips, threshold=0.90):
    if ip in benchmarking_ips:
        return  # Während Benchmark keine Drift-Prüfung

    q = rolling_eff.get(ip, deque(maxlen=ROLLING_WINDOW_SIZE))
    q.append(current_eff)
    rolling_eff[ip] = q

    smoothed = sum(q) / len(q)

    if smoothed < best_eff * threshold:
        log_event(ip, "EFFICIENCY_DRIFT", f"Live Eff: {smoothed:.2f} < {best_eff:.2f}")
        write_logfile_entry(f"[{ip}] Drift erkannt: {smoothed:.2f} < {best_eff:.2f}")

def apply_adaptive_settings(ip, temp, current_freq, current_volt):
    new_freq = max(current_freq - 25, 600)
    new_volt = max(current_volt - 10, 700)
    payload = {
        "coreVoltage": new_volt,
        "frequency": new_freq,
        "autofanspeed": True
    }
    try:
        requests.patch(f"http://{ip}/api/system", json=payload, timeout=5)
        log_event(ip, "ADAPTIVE_SETTING", f"{new_freq} MHz / {new_volt} mV @ {temp} °C")
        write_logfile_entry(f"[{ip}] Adaptive Settings gesetzt: {payload}")
    except Exception as e:
        write_logfile_entry(f"[{ip}] Adaptive Setting-Fehler: {e}")

def set_fallback(ip):
    try:
        payload = {
            "coreVoltage": fallback["coreVoltage"],
            "frequency": fallback["frequency"],
            "fanspeed": 100,
            "autofanspeed": False
        }
        requests.patch(f"http://{ip}/api/system", json=payload, timeout=5)
        requests.post(f"http://{ip}/api/system/restart", json={}, timeout=5)
        write_logfile_entry(f"[{ip}] Fallback gesetzt + Reboot")
        log_event(ip, "FALLBACK_SET", f"{fallback['frequency']} MHz / {fallback['coreVoltage']} mV")
    except Exception as e:
        write_logfile_entry(f"[{ip}] Fallback-Fehler: {e}")

def update_best_settings_from_benchmarks():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT ip, frequency, coreVoltage, efficiencyJTH
        FROM benchmark_results AS b1
        WHERE efficiencyJTH = (
            SELECT MIN(efficiencyJTH)
            FROM benchmark_results AS b2
            WHERE b1.ip = b2.ip
        )
        GROUP BY ip
    """)
    bests = cursor.fetchall()

    for ip, freq, volt, eff in bests:
        cursor.execute(f"""
            INSERT OR REPLACE INTO {tuning_table}
            (ip, frequency, coreVoltage, avgEfficiency, failed)
            VALUES (?, ?, ?, ?, 0)
        """, (ip, freq, volt, eff))
        log_event(ip, "BEST_SETTINGS_UPDATED", f"{freq} MHz / {volt} mV, Eff={eff:.2f}",
                  conn=conn, cursor=cursor)

    conn.commit()
    conn.close()
