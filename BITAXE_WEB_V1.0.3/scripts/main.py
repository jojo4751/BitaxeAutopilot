import time
import sqlite3
from datetime import datetime
from config.config_loader import load_config
from scripts.fetcher import fetch_data
from scripts.protocol_utils import write_logfile_entry

cfg = load_config()

ips = cfg["config"]["ips"]
log_interval = cfg["config"]["log_interval_sec"]
db_path = cfg["paths"]["database"]
LOG_TABLE = cfg["paths"]["log_table"]
LOGFILE_PATH = cfg["paths"]["logfile"]

# SQLite-Verbindung & Tabelle sicherstellen
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {LOG_TABLE} (
        timestamp TEXT,
        ip TEXT,
        hostname TEXT,
        temp REAL,
        hashRate REAL,
        power REAL,
        voltage REAL,
        frequency INTEGER,
        coreVoltage INTEGER,
        fanrpm INTEGER,
        sharesAccepted INTEGER,
        sharesRejected INTEGER,
        uptime INTEGER,
        version TEXT
    )
''')
conn.commit()

def log_to_sqlite(ip, data):
    timestamp = datetime.now().isoformat()
    row = (
        timestamp,
        ip,
        data.get("hostname"),
        round(data.get("temp", 0), 1),
        round(data.get("hashRate", 0), 2),
        round(data.get("power", 0), 2),
        round(data.get("voltage", 0), 2),
        data.get("frequency"),
        data.get("coreVoltage"),
        data.get("fanrpm"),
        data.get("sharesAccepted"),
        data.get("sharesRejected"),
        data.get("uptimeSeconds"),
        data.get("version")
    )

    cursor.execute(f'''
        INSERT INTO {LOG_TABLE} (
            timestamp, ip, hostname, temp, hashRate, power, voltage,
            frequency, coreVoltage, fanrpm, sharesAccepted, sharesRejected,
            uptime, version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', row)
    conn.commit()

    write_logfile_entry("|".join(map(str, row)))

def main():
    while True:
        for ip in ips:
            data = fetch_data(ip)
            if data:
                log_to_sqlite(ip, data)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {ip} erfolgreich geloggt")
        print(f"Warte {log_interval} Sekunden...\n")
        time.sleep(log_interval)

if __name__ == "__main__":
    main()
