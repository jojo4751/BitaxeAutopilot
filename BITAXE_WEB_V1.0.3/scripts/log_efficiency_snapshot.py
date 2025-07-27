from datetime import datetime
from config.config_loader import load_config

cfg = load_config()

def log_efficiency_snapshot(ip, data):
    import sqlite3
    from scripts.protocol_utils import write_logfile_entry

    db_path = cfg["paths"]["database"]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        timestamp = datetime.now().isoformat()
        hostname = data.get("hostname", ip)
        temp = data.get("temp", 0)
        hashRate = data.get("hashRate", 0)
        power = data.get("power", 0)
        sharesAccepted = data.get("sharesAccepted", 0)
        sharesRejected = data.get("sharesRejected", 0)
        freq = data.get("frequency", 0)
        volt = data.get("coreVoltage", 0)
        uptime = data.get("uptime", 0)

        gh_per_watt = hashRate / power if power > 0 else None
        watt_per_share = power / sharesAccepted if sharesAccepted > 0 else None
        shares_per_gh = sharesAccepted / hashRate if hashRate > 0 else None

        row = (
            timestamp, ip, hostname, temp, hashRate, power,
            sharesAccepted, sharesRejected, freq, volt,
            watt_per_share, gh_per_watt, shares_per_gh, uptime
        )

        cursor.execute("""
            INSERT INTO efficiency_markers (
                timestamp, ip, hostname, temp, hashRate, power,
                sharesAccepted, sharesRejected, frequency, coreVoltage,
                watt_per_share, gh_per_watt, shares_per_gh, uptime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, row)

        conn.commit()
    except Exception as e:
        write_logfile_entry(f"[{ip}] Fehler beim Schreiben efficiency_markers: {e}")
    finally:
        conn.close()
