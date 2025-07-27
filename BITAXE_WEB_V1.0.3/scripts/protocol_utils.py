# protocol_utils.py

from config.config_loader import load_config
from datetime import datetime
import os
import shutil
import sqlite3

cfg = load_config()
LOGFILE_PATH = cfg["paths"]["logfile"]
db_path = cfg["paths"]["database"]
PROTOCOL_TABLE = cfg["paths"]["protocol_table"]
MAX_LOGFILE_SIZE_MB = 5  # Grenze z.B. 5 MB

def write_logfile_entry(entry):
    timestamp = datetime.now().isoformat()

    # 1) Prüfen, ob Datei zu groß
    if os.path.exists(LOGFILE_PATH):
        size_mb = os.path.getsize(LOGFILE_PATH) / (1024 * 1024)
        if size_mb >= MAX_LOGFILE_SIZE_MB:
            rotate_logfile()

    # 2) Anfügen
    with open(LOGFILE_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {entry}\n")

def log_event(ip, event_type, description, conn=None, cursor=None):

    close_conn = False
    if conn is None or cursor is None:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        close_conn = True

    timestamp = datetime.now().isoformat()
    hostname = ip  # oder extra Parameter, wenn du magst

    cursor.execute(f'''
        INSERT INTO {PROTOCOL_TABLE}
        (timestamp, ip, hostname, event_type, description)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, ip, hostname, event_type, description))

    if close_conn:
        conn.commit()
        conn.close()

def rotate_logfile():
    backup_path = LOGFILE_PATH + ".1"

    # Falls altes Backup existiert, löschen
    if os.path.exists(backup_path):
        os.remove(backup_path)

    # Aktuelle Datei umbenennen
    shutil.move(LOGFILE_PATH, backup_path)

    # Neue leere Logfile anlegen (optional, wird sowieso beim ersten write neu erstellt)
    with open(LOGFILE_PATH, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Logfile rotiert.\n")