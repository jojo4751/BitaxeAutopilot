import sqlite3
import pandas as pd
from config.config_loader import load_config
import json
import plotly.graph_objs as go
from collections import defaultdict
from datetime import datetime
from utils.plot_utils import make_plotly_traces

cfg = load_config()
DB_PATH = cfg["paths"]["database"]
LOG_TABLE = cfg["paths"]["log_table"]
PROTOCOL_TABLE = cfg["paths"]["protocol_table"]
IP_LIST = cfg["config"]["ips"]

def get_latest_status():
    try:
        query = f"""
            SELECT *
            FROM {LOG_TABLE}
            WHERE timestamp IN (
                SELECT MAX(timestamp) FROM {LOG_TABLE} GROUP BY ip
            )
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[get_latest_status] Fehler: {e}")
        return []

def get_event_log(limit=100):
    try:
        query = f"""
            SELECT timestamp, ip, hostname, event_type, description
            FROM {PROTOCOL_TABLE}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"[get_event_log] Fehler: {e}")
        return []

def get_history_data(start: datetime, end: datetime) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    result = {}
    for ip in IP_LIST:
        cursor.execute("""
            SELECT
                timestamp,
                ip,
                hostname,
                temp,
                hashRate,
                power,
                frequency,
                coreVoltage,
                sharesAccepted,
                sharesRejected,
                watt_per_share,
                gh_per_watt,
                shares_per_gh,
                uptime,
                shares_accepted_delta,
                shares_rejected_delta,
                shares_per_hour,
                gh_per_share
            FROM efficiency_markers
            WHERE ip = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """, (ip, start, end))
        rows = cursor.fetchall()
        # rows in Plotly-Traces umwandeln:
        traces = make_plotly_traces(rows)
        result[ip] = {'traces': traces}
    conn.close()
    return result

def get_top_settings():
    try:
        query = f"""
            SELECT ip, frequency, coreVoltage, avgEfficiency
            FROM {cfg["paths"]["tuning_table"]}
            WHERE failed = 0
            ORDER BY ip, avgEfficiency DESC
        """
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        df = pd.read_sql_query(query, conn)
        conn.close()

        print("Top-Settings (RAW):")
        print(df)

        top_per_ip = df.groupby("ip").first().reset_index()
        print("Top-Settings (GEFILTERT):")
        print(top_per_ip)

        return top_per_ip.to_dict(orient="records")

    except Exception as e:
        print(f"[get_top_settings] Fehler: {e}")
        return []
