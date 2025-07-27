import sqlite3
import os
import json
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import requests

from config.config_loader import load_config
from utils.db_access import get_latest_status, get_event_log, get_history_data, get_top_settings
from utils.plot_utils import make_plotly_traces
from scripts.protocol_utils import log_event, write_logfile_entry
from scripts.benchmark_runner import run_benchmark

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback-secret-key")

cfg = load_config()

# ----------------------------
# DASHBOARD + DETAILS
# ----------------------------

@app.route("/")
@app.route("/status")
def status():
    miners = get_latest_status()
    return render_template("status.html", miners=miners)

@app.route("/dashboard")
def dashboard():
    miners = get_latest_status()
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = get_history_data(start=start, end=end)
    return render_template("dashboard.html", miners=miners, plot_data=plot_data)

@app.route("/dashboard/<ip>")
def miner_dashboard(ip):
    miners = cfg["config"]["ips"]
    color = cfg["visual"]["colors"].get(ip, "#3498db")
    status = next((m for m in get_latest_status() if m["ip"] == ip), None)
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = get_history_data(start=start, end=end).get(ip, {"traces": []})

    conn = sqlite3.connect(cfg["paths"]["database"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, timestamp
        FROM benchmark_results
        WHERE ip = ?
        ORDER BY timestamp DESC LIMIT 10
    """, (ip,))
    benchmarks = cursor.fetchall()
    conn.close()

    return render_template("miner_dashboard.html",
                           miner=status,
                           ip=ip,
                           miners=miners,
                           color=color,
                           plot_data=plot_data,
                           benchmarks=benchmarks)

# ----------------------------
# MANUELLE STEUERUNG
# ----------------------------

@app.route("/control", methods=["GET", "POST"])
def control():
    freq_list = cfg["settings"]["freq_list"]
    volt_list = cfg["settings"]["volt_list"]
    ips = cfg["config"]["ips"]
    message = ""

    if request.method == "POST":
        ip = request.form["ip"]
        freq = int(request.form["frequency"])
        volt = int(request.form["voltage"])
        try:
            payload = {"frequency": freq, "coreVoltage": volt, "autofanspeed": True}
            res = requests.patch(f"http://{ip}/api/system", json=payload, timeout=5)
            if res.ok:
                message = f"Settings erfolgreich gesetzt: {ip} → {freq} MHz @ {volt} mV"
                log_event(ip, "MANUAL_SET", message)
            else:
                message = f"Fehler: {res.status_code} beim Setzen für {ip}"
        except Exception as e:
            message = f"Fehler bei Verbindung zu {ip}: {e}"

    return render_template("control.html",
                           freq_list=freq_list,
                           volt_list=volt_list,
                           ips=ips,
                           message=message,
                           benchmark_interval=cfg["settings"]["benchmark_interval_sec"])

# ----------------------------
# BENCHMARKS STARTEN
# ----------------------------

@app.route("/benchmark", methods=["POST"])
def start_benchmark():
    ip = request.form["ip"]
    frequency = int(request.form["frequency"])
    voltage = int(request.form["voltage"])
    benchmark_time = int(request.form.get("benchmark_time", 600))

    threading.Thread(target=run_benchmark, args=(ip, voltage, frequency, benchmark_time)).start()
    flash(f"Benchmark gestartet für {ip} → {frequency} MHz @ {voltage} mV für {benchmark_time}s")
    return redirect(url_for("control"))

@app.route("/benchmark-multi", methods=["POST"])
def start_multi_benchmark():
    selected_ips = request.form.getlist("ips")
    frequency = int(request.form["frequency"])
    voltage = int(request.form["voltage"])
    benchmark_time = int(request.form.get("benchmark_time", 600))

    for ip in selected_ips:
        threading.Thread(target=run_benchmark, args=(ip, voltage, frequency, benchmark_time)).start()
        log_event(ip, "MULTI_BENCHMARK_STARTED", f"{frequency} MHz @ {voltage} mV für {benchmark_time}s")

    flash(f"Multi-Benchmark gestartet für: {', '.join(selected_ips)}")
    return redirect(url_for("control"))

# ----------------------------
# BENCHMARK-ERGENISSE + TOOLS
# ----------------------------

@app.route("/benchmarks")
def benchmarks():
    conn = sqlite3.connect(cfg["paths"]["database"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ip, frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, timestamp, duration
        FROM benchmark_results
        ORDER BY averageHashRate DESC LIMIT 50
    """)
    rows = cursor.fetchall()
    conn.close()
    return render_template("benchmarks.html", rows=rows)

@app.route("/download/<ip>.csv")
def download_csv(ip):
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    conn = sqlite3.connect(cfg["paths"]["database"])
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM efficiency_markers
        WHERE ip = ?
        AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    """, (ip, start_str, end_str))
    rows = cursor.fetchall()
    headers = [d[0] for d in cursor.description]
    conn.close()

    def generate():
        yield ','.join(headers) + '\n'
        for row in rows:
            yield ','.join(str(v) if v is not None else '' for v in row) + '\n'

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename={ip}.csv"})

# ----------------------------
# EVENTS, TOP SETTINGS & API
# ----------------------------

@app.route("/events")
def events():
    events = get_event_log(limit=100)
    return render_template("events.html", events=events)

@app.route("/top-settings")
def top_settings():
    top = get_top_settings()
    return render_template("top_settings.html", top_settings=top)

@app.route("/api/status")
def api_status():
    return jsonify(get_latest_status())

@app.route("/api/events")
def api_events():
    return jsonify(get_event_log(limit=100))

@app.route("/set-benchmark-interval", methods=["POST"])
def set_benchmark_interval():
    new_interval = int(request.form["interval"])
    with open(cfg["paths"]["config"], "r") as f:
        data = json.load(f)
    data["settings"]["benchmark_interval_sec"] = new_interval
    with open(cfg["paths"]["config"], "w") as f:
        json.dump(data, f, indent=2)
    flash(f"Benchmark-Intervall auf {new_interval} Sekunden geändert.")
    return redirect(url_for("control"))

@app.route('/history')
def history():
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    if start_str and end_str:
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str)
    else:
        end = datetime.utcnow()
        start = end - timedelta(hours=48)
        start_str = start.isoformat(timespec='minutes')
        end_str = end.isoformat(timespec='minutes')
    plot_data = get_history_data(start=start, end=end)
    return render_template("history.html",
                           miners=plot_data.keys(),
                           plot_data=plot_data,
                           selected_start=start_str,
                           selected_end=end_str)

if __name__ == "__main__":
    app.run(debug=True)
