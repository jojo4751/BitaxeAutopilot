import os
import json
import requests
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response

from services.service_container import get_container
from utils.plot_utils import make_plotly_traces
from utils.rate_limiter import init_rate_limiting
from api.swagger_config import setup_api_documentation
from api.v1.blueprints import register_api_blueprint

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback-secret-key")

# Setup API documentation and endpoints
setup_api_documentation(app)
register_api_blueprint(app)

# Initialize rate limiting
init_rate_limiting(app)

# Initialize services
container = get_container()
config_service = container.get_config_service()
database_service = container.get_database_service()
miner_service = container.get_miner_service()
benchmark_service = container.get_benchmark_service()
autopilot_service = container.get_autopilot_service()

# Helper functions
def log_event(ip, event_type, message):
    """Log event to database - stub implementation"""
    print(f"[{event_type}] {ip}: {message}")

def run_benchmark(ip, voltage, frequency, benchmark_time):
    """Run benchmark - stub implementation"""
    print(f"Running benchmark for {ip}: {frequency}MHz @ {voltage}mV for {benchmark_time}s")

# ----------------------------
# DASHBOARD + DETAILS
# ----------------------------

@app.route("/")
@app.route("/status")
def status():
    miners = database_service.get_latest_status()
    return render_template("status.html", miners=miners)

@app.route("/dashboard")
def dashboard():
    miners = database_service.get_latest_status()
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = database_service.get_history_data(start=start, end=end)
    return render_template("dashboard.html", miners=miners, plot_data=plot_data)

@app.route("/dashboard/<ip>")
def miner_dashboard(ip):
    miners = config_service.ips
    color = config_service.get_miner_color(ip)
    status = next((m for m in database_service.get_latest_status() if m["ip"] == ip), None)
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = database_service.get_history_data(start=start, end=end).get(ip, {"traces": []})
    benchmarks = database_service.get_benchmark_results_for_ip(ip, 10)

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
    config = config_service.get_config()
    freq_list = config.get("settings", {}).get("freq_list", [])
    volt_list = config.get("settings", {}).get("volt_list", [])
    ips = config.get("config", {}).get("ips", [])
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
                           benchmark_interval=config.get("settings", {}).get("benchmark_interval_sec", 600))

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

    started_ips = benchmark_service.start_multi_benchmark(selected_ips, frequency, voltage, benchmark_time)
    
    if started_ips:
        flash(f"Multi-Benchmark gestartet für: {', '.join(started_ips)}")
    else:
        flash("Fehler beim Starten der Multi-Benchmarks")
    return redirect(url_for("control"))

# ----------------------------
# BENCHMARK-ERGENISSE + TOOLS
# ----------------------------

@app.route("/benchmarks")
def benchmarks():
    rows = database_service.get_benchmark_results(50)
    return render_template("benchmarks.html", rows=rows)

@app.route("/download/<ip>.csv")
def download_csv(ip):
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    headers, rows = database_service.get_efficiency_data_for_export(ip, start_str, end_str)

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
    events = database_service.get_event_log(limit=100)
    return render_template("events.html", events=events)

@app.route("/top-settings")
def top_settings():
    top = database_service.get_top_settings()
    return render_template("top_settings.html", top_settings=top)

@app.route("/api/status")
def api_status():
    return jsonify(database_service.get_latest_status())

@app.route("/api/events")
def api_events():
    return jsonify(database_service.get_event_log(limit=100))

@app.route("/set-benchmark-interval", methods=["POST"])
def set_benchmark_interval():
    new_interval = int(request.form["interval"])
    config_service.set("settings.benchmark_interval_sec", new_interval)
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
    plot_data = database_service.get_history_data(start=start, end=end)
    return render_template("history.html",
                           miners=plot_data.keys(),
                           plot_data=plot_data,
                           selected_start=start_str,
                           selected_end=end_str)

@app.teardown_appcontext
def shutdown_services(error):
    """Cleanup services on app shutdown"""
    pass

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        container.shutdown()
