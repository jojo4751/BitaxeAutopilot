"""
BitAxe Web Management System V2.0.0
Web route handlers for the main application
"""

import requests
import threading
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, jsonify, Response


def register_routes(app):
    """Register all web routes with the Flask application"""
    
    @app.route("/")
    @app.route("/status")
    def status():
        """Main status page showing all miners"""
        database_service = app.container.get_database_service()
        miners = database_service.get_latest_status()
        return render_template("status.html", miners=miners)

    @app.route("/dashboard")
    def dashboard():
        """Real-time dashboard with charts and metrics"""
        miners = app.container.get_database_service().get_latest_status()
        end = datetime.utcnow()
        start = end - timedelta(hours=6)
        plot_data = app.container.get_database_service().get_history_data(start=start, end=end)
        return render_template("dashboard.html", miners=miners, plot_data=plot_data)

    @app.route("/dashboard/<ip>")
    def miner_dashboard(ip):
        """Individual miner detailed dashboard"""
        config_service = app.container.get_config_service()
        database_service = app.container.get_database_service()
        
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

    @app.route("/control", methods=["GET", "POST"])
    def control():
        """Miner control interface for manual adjustments"""
        config_service = app.container.get_config_service()
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
                    message = f"Settings successfully applied: {ip} → {freq} MHz @ {volt} mV"
                    _log_event(app, ip, "MANUAL_SET", message)
                else:
                    message = f"Error: {res.status_code} when setting {ip}"
            except Exception as e:
                message = f"Connection error to {ip}: {e}"

        return render_template("control.html",
                             freq_list=freq_list,
                             volt_list=volt_list,
                             ips=ips,
                             message=message,
                             benchmark_interval=config.get("settings", {}).get("benchmark_interval_sec", 600))

    @app.route("/benchmark", methods=["POST"])
    def start_benchmark():
        """Start benchmark for a single miner"""
        ip = request.form["ip"]
        frequency = int(request.form["frequency"])
        voltage = int(request.form["voltage"])
        benchmark_time = int(request.form.get("benchmark_time", 600))

        # Use the benchmark service to start the benchmark
        success = app.container.get_benchmark_service().start_benchmark(ip, frequency, voltage, benchmark_time)
        
        if success:
            flash(f"Benchmark started for {ip} → {frequency} MHz @ {voltage} mV for {benchmark_time}s")
        else:
            flash(f"Failed to start benchmark for {ip}")
            
        return redirect(url_for("control"))

    @app.route("/benchmark-multi", methods=["POST"])
    def start_multi_benchmark():
        """Start benchmark for multiple miners"""
        selected_ips = request.form.getlist("ips")
        frequency = int(request.form["frequency"])
        voltage = int(request.form["voltage"])
        benchmark_time = int(request.form.get("benchmark_time", 600))

        started_ips = app.container.get_benchmark_service().start_multi_benchmark(
            selected_ips, frequency, voltage, benchmark_time)
        
        if started_ips:
            flash(f"Multi-benchmark started for: {', '.join(started_ips)}")
        else:
            flash("Error starting multi-benchmarks")
            
        return redirect(url_for("control"))

    @app.route("/benchmarks")
    def benchmarks():
        """Benchmark results page"""
        rows = app.container.get_database_service().get_benchmark_results(50)
        return render_template("benchmarks.html", rows=rows)

    @app.route("/events")
    def events():
        """System events and logs page"""
        events = app.container.get_database_service().get_event_log(limit=100)
        return render_template("events.html", events=events)

    @app.route("/top-settings")
    def top_settings():
        """Best performing settings page"""
        top = app.container.get_database_service().get_top_settings()
        return render_template("top_settings.html", top_settings=top)

    @app.route("/history")
    def history():
        """Historical data analysis page"""
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
            
        plot_data = app.container.get_database_service().get_history_data(start=start, end=end)
        
        return render_template("history.html",
                             miners=plot_data.keys(),
                             plot_data=plot_data,
                             selected_start=start_str,
                             selected_end=end_str)

    @app.route("/download/<ip>.csv")
    def download_csv(ip):
        """Export miner data as CSV"""
        start_str = request.args.get('start')
        end_str = request.args.get('end')
        headers, rows = app.container.get_database_service().get_efficiency_data_for_export(
            ip, start_str, end_str)

        def generate():
            yield ','.join(headers) + '\n'
            for row in rows:
                yield ','.join(str(v) if v is not None else '' for v in row) + '\n'

        return Response(generate(), mimetype='text/csv',
                       headers={"Content-Disposition": f"attachment;filename={ip}.csv"})

    @app.route("/set-benchmark-interval", methods=["POST"])
    def set_benchmark_interval():
        """Update benchmark interval setting"""
        new_interval = int(request.form["interval"])
        app.container.get_config_service().set("settings.benchmark_interval_sec", new_interval)
        flash(f"Benchmark interval changed to {new_interval} seconds.")
        return redirect(url_for("control"))

    # Simple API endpoints for AJAX calls
    @app.route("/api/status")
    def api_status():
        """API endpoint for current miner status"""
        return jsonify(app.container.get_database_service().get_latest_status())

    @app.route("/api/events")
    def api_events():
        """API endpoint for recent events"""
        return jsonify(app.container.get_database_service().get_event_log(limit=100))


def _log_event(app, ip, event_type, message):
    """Helper function to log events to database"""
    try:
        # Use the database service to log events
        app.container.get_database_service().log_event(ip, event_type, message)
    except Exception as e:
        print(f"[{event_type}] {ip}: {message} (logging failed: {e})")