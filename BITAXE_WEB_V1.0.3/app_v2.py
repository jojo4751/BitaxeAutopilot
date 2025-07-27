import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response

from logging.structured_logger import configure_logging, get_logger
from services.service_container_v2 import get_container_v2
from utils.error_handlers import register_flask_error_handlers, route_error_boundary, api_error_boundary
from health.health_checks import get_health_manager, initialize_health_checks
from utils.plot_utils import make_plotly_traces

# Configure structured logging
configure_logging(
    service_name="bitaxe-web",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "data/logs/app.log")
)

logger = get_logger("bitaxe.app")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fallback-secret-key")

# Initialize services
container = get_container_v2()
config_service = container.get_config_service()
database_service = container.get_database_service()
miner_service = container.get_miner_service()
benchmark_service = container.get_benchmark_service()
autopilot_service = container.get_autopilot_service()

# Register error handlers
register_flask_error_handlers(app)

# Initialize health checks
health_manager = initialize_health_checks(container)

logger.info("BITAXE Web Application starting up",
           service_version="1.0.3",
           total_miners=len(config_service.ips))


# ----------------------------
# HEALTH CHECK ENDPOINTS
# ----------------------------

@app.route("/health")
@api_error_boundary
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "bitaxe-web",
        "version": "1.0.3"
    })


@app.route("/health/detailed")
@api_error_boundary
def detailed_health_check():
    """Detailed health check with component status"""
    health_status = health_manager.get_overall_health()
    return jsonify(health_status)


@app.route("/health/<component>")
@api_error_boundary
def component_health_check(component):
    """Health check for specific component"""
    health_status = health_manager.get_health_status(component)
    return jsonify(health_status)


# ----------------------------
# DASHBOARD + DETAILS
# ----------------------------

@app.route("/")
@app.route("/status")
@route_error_boundary(fallback_template="error.html")
def status():
    """Main status page with error boundary"""
    logger.info("Status page requested")
    miners = database_service.get_latest_status()
    
    logger.debug("Status data retrieved",
                miners_count=len(miners))
    
    return render_template("status.html", miners=miners)


@app.route("/dashboard")
@route_error_boundary(fallback_template="error.html")
def dashboard():
    """Dashboard page with charts"""
    logger.info("Dashboard page requested")
    
    miners = database_service.get_latest_status()
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = database_service.get_history_data(start=start, end=end)
    
    logger.debug("Dashboard data retrieved",
                miners_count=len(miners),
                plot_data_keys=list(plot_data.keys()),
                time_range_hours=6)
    
    return render_template("dashboard.html", miners=miners, plot_data=plot_data)


@app.route("/dashboard/<ip>")
@route_error_boundary(fallback_template="error.html")
def miner_dashboard(ip):
    """Individual miner dashboard"""
    logger.info("Miner dashboard requested", miner_ip=ip)
    
    miners = config_service.ips
    color = config_service.get_miner_color(ip)
    status = next((m for m in database_service.get_latest_status() if m["ip"] == ip), None)
    
    if not status:
        logger.warning("Miner not found in latest status", miner_ip=ip)
    
    end = datetime.utcnow()
    start = end - timedelta(hours=6)
    plot_data = database_service.get_history_data(start=start, end=end).get(ip, {"traces": []})
    benchmarks = database_service.get_benchmark_results_for_ip(ip, 10)
    
    logger.debug("Miner dashboard data retrieved",
                miner_ip=ip,
                has_status=status is not None,
                plot_traces_count=len(plot_data.get("traces", [])),
                benchmarks_count=len(benchmarks))
    
    return render_template("miner_dashboard.html",
                           miner=status,
                           ip=ip,
                           miners=miners,
                           color=color,
                           plot_data=plot_data,
                           benchmarks=benchmarks)


# ----------------------------
# MINER CONTROL
# ----------------------------

@app.route("/control", methods=["GET", "POST"])
@route_error_boundary(fallback_template="error.html")
def control():
    """Miner control page"""
    freq_list = config_service.freq_list
    volt_list = config_service.volt_list
    ips = config_service.ips
    message = ""

    if request.method == "POST":
        ip = request.form["ip"]
        freq = int(request.form["frequency"])
        volt = int(request.form["voltage"])
        
        logger.info("Manual miner settings requested",
                   miner_ip=ip,
                   frequency=freq,
                   voltage=volt)
        
        try:
            success = miner_service.set_miner_settings(ip, freq, volt)
            if success:
                message = f"Settings successfully applied: {ip} → {freq} MHz @ {volt} mV"
                logger.info("Manual settings applied successfully",
                           miner_ip=ip,
                           frequency=freq,
                           voltage=volt)
            else:
                message = f"Failed to apply settings for {ip}"
                logger.warning("Manual settings application failed",
                             miner_ip=ip,
                             frequency=freq,
                             voltage=volt)
        except Exception as e:
            message = f"Error applying settings for {ip}: {str(e)}"
            logger.error("Manual settings application error",
                        miner_ip=ip,
                        frequency=freq,
                        voltage=volt,
                        error=str(e))

    return render_template("control.html",
                           freq_list=freq_list,
                           volt_list=volt_list,
                           ips=ips,
                           message=message,
                           benchmark_interval=config_service.benchmark_interval)


# ----------------------------
# BENCHMARK ENDPOINTS
# ----------------------------

@app.route("/benchmark", methods=["POST"])
@route_error_boundary(fallback_template="error.html")
def start_benchmark():
    """Start single miner benchmark"""
    ip = request.form["ip"]
    frequency = int(request.form["frequency"])
    voltage = int(request.form["voltage"])
    benchmark_time = int(request.form.get("benchmark_time", 600))

    logger.info("Benchmark start requested",
               miner_ip=ip,
               frequency=frequency,
               voltage=voltage,
               duration=benchmark_time)

    success = benchmark_service.start_benchmark(ip, frequency, voltage, benchmark_time)
    if success:
        flash(f"Benchmark started for {ip} → {frequency} MHz @ {voltage} mV for {benchmark_time}s")
        logger.info("Benchmark started successfully",
                   miner_ip=ip,
                   frequency=frequency,
                   voltage=voltage,
                   duration=benchmark_time)
    else:
        flash(f"Failed to start benchmark for {ip}")
        logger.error("Benchmark start failed",
                    miner_ip=ip,
                    frequency=frequency,
                    voltage=voltage)
    
    return redirect(url_for("control"))


@app.route("/benchmark-multi", methods=["POST"])
@route_error_boundary(fallback_template="error.html")
def start_multi_benchmark():
    """Start multi-miner benchmark"""
    selected_ips = request.form.getlist("ips")
    frequency = int(request.form["frequency"])
    voltage = int(request.form["voltage"])
    benchmark_time = int(request.form.get("benchmark_time", 600))

    logger.info("Multi-benchmark start requested",
               miner_ips=selected_ips,
               frequency=frequency,
               voltage=voltage,
               duration=benchmark_time)

    started_ips = benchmark_service.start_multi_benchmark(selected_ips, frequency, voltage, benchmark_time)
    
    if started_ips:
        flash(f"Multi-benchmark started for: {', '.join(started_ips)}")
        logger.info("Multi-benchmark started successfully",
                   requested_ips=selected_ips,
                   started_ips=started_ips,
                   frequency=frequency,
                   voltage=voltage)
    else:
        flash("Failed to start multi-benchmarks")
        logger.error("Multi-benchmark start failed",
                    requested_ips=selected_ips,
                    frequency=frequency,
                    voltage=voltage)
    
    return redirect(url_for("control"))


@app.route("/benchmarks")
@route_error_boundary(fallback_template="error.html")
def benchmarks():
    """Benchmark results page"""
    logger.info("Benchmark results page requested")
    
    rows = database_service.get_benchmark_results(50)
    
    logger.debug("Benchmark results retrieved",
                results_count=len(rows))
    
    return render_template("benchmarks.html", rows=rows)


# ----------------------------
# DATA EXPORT
# ----------------------------

@app.route("/download/<ip>.csv")
@route_error_boundary(return_json=True)
def download_csv(ip):
    """Export miner data as CSV"""
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    
    logger.info("CSV export requested",
               miner_ip=ip,
               start_time=start_str,
               end_time=end_str)
    
    headers, rows = database_service.get_efficiency_data_for_export(ip, start_str, end_str)

    def generate():
        yield ','.join(headers) + '\n'
        for row in rows:
            yield ','.join(str(v) if v is not None else '' for v in row) + '\n'

    logger.info("CSV export generated",
               miner_ip=ip,
               rows_exported=len(rows))

    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment;filename={ip}.csv"})


# ----------------------------
# SYSTEM PAGES
# ----------------------------

@app.route("/events")
@route_error_boundary(fallback_template="error.html")
def events():
    """System events page"""
    logger.info("Events page requested")
    
    events = database_service.get_event_log(limit=100)
    
    logger.debug("Events retrieved",
                events_count=len(events))
    
    return render_template("events.html", events=events)


@app.route("/top-settings")
@route_error_boundary(fallback_template="error.html")
def top_settings():
    """Top performing settings page"""
    logger.info("Top settings page requested")
    
    top = database_service.get_top_settings()
    
    logger.debug("Top settings retrieved",
                settings_count=len(top))
    
    return render_template("top_settings.html", top_settings=top)


@app.route('/history')
@route_error_boundary(fallback_template="error.html")
def history():
    """Historical data page"""
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
    
    logger.info("History page requested",
               start_time=start_str,
               end_time=end_str)
    
    plot_data = database_service.get_history_data(start=start, end=end)
    
    logger.debug("History data retrieved",
                miners_count=len(plot_data),
                time_range_hours=(end - start).total_seconds() / 3600)
    
    return render_template("history.html",
                           miners=plot_data.keys(),
                           plot_data=plot_data,
                           selected_start=start_str,
                           selected_end=end_str)


# ----------------------------
# API ENDPOINTS
# ----------------------------

@app.route("/api/status")
@api_error_boundary
def api_status():
    """API endpoint for miner status"""
    logger.debug("API status endpoint requested")
    return jsonify(database_service.get_latest_status())


@app.route("/api/events")
@api_error_boundary
def api_events():
    """API endpoint for events"""
    limit = request.args.get('limit', 100, type=int)
    logger.debug("API events endpoint requested", limit=limit)
    return jsonify(database_service.get_event_log(limit=limit))


@app.route("/api/miners/summary")
@api_error_boundary
def api_miners_summary():
    """API endpoint for miners summary"""
    logger.debug("API miners summary requested")
    summary = miner_service.get_miners_summary()
    return jsonify(summary)


@app.route("/api/benchmarks/status")
@api_error_boundary
def api_benchmark_status():
    """API endpoint for benchmark status"""
    logger.debug("API benchmark status requested")
    status = benchmark_service.get_benchmark_status()
    return jsonify(status)


@app.route("/api/autopilot/status")
@api_error_boundary
def api_autopilot_status():
    """API endpoint for autopilot status"""
    logger.debug("API autopilot status requested")
    status = autopilot_service.get_autopilot_status()
    return jsonify(status)


# ----------------------------
# CONFIGURATION ENDPOINTS
# ----------------------------

@app.route("/set-benchmark-interval", methods=["POST"])
@route_error_boundary(fallback_template="error.html")
def set_benchmark_interval():
    """Update benchmark interval"""
    new_interval = int(request.form["interval"])
    
    logger.info("Benchmark interval update requested",
               new_interval=new_interval)
    
    config_service.set("settings.benchmark_interval_sec", new_interval)
    flash(f"Benchmark interval changed to {new_interval} seconds.")
    
    logger.info("Benchmark interval updated successfully",
               new_interval=new_interval)
    
    return redirect(url_for("control"))


# ----------------------------
# APPLICATION LIFECYCLE
# ----------------------------

@app.before_first_request
def initialize_app():
    """Initialize application components"""
    logger.info("Initializing BITAXE Web Application",
               miners_configured=len(config_service.ips),
               database_path=config_service.database_path)


@app.teardown_appcontext
def cleanup_request(error):
    """Cleanup after each request"""
    if error:
        logger.error("Request completed with error",
                    error=str(error),
                    error_type=type(error).__name__)


@app.route("/shutdown", methods=["POST"])
@api_error_boundary
def shutdown():
    """Graceful shutdown endpoint"""
    logger.info("Shutdown requested")
    
    # Stop health monitoring
    health_manager.stop_monitoring()
    
    # Shutdown services
    container.shutdown()
    
    logger.info("Application shutdown completed")
    
    return jsonify({"message": "Application shutdown initiated"})


if __name__ == "__main__":
    try:
        logger.info("Starting BITAXE Web Application",
                   host="127.0.0.1",
                   port=5000,
                   debug=True)
        
        app.run(host="127.0.0.1", port=5000, debug=True)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception("Application startup failed", error=str(e))
    finally:
        logger.info("Cleaning up resources")
        health_manager.stop_monitoring()
        container.shutdown()