"""
BitAxe Web Management System V2.0.0
Web route handlers for the main application
"""

import requests
import threading
import logging
from datetime import datetime, timedelta
from flask import render_template, request, redirect, url_for, flash, jsonify, Response

logger = logging.getLogger(__name__)


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
    
    @app.route("/analytics")
    @app.route("/realtime")
    def realtime_dashboard():
        """Advanced real-time analytics dashboard"""
        try:
            # Get initial data for the dashboard
            database_service = app.container.get_database_service()
            analytics_service = app.container.get_analytics_service()
            
            # Get latest miner status
            latest_miners = database_service.get_latest_status()
            
            # Get active alerts
            active_alerts = database_service.get_active_alerts()
            
            # Get system stats
            system_stats = {}
            try:
                system_stats = database_service.get_database_stats()
            except Exception as e:
                logger.warning(f"Could not get system stats: {e}")
            
            # Get fleet profitability
            profitability_data = {}
            try:
                recent_time = datetime.utcnow() - timedelta(hours=1)
                profitability_records = database_service.get_profitability_data(start_time=recent_time)
                
                if profitability_records:
                    total_profit = sum(r.get('estimated_daily_profit_usd', 0) for r in profitability_records)
                    total_revenue = sum(r.get('estimated_daily_usd', 0) for r in profitability_records)
                    
                    profitability_data = {
                        'total_daily_profit': total_profit,
                        'total_daily_revenue': total_revenue,
                        'active_miners': len(set(r.get('ip') for r in profitability_records))
                    }
            except Exception as e:
                logger.warning(f"Could not get profitability data: {e}")
            
            return render_template("realtime_dashboard.html", 
                                 miners=latest_miners,
                                 alerts=active_alerts,
                                 system_stats=system_stats,
                                 profitability=profitability_data)
        except Exception as e:
            logger.error(f"Error loading realtime dashboard: {e}")
            flash(f"Error loading dashboard: {e}", "error")
            return redirect(url_for("status"))

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
    
    # Advanced Analytics Routes
    @app.route("/analytics/statistical")
    def statistical_analysis():
        """Statistical analysis dashboard"""
        try:
            statistical_service = app.container.get_statistical_service()
            config_service = app.container.get_config_service()
            
            # Get all miner IPs
            miner_ips = config_service.ips
            
            # Get time range from query parameters
            hours = int(request.args.get('hours', 24))
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Generate performance profiles for all miners
            profiles = {}
            for ip in miner_ips:
                profile = statistical_service.analyze_performance_profile(ip, start_time, end_time)
                if profile:
                    profiles[ip] = profile.to_dict()
            
            # Compare miners if we have multiple profiles
            comparison_data = {}
            if len(profiles) > 1:
                comparison_data = statistical_service.compare_miners(list(profiles.keys()), start_time, end_time)
            
            return render_template("analytics/statistical.html",
                                 profiles=profiles,
                                 comparison_data=comparison_data,
                                 analysis_period={'start': start_time.isoformat(), 'end': end_time.isoformat(), 'hours': hours})
                                 
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            flash(f"Error loading statistical analysis: {e}", "error")
            return redirect(url_for("status"))
    
    @app.route("/analytics/predictive")
    def predictive_analysis():
        """Predictive analysis dashboard"""
        try:
            predictive_service = app.container.get_predictive_service()
            config_service = app.container.get_config_service()
            
            # Get all miner IPs
            miner_ips = config_service.ips
            
            # Get predictions for all miners
            predictions = {}
            forecasts = {}
            
            for ip in miner_ips:
                # Get failure prediction
                failure_prediction = predictive_service.predict_next_failure(ip)
                if 'error' not in failure_prediction:
                    predictions[ip] = failure_prediction
                
                # Get metric forecasts
                forecasts[ip] = {}
                for metric in ['hashRate', 'temp', 'power', 'efficiency']:
                    forecast = predictive_service.forecast_metric(ip, metric, 24)  # 24 hour forecast
                    if forecast.model_accuracy > 0:
                        forecasts[ip][metric] = {
                            'timestamps': [ts.isoformat() for ts in forecast.timestamps],
                            'values': forecast.values,
                            'confidence_intervals': forecast.confidence_intervals,
                            'model_accuracy': forecast.model_accuracy,
                            'assumptions': forecast.assumptions
                        }
            
            return render_template("analytics/predictive.html",
                                 predictions=predictions,
                                 forecasts=forecasts)
                                 
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            flash(f"Error loading predictive analysis: {e}", "error")
            return redirect(url_for("status"))
    
    @app.route("/analytics/reports")
    def reports_dashboard():
        """Reports dashboard"""
        try:
            reporting_service = app.container.get_reporting_service()
            
            # Get available reports
            reports = reporting_service.get_available_reports()
            
            # Get recent reports
            recent_reports = []
            for report_id in reports.get('recent_reports', []):
                report = reporting_service.get_report(report_id)
                if report:
                    recent_reports.append(report)
            
            return render_template("analytics/reports.html",
                                 reports=reports,
                                 recent_reports=recent_reports)
                                 
        except Exception as e:
            logger.error(f"Error in reports dashboard: {e}")
            flash(f"Error loading reports dashboard: {e}", "error")
            return redirect(url_for("status"))
    
    @app.route("/analytics/fleet-comparison")
    def fleet_comparison():
        """Fleet comparison analysis"""
        try:
            statistical_service = app.container.get_statistical_service()
            config_service = app.container.get_config_service()
            
            # Get time range from query parameters
            hours = int(request.args.get('hours', 168))  # Default 7 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Get all miner IPs
            miner_ips = config_service.ips
            
            # Perform fleet comparison
            comparison_data = statistical_service.compare_miners(miner_ips, start_time, end_time)
            
            return render_template("analytics/fleet_comparison.html",
                                 comparison_data=comparison_data,
                                 analysis_period={'start': start_time.isoformat(), 'end': end_time.isoformat(), 'hours': hours})
                                 
        except Exception as e:
            logger.error(f"Error in fleet comparison: {e}")
            flash(f"Error loading fleet comparison: {e}", "error")
            return redirect(url_for("status"))
    
    # API endpoints for analytics
    @app.route("/api/analytics/performance-profile/<ip>")
    def api_performance_profile(ip):
        """API endpoint for miner performance profile"""
        try:
            statistical_service = app.container.get_statistical_service()
            
            # Get time range from query parameters
            hours = int(request.args.get('hours', 24))
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            profile = statistical_service.analyze_performance_profile(ip, start_time, end_time)
            
            if profile:
                return jsonify(profile.to_dict())
            else:
                return jsonify({'error': 'No profile data available'}), 404
                
        except Exception as e:
            logger.error(f"Error getting performance profile for {ip}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route("/api/analytics/predict-failure/<ip>")
    def api_predict_failure(ip):
        """API endpoint for failure prediction"""
        try:
            predictive_service = app.container.get_predictive_service()
            prediction = predictive_service.predict_next_failure(ip)
            return jsonify(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting failure for {ip}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route("/api/analytics/forecast/<ip>/<metric>")
    def api_forecast_metric(ip, metric):
        """API endpoint for metric forecasting"""
        try:
            predictive_service = app.container.get_predictive_service()
            hours = int(request.args.get('hours', 24))
            
            forecast = predictive_service.forecast_metric(ip, metric, hours)
            
            return jsonify({
                'metric': forecast.metric,
                'timestamps': [ts.isoformat() for ts in forecast.timestamps],
                'values': forecast.values,
                'confidence_intervals': forecast.confidence_intervals,
                'model_accuracy': forecast.model_accuracy,
                'assumptions': forecast.assumptions
            })
            
        except Exception as e:
            logger.error(f"Error forecasting {metric} for {ip}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route("/api/analytics/generate-report", methods=['POST'])
    def api_generate_report():
        """API endpoint to generate a new report"""
        try:
            reporting_service = app.container.get_reporting_service()
            
            data = request.get_json()
            report_type = data.get('type', 'daily')
            
            if report_type == 'daily':
                report = reporting_service.generate_daily_report()
            elif report_type == 'weekly':
                report = reporting_service.generate_weekly_report()
            elif report_type == 'performance':
                ip = data.get('ip')
                if not ip:
                    return jsonify({'error': 'IP address required for performance report'}), 400
                report = reporting_service.generate_performance_report(ip)
            else:
                return jsonify({'error': f'Unknown report type: {report_type}'}), 400
            
            return jsonify(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return jsonify({'error': str(e)}), 500


def _log_event(app, ip, event_type, message):
    """Helper function to log events to database"""
    try:
        # Use the database service to log events
        app.container.get_database_service().log_event(ip, event_type, message)
    except Exception as e:
        print(f"[{event_type}] {ip}: {message} (logging failed: {e})")