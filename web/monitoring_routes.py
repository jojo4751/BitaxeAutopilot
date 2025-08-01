#!/usr/bin/env python3
"""
BitAxe V2.0.0 - Monitoring Dashboard Routes
Provides web interface routes for system monitoring, health status, and metrics visualization
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from flask import Blueprint, render_template, jsonify, request
from pathlib import Path

# Create blueprint for monitoring routes
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/monitoring')

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_FILE = BASE_DIR / "config" / "monitoring.conf"
DATABASE_PATH = "/opt/bitaxe-web/data/bitaxe_data.db"
LOG_DIR = "/var/log/bitaxe"

def load_monitoring_config():
    """Load monitoring configuration"""
    config = {}
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key] = value.strip('"')
        return config
    except Exception as e:
        print(f"Error loading monitoring config: {e}")
        return {}

def get_database_connection():
    """Get database connection"""
    try:
        return sqlite3.connect(DATABASE_PATH)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def execute_query(query, params=None):
    """Execute database query safely"""
    conn = get_database_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"Query execution error: {e}")
        if conn:
            conn.close()
        return []

# Routes
@monitoring_bp.route('/')
def monitoring_dashboard():
    """Main monitoring dashboard"""
    config = load_monitoring_config()
    
    # Get latest system metrics
    metrics_query = """
    SELECT metric_type, metric_name, metric_value, metric_unit, timestamp
    FROM system_metrics 
    WHERE timestamp > datetime('now', '-5 minutes')
    ORDER BY timestamp DESC
    """
    
    metrics_data = execute_query(metrics_query)
    
    # Organize metrics by type
    metrics = {
        'system': {},
        'application': {},
        'database': {},
        'mining': {}
    }
    
    for metric_type, metric_name, metric_value, metric_unit, timestamp in metrics_data:
        if metric_type not in metrics:
            metrics[metric_type] = {}
        
        metrics[metric_type][metric_name] = {
            'value': metric_value,
            'unit': metric_unit,
            'timestamp': timestamp
        }
    
    # Get service status
    service_status = get_service_status()
    
    # Get system health
    health_status = get_health_status()
    
    return render_template('monitoring/dashboard.html',
                         metrics=metrics,
                         services=service_status,
                         health=health_status,
                         config=config)

@monitoring_bp.route('/health')
def health_status():
    """System health status page"""
    health_data = get_health_status()
    return render_template('monitoring/health.html', health=health_data)

@monitoring_bp.route('/metrics')
def metrics_page():
    """Detailed metrics visualization page"""
    # Get time range from request
    hours = request.args.get('hours', 24, type=int)
    
    # Query metrics for the specified time range
    metrics_query = """
    SELECT metric_type, metric_name, metric_value, metric_unit, timestamp
    FROM system_metrics 
    WHERE timestamp > datetime('now', '-{} hours')
    ORDER BY timestamp ASC
    """.format(hours)
    
    metrics_data = execute_query(metrics_query)
    
    # Organize data for charts
    chart_data = {}
    for metric_type, metric_name, metric_value, metric_unit, timestamp in metrics_data:
        key = f"{metric_type}.{metric_name}"
        if key not in chart_data:
            chart_data[key] = {
                'name': metric_name.replace('_', ' ').title(),
                'unit': metric_unit,
                'type': metric_type,
                'timestamps': [],
                'values': []
            }
        
        chart_data[key]['timestamps'].append(timestamp)
        chart_data[key]['values'].append(metric_value)
    
    return render_template('monitoring/metrics.html', 
                         chart_data=chart_data, 
                         hours=hours)

@monitoring_bp.route('/logs')
def logs_page():
    """System logs viewer"""
    # Get recent log entries from various log files
    log_files = get_log_files()
    return render_template('monitoring/logs.html', log_files=log_files)

@monitoring_bp.route('/backups')
def backups_page():
    """Backup status and history"""
    backup_status = get_backup_status()
    return render_template('monitoring/backups.html', backups=backup_status)

# API Endpoints
@monitoring_bp.route('/api/health')
def api_health():
    """API endpoint for health status"""
    health_data = get_health_status()
    return jsonify(health_data)

@monitoring_bp.route('/api/metrics')
def api_metrics():
    """API endpoint for current metrics"""
    hours = request.args.get('hours', 1, type=int)
    
    metrics_query = """
    SELECT metric_type, metric_name, metric_value, metric_unit, timestamp
    FROM system_metrics 
    WHERE timestamp > datetime('now', '-{} hours')
    ORDER BY timestamp DESC
    """.format(hours)
    
    metrics_data = execute_query(metrics_query)
    
    result = []
    for metric_type, metric_name, metric_value, metric_unit, timestamp in metrics_data:
        result.append({
            'type': metric_type,
            'name': metric_name,
            'value': metric_value,
            'unit': metric_unit,
            'timestamp': timestamp
        })
    
    return jsonify(result)

@monitoring_bp.route('/api/services')
def api_services():
    """API endpoint for service status"""
    service_status = get_service_status()
    return jsonify(service_status)

@monitoring_bp.route('/api/logs/<log_type>')
def api_logs(log_type):
    """API endpoint for log data"""
    lines = request.args.get('lines', 100, type=int)
    
    log_files = {
        'health': f"{LOG_DIR}/health_check.log",
        'backup': f"{LOG_DIR}/backup.log",
        'metrics': f"{LOG_DIR}/metrics.log",
        'maintenance': f"{LOG_DIR}/maintenance.log"
    }
    
    if log_type not in log_files:
        return jsonify({'error': 'Invalid log type'}), 400
    
    log_content = read_log_file(log_files[log_type], lines)
    return jsonify({'content': log_content})

# Helper Functions
def get_service_status():
    """Get status of monitored services"""
    config = load_monitoring_config()
    services = config.get('MONITORED_SERVICES', '').split()
    
    status = {}
    for service in services:
        # This would normally check systemctl status
        # For now, return mock data
        status[service] = {
            'status': 'running',
            'uptime': '2d 5h 30m',
            'memory': '45MB',
            'cpu': '2.3%'
        }
    
    return status

def get_health_status():
    """Get overall system health status"""
    try:
        # Try to read the latest health report
        health_file = f"{LOG_DIR}/health_status.json"
        if os.path.exists(health_file):
            with open(health_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading health status: {e}")
    
    # Return default health status
    return {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'overall_status': 'UNKNOWN',
        'system_metrics': {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'temperature': 'N/A',
            'load_average': '0.00'
        },
        'services': {},
        'uptime': 'Unknown',
        'last_check': 'Never'
    }

def get_backup_status():
    """Get backup status and history"""
    try:
        config = load_monitoring_config()
        backup_dir = config.get('BACKUP_LOCAL_DIR', '/opt/bitaxe-web/backups')
        
        backups = []
        if os.path.exists(backup_dir):
            for file in os.listdir(backup_dir):
                if file.endswith('.db.gz') or file.endswith('.tar.gz'):
                    file_path = os.path.join(backup_dir, file)
                    stat = os.stat(file_path)
                    
                    backups.append({
                        'filename': file,
                        'size': f"{stat.st_size / 1024 / 1024:.1f}MB",
                        'created': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'database' if file.endswith('.db.gz') else 'csv_export'
                    })
        
        # Sort by creation time, newest first
        backups.sort(key=lambda x: x['created'], reverse=True)
        
        return {
            'enabled': config.get('BACKUP_ENABLED', 'false').lower() == 'true',
            'laptop_sync': config.get('LAPTOP_BACKUP_ENABLED', 'false').lower() == 'true',
            'recent_backups': backups[:10],
            'total_backups': len(backups)
        }
    
    except Exception as e:
        print(f"Error getting backup status: {e}")
        return {
            'enabled': False,
            'laptop_sync': False,
            'recent_backups': [],
            'total_backups': 0
        }

def get_log_files():
    """Get available log files"""
    log_files = {}
    
    log_types = {
        'Health Check': f"{LOG_DIR}/health_check.log",
        'Backup': f"{LOG_DIR}/backup.log",
        'Metrics': f"{LOG_DIR}/metrics.log",
        'Maintenance': f"{LOG_DIR}/maintenance.log",
        'Log Cleanup': f"{LOG_DIR}/log_cleanup.log"
    }
    
    for name, path in log_types.items():
        if os.path.exists(path):
            stat = os.stat(path)
            log_files[name] = {
                'path': path,
                'size': f"{stat.st_size / 1024:.1f}KB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return log_files

def read_log_file(file_path, lines=100):
    """Read last N lines from log file"""
    try:
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, 'r') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    except Exception as e:
        print(f"Error reading log file {file_path}: {e}")
        return [f"Error reading log file: {e}"]