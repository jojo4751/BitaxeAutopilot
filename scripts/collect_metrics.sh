#!/bin/bash
# BitAxe V2.0.0 - Performance Metrics Collection Script
# Collects system and application metrics for monitoring and analysis

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"

# Load configuration
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Logging setup
LOG_FILE="$LOG_DIR/metrics.log"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Database connection function
execute_sql() {
    local sql="$1"
    sqlite3 "$DATABASE_PATH" "$sql"
}

# Initialize metrics table if not exists
init_metrics_table() {
    execute_sql "
    CREATE TABLE IF NOT EXISTS system_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        metric_type TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_unit TEXT,
        additional_data TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_metrics_type_name ON system_metrics(metric_type, metric_name);
    "
}

# Collect system CPU metrics
collect_cpu_metrics() {
    log "Collecting CPU metrics..."
    
    # CPU usage percentage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # Load averages
    local load_avg=$(cat /proc/loadavg)
    local load_1min=$(echo "$load_avg" | cut -d' ' -f1)
    local load_5min=$(echo "$load_avg" | cut -d' ' -f2)
    local load_15min=$(echo "$load_avg" | cut -d' ' -f3)
    
    # CPU temperature (if available)
    local cpu_temp="NULL"
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        cpu_temp=$(awk '{print $1/1000}' /sys/class/thermal/thermal_zone0/temp)
    fi
    
    # Insert CPU metrics
    execute_sql "
    INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
    ('system', 'cpu_usage_percent', $cpu_usage, '%'),
    ('system', 'load_average_1min', $load_1min, 'load'),
    ('system', 'load_average_5min', $load_5min, 'load'),
    ('system', 'load_average_15min', $load_15min, 'load');
    "
    
    if [[ "$cpu_temp" != "NULL" ]]; then
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('system', 'cpu_temperature', $cpu_temp, '°C');
        "
    fi
}

# Collect memory metrics
collect_memory_metrics() {
    log "Collecting memory metrics..."
    
    # Memory information
    local mem_info=$(free -b | awk 'NR==2{printf "%.2f %.2f %.2f %.2f", $3/$2*100, $2/1024/1024/1024, $3/1024/1024/1024, $7/1024/1024/1024}')
    local mem_usage_percent=$(echo "$mem_info" | cut -d' ' -f1)
    local mem_total_gb=$(echo "$mem_info" | cut -d' ' -f2)
    local mem_used_gb=$(echo "$mem_info" | cut -d' ' -f3)
    local mem_available_gb=$(echo "$mem_info" | cut -d' ' -f4)
    
    # Swap information
    local swap_info=$(free -b | awk 'NR==3{printf "%.2f %.2f", ($2 > 0 ? $3/$2*100 : 0), $2/1024/1024/1024}')
    local swap_usage_percent=$(echo "$swap_info" | cut -d' ' -f1)
    local swap_total_gb=$(echo "$swap_info" | cut -d' ' -f2)
    
    # Insert memory metrics
    execute_sql "
    INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
    ('system', 'memory_usage_percent', $mem_usage_percent, '%'),
    ('system', 'memory_total', $mem_total_gb, 'GB'),
    ('system', 'memory_used', $mem_used_gb, 'GB'),
    ('system', 'memory_available', $mem_available_gb, 'GB'),
    ('system', 'swap_usage_percent', $swap_usage_percent, '%'),
    ('system', 'swap_total', $swap_total_gb, 'GB');
    "
}

# Collect disk metrics
collect_disk_metrics() {
    log "Collecting disk metrics..."
    
    # Disk usage for main partition
    local disk_info=$(df "$BASE_DIR" | awk 'NR==2{printf "%.2f %.2f %.2f %.2f", $5, $2/1024/1024, $3/1024/1024, $4/1024/1024}')
    local disk_usage_percent=$(echo "$disk_info" | cut -d' ' -f1)
    local disk_total_gb=$(echo "$disk_info" | cut -d' ' -f2)
    local disk_used_gb=$(echo "$disk_info" | cut -d' ' -f3)
    local disk_available_gb=$(echo "$disk_info" | cut -d' ' -f4)
    
    # Disk I/O statistics (if available)
    local disk_reads="NULL"
    local disk_writes="NULL"
    if [[ -f "/proc/diskstats" ]]; then
        local main_disk=$(df "$BASE_DIR" | awk 'NR==2{print $1}' | sed 's|/dev/||' | sed 's|[0-9]*$||')
        if [[ -n "$main_disk" ]]; then
            local io_stats=$(awk -v disk="$main_disk" '$3==disk{print $6" "$10}' /proc/diskstats)
            if [[ -n "$io_stats" ]]; then
                disk_reads=$(echo "$io_stats" | cut -d' ' -f1)
                disk_writes=$(echo "$io_stats" | cut -d' ' -f2)
            fi
        fi
    fi
    
    # Insert disk metrics
    execute_sql "
    INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
    ('system', 'disk_usage_percent', $disk_usage_percent, '%'),
    ('system', 'disk_total', $disk_total_gb, 'GB'),
    ('system', 'disk_used', $disk_used_gb, 'GB'),
    ('system', 'disk_available', $disk_available_gb, 'GB');
    "
    
    if [[ "$disk_reads" != "NULL" && "$disk_writes" != "NULL" ]]; then
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('system', 'disk_reads_total', $disk_reads, 'sectors'),
        ('system', 'disk_writes_total', $disk_writes, 'sectors');
        "
    fi
}

# Collect network metrics
collect_network_metrics() {
    log "Collecting network metrics..."
    
    # Network interface statistics
    local interface=$(ip route | awk '/default/ { print $5; exit }')
    if [[ -n "$interface" ]]; then
        local net_stats=$(awk -v iface="$interface:" '$0 ~ iface {print $2" "$10}' /proc/net/dev)
        if [[ -n "$net_stats" ]]; then
            local bytes_received=$(echo "$net_stats" | cut -d' ' -f1)
            local bytes_transmitted=$(echo "$net_stats" | cut -d' ' -f2)
            
            # Convert to MB
            local mb_received=$(awk "BEGIN {printf \"%.2f\", $bytes_received/1024/1024}")
            local mb_transmitted=$(awk "BEGIN {printf \"%.2f\", $bytes_transmitted/1024/1024}")
            
            execute_sql "
            INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit, additional_data) VALUES
            ('system', 'network_bytes_received', $mb_received, 'MB', '$interface'),
            ('system', 'network_bytes_transmitted', $mb_transmitted, 'MB', '$interface');
            "
        fi
    fi
}

# Collect application-specific metrics
collect_application_metrics() {
    log "Collecting application metrics..."
    
    # Check if web service is running and get response time
    if systemctl is-active --quiet "bitaxe-web"; then
        local api_response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://localhost:5000/api/status" 2>/dev/null || echo "NULL")
        
        if [[ "$api_response_time" != "NULL" ]]; then
            # Convert to milliseconds
            local response_time_ms=$(awk "BEGIN {printf \"%.0f\", $api_response_time*1000}")
            
            execute_sql "
            INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
            ('application', 'api_response_time', $response_time_ms, 'ms');
            "
        fi
        
        # Service status metrics (1 = running, 0 = stopped)
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('application', 'bitaxe_web_status', 1, 'boolean');
        "
    else
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('application', 'bitaxe_web_status', 0, 'boolean');
        "
    fi
    
    # Check other services
    for service in bitaxe-autopilot bitaxe-logger; do
        local status_value=0
        if systemctl is-active --quiet "$service"; then
            status_value=1
        fi
        
        local service_name=$(echo "$service" | tr '-' '_')
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('application', '${service_name}_status', $status_value, 'boolean');
        "
    done
}

# Collect database metrics
collect_database_metrics() {
    log "Collecting database metrics..."
    
    if [[ -f "$DATABASE_PATH" ]]; then
        # Database file size
        local db_size_mb=$(du -m "$DATABASE_PATH" | cut -f1)
        
        # Count of records in main tables
        local logs_count=$(execute_sql "SELECT COUNT(*) FROM logs;" 2>/dev/null || echo "0")
        local benchmarks_count=$(execute_sql "SELECT COUNT(*) FROM benchmark_results;" 2>/dev/null || echo "0")
        local events_count=$(execute_sql "SELECT COUNT(*) FROM protocol;" 2>/dev/null || echo "0")
        
        # Database performance test (simple query time)
        local start_time=$(date +%s%N)
        execute_sql "SELECT COUNT(*) FROM logs WHERE timestamp > datetime('now', '-1 hour');" >/dev/null 2>&1 || true
        local end_time=$(date +%s%N)
        local query_time_ms=$(( (end_time - start_time) / 1000000 ))
        
        # Insert database metrics
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('database', 'db_file_size', $db_size_mb, 'MB'),
        ('database', 'logs_count', $logs_count, 'records'),
        ('database', 'benchmarks_count', $benchmarks_count, 'records'),
        ('database', 'events_count', $events_count, 'records'),
        ('database', 'query_response_time', $query_time_ms, 'ms');
        "
    fi
}

# Collect mining performance metrics from existing data
collect_mining_metrics() {
    log "Collecting mining performance metrics..."
    
    # Get latest mining data aggregates
    local mining_stats=$(execute_sql "
    SELECT 
        COUNT(DISTINCT ip) as active_miners,
        AVG(CASE WHEN hashRate > 0 THEN hashRate END) as avg_hashrate,
        SUM(CASE WHEN hashRate > 0 THEN hashRate END) as total_hashrate,
        AVG(CASE WHEN temp > 0 THEN temp END) as avg_temperature,
        AVG(CASE WHEN power > 0 THEN power END) as avg_power,
        AVG(CASE WHEN hashRate > 0 AND power > 0 THEN hashRate/power END) as avg_efficiency
    FROM logs 
    WHERE timestamp > datetime('now', '-5 minutes')
    AND hashRate IS NOT NULL;
    " 2>/dev/null || echo "0|0|0|0|0|0")
    
    if [[ "$mining_stats" != "0|0|0|0|0|0" ]]; then
        local active_miners=$(echo "$mining_stats" | cut -d'|' -f1)
        local avg_hashrate=$(echo "$mining_stats" | cut -d'|' -f2)
        local total_hashrate=$(echo "$mining_stats" | cut -d'|' -f3)
        local avg_temperature=$(echo "$mining_stats" | cut -d'|' -f4)
        local avg_power=$(echo "$mining_stats" | cut -d'|' -f5)
        local avg_efficiency=$(echo "$mining_stats" | cut -d'|' -f6)
        
        # Handle NULL values
        avg_hashrate=${avg_hashrate:-0}
        total_hashrate=${total_hashrate:-0}
        avg_temperature=${avg_temperature:-0}
        avg_power=${avg_power:-0}
        avg_efficiency=${avg_efficiency:-0}
        
        execute_sql "
        INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
        ('mining', 'active_miners', $active_miners, 'count'),
        ('mining', 'average_hashrate', $avg_hashrate, 'GH/s'),
        ('mining', 'total_hashrate', $total_hashrate, 'GH/s'),
        ('mining', 'average_temperature', $avg_temperature, '°C'),
        ('mining', 'average_power', $avg_power, 'W'),
        ('mining', 'average_efficiency', $avg_efficiency, 'GH/W');
        "
    fi
}

# Cleanup old metrics data
cleanup_old_metrics() {
    local retention_hours=$((METRICS_HISTORY_DAYS * 24))
    
    execute_sql "
    DELETE FROM system_metrics 
    WHERE timestamp < datetime('now', '-$retention_hours hours');
    "
    
    local deleted_count=$(execute_sql "SELECT changes();")
    if [[ $deleted_count -gt 0 ]]; then
        log "Cleaned up $deleted_count old metrics records"
    fi
}

# Generate metrics summary
generate_metrics_summary() {
    local summary_file="$LOG_DIR/metrics_summary.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Get latest metrics for each type
    local latest_metrics=$(execute_sql "
    SELECT metric_type, metric_name, metric_value, metric_unit
    FROM system_metrics 
    WHERE timestamp > datetime('now', '-2 minutes')
    ORDER BY timestamp DESC;
    ")
    
    # Create JSON summary (simplified version)
    cat > "$summary_file" <<EOF
{
  "timestamp": "$timestamp",
  "collection_status": "completed",
  "metrics_collected": true,
  "retention_days": $METRICS_HISTORY_DAYS
}
EOF
    
    log "Metrics summary generated: $summary_file"
}

# Main execution
main() {
    if [[ "$SYSTEM_MONITORING_ENABLED" != "true" ]]; then
        exit 0
    fi
    
    log "=== BitAxe Metrics Collection Started ==="
    
    # Initialize database table
    init_metrics_table
    
    # Collect all metrics
    collect_cpu_metrics
    collect_memory_metrics
    collect_disk_metrics
    collect_network_metrics
    collect_application_metrics
    collect_database_metrics
    collect_mining_metrics
    
    # Cleanup and reporting
    cleanup_old_metrics
    generate_metrics_summary
    
    log "=== Metrics Collection Completed ==="
}

# Run main function
main "$@"