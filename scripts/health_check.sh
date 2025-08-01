#!/bin/bash
# BitAxe V2.0.0 - Comprehensive Health Monitoring Script
# Monitors services, system resources, and API endpoints with auto-restart

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
LOG_FILE="$LOG_DIR/health_check.log"
mkdir -p "$LOG_DIR"

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "WARN" "$1"; }
log_error() { log "ERROR" "$1"; }
log_success() { log "SUCCESS" "$1"; }

# Service restart tracking
RESTART_STATE_DIR="/tmp/bitaxe_health"
mkdir -p "$RESTART_STATE_DIR"

get_restart_count() {
    local service="$1"
    local count_file="$RESTART_STATE_DIR/${service}_restart_count"
    if [[ -f "$count_file" ]]; then
        cat "$count_file"
    else
        echo "0"
    fi
}

increment_restart_count() {
    local service="$1"
    local count_file="$RESTART_STATE_DIR/${service}_restart_count"
    local current_count=$(get_restart_count "$service")
    echo $((current_count + 1)) > "$count_file"
}

reset_restart_count() {
    local service="$1"
    local count_file="$RESTART_STATE_DIR/${service}_restart_count"
    echo "0" > "$count_file"
}

get_last_restart_time() {
    local service="$1"
    local time_file="$RESTART_STATE_DIR/${service}_last_restart"
    if [[ -f "$time_file" ]]; then
        cat "$time_file"
    else
        echo "0"
    fi
}

set_last_restart_time() {
    local service="$1"
    local time_file="$RESTART_STATE_DIR/${service}_last_restart"
    date +%s > "$time_file"
}

# Check if service should be restarted
can_restart_service() {
    local service="$1"
    local current_time=$(date +%s)
    local last_restart=$(get_last_restart_time "$service")
    local restart_count=$(get_restart_count "$service")
    
    # Check cooldown period
    if [[ $((current_time - last_restart)) -lt $RESTART_COOLDOWN ]]; then
        log_warn "Service $service is in cooldown period"
        return 1
    fi
    
    # Check max restart attempts
    if [[ $restart_count -ge $MAX_RESTART_ATTEMPTS ]]; then
        log_error "Service $service has exceeded max restart attempts ($MAX_RESTART_ATTEMPTS)"
        return 1
    fi
    
    return 0
}

# Check systemd service status
check_service_status() {
    local service="$1"
    
    if systemctl is-active --quiet "$service"; then
        log_info "✓ Service $service is running"
        reset_restart_count "$service"
        return 0
    else
        local status=$(systemctl is-active "$service" 2>/dev/null || echo "unknown")
        log_error "✗ Service $service is $status"
        return 1
    fi
}

# Restart a failed service
restart_service() {
    local service="$1"
    
    if [[ "$AUTO_RESTART_ENABLED" != "true" ]]; then
        log_warn "Auto-restart disabled for $service"
        return 1
    fi
    
    if ! can_restart_service "$service"; then
        return 1
    fi
    
    log_warn "Attempting to restart service: $service"
    
    if systemctl restart "$service"; then
        increment_restart_count "$service"
        set_last_restart_time "$service"
        
        # Wait a moment and check if it started successfully
        sleep 5
        if systemctl is-active --quiet "$service"; then
            log_success "Service $service restarted successfully"
            return 0
        else
            log_error "Service $service failed to start after restart"
            return 1
        fi
    else
        log_error "Failed to restart service: $service"
        return 1
    fi
}

# Check API health endpoint
check_api_health() {
    local api_url="http://localhost:5000/api/v1/health"
    local timeout="$API_HEALTH_TIMEOUT"
    
    log_info "Checking API health endpoint..."
    
    if curl -s --max-time "$timeout" "$api_url" >/dev/null 2>&1; then
        log_success "✓ API health endpoint responding"
        return 0
    else
        log_error "✗ API health endpoint not responding"
        return 1
    fi
}

# Get system metrics
get_system_metrics() {
    local metrics_file="/tmp/bitaxe_system_metrics"
    
    # CPU usage (1-minute average)
    local cpu_usage=$(awk '{print 100-$4}' <(head -1 /proc/stat; sleep 1; head -1 /proc/stat) | tail -1 | cut -d'.' -f1)
    
    # Memory usage
    local memory_info=$(free | awk 'NR==2{printf "%.0f %.0f", $3/$2*100, $2/1024/1024}')
    local memory_usage=$(echo "$memory_info" | cut -d' ' -f1)
    local total_memory=$(echo "$memory_info" | cut -d' ' -f2)
    
    # Disk usage
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    
    # System temperature (if available)
    local temperature="N/A"
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        temperature=$(($(cat /sys/class/thermal/thermal_zone0/temp) / 1000))
    fi
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d',' -f1 | xargs)
    
    # Save metrics
    cat > "$metrics_file" <<EOF
cpu_usage=$cpu_usage
memory_usage=$memory_usage
total_memory=$total_memory
disk_usage=$disk_usage
temperature=$temperature
load_avg=$load_avg
timestamp=$(date +%s)
EOF
    
    echo "$metrics_file"
}

# Check system resource thresholds
check_system_resources() {
    local metrics_file=$(get_system_metrics)
    source "$metrics_file"
    
    log_info "System Resources Check:"
    log_info "  CPU Usage: ${cpu_usage}%"
    log_info "  Memory Usage: ${memory_usage}% (${total_memory}GB total)"
    log_info "  Disk Usage: ${disk_usage}%"
    log_info "  Temperature: ${temperature}°C"
    log_info "  Load Average: ${load_avg}"
    
    local critical_issues=0
    local warnings=0
    
    # Check CPU usage
    if [[ $cpu_usage -ge $CPU_CRITICAL_THRESHOLD ]]; then
        log_error "CRITICAL: CPU usage is ${cpu_usage}% (threshold: ${CPU_CRITICAL_THRESHOLD}%)"
        ((critical_issues++))
    elif [[ $cpu_usage -ge $CPU_WARNING_THRESHOLD ]]; then
        log_warn "WARNING: CPU usage is ${cpu_usage}% (threshold: ${CPU_WARNING_THRESHOLD}%)"
        ((warnings++))
    fi
    
    # Check memory usage
    if [[ $memory_usage -ge $MEMORY_CRITICAL_THRESHOLD ]]; then
        log_error "CRITICAL: Memory usage is ${memory_usage}% (threshold: ${MEMORY_CRITICAL_THRESHOLD}%)"
        ((critical_issues++))
    elif [[ $memory_usage -ge $MEMORY_WARNING_THRESHOLD ]]; then
        log_warn "WARNING: Memory usage is ${memory_usage}% (threshold: ${MEMORY_WARNING_THRESHOLD}%)"
        ((warnings++))
    fi
    
    # Check disk usage
    if [[ $disk_usage -ge $DISK_CRITICAL_THRESHOLD ]]; then
        log_error "CRITICAL: Disk usage is ${disk_usage}% (threshold: ${DISK_CRITICAL_THRESHOLD}%)"
        ((critical_issues++))
    elif [[ $disk_usage -ge $DISK_WARNING_THRESHOLD ]]; then
        log_warn "WARNING: Disk usage is ${disk_usage}% (threshold: ${DISK_WARNING_THRESHOLD}%)"
        ((warnings++))
    fi
    
    # Check temperature (if available)
    if [[ "$temperature" != "N/A" ]]; then
        if [[ $temperature -ge $TEMP_CRITICAL_THRESHOLD ]]; then
            log_error "CRITICAL: System temperature is ${temperature}°C (threshold: ${TEMP_CRITICAL_THRESHOLD}°C)"
            ((critical_issues++))
        elif [[ $temperature -ge $TEMP_WARNING_THRESHOLD ]]; then
            log_warn "WARNING: System temperature is ${temperature}°C (threshold: ${TEMP_WARNING_THRESHOLD}°C)"
            ((warnings++))
        fi
    fi
    
    # Log summary
    if [[ $critical_issues -gt 0 ]]; then
        log_error "System health check failed: $critical_issues critical issue(s), $warnings warning(s)"
        return 2  # Critical
    elif [[ $warnings -gt 0 ]]; then
        log_warn "System health check completed with warnings: $warnings warning(s)"
        return 1  # Warning
    else
        log_success "System health check passed: All resources within normal limits"
        return 0  # OK
    fi
}

# Generate health report
generate_health_report() {
    local overall_status="$1"
    local report_file="$LOG_DIR/health_status.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Get current metrics
    local metrics_file=$(get_system_metrics)
    source "$metrics_file"
    
    # Check service statuses
    local services_status="{"
    for service in $MONITORED_SERVICES; do
        if systemctl is-active --quiet "$service"; then
            services_status+="\"$service\":\"running\","
        else
            services_status+="\"$service\":\"failed\","
        fi
    done
    services_status="${services_status%,}}"
    
    # Generate JSON report
    cat > "$report_file" <<EOF
{
  "timestamp": "$timestamp",
  "overall_status": "$overall_status",
  "system_metrics": {
    "cpu_usage": $cpu_usage,
    "memory_usage": $memory_usage,
    "disk_usage": $disk_usage,
    "temperature": "$temperature",
    "load_average": "$load_avg"
  },
  "services": $services_status,
  "uptime": "$(uptime -p)",
  "last_check": "$timestamp"
}
EOF
    
    log_info "Health report generated: $report_file"
}

# Main health check execution
main() {
    if [[ "$HEALTH_CHECK_ENABLED" != "true" ]]; then
        exit 0
    fi
    
    log_info "=== BitAxe Health Check Started ==="
    
    local overall_status="OK"
    local failed_services=0
    local critical_resources=0
    
    # Check all monitored services
    for service in $MONITORED_SERVICES; do
        if ! check_service_status "$service"; then
            if restart_service "$service"; then
                log_success "Service $service recovered"
            else
                ((failed_services++))
                overall_status="CRITICAL"
            fi
        fi
    done
    
    # Check API health (only if web service is running)
    if systemctl is-active --quiet "bitaxe-web"; then
        if ! check_api_health; then
            if restart_service "bitaxe-web"; then
                log_success "Web service recovered"
            else
                overall_status="CRITICAL"
            fi
        fi
    fi
    
    # Check system resources
    check_system_resources
    local resource_status=$?
    
    if [[ $resource_status -eq 2 ]]; then
        overall_status="CRITICAL"
        ((critical_resources++))
    elif [[ $resource_status -eq 1 && "$overall_status" == "OK" ]]; then
        overall_status="WARNING"
    fi
    
    # Generate health report
    generate_health_report "$overall_status"
    
    # Final status
    log_info "=== Health Check Summary ==="
    log_info "Overall Status: $overall_status"
    log_info "Failed Services: $failed_services"
    log_info "Resource Issues: $([[ $critical_resources -gt 0 ]] && echo "Critical" || echo "None")"
    log_info "=== Health Check Completed ==="
    
    # Exit with appropriate code
    case "$overall_status" in
        "OK") exit 0 ;;
        "WARNING") exit 1 ;;
        "CRITICAL") exit 2 ;;
    esac
}

# Run main function
main "$@"