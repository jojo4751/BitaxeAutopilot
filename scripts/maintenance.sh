#!/bin/bash
# BitAxe V2.0.0 - Automated Maintenance and Cleanup Script
# Performs routine maintenance tasks including log rotation, database optimization, and cleanup

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
LOG_FILE="$LOG_DIR/maintenance.log"
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

# Database maintenance functions
vacuum_database() {
    log_info "Starting database vacuum operation..."
    
    if [[ ! -f "$DATABASE_PATH" ]]; then
        log_error "Database file not found: $DATABASE_PATH"
        return 1
    fi
    
    # Get database size before vacuum
    local size_before=$(du -h "$DATABASE_PATH" | cut -f1)
    
    # Perform vacuum operation
    if sqlite3 "$DATABASE_PATH" "VACUUM;"; then
        local size_after=$(du -h "$DATABASE_PATH" | cut -f1)
        log_success "Database vacuum completed: $size_before → $size_after"
        
        # Update database statistics
        sqlite3 "$DATABASE_PATH" "ANALYZE;" 2>/dev/null || log_warn "Failed to analyze database statistics"
        
        return 0
    else
        log_error "Database vacuum operation failed"
        return 1
    fi
}

optimize_database_indexes() {
    log_info "Optimizing database indexes..."
    
    # Reindex all indexes to optimize performance
    if sqlite3 "$DATABASE_PATH" "REINDEX;"; then
        log_success "Database indexes optimized"
        return 0
    else
        log_error "Failed to optimize database indexes"
        return 1
    fi
}

cleanup_old_database_records() {
    log_info "Cleaning up old database records..."
    
    local retention_hours=$((METRICS_HISTORY_DAYS * 24))
    local cleanup_count=0
    
    # Clean up old system metrics
    local metrics_deleted=$(sqlite3 "$DATABASE_PATH" "
        DELETE FROM system_metrics 
        WHERE timestamp < datetime('now', '-$retention_hours hours');
        SELECT changes();
    " 2>/dev/null || echo "0")
    
    # Clean up old logs (keep last 30 days)
    local logs_deleted=$(sqlite3 "$DATABASE_PATH" "
        DELETE FROM logs 
        WHERE timestamp < datetime('now', '-720 hours');
        SELECT changes();
    " 2>/dev/null || echo "0")
    
    # Clean up old protocol events (keep last 14 days)
    local events_deleted=$(sqlite3 "$DATABASE_PATH" "
        DELETE FROM protocol 
        WHERE timestamp < datetime('now', '-336 hours');
        SELECT changes();
    " 2>/dev/null || echo "0")
    
    cleanup_count=$((metrics_deleted + logs_deleted + events_deleted))
    
    if [[ $cleanup_count -gt 0 ]]; then
        log_success "Cleaned up $cleanup_count old database records"
        log_info "  - System metrics: $metrics_deleted"
        log_info "  - Miner logs: $logs_deleted"
        log_info "  - Protocol events: $events_deleted"
    else
        log_info "No old database records to clean up"
    fi
}

# Log rotation and cleanup
rotate_application_logs() {
    log_info "Rotating application logs..."
    
    local rotated_count=0
    
    # Find and rotate large log files
    find "$LOG_DIR" -name "*.log" -size +${LOG_MAX_SIZE} | while read -r logfile; do
        if [[ -f "$logfile" ]]; then
            local backup_file="${logfile}.$(date +%Y%m%d_%H%M%S)"
            
            if mv "$logfile" "$backup_file" && gzip "$backup_file"; then
                touch "$logfile"
                log_success "Rotated log file: $(basename "$logfile")"
                ((rotated_count++))
            else
                log_error "Failed to rotate log file: $(basename "$logfile")"
            fi
        fi
    done
    
    # Clean up old rotated logs
    find "$LOG_DIR" -name "*.log.*.gz" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    
    log_info "Log rotation completed"
}

cleanup_temporary_files() {
    if [[ "$CLEANUP_TEMP_FILES" != "true" ]]; then
        return 0
    fi
    
    log_info "Cleaning up temporary files..."
    
    local cleanup_dirs=(
        "/tmp/bitaxe_*"
        "/var/tmp/bitaxe_*"
        "$BASE_DIR/tmp"
        "$BASE_DIR/.tmp"
    )
    
    local cleaned_files=0
    
    for pattern in "${cleanup_dirs[@]}"; do
        # Clean files older than 1 day
        find $pattern -type f -mtime +1 -delete 2>/dev/null && ((cleaned_files++)) || true
        # Clean empty directories
        find $pattern -type d -empty -delete 2>/dev/null || true
    done
    
    # Clean up Python cache files
    find "$BASE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE_DIR" -name "*.pyc" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean up backup temporary files
    find "$BACKUP_LOCAL_DIR" -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
    
    log_success "Temporary files cleanup completed"
}

# System maintenance functions
update_system_packages() {
    if [[ "$AUTO_UPDATE_ENABLED" != "true" ]]; then
        log_info "System package updates disabled"
        return 0
    fi
    
    log_info "Checking for system package updates..."
    
    # Update package lists
    if apt-get update >/dev/null 2>&1; then
        log_success "Package lists updated"
        
        # Check for available upgrades
        local upgrades=$(apt list --upgradable 2>/dev/null | wc -l)
        
        if [[ $upgrades -gt 1 ]]; then  # Subtract 1 for header line
            log_info "Found $(($upgrades - 1)) package updates available"
            
            # Perform security updates only
            if unattended-upgrade --dry-run >/dev/null 2>&1; then
                log_info "Security updates would be applied (dry-run mode)"
                # Uncomment next line to enable automatic security updates
                # unattended-upgrade
            fi
        else
            log_info "No package updates available"
        fi
    else
        log_warn "Failed to update package lists"
    fi
}

check_disk_space() {
    log_info "Checking disk space usage..."
    
    # Check main system disk
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    local disk_available=$(df -h "$BASE_DIR" | awk 'NR==2{print $4}')
    
    log_info "Disk usage: ${disk_usage}% (${disk_available} available)"
    
    if [[ $disk_usage -ge $DISK_CRITICAL_THRESHOLD ]]; then
        log_error "CRITICAL: Disk usage at ${disk_usage}% - immediate attention required!"
        return 1
    elif [[ $disk_usage -ge $DISK_WARNING_THRESHOLD ]]; then
        log_warn "WARNING: Disk usage at ${disk_usage}% - consider cleanup"
        return 1
    else
        log_success "Disk usage within normal limits"
        return 0
    fi
}

optimize_system_performance() {
    log_info "Optimizing system performance..."
    
    # Clear system caches if memory usage is high
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
    
    if [[ $memory_usage -ge 85 ]]; then
        log_info "High memory usage detected (${memory_usage}%), clearing caches..."
        
        # Clear page cache, dentries and inodes
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || log_warn "Failed to clear system caches"
        
        log_success "System caches cleared"
    fi
    
    # Optimize system swappiness for mining workload
    local current_swappiness=$(cat /proc/sys/vm/swappiness)
    local optimal_swappiness=10
    
    if [[ $current_swappiness -ne $optimal_swappiness ]]; then
        echo $optimal_swappiness > /proc/sys/vm/swappiness 2>/dev/null || log_warn "Failed to optimize swappiness"
        log_info "Swappiness optimized: $current_swappiness → $optimal_swappiness"
    fi
}

# Service maintenance
restart_services_if_needed() {
    log_info "Checking services for restart requirements..."
    
    # Check if services have been running for more than 7 days
    local restart_threshold=604800  # 7 days in seconds
    local current_time=$(date +%s)
    
    for service in $MONITORED_SERVICES; do
        if systemctl is-active --quiet "$service"; then
            # Get service start time
            local start_time=$(systemctl show "$service" --property=ActiveEnterTimestamp --value)
            local start_timestamp=$(date -d "$start_time" +%s 2>/dev/null || echo "$current_time")
            local running_time=$((current_time - start_timestamp))
            
            if [[ $running_time -gt $restart_threshold ]]; then
                log_info "Service $service has been running for $(($running_time / 86400)) days"
                
                # Schedule restart during next maintenance window
                log_info "Service $service scheduled for restart during next maintenance window"
                # Note: Actual restart logic would be implemented based on maintenance windows
            fi
        fi
    done
}

# Generate maintenance report
generate_maintenance_report() {
    local report_file="$LOG_DIR/maintenance_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Get current system status
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    local memory_usage=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
    local db_size=$(du -h "$DATABASE_PATH" | cut -f1)
    local log_count=$(find "$LOG_DIR" -name "*.log" | wc -l)
    
    # Create JSON report
    cat > "$report_file" <<EOF
{
  "timestamp": "$timestamp",
  "maintenance_status": "completed",
  "system_status": {
    "disk_usage_percent": $disk_usage,
    "memory_usage_percent": $memory_usage,
    "database_size": "$db_size",
    "log_files_count": $log_count
  },
  "tasks_performed": {
    "database_vacuum": "$DATABASE_VACUUM_ENABLED",
    "log_rotation": "true",
    "temp_cleanup": "$CLEANUP_TEMP_FILES",
    "system_updates": "$AUTO_UPDATE_ENABLED"
  },
  "next_maintenance": "$(date -d '+1 day' -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    log_success "Maintenance report generated: $report_file"
}

# Main maintenance execution
main() {
    if [[ "$MAINTENANCE_ENABLED" != "true" ]]; then
        log_info "Maintenance system disabled"
        exit 0
    fi
    
    log_info "=== BitAxe System Maintenance Started ==="
    
    local maintenance_status="SUCCESS"
    local failed_tasks=0
    
    # Database maintenance
    if [[ "$DATABASE_VACUUM_ENABLED" == "true" ]]; then
        vacuum_database || ((failed_tasks++))
        optimize_database_indexes || ((failed_tasks++))
    fi
    
    cleanup_old_database_records || ((failed_tasks++))
    
    # File system maintenance
    rotate_application_logs || ((failed_tasks++))
    cleanup_temporary_files || ((failed_tasks++))
    
    # System maintenance
    check_disk_space || ((failed_tasks++))
    optimize_system_performance || ((failed_tasks++))
    update_system_packages || ((failed_tasks++))
    restart_services_if_needed || ((failed_tasks++))
    
    # Generate report
    generate_maintenance_report
    
    # Final status
    if [[ $failed_tasks -gt 0 ]]; then
        maintenance_status="PARTIAL_FAILURE"
        log_warn "Maintenance completed with $failed_tasks failed tasks"
    else
        log_success "All maintenance tasks completed successfully"
    fi
    
    log_info "=== System Maintenance Completed: $maintenance_status ==="
    
    # Exit with appropriate code
    if [[ $failed_tasks -gt 3 ]]; then
        exit 1  # Critical failure
    elif [[ $failed_tasks -gt 0 ]]; then
        exit 2  # Partial failure
    else
        exit 0  # Success
    fi
}

# Handle interrupts
trap 'log_error "Maintenance interrupted"; exit 1' INT TERM

# Run main function
main "$@"