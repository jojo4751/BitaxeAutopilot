#!/bin/bash
# BitAxe V2.0.0 - Advanced Log Management and Cleanup Script
# Handles log rotation, compression, archival, and intelligent cleanup

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
LOG_FILE="$LOG_DIR/log_cleanup.log"
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

# Log discovery and analysis
discover_log_files() {
    log_info "Discovering log files across the system..."
    
    local log_locations=(
        "$LOG_DIR"
        "$BASE_DIR/logs"
        "/var/log/bitaxe"
        "/opt/bitaxe-web/logs"
        "/tmp"
    )
    
    local discovered_logs=()
    
    # Find all log files in known locations
    for location in "${log_locations[@]}"; do
        if [[ -d "$location" ]]; then
            while IFS= read -r -d '' logfile; do
                discovered_logs+=("$logfile")
            done < <(find "$location" -name "*.log" -type f -print0 2>/dev/null || true)
        fi
    done
    
    # Find Python application logs
    while IFS= read -r -d '' logfile; do
        discovered_logs+=("$logfile")
    done < <(find "$BASE_DIR" -name "*.log" -o -name "*.out" -o -name "bitaxe*.log*" -type f -print0 2>/dev/null || true)
    
    # Save discovered logs to temporary file
    printf '%s\n' "${discovered_logs[@]}" | sort -u > "/tmp/bitaxe_discovered_logs"
    
    local log_count=${#discovered_logs[@]}
    log_success "Discovered $log_count log files"
    
    return 0
}

analyze_log_files() {
    log_info "Analyzing log files for cleanup decisions..."
    
    local total_size=0
    local large_logs=()
    local old_logs=()
    local empty_logs=()
    
    if [[ ! -f "/tmp/bitaxe_discovered_logs" ]]; then
        log_error "No discovered logs file found"
        return 1
    fi
    
    while IFS= read -r logfile; do
        if [[ -f "$logfile" ]]; then
            local size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile" 2>/dev/null || echo "0")
            local age_days=$(( ($(date +%s) - $(stat -f%m "$logfile" 2>/dev/null || stat -c%Y "$logfile" 2>/dev/null || echo "0")) / 86400 ))
            
            total_size=$((total_size + size))
            
            # Categorize logs
            if [[ $size -eq 0 ]]; then
                empty_logs+=("$logfile")
            elif [[ $size -gt $((10 * 1024 * 1024)) ]]; then  # > 10MB
                large_logs+=("$logfile")
            fi
            
            if [[ $age_days -gt $LOG_RETENTION_DAYS ]]; then
                old_logs+=("$logfile")
            fi
        fi
    done < "/tmp/bitaxe_discovered_logs"
    
    # Convert total size to human readable
    local total_size_mb=$((total_size / 1024 / 1024))
    
    # Save analysis results
    cat > "/tmp/bitaxe_log_analysis" <<EOF
total_size_mb=$total_size_mb
large_logs_count=${#large_logs[@]}
old_logs_count=${#old_logs[@]}
empty_logs_count=${#empty_logs[@]}
EOF
    
    # Save specific file lists
    printf '%s\n' "${large_logs[@]}" > "/tmp/bitaxe_large_logs" 2>/dev/null || touch "/tmp/bitaxe_large_logs"
    printf '%s\n' "${old_logs[@]}" > "/tmp/bitaxe_old_logs" 2>/dev/null || touch "/tmp/bitaxe_old_logs"
    printf '%s\n' "${empty_logs[@]}" > "/tmp/bitaxe_empty_logs" 2>/dev/null || touch "/tmp/bitaxe_empty_logs"
    
    log_info "Log analysis completed:"
    log_info "  Total size: ${total_size_mb}MB"
    log_info "  Large logs (>10MB): ${#large_logs[@]}"
    log_info "  Old logs (>${LOG_RETENTION_DAYS} days): ${#old_logs[@]}"
    log_info "  Empty logs: ${#empty_logs[@]}"
    
    return 0
}

rotate_large_logs() {
    log_info "Rotating large log files..."
    
    if [[ ! -f "/tmp/bitaxe_large_logs" ]]; then
        log_info "No large logs found for rotation"
        return 0
    fi
    
    local rotated_count=0
    
    while IFS= read -r logfile; do
        if [[ -f "$logfile" && -s "$logfile" ]]; then
            local timestamp=$(date '+%Y%m%d_%H%M%S')
            local rotated_file="${logfile}.${timestamp}"
            
            log_info "Rotating large log: $(basename "$logfile")"
            
            # Move current log to rotated name
            if mv "$logfile" "$rotated_file"; then
                # Create new empty log file
                touch "$logfile"
                
                # Set appropriate permissions
                if [[ "$logfile" == *"/var/log/"* ]]; then
                    chown syslog:adm "$logfile" 2>/dev/null || true
                    chmod 640 "$logfile" 2>/dev/null || true
                else
                    chown bitaxe:bitaxe "$logfile" 2>/dev/null || true
                    chmod 644 "$logfile" 2>/dev/null || true
                fi
                
                # Compress rotated file
                if gzip "$rotated_file"; then
                    log_success "Rotated and compressed: $(basename "$logfile")"
                    ((rotated_count++))
                else
                    log_warn "Rotated but failed to compress: $(basename "$logfile")"
                fi
                
                # Send signal to services to reopen log files
                if [[ "$logfile" == *"bitaxe"* ]]; then
                    pkill -HUP -f "python.*bitaxe" 2>/dev/null || true
                fi
            else
                log_error "Failed to rotate log: $(basename "$logfile")"
            fi
        fi
    done < "/tmp/bitaxe_large_logs"
    
    log_success "Rotated $rotated_count large log files"
}

compress_old_logs() {
    log_info "Compressing old uncompressed log files..."
    
    local compressed_count=0
    
    # Find uncompressed rotated logs older than 1 day
    find "$LOG_DIR" -name "*.log.*" ! -name "*.gz" -mtime +1 -type f | while read -r logfile; do
        if [[ -f "$logfile" ]]; then
            log_info "Compressing old log: $(basename "$logfile")"
            
            if gzip "$logfile"; then
                log_success "Compressed: $(basename "$logfile")"
                ((compressed_count++))
            else
                log_warn "Failed to compress: $(basename "$logfile")"
            fi
        fi
    done
    
    log_info "Compressed additional old log files"
}

archive_logs() {
    log_info "Archiving very old log files..."
    
    local archive_dir="$BASE_DIR/archive/logs"
    local archive_threshold=$((LOG_RETENTION_DAYS + 7))  # Archive logs older than retention + 7 days
    
    mkdir -p "$archive_dir"
    
    local archived_count=0
    
    # Find logs older than archive threshold
    find "$LOG_DIR" -name "*.log.*.gz" -mtime +$archive_threshold -type f | while read -r logfile; do
        if [[ -f "$logfile" ]]; then
            local year_month=$(date -r "$logfile" '+%Y-%m' 2>/dev/null || date '+%Y-%m')
            local archive_subdir="$archive_dir/$year_month"
            
            mkdir -p "$archive_subdir"
            
            log_info "Archiving old log: $(basename "$logfile")"
            
            if mv "$logfile" "$archive_subdir/"; then
                log_success "Archived: $(basename "$logfile")"
                ((archived_count++))
            else
                log_warn "Failed to archive: $(basename "$logfile")"
            fi
        fi
    done
    
    # Create archive index
    if [[ $archived_count -gt 0 ]]; then
        find "$archive_dir" -name "*.gz" -exec ls -lh {} \; > "$archive_dir/archive_index.txt"
        log_success "Archived $archived_count log files"
    fi
}

cleanup_empty_logs() {
    log_info "Cleaning up empty log files..."
    
    if [[ ! -f "/tmp/bitaxe_empty_logs" ]]; then
        log_info "No empty logs found"
        return 0
    fi
    
    local cleaned_count=0
    
    while IFS= read -r logfile; do
        if [[ -f "$logfile" && ! -s "$logfile" ]]; then
            # Only remove empty logs that are older than 1 day
            local age_days=$(( ($(date +%s) - $(stat -f%m "$logfile" 2>/dev/null || stat -c%Y "$logfile" 2>/dev/null || echo "0")) / 86400 ))
            
            if [[ $age_days -gt 1 ]]; then
                log_info "Removing empty log: $(basename "$logfile")"
                
                if rm "$logfile"; then
                    log_success "Removed empty log: $(basename "$logfile")"
                    ((cleaned_count++))
                else
                    log_warn "Failed to remove empty log: $(basename "$logfile")"
                fi
            fi
        fi
    done < "/tmp/bitaxe_empty_logs"
    
    log_success "Cleaned up $cleaned_count empty log files"
}

delete_old_logs() {
    log_info "Deleting logs older than retention period..."
    
    if [[ ! -f "/tmp/bitaxe_old_logs" ]]; then
        log_info "No old logs found for deletion"
        return 0
    fi
    
    local deleted_count=0
    
    while IFS= read -r logfile; do
        if [[ -f "$logfile" ]]; then
            log_info "Deleting old log: $(basename "$logfile")"
            
            if rm "$logfile"; then
                log_success "Deleted old log: $(basename "$logfile")"
                ((deleted_count++))
            else
                log_warn "Failed to delete old log: $(basename "$logfile")"
            fi
        fi
    done < "/tmp/bitaxe_old_logs"
    
    log_success "Deleted $deleted_count old log files"
}

cleanup_system_logs() {
    log_info "Cleaning up system logs..."
    
    # Clean journal logs (keep last 7 days)
    if command -v journalctl >/dev/null 2>&1; then
        journalctl --vacuum-time=7d >/dev/null 2>&1 || log_warn "Failed to clean journal logs"
        log_success "System journal logs cleaned"
    fi
    
    # Clean Apache/Nginx logs if they exist
    for log_dir in /var/log/apache2 /var/log/nginx; do
        if [[ -d "$log_dir" ]]; then
            find "$log_dir" -name "*.log.*" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
            log_info "Cleaned logs in $log_dir"
        fi
    done
    
    # Clean syslog old files
    find /var/log -name "syslog.*" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    find /var/log -name "kern.log.*" -mtime +$LOG_RETENTION_DAYS -delete 2>/dev/null || true
    
    log_success "System logs cleanup completed"
}

generate_cleanup_report() {
    local report_file="$LOG_DIR/log_cleanup_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Load analysis results
    source "/tmp/bitaxe_log_analysis" 2>/dev/null || {
        total_size_mb=0
        large_logs_count=0
        old_logs_count=0
        empty_logs_count=0
    }
    
    # Get current log statistics
    local current_logs=$(find "$LOG_DIR" -name "*.log*" -type f | wc -l)
    local current_size=$(du -sm "$LOG_DIR" 2>/dev/null | cut -f1 || echo "0")
    
    # Create JSON report
    cat > "$report_file" <<EOF
{
  "timestamp": "$timestamp",
  "cleanup_status": "completed",
  "pre_cleanup": {
    "total_size_mb": $total_size_mb,
    "large_logs_count": $large_logs_count,
    "old_logs_count": $old_logs_count,
    "empty_logs_count": $empty_logs_count
  },
  "post_cleanup": {
    "current_logs": $current_logs,
    "current_size_mb": $current_size
  },
  "actions_performed": {
    "log_rotation": true,
    "compression": true,
    "archival": true,
    "deletion": true
  },
  "retention_policy": {
    "retention_days": $LOG_RETENTION_DAYS,
    "max_size": "$LOG_MAX_SIZE"
  },
  "next_cleanup": "$(date -d '+1 day' -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    log_success "Log cleanup report generated: $report_file"
}

cleanup_temporary_files() {
    # Clean up temporary files created by this script
    rm -f /tmp/bitaxe_discovered_logs
    rm -f /tmp/bitaxe_log_analysis
    rm -f /tmp/bitaxe_large_logs
    rm -f /tmp/bitaxe_old_logs
    rm -f /tmp/bitaxe_empty_logs
}

# Main cleanup execution
main() {
    log_info "=== BitAxe Log Cleanup Started ==="
    
    # Discovery and analysis phase
    discover_log_files
    analyze_log_files
    
    # Cleanup phases
    rotate_large_logs
    compress_old_logs
    archive_logs
    cleanup_empty_logs
    delete_old_logs
    cleanup_system_logs
    
    # Reporting
    generate_cleanup_report
    
    # Cleanup
    cleanup_temporary_files
    
    log_info "=== Log Cleanup Completed Successfully ==="
}

# Handle interrupts
trap 'log_error "Log cleanup interrupted"; cleanup_temporary_files; exit 1' INT TERM

# Run main function
main "$@"