#!/bin/bash
# BitAxe V2.0.0 - System Update and Security Patch Script
# Handles system updates with service coordination and rollback capability

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
LOG_FILE="$LOG_DIR/system_update.log"
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

# Update management functions
check_system_status() {
    log_info "Checking system status before updates..."
    
    # Check if all critical services are running
    local failed_services=0
    for service in $MONITORED_SERVICES; do
        if ! systemctl is-active --quiet "$service"; then
            log_warn "Service $service is not running"
            ((failed_services++))
        fi
    done
    
    if [[ $failed_services -gt 0 ]]; then
        log_error "System not ready for updates: $failed_services services are down"
        return 1
    fi
    
    # Check system resources
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -ge 90 ]]; then
        log_error "Insufficient disk space for updates: ${disk_usage}% used"
        return 1
    fi
    
    log_success "System status check passed"
    return 0
}

create_system_snapshot() {
    log_info "Creating system snapshot before updates..."
    
    local snapshot_dir="/opt/bitaxe-web/snapshots"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local snapshot_path="$snapshot_dir/pre_update_$timestamp"
    
    mkdir -p "$snapshot_path"
    
    # Backup critical configuration files
    cp -r "$BASE_DIR/config" "$snapshot_path/" 2>/dev/null || log_warn "Failed to backup config directory"
    
    # Backup systemd service files
    mkdir -p "$snapshot_path/systemd"
    cp /etc/systemd/system/bitaxe-* "$snapshot_path/systemd/" 2>/dev/null || log_warn "Failed to backup systemd files"
    
    # Backup current package list
    dpkg --get-selections > "$snapshot_path/package_list.txt"
    
    # Create database backup
    if [[ -f "$DATABASE_PATH" ]]; then
        sqlite3 "$DATABASE_PATH" ".backup $snapshot_path/database_backup.db"
        log_success "Database backup created"
    fi
    
    # Save system information
    cat > "$snapshot_path/system_info.txt" <<EOF
Snapshot created: $(date)
Kernel version: $(uname -r)
OS version: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
Uptime: $(uptime)
BitAxe services status:
$(systemctl status bitaxe-* --no-pager -l 2>/dev/null || echo "No BitAxe services found")
EOF
    
    echo "$snapshot_path" > "/tmp/bitaxe_last_snapshot"
    log_success "System snapshot created: $snapshot_path"
    
    # Clean up old snapshots (keep last 5)
    find "$snapshot_dir" -name "pre_update_*" -type d | sort -r | tail -n +6 | xargs rm -rf 2>/dev/null || true
}

stop_bitaxe_services() {
    log_info "Stopping BitAxe services for update..."
    
    local stopped_services=()
    
    for service in $MONITORED_SERVICES; do
        if systemctl is-active --quiet "$service"; then
            if systemctl stop "$service"; then
                stopped_services+=("$service")
                log_success "Stopped service: $service"
            else
                log_error "Failed to stop service: $service"
                return 1
            fi
        fi
    done
    
    # Save list of stopped services for restart
    printf '%s\n' "${stopped_services[@]}" > "/tmp/bitaxe_stopped_services"
    
    log_info "All BitAxe services stopped successfully"
}

start_bitaxe_services() {
    log_info "Starting BitAxe services after update..."
    
    if [[ -f "/tmp/bitaxe_stopped_services" ]]; then
        while IFS= read -r service; do
            if systemctl start "$service"; then
                log_success "Started service: $service"
                
                # Wait a moment and verify it's running
                sleep 2
                if systemctl is-active --quiet "$service"; then
                    log_success "Service $service is running properly"
                else
                    log_error "Service $service started but not running properly"
                fi
            else
                log_error "Failed to start service: $service"
            fi
        done < "/tmp/bitaxe_stopped_services"
        
        rm -f "/tmp/bitaxe_stopped_services"
    fi
}

perform_system_update() {
    log_info "Performing system package updates..."
    
    # Update package lists
    if ! apt-get update; then
        log_error "Failed to update package lists"
        return 1
    fi
    
    log_success "Package lists updated"
    
    # Check for available updates
    local updates=$(apt list --upgradable 2>/dev/null | grep -c upgradable || echo "0")
    
    if [[ $updates -eq 0 ]]; then
        log_info "No package updates available"
        return 0
    fi
    
    log_info "Found $updates package updates available"
    
    # Perform security updates first
    log_info "Installing security updates..."
    if unattended-upgrade --debug 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Security updates completed"
    else
        log_warn "Security updates completed with warnings"
    fi
    
    # Perform other critical updates
    log_info "Installing other system updates..."
    if DEBIAN_FRONTEND=noninteractive apt-get -y upgrade; then
        log_success "System updates completed"
    else
        log_error "System updates failed"
        return 1
    fi
    
    # Clean up package cache
    apt-get autoremove -y >/dev/null 2>&1 || true
    apt-get autoclean >/dev/null 2>&1 || true
    
    log_success "Package cache cleaned"
}

update_python_dependencies() {
    log_info "Updating Python dependencies..."
    
    local requirements_file="$BASE_DIR/requirements.txt"
    local venv_path="$BASE_DIR/venv"
    
    if [[ -f "$requirements_file" ]]; then
        # Activate virtual environment if it exists
        if [[ -d "$venv_path" ]]; then
            source "$venv_path/bin/activate"
            log_info "Virtual environment activated"
        fi
        
        # Update pip first
        if pip install --upgrade pip >/dev/null 2>&1; then
            log_success "pip updated"
        fi
        
        # Update all dependencies
        if pip install --upgrade -r "$requirements_file" >/dev/null 2>&1; then
            log_success "Python dependencies updated"
        else
            log_warn "Some Python dependencies failed to update"
        fi
        
        # Deactivate virtual environment
        if [[ -d "$venv_path" ]]; then
            deactivate 2>/dev/null || true
        fi
    else
        log_info "No requirements.txt found, skipping Python dependency updates"
    fi
}

verify_system_integrity() {
    log_info "Verifying system integrity after updates..."
    
    local integrity_issues=0
    
    # Check if all critical services can start
    for service in $MONITORED_SERVICES; do
        if ! systemctl is-active --quiet "$service"; then
            log_error "Service $service is not running after update"
            ((integrity_issues++))
        fi
    done
    
    # Test API endpoints
    if systemctl is-active --quiet "bitaxe-web"; then
        if ! curl -s --max-time 10 "http://localhost:5000/api/v1/health" >/dev/null 2>&1; then
            log_error "Web API not responding after update"
            ((integrity_issues++))
        else
            log_success "Web API responding correctly"
        fi
    fi
    
    # Check database integrity
    if [[ -f "$DATABASE_PATH" ]]; then
        if sqlite3 "$DATABASE_PATH" "PRAGMA integrity_check;" | grep -q "ok"; then
            log_success "Database integrity verified"
        else
            log_error "Database integrity check failed"
            ((integrity_issues++))
        fi
    fi
    
    # Check disk space after updates
    local disk_usage=$(df "$BASE_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -ge 85 ]]; then
        log_warn "Disk usage high after updates: ${disk_usage}%"
        ((integrity_issues++))
    fi
    
    if [[ $integrity_issues -eq 0 ]]; then
        log_success "System integrity verification passed"
        return 0
    else
        log_error "System integrity verification failed: $integrity_issues issues found"
        return 1
    fi
}

rollback_system() {
    log_error "Initiating system rollback..."
    
    local snapshot_path
    if [[ -f "/tmp/bitaxe_last_snapshot" ]]; then
        snapshot_path=$(cat "/tmp/bitaxe_last_snapshot")
    else
        log_error "No snapshot information found for rollback"
        return 1
    fi
    
    if [[ ! -d "$snapshot_path" ]]; then
        log_error "Snapshot directory not found: $snapshot_path"
        return 1
    fi
    
    log_info "Rolling back from snapshot: $snapshot_path"
    
    # Stop services
    stop_bitaxe_services
    
    # Restore configuration files
    if [[ -d "$snapshot_path/config" ]]; then
        cp -r "$snapshot_path/config/"* "$BASE_DIR/config/" 2>/dev/null || log_warn "Failed to restore config files"
        log_success "Configuration files restored"
    fi
    
    # Restore database if it exists and current database is corrupted
    if [[ -f "$snapshot_path/database_backup.db" ]]; then
        if ! sqlite3 "$DATABASE_PATH" "PRAGMA integrity_check;" | grep -q "ok" 2>/dev/null; then
            cp "$snapshot_path/database_backup.db" "$DATABASE_PATH"
            log_success "Database restored from snapshot"
        fi
    fi
    
    # Restore systemd files
    if [[ -d "$snapshot_path/systemd" ]]; then
        cp "$snapshot_path/systemd/"* /etc/systemd/system/ 2>/dev/null || log_warn "Failed to restore systemd files"
        systemctl daemon-reload
        log_success "Systemd files restored"
    fi
    
    # Restart services
    start_bitaxe_services
    
    log_success "System rollback completed"
}

generate_update_report() {
    local update_status="$1"
    local report_file="$LOG_DIR/update_report.json"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Get current system information
    local kernel_version=$(uname -r)
    local os_version=$(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")
    local uptime=$(uptime -p)
    
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
    
    # Create JSON report
    cat > "$report_file" <<EOF
{
  "timestamp": "$timestamp",
  "update_status": "$update_status",
  "system_info": {
    "kernel_version": "$kernel_version",
    "os_version": "$os_version",
    "uptime": "$uptime"
  },
  "services": $services_status,
  "reboot_required": $(test -f /var/run/reboot-required && echo "true" || echo "false"),
  "next_update_check": "$(date -d '+1 week' -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    log_success "Update report generated: $report_file"
    
    # Check if reboot is required
    if [[ -f /var/run/reboot-required ]]; then
        log_warn "SYSTEM REBOOT REQUIRED - Some updates require a system restart"
        echo "System reboot required after updates" > "$LOG_DIR/reboot_required.flag"
    fi
}

# Main update execution
main() {
    if [[ "$AUTO_UPDATE_ENABLED" != "true" ]]; then
        log_info "Automatic updates disabled"
        exit 0
    fi
    
    log_info "=== BitAxe System Update Started ==="
    
    local update_status="FAILED"
    local rollback_needed=false
    
    # Pre-update checks
    if ! check_system_status; then
        log_error "Pre-update system checks failed"
        exit 1
    fi
    
    # Create system snapshot
    create_system_snapshot
    
    # Stop services
    if ! stop_bitaxe_services; then
        log_error "Failed to stop services for update"
        exit 1
    fi
    
    # Perform updates
    if perform_system_update; then
        log_success "System updates completed"
        
        # Update Python dependencies
        update_python_dependencies
        
        update_status="SUCCESS"
    else
        log_error "System updates failed"
        rollback_needed=true
    fi
    
    # Restart services
    start_bitaxe_services
    
    # Verify system integrity
    if ! verify_system_integrity; then
        log_error "Post-update integrity check failed"
        rollback_needed=true
        update_status="INTEGRITY_FAILED"
    fi
    
    # Perform rollback if needed
    if [[ "$rollback_needed" == "true" ]]; then
        rollback_system
        update_status="ROLLED_BACK"
    fi
    
    # Generate report
    generate_update_report "$update_status"
    
    log_info "=== System Update Completed: $update_status ==="
    
    # Exit with appropriate code
    case "$update_status" in
        "SUCCESS") exit 0 ;;
        "ROLLED_BACK") exit 2 ;;
        *) exit 1 ;;
    esac
}

# Handle interrupts
trap 'log_error "Update process interrupted"; start_bitaxe_services; exit 1' INT TERM

# Run main function
main "$@"