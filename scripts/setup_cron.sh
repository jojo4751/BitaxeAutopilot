#!/bin/bash
# BitAxe V2.0.0 - Cron Jobs Setup Script
# Configures automated scheduling for all monitoring and maintenance tasks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "[$(date '+%H:%M:%S')] $1"
}

log_success() {
    echo -e "[$(date '+%H:%M:%S')] ${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "[$(date '+%H:%M:%S')] ${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "[$(date '+%H:%M:%S')] ${RED}✗${NC} $1"
}

log_info() {
    echo -e "[$(date '+%H:%M:%S')] ${BLUE}ℹ${NC} $1"
}

# Load configuration
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        log_success "Configuration loaded from $CONFIG_FILE"
    else
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
}

# Check if running as root
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root to setup system cron jobs"
        log_info "Please run: sudo $0"
        exit 1
    fi
    log_success "Running with root permissions"
}

# Backup existing cron jobs
backup_existing_cron() {
    local backup_dir="/opt/bitaxe-web/backups/cron"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    mkdir -p "$backup_dir"
    
    # Backup root crontab
    if crontab -l >/dev/null 2>&1; then
        crontab -l > "$backup_dir/root_crontab_$timestamp.bak"
        log_success "Root crontab backed up to $backup_dir/root_crontab_$timestamp.bak"
    fi
    
    # Backup bitaxe user crontab if exists
    if id "bitaxe" >/dev/null 2>&1 && crontab -u bitaxe -l >/dev/null 2>&1; then
        crontab -u bitaxe -l > "$backup_dir/bitaxe_crontab_$timestamp.bak"
        log_success "BitAxe user crontab backed up"
    fi
}

# Create systemd timer services (preferred over cron for service management)
create_systemd_timers() {
    log_info "Creating systemd timer services..."
    
    # Health Check Timer
    cat > /etc/systemd/system/bitaxe-health-check.timer <<EOF
[Unit]
Description=BitAxe Health Check Timer
Requires=bitaxe-health-check.service

[Timer]
OnCalendar=*:*:0/5
Persistent=true
AccuracySec=30s

[Install]
WantedBy=timers.target
EOF

    cat > /etc/systemd/system/bitaxe-health-check.service <<EOF
[Unit]
Description=BitAxe Health Check Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/health_check.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Metrics Collection Timer
    cat > /etc/systemd/system/bitaxe-metrics.timer <<EOF
[Unit]
Description=BitAxe Metrics Collection Timer
Requires=bitaxe-metrics.service

[Timer]
OnCalendar=*:*:0
Persistent=true
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF

    cat > /etc/systemd/system/bitaxe-metrics.service <<EOF
[Unit]
Description=BitAxe Metrics Collection Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/collect_metrics.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    # Daily Backup Timer
    cat > /etc/systemd/system/bitaxe-backup.timer <<EOF
[Unit]
Description=BitAxe Backup Timer
Requires=bitaxe-backup.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
EOF

    cat > /etc/systemd/system/bitaxe-backup.service <<EOF
[Unit]
Description=BitAxe Backup Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/backup_sync.sh
StandardOutput=journal
StandardError=journal
TimeoutSec=3600

[Install]
WantedBy=multi-user.target
EOF

    # Weekly Maintenance Timer
    cat > /etc/systemd/system/bitaxe-maintenance.timer <<EOF
[Unit]
Description=BitAxe Maintenance Timer
Requires=bitaxe-maintenance.service

[Timer]
OnCalendar=Sun 02:00
Persistent=true
RandomizedDelaySec=3600

[Install]
WantedBy=timers.target
EOF

    cat > /etc/systemd/system/bitaxe-maintenance.service <<EOF
[Unit]
Description=BitAxe Maintenance Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/maintenance.sh
StandardOutput=journal
StandardError=journal
TimeoutSec=7200

[Install]
WantedBy=multi-user.target
EOF

    # Daily Log Cleanup Timer
    cat > /etc/systemd/system/bitaxe-log-cleanup.timer <<EOF
[Unit]
Description=BitAxe Log Cleanup Timer
Requires=bitaxe-log-cleanup.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=600

[Install]
WantedBy=timers.target
EOF

    cat > /etc/systemd/system/bitaxe-log-cleanup.service <<EOF
[Unit]
Description=BitAxe Log Cleanup Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/log_cleanup.sh
StandardOutput=journal
StandardError=journal
TimeoutSec=1800

[Install]
WantedBy=multi-user.target
EOF

    # Monthly System Update Timer (if enabled)
    if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
        cat > /etc/systemd/system/bitaxe-system-update.timer <<EOF
[Unit]
Description=BitAxe System Update Timer
Requires=bitaxe-system-update.service

[Timer]
OnCalendar=monthly
Persistent=true
RandomizedDelaySec=86400

[Install]
WantedBy=timers.target
EOF

        cat > /etc/systemd/system/bitaxe-system-update.service <<EOF
[Unit]
Description=BitAxe System Update Service
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=$SCRIPT_DIR/system_update.sh
StandardOutput=journal
StandardError=journal
TimeoutSec=7200

[Install]
WantedBy=multi-user.target
EOF
    fi

    log_success "Systemd timer services created"
}

# Enable and start systemd timers
enable_systemd_timers() {
    log_info "Enabling and starting systemd timers..."
    
    # Reload systemd configuration
    systemctl daemon-reload
    
    local timers=(
        "bitaxe-health-check.timer"
        "bitaxe-metrics.timer"
        "bitaxe-backup.timer"
        "bitaxe-maintenance.timer"
        "bitaxe-log-cleanup.timer"
    )
    
    if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
        timers+=("bitaxe-system-update.timer")
    fi
    
    for timer in "${timers[@]}"; do
        if systemctl enable "$timer"; then
            log_success "Enabled $timer"
        else
            log_error "Failed to enable $timer"
        fi
        
        if systemctl start "$timer"; then
            log_success "Started $timer"
        else
            log_error "Failed to start $timer"
        fi
    done
    
    log_success "All systemd timers enabled and started"
}

# Create fallback cron jobs (as backup to systemd timers)
create_fallback_cron() {
    log_info "Creating fallback cron jobs..."
    
    local cron_file="/etc/cron.d/bitaxe-monitoring"
    
    cat > "$cron_file" <<EOF
# BitAxe V2.0.0 - Monitoring and Maintenance Cron Jobs
# These serve as fallback if systemd timers fail

SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# Health check every 5 minutes
*/5 * * * * root $SCRIPT_DIR/health_check.sh >/dev/null 2>&1

# Metrics collection every minute
* * * * * root $SCRIPT_DIR/collect_metrics.sh >/dev/null 2>&1

# Daily backup at 2:30 AM
30 2 * * * root $SCRIPT_DIR/backup_sync.sh >/dev/null 2>&1

# Weekly maintenance on Sunday at 3:00 AM
0 3 * * 0 root $SCRIPT_DIR/maintenance.sh >/dev/null 2>&1

# Daily log cleanup at 1:00 AM
0 1 * * * root $SCRIPT_DIR/log_cleanup.sh >/dev/null 2>&1

EOF

    # Add system update cron if enabled
    if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
        echo "# Monthly system update on first Sunday at 4:00 AM" >> "$cron_file"
        echo "0 4 1-7 * 0 root $SCRIPT_DIR/system_update.sh >/dev/null 2>&1" >> "$cron_file"
    fi
    
    chmod 644 "$cron_file"
    log_success "Fallback cron jobs created in $cron_file"
}

# Create user-specific cron jobs for non-root tasks
create_user_cron() {
    log_info "Creating user-specific cron jobs..."
    
    # Check if bitaxe user exists
    if ! id "bitaxe" >/dev/null 2>&1; then
        log_info "BitAxe user does not exist, creating..."
        useradd -r -s /bin/bash -d /opt/bitaxe-web bitaxe
        log_success "BitAxe user created"
    fi
    
    # Create temp cron file for bitaxe user
    local temp_cron="/tmp/bitaxe_user_cron"
    
    cat > "$temp_cron" <<EOF
# BitAxe User Cron Jobs
# Application-specific monitoring tasks

# Check application logs every 10 minutes
*/10 * * * * find $LOG_DIR -name "*.log" -size +100M -exec echo "Large log file: {}" \; 2>/dev/null

# Clear temporary files every hour
0 * * * * find /tmp -name "bitaxe_*" -mtime +1 -delete 2>/dev/null

EOF
    
    # Install cron for bitaxe user
    if crontab -u bitaxe "$temp_cron"; then
        log_success "User cron jobs installed for bitaxe user"
    else
        log_warning "Failed to install user cron jobs"
    fi
    
    rm -f "$temp_cron"
}

# Validate cron setup
validate_cron_setup() {
    log_info "Validating cron setup..."
    
    # Check systemd timers
    local timers=(
        "bitaxe-health-check.timer"
        "bitaxe-metrics.timer"
        "bitaxe-backup.timer"
        "bitaxe-maintenance.timer"
        "bitaxe-log-cleanup.timer"
    )
    
    local failed_timers=0
    for timer in "${timers[@]}"; do
        if systemctl is-enabled "$timer" >/dev/null 2>&1; then
            if systemctl is-active "$timer" >/dev/null 2>&1; then
                log_success "$timer is enabled and active"
            else
                log_warning "$timer is enabled but not active"
                ((failed_timers++))
            fi
        else
            log_error "$timer is not enabled"
            ((failed_timers++))
        fi
    done
    
    # Check cron service
    if systemctl is-active cron >/dev/null 2>&1 || systemctl is-active crond >/dev/null 2>&1; then
        log_success "Cron service is running"
    else
        log_warning "Cron service is not running"
    fi
    
    # Check if scripts are executable
    local scripts=(
        "health_check.sh"
        "collect_metrics.sh"
        "backup_sync.sh"
        "maintenance.sh"
        "log_cleanup.sh"
    )
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [[ -x "$script_path" ]]; then
            log_success "$script is executable"
        else
            log_warning "$script is not executable"
            chmod +x "$script_path" 2>/dev/null || log_error "Failed to make $script executable"
        fi
    done
    
    if [[ $failed_timers -eq 0 ]]; then
        log_success "All cron validation checks passed"
        return 0
    else
        log_warning "Some cron validation checks failed"
        return 1
    fi
}

# Display schedule summary
show_schedule_summary() {
    log_info "=== BitAxe Monitoring Schedule Summary ==="
    echo
    echo "Systemd Timers:"
    echo "  Health Check:     Every 5 minutes"
    echo "  Metrics Collection: Every minute"
    echo "  Backup:           Daily at 2:00 AM (randomized ±30min)"
    echo "  Maintenance:      Weekly on Sunday at 2:00 AM (randomized ±1hr)"
    echo "  Log Cleanup:      Daily at midnight (randomized ±10min)"
    
    if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
        echo "  System Updates:   Monthly (randomized within first week)"
    else
        echo "  System Updates:   Disabled"
    fi
    
    echo
    echo "Fallback Cron Jobs:"
    echo "  Health Check:     */5 * * * *"
    echo "  Metrics:          * * * * *"
    echo "  Backup:           30 2 * * *"
    echo "  Maintenance:      0 3 * * 0"
    echo "  Log Cleanup:      0 1 * * *"
    
    if [[ "$AUTO_UPDATE_ENABLED" == "true" ]]; then
        echo "  System Updates:   0 4 1-7 * 0"
    fi
    
    echo
    echo "Management Commands:"
    echo "  View timer status:    systemctl list-timers 'bitaxe-*'"
    echo "  View logs:           journalctl -u bitaxe-*.service"
    echo "  Stop all timers:     systemctl stop bitaxe-*.timer"
    echo "  Start all timers:    systemctl start bitaxe-*.timer"
    echo
}

# Create management scripts
create_management_scripts() {
    log_info "Creating timer management scripts..."
    
    # Timer control script
    cat > "$SCRIPT_DIR/control_timers.sh" <<'EOF'
#!/bin/bash
# BitAxe Timer Control Script

case "$1" in
    start)
        echo "Starting all BitAxe timers..."
        systemctl start bitaxe-*.timer
        echo "All timers started"
        ;;
    stop)
        echo "Stopping all BitAxe timers..."
        systemctl stop bitaxe-*.timer
        echo "All timers stopped"
        ;;
    restart)
        echo "Restarting all BitAxe timers..."
        systemctl restart bitaxe-*.timer
        echo "All timers restarted"
        ;;
    status)
        echo "BitAxe Timer Status:"
        systemctl list-timers 'bitaxe-*'
        ;;
    logs)
        service_name="${2:-}"
        if [[ -n "$service_name" ]]; then
            journalctl -u "bitaxe-$service_name.service" -f
        else
            echo "Available services: health-check, metrics, backup, maintenance, log-cleanup"
            echo "Usage: $0 logs <service-name>"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs [service-name]}"
        exit 1
        ;;
esac
EOF

    chmod +x "$SCRIPT_DIR/control_timers.sh"
    log_success "Timer control script created: $SCRIPT_DIR/control_timers.sh"
    
    # Schedule viewer script
    cat > "$SCRIPT_DIR/view_schedule.sh" <<'EOF'
#!/bin/bash
# BitAxe Schedule Viewer

echo "=== BitAxe Monitoring Schedule ==="
echo
echo "Current Timer Status:"
systemctl list-timers 'bitaxe-*' --no-pager
echo
echo "Next Scheduled Runs:"
systemctl list-timers 'bitaxe-*' --no-pager | grep NEXT
echo
echo "Recent Service Logs (last 10 entries):"
journalctl -u 'bitaxe-*.service' -n 10 --no-pager
EOF

    chmod +x "$SCRIPT_DIR/view_schedule.sh"
    log_success "Schedule viewer script created: $SCRIPT_DIR/view_schedule.sh"
}

# Main execution
main() {
    log_info "=== BitAxe Cron Setup Started ==="
    
    check_permissions
    load_config
    backup_existing_cron
    
    create_systemd_timers
    enable_systemd_timers
    create_fallback_cron
    create_user_cron
    create_management_scripts
    
    if validate_cron_setup; then
        log_success "Cron setup completed successfully!"
        show_schedule_summary
        
        echo
        log_info "Next steps:"
        echo "1. Check timer status: systemctl list-timers 'bitaxe-*'"
        echo "2. View logs: journalctl -u bitaxe-health-check.service"
        echo "3. Control timers: $SCRIPT_DIR/control_timers.sh status"
        echo "4. View schedule: $SCRIPT_DIR/view_schedule.sh"
    else
        log_error "Cron setup completed with warnings"
        exit 1
    fi
}

# Handle interrupts
trap 'log_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"