#!/bin/bash
# BitAxe V2.0.0 - Schedule Viewer Script
# Displays comprehensive information about all scheduled monitoring tasks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"

# Load configuration if available
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE" 2>/dev/null || true
fi

print_header() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}               BitAxe V2.0.0 - Monitoring Schedule${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo
}

print_section() {
    local title="$1"
    echo -e "${BLUE}=== $title ===${NC}"
    echo
}

# Show current systemd timer status
show_timer_status() {
    print_section "Systemd Timer Status"
    
    if ! command -v systemctl >/dev/null 2>&1; then
        echo -e "${RED}systemctl not available${NC}"
        return 1
    fi
    
    # Check if any BitAxe timers exist
    local timer_count
    timer_count=$(systemctl list-unit-files 'bitaxe-*.timer' --no-legend 2>/dev/null | wc -l)
    
    if [[ $timer_count -eq 0 ]]; then
        echo -e "${YELLOW}No BitAxe timers found. Run setup_cron.sh to install.${NC}"
        echo
        return 1
    fi
    
    # Show active timers with next run times
    echo -e "${GREEN}Active Timers:${NC}"
    systemctl list-timers 'bitaxe-*' --no-pager 2>/dev/null || {
        echo -e "${YELLOW}No active timers found${NC}"
        echo
        return 1
    }
    
    echo
    
    # Show individual timer details
    echo -e "${GREEN}Timer Details:${NC}"
    local timers=(
        "bitaxe-health-check.timer"
        "bitaxe-metrics.timer"
        "bitaxe-backup.timer"
        "bitaxe-maintenance.timer"
        "bitaxe-log-cleanup.timer"
        "bitaxe-system-update.timer"
    )
    
    for timer in "${timers[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            local enabled
            local active
            enabled=$(systemctl is-enabled "$timer" 2>/dev/null || echo "disabled")
            active=$(systemctl is-active "$timer" 2>/dev/null || echo "inactive")
            
            local status_color
            case "$active" in
                "active") status_color="$GREEN" ;;
                "inactive") status_color="$YELLOW" ;;
                *) status_color="$RED" ;;
            esac
            
            printf "  %-30s ${status_color}%s${NC} (%s)\n" "$timer" "$active" "$enabled"
        fi
    done
    
    echo
}

# Show schedule configuration
show_schedule_config() {
    print_section "Configured Schedule"
    
    local schedules=(
        "Health Check|Every 5 minutes|Monitors system health and restarts failed services"
        "Metrics Collection|Every minute|Collects system and mining performance metrics"
        "Database Backup|Daily at 2:00 AM|Creates database backup and syncs to laptop"
        "System Maintenance|Weekly on Sunday at 2:00 AM|Performs database optimization and cleanup"
        "Log Cleanup|Daily at midnight|Rotates and archives log files"
    )
    
    if [[ "${AUTO_UPDATE_ENABLED:-false}" == "true" ]]; then
        schedules+=("System Updates|Monthly|Installs security and system updates")
    fi
    
    printf "%-20s %-25s %s\n" "Task" "Schedule" "Description"
    echo "-------------------- ------------------------- --------------------------------------------------"
    
    for schedule in "${schedules[@]}"; do
        IFS='|' read -r task timing description <<< "$schedule"
        printf "%-20s %-25s %s\n" "$task" "$timing" "$description"
    done
    
    echo
}

# Show recent service executions
show_recent_executions() {
    print_section "Recent Service Executions"
    
    if ! command -v journalctl >/dev/null 2>&1; then
        echo -e "${RED}journalctl not available${NC}"
        echo
        return 1
    fi
    
    echo -e "${GREEN}Last 10 service executions:${NC}"
    journalctl -u 'bitaxe-*.service' -n 10 --no-pager --output=short-precise 2>/dev/null || {
        echo -e "${YELLOW}No recent service executions found${NC}"
        echo
        return 1
    }
    
    echo
}

# Show next scheduled runs
show_next_runs() {
    print_section "Next Scheduled Runs"
    
    if ! command -v systemctl >/dev/null 2>&1; then
        echo -e "${RED}systemctl not available${NC}"
        echo
        return 1
    fi
    
    local next_runs
    next_runs=$(systemctl list-timers 'bitaxe-*' --no-pager 2>/dev/null | grep -E "NEXT|next" | head -10)
    
    if [[ -n "$next_runs" ]]; then
        echo -e "${GREEN}Upcoming executions:${NC}"
        echo "$next_runs"
    else
        echo -e "${YELLOW}No upcoming executions found${NC}"
    fi
    
    echo
}

# Show service health
show_service_health() {
    print_section "Service Health Status"
    
    local services=(
        "bitaxe-health-check.service|Health monitoring service"
        "bitaxe-metrics.service|Metrics collection service"
        "bitaxe-backup.service|Backup service"
        "bitaxe-maintenance.service|Maintenance service"
        "bitaxe-log-cleanup.service|Log cleanup service"
        "bitaxe-system-update.service|System update service"
    )
    
    printf "%-35s %-20s %s\n" "Service" "Last Result" "Description"
    echo "----------------------------------- -------------------- ----------------------------------"
    
    for service_info in "${services[@]}"; do
        IFS='|' read -r service description <<< "$service_info"
        
        if systemctl list-unit-files "$service" >/dev/null 2>&1; then
            # Get last execution result
            local last_result
            last_result=$(systemctl show "$service" -p ExecMainStatus --value 2>/dev/null || echo "unknown")
            
            local result_color
            local result_text
            case "$last_result" in
                "0") result_color="$GREEN"; result_text="Success" ;;
                "") result_color="$YELLOW"; result_text="Never run" ;;
                "unknown") result_color="$YELLOW"; result_text="Unknown" ;;
                *) result_color="$RED"; result_text="Failed ($last_result)" ;;
            esac
            
            printf "%-35s ${result_color}%-20s${NC} %s\n" "$service" "$result_text" "$description"
        else
            printf "%-35s ${YELLOW}%-20s${NC} %s\n" "$service" "Not installed" "$description"
        fi
    done
    
    echo
}

# Show disk usage and log information
show_system_info() {
    print_section "System Information"
    
    # System uptime
    if command -v uptime >/dev/null 2>&1; then
        echo -e "${GREEN}System Uptime:${NC}"
        uptime
        echo
    fi
    
    # Disk usage
    echo -e "${GREEN}Disk Usage:${NC}"
    if [[ -d "$BASE_DIR" ]]; then
        df -h "$BASE_DIR" 2>/dev/null || df -h / 2>/dev/null
    else
        df -h / 2>/dev/null
    fi
    echo
    
    # Log directory size
    if [[ -d "${LOG_DIR:-/var/log/bitaxe}" ]]; then
        echo -e "${GREEN}Log Directory Usage:${NC}"
        du -sh "${LOG_DIR:-/var/log/bitaxe}" 2>/dev/null || echo "Log directory not accessible"
        echo
    fi
    
    # Database size
    local db_path="${DATABASE_PATH:-/opt/bitaxe-web/data/bitaxe_data.db}"
    if [[ -f "$db_path" ]]; then
        echo -e "${GREEN}Database Size:${NC}"
        ls -lh "$db_path" | awk '{print $5, $9}'
        echo
    fi
}

# Show configuration summary
show_config_summary() {
    print_section "Configuration Summary"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        echo -e "${GREEN}Configuration file: ${NC}$CONFIG_FILE"
        echo
        
        # Show key configuration values
        local configs=(
            "BACKUP_ENABLED|Backup System"
            "HEALTH_CHECK_ENABLED|Health Monitoring"
            "SYSTEM_MONITORING_ENABLED|System Metrics"
            "AUTO_UPDATE_ENABLED|Automatic Updates"
            "LAPTOP_BACKUP_ENABLED|Laptop Sync"
        )
        
        printf "%-25s %s\n" "Setting" "Status"
        echo "------------------------- --------"
        
        for config_info in "${configs[@]}"; do
            IFS='|' read -r config_key config_desc <<< "$config_info"
            local config_value
            config_value=$(grep "^$config_key=" "$CONFIG_FILE" 2>/dev/null | cut -d'=' -f2 | tr -d '"' || echo "unknown")
            
            local status_color
            case "$config_value" in
                "true") status_color="$GREEN" ;;
                "false") status_color="$YELLOW" ;;
                *) status_color="$RED" ;;
            esac
            
            printf "%-25s ${status_color}%s${NC}\n" "$config_desc" "$config_value"
        done
        
        echo
    else
        echo -e "${YELLOW}Configuration file not found: $CONFIG_FILE${NC}"
        echo
    fi
}

# Show available management commands
show_management_commands() {
    print_section "Management Commands"
    
    local script_dir="${SCRIPT_DIR}"
    
    echo -e "${GREEN}Available management scripts:${NC}"
    echo
    
    local commands=(
        "control_timers.sh start|Start all monitoring timers"
        "control_timers.sh stop|Stop all monitoring timers"
        "control_timers.sh status|Show detailed timer status"
        "control_timers.sh logs health-check|View health check logs"
        "setup_cron.sh|Install/reinstall monitoring schedule"
        "health_check.sh|Run manual health check"
        "collect_metrics.sh|Run manual metrics collection"
        "backup_sync.sh|Run manual backup"
        "maintenance.sh|Run manual maintenance"
    )
    
    for command_info in "${commands[@]}"; do
        IFS='|' read -r command description <<< "$command_info"
        local script_path="$script_dir/$command"
        
        if [[ -x "$script_path" ]]; then
            printf "  ${CYAN}%-30s${NC} %s\n" "$command" "$description"
        else
            printf "  ${YELLOW}%-30s${NC} %s (not executable)\n" "$command" "$description"
        fi
    done
    
    echo
    echo -e "${GREEN}System commands:${NC}"
    echo "  systemctl list-timers 'bitaxe-*'     Show timer schedule"
    echo "  journalctl -u bitaxe-*.service       Show service logs"
    echo "  systemctl status bitaxe-*.timer      Show timer status"
    echo
}

# Main execution
main() {
    local option="${1:-all}"
    
    case "$option" in
        timers|timer)
            print_header
            show_timer_status
            ;;
        schedule|config)
            print_header
            show_schedule_config
            ;;
        recent|executions)
            print_header
            show_recent_executions
            ;;
        next|upcoming)
            print_header
            show_next_runs
            ;;
        health|status)
            print_header
            show_service_health
            ;;
        system|info)
            print_header
            show_system_info
            ;;
        config|configuration)
            print_header
            show_config_summary
            ;;
        commands|help)
            print_header
            show_management_commands
            ;;
        all|*)
            print_header
            show_timer_status
            show_schedule_config
            show_next_runs
            show_recent_executions
            show_service_health
            show_system_info
            show_config_summary
            show_management_commands
            ;;
    esac
}

# Show help if requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "BitAxe Schedule Viewer"
    echo "====================="
    echo
    echo "Usage: $0 [option]"
    echo
    echo "Options:"
    echo "  all        - Show complete schedule information (default)"
    echo "  timers     - Show systemd timer status"
    echo "  schedule   - Show configured schedule"
    echo "  recent     - Show recent service executions"
    echo "  next       - Show next scheduled runs"
    echo "  health     - Show service health status"
    echo "  system     - Show system information"
    echo "  config     - Show configuration summary"
    echo "  commands   - Show available management commands"
    echo
    exit 0
fi

# Run main function
main "$@"