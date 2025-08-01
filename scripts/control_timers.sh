#!/bin/bash
# BitAxe V2.0.0 - Timer Control Script
# Provides easy management of all BitAxe systemd timers

set -euo pipefail

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

# BitAxe timer services
BITAXE_TIMERS=(
    "bitaxe-health-check.timer"
    "bitaxe-metrics.timer"
    "bitaxe-backup.timer"
    "bitaxe-maintenance.timer"
    "bitaxe-log-cleanup.timer"
    "bitaxe-system-update.timer"
)

# Check if running with appropriate permissions
check_permissions() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script requires root permissions to control systemd timers"
        log_info "Please run: sudo $0 $*"
        exit 1
    fi
}

# Start all BitAxe timers
start_timers() {
    log_info "Starting all BitAxe timers..."
    
    local started=0
    local failed=0
    
    for timer in "${BITAXE_TIMERS[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            if systemctl start "$timer" 2>/dev/null; then
                log_success "Started $timer"
                ((started++))
            else
                log_error "Failed to start $timer"
                ((failed++))
            fi
        else
            log_warning "$timer not found (may not be installed)"
        fi
    done
    
    log_info "Timer start summary: $started started, $failed failed"
    
    if [[ $failed -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

# Stop all BitAxe timers
stop_timers() {
    log_info "Stopping all BitAxe timers..."
    
    local stopped=0
    local failed=0
    
    for timer in "${BITAXE_TIMERS[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            if systemctl stop "$timer" 2>/dev/null; then
                log_success "Stopped $timer"
                ((stopped++))
            else
                log_warning "Failed to stop $timer (may already be stopped)"
            fi
        else
            log_warning "$timer not found"
        fi
    done
    
    log_info "Timer stop summary: $stopped stopped"
    return 0
}

# Restart all BitAxe timers
restart_timers() {
    log_info "Restarting all BitAxe timers..."
    
    stop_timers
    sleep 2
    start_timers
    
    log_success "Timer restart completed"
}

# Show timer status
show_status() {
    log_info "BitAxe Timer Status:"
    echo
    
    # Show detailed timer status
    systemctl list-timers 'bitaxe-*' --no-pager 2>/dev/null || {
        log_warning "No BitAxe timers found or systemctl unavailable"
        return 1
    }
    
    echo
    log_info "Individual Timer Status:"
    
    for timer in "${BITAXE_TIMERS[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            local status=$(systemctl is-active "$timer" 2>/dev/null || echo "inactive")
            local enabled=$(systemctl is-enabled "$timer" 2>/dev/null || echo "disabled")
            
            case "$status" in
                "active")
                    log_success "$timer: $status ($enabled)"
                    ;;
                "inactive")
                    log_warning "$timer: $status ($enabled)"
                    ;;
                *)
                    log_error "$timer: $status ($enabled)"
                    ;;
            esac
        else
            log_warning "$timer: not installed"
        fi
    done
}

# Show timer logs
show_logs() {
    local service_name="$1"
    
    if [[ -z "$service_name" ]]; then
        log_info "Available services:"
        echo "  health-check  - System health monitoring"
        echo "  metrics       - Performance metrics collection" 
        echo "  backup        - Database backup and sync"
        echo "  maintenance   - System maintenance tasks"
        echo "  log-cleanup   - Log rotation and cleanup"
        echo "  system-update - System package updates"
        echo
        log_info "Usage: $0 logs <service-name>"
        echo "Example: $0 logs health-check"
        return 1
    fi
    
    local full_service_name="bitaxe-$service_name.service"
    
    if systemctl list-unit-files "$full_service_name" >/dev/null 2>&1; then
        log_info "Showing logs for $full_service_name (Press Ctrl+C to exit)..."
        echo
        journalctl -u "$full_service_name" -f --no-pager
    else
        log_error "Service $full_service_name not found"
        return 1
    fi
}

# Show recent service executions
show_recent() {
    log_info "Recent BitAxe Service Executions:"
    echo
    
    journalctl -u 'bitaxe-*.service' -n 20 --no-pager --output=short-precise
}

# Enable all BitAxe timers
enable_timers() {
    log_info "Enabling all BitAxe timers..."
    
    local enabled=0
    local failed=0
    
    for timer in "${BITAXE_TIMERS[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            if systemctl enable "$timer" 2>/dev/null; then
                log_success "Enabled $timer"
                ((enabled++))
            else
                log_error "Failed to enable $timer"
                ((failed++))
            fi
        else
            log_warning "$timer not found"
        fi
    done
    
    log_info "Timer enable summary: $enabled enabled, $failed failed"
    
    if [[ $failed -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

# Disable all BitAxe timers
disable_timers() {
    log_info "Disabling all BitAxe timers..."
    
    local disabled=0
    
    for timer in "${BITAXE_TIMERS[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            if systemctl disable "$timer" 2>/dev/null; then
                log_success "Disabled $timer"
                ((disabled++))
            else
                log_warning "Failed to disable $timer"
            fi
        else
            log_warning "$timer not found"
        fi
    done
    
    log_info "Timer disable summary: $disabled disabled"
    return 0
}

# Test all timer services (dry run)
test_services() {
    log_info "Testing all BitAxe services (dry run)..."
    echo
    
    local services=(
        "bitaxe-health-check.service"
        "bitaxe-metrics.service"
        "bitaxe-backup.service"
        "bitaxe-maintenance.service"
        "bitaxe-log-cleanup.service"
        "bitaxe-system-update.service"
    )
    
    for service in "${services[@]}"; do
        if systemctl list-unit-files "$service" >/dev/null 2>&1; then
            log_info "Testing $service..."
            
            if systemctl start "$service" 2>/dev/null; then
                # Wait a moment for service to complete
                sleep 2
                
                local status=$(systemctl is-active "$service" 2>/dev/null || echo "inactive")
                if [[ "$status" == "inactive" ]]; then
                    log_success "$service completed successfully"
                else
                    log_warning "$service is still running"
                fi
            else
                log_error "Failed to start $service"
            fi
        else
            log_warning "$service not found"
        fi
    done
    
    log_info "Service testing completed"
}

# Show help information
show_help() {
    echo "BitAxe Timer Control Script"
    echo "=========================="
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  start      - Start all BitAxe timers"
    echo "  stop       - Stop all BitAxe timers"  
    echo "  restart    - Restart all BitAxe timers"
    echo "  status     - Show timer status and schedule"
    echo "  logs       - Show logs for a specific service"
    echo "  recent     - Show recent service executions"
    echo "  enable     - Enable all BitAxe timers"
    echo "  disable    - Disable all BitAxe timers"
    echo "  test       - Test all services (manual execution)"
    echo "  help       - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 status                    # Show all timer status"
    echo "  $0 logs health-check         # Show health check logs"
    echo "  $0 start                     # Start all timers"
    echo "  $0 test                      # Test all services"
    echo
    echo "Log Services:"
    echo "  health-check, metrics, backup, maintenance, log-cleanup, system-update"
    echo
}

# Main execution
main() {
    local command="${1:-help}"
    
    case "$command" in
        start)
            check_permissions
            start_timers
            ;;
        stop)
            check_permissions
            stop_timers
            ;;
        restart)
            check_permissions
            restart_timers
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        recent)
            show_recent
            ;;
        enable)
            check_permissions
            enable_timers
            ;;
        disable)
            check_permissions
            disable_timers
            ;;
        test)
            check_permissions
            test_services
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo
            show_help
            exit 1
            ;;
    esac
}

# Handle interrupts gracefully
trap 'log_warning "Operation interrupted"; exit 1' INT TERM

# Run main function with all arguments
main "$@"