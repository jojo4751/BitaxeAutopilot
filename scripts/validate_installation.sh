#!/bin/bash
# BitAxe V2.0.0 - Installation Validation Script
# Quick validation of monitoring system installation

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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}           BitAxe V2.0.0 - Installation Validation${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo
}

print_section() {
    local title="$1"
    echo -e "${BLUE}=== $title ===${NC}"
    echo
}

check_item() {
    local description="$1"
    local status="$2"
    local details="${3:-}"
    
    printf "%-50s " "$description"
    
    case "$status" in
        "OK")
            echo -e "${GREEN}✓ OK${NC}"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠ WARNING${NC}"
            ;;
        "FAIL")
            echo -e "${RED}✗ FAILED${NC}"
            ;;
        *)
            echo -e "${YELLOW}? UNKNOWN${NC}"
            ;;
    esac
    
    if [[ -n "$details" ]]; then
        echo "   $details"
    fi
}

# Validation functions
validate_configuration() {
    print_section "Configuration Validation"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        check_item "Configuration file exists" "OK" "$CONFIG_FILE"
        
        # Try to source the configuration
        if source "$CONFIG_FILE" 2>/dev/null; then
            check_item "Configuration file is valid" "OK"
            
            # Check key settings
            if [[ "${BACKUP_ENABLED:-}" == "true" ]]; then
                check_item "Backup system enabled" "OK"
            else
                check_item "Backup system enabled" "WARN" "Backups are disabled"
            fi
            
            if [[ "${HEALTH_CHECK_ENABLED:-}" == "true" ]]; then
                check_item "Health monitoring enabled" "OK"
            else
                check_item "Health monitoring enabled" "WARN" "Health checks are disabled"
            fi
            
            if [[ "${SYSTEM_MONITORING_ENABLED:-}" == "true" ]]; then
                check_item "System metrics enabled" "OK"
            else
                check_item "System metrics enabled" "WARN" "System monitoring is disabled"
            fi
            
        else
            check_item "Configuration file is valid" "FAIL" "Syntax error in configuration"
        fi
    else
        check_item "Configuration file exists" "FAIL" "$CONFIG_FILE"
    fi
    
    echo
}

validate_scripts() {
    print_section "Script Validation"
    
    local scripts=(
        "health_check.sh:Health monitoring script"
        "collect_metrics.sh:Metrics collection script"
        "backup_sync.sh:Backup and sync script"
        "maintenance.sh:System maintenance script"
        "log_cleanup.sh:Log rotation script"
        "setup_cron.sh:Scheduling setup script"
        "control_timers.sh:Timer control script"
        "view_schedule.sh:Schedule viewer script"
    )
    
    for script_info in "${scripts[@]}"; do
        IFS=':' read -r script_name script_desc <<< "$script_info"
        local script_path="$SCRIPT_DIR/$script_name"
        
        if [[ -f "$script_path" ]]; then
            if [[ -x "$script_path" ]]; then
                check_item "$script_desc" "OK"
            else
                check_item "$script_desc" "WARN" "Not executable"
            fi
        else
            check_item "$script_desc" "FAIL" "File not found"
        fi
    done
    
    echo
}

validate_directories() {
    print_section "Directory Structure"
    
    # Load configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
    fi
    
    local directories=(
        "$BASE_DIR/config:Configuration directory"
        "$BASE_DIR/scripts:Scripts directory"
        "$BASE_DIR/web:Web interface directory"
        "${LOG_DIR:-/var/log/bitaxe}:Log directory"
        "${BACKUP_LOCAL_DIR:-/opt/bitaxe-web/backups}:Backup directory"
    )
    
    for dir_info in "${directories[@]}"; do
        IFS=':' read -r dir_path dir_desc <<< "$dir_info"
        
        if [[ -d "$dir_path" ]]; then
            if [[ -w "$dir_path" ]]; then
                check_item "$dir_desc" "OK"
            else
                check_item "$dir_desc" "WARN" "Not writable"
            fi
        else
            check_item "$dir_desc" "FAIL" "Directory does not exist"
        fi
    done
    
    echo
}

validate_database() {
    print_section "Database Validation"
    
    # Load configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
    fi
    
    local db_path="${DATABASE_PATH:-/opt/bitaxe-web/data/bitaxe_data.db}"
    
    if [[ -f "$db_path" ]]; then
        check_item "Database file exists" "OK" "$db_path"
        
        # Test database access
        if sqlite3 "$db_path" ".tables" >/dev/null 2>&1; then
            check_item "Database is accessible" "OK"
            
            # Check for system_metrics table
            if sqlite3 "$db_path" ".schema system_metrics" >/dev/null 2>&1; then
                check_item "System metrics table exists" "OK"
            else
                check_item "System metrics table exists" "WARN" "Table will be created on first run"
            fi
            
            # Check database size
            local db_size_mb
            db_size_mb=$(du -m "$db_path" | cut -f1)
            check_item "Database size" "OK" "${db_size_mb}MB"
            
        else
            check_item "Database is accessible" "FAIL" "Cannot access database"
        fi
    else
        check_item "Database file exists" "FAIL" "$db_path not found"
    fi
    
    echo
}

validate_systemd() {
    print_section "Systemd Integration"
    
    if ! command -v systemctl >/dev/null 2>&1; then
        check_item "systemctl available" "FAIL" "systemctl command not found"
        echo
        return 1
    fi
    
    check_item "systemctl available" "OK"
    
    local timers=(
        "bitaxe-health-check.timer:Health check timer"
        "bitaxe-metrics.timer:Metrics collection timer"
        "bitaxe-backup.timer:Backup timer"
        "bitaxe-maintenance.timer:Maintenance timer"
        "bitaxe-log-cleanup.timer:Log cleanup timer"
    )
    
    local installed_count=0
    local enabled_count=0
    local active_count=0
    
    for timer_info in "${timers[@]}"; do
        IFS=':' read -r timer_name timer_desc <<< "$timer_info"
        
        if systemctl list-unit-files "$timer_name" >/dev/null 2>&1; then
            ((installed_count++))
            
            local enabled_status="WARN"
            local active_status="WARN"
            
            if systemctl is-enabled "$timer_name" >/dev/null 2>&1; then
                enabled_status="OK"
                ((enabled_count++))
            fi
            
            if systemctl is-active "$timer_name" >/dev/null 2>&1; then
                active_status="OK"
                ((active_count++))
            fi
            
            check_item "$timer_desc" "$enabled_status" "Enabled: $(systemctl is-enabled "$timer_name" 2>/dev/null)"
        else
            check_item "$timer_desc" "WARN" "Not installed"
        fi
    done
    
    echo
    echo "Summary: $installed_count installed, $enabled_count enabled, $active_count active"
    echo
}

validate_web_interface() {
    print_section "Web Interface"
    
    local web_files=(
        "$BASE_DIR/web/monitoring_routes.py:Flask routes"
        "$BASE_DIR/web/templates/monitoring/dashboard.html:Dashboard template"
        "$BASE_DIR/web/templates/monitoring/health.html:Health page template"
        "$BASE_DIR/web/templates/monitoring/metrics.html:Metrics page template"
    )
    
    for file_info in "${web_files[@]}"; do
        IFS=':' read -r file_path file_desc <<< "$file_info"
        
        if [[ -f "$file_path" ]]; then
            check_item "$file_desc" "OK"
        else
            check_item "$file_desc" "FAIL" "File not found"
        fi
    done
    
    echo
}

validate_dependencies() {
    print_section "System Dependencies"
    
    local commands=(
        "sqlite3:SQLite database"
        "curl:HTTP client"
        "ssh:SSH client"
        "gzip:Compression utility"
        "find:File search utility"
        "awk:Text processing"
        "systemctl:Systemd control"
    )
    
    for cmd_info in "${commands[@]}"; do
        IFS=':' read -r cmd_name cmd_desc <<< "$cmd_info"
        
        if command -v "$cmd_name" >/dev/null 2>&1; then
            local version
            case "$cmd_name" in
                "sqlite3")
                    version=$(sqlite3 --version | cut -d' ' -f1)
                    ;;
                "curl")
                    version=$(curl --version | head -1 | cut -d' ' -f2)
                    ;;
                *)
                    version="available"
                    ;;
            esac
            check_item "$cmd_desc" "OK" "$version"
        else
            check_item "$cmd_desc" "FAIL" "Command not found"
        fi
    done
    
    echo
}

validate_permissions() {
    print_section "File Permissions"
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        check_item "Running as root" "OK" "Full permissions available"
    else
        check_item "Running as root" "WARN" "Some operations may require sudo"
    fi
    
    # Check script directory permissions
    if [[ -w "$SCRIPT_DIR" ]]; then
        check_item "Script directory writable" "OK"
    else
        check_item "Script directory writable" "WARN" "May need sudo for script updates"
    fi
    
    # Check log directory permissions
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
        local log_dir="${LOG_DIR:-/var/log/bitaxe}"
        
        if [[ -d "$log_dir" ]]; then
            if [[ -w "$log_dir" ]]; then
                check_item "Log directory writable" "OK"
            else
                check_item "Log directory writable" "WARN" "Logs may not be written"
            fi
        else
            check_item "Log directory exists" "WARN" "Will be created on first run"
        fi
    fi
    
    echo
}

validate_network() {
    print_section "Network Connectivity"
    
    # Test local connectivity
    if ping -c 1 127.0.0.1 >/dev/null 2>&1; then
        check_item "Local network" "OK"
    else
        check_item "Local network" "FAIL" "Cannot ping localhost"
    fi
    
    # Test internet connectivity (for updates)
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        check_item "Internet connectivity" "OK"
    else
        check_item "Internet connectivity" "WARN" "No internet access (updates disabled)"
    fi
    
    # Test SSH backup if configured
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
        
        if [[ "${LAPTOP_BACKUP_ENABLED:-false}" == "true" ]]; then
            local laptop_host="${LAPTOP_HOST:-}"
            local laptop_user="${LAPTOP_USER:-}"
            
            if [[ -n "$laptop_host" && -n "$laptop_user" ]]; then
                if timeout 5 ssh -o ConnectTimeout=3 -o BatchMode=yes "$laptop_user@$laptop_host" "echo test" >/dev/null 2>&1; then
                    check_item "SSH backup connection" "OK" "$laptop_user@$laptop_host"
                else
                    check_item "SSH backup connection" "WARN" "Cannot connect to $laptop_user@$laptop_host"
                fi
            else
                check_item "SSH backup connection" "WARN" "SSH backup not fully configured"
            fi
        else
            check_item "SSH backup connection" "OK" "SSH backup disabled"
        fi
    fi
    
    echo
}

generate_summary() {
    print_section "Installation Summary"
    
    echo "Installation validation completed."
    echo
    echo "Key findings:"
    
    # Configuration status
    if [[ -f "$CONFIG_FILE" ]]; then
        echo "✓ Configuration file present and valid"
    else
        echo "✗ Configuration file missing or invalid"
    fi
    
    # Script status
    local executable_scripts
    executable_scripts=$(find "$SCRIPT_DIR" -name "*.sh" -executable | wc -l)
    echo "✓ $executable_scripts executable scripts found"
    
    # Systemd status
    if command -v systemctl >/dev/null 2>&1; then
        local timer_count
        timer_count=$(systemctl list-unit-files 'bitaxe-*.timer' --no-legend 2>/dev/null | wc -l)
        if [[ $timer_count -gt 0 ]]; then
            echo "✓ $timer_count systemd timers installed"
        else
            echo "⚠ No systemd timers installed (run setup_cron.sh)"
        fi
    else
        echo "⚠ systemctl not available"
    fi
    
    # Database status
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
        local db_path="${DATABASE_PATH:-/opt/bitaxe-web/data/bitaxe_data.db}"
        if [[ -f "$db_path" ]]; then
            echo "✓ Database file accessible"
        else
            echo "⚠ Database file not found"
        fi
    fi
    
    echo
    echo "Next steps:"
    echo "1. If systemd timers are not installed: sudo ./scripts/setup_cron.sh"
    echo "2. Start monitoring: sudo ./scripts/control_timers.sh start"
    echo "3. Check status: ./scripts/view_schedule.sh"
    echo "4. Run full test suite: sudo ./scripts/test_monitoring.sh"
    echo
}

# Main execution
main() {
    print_header
    
    validate_configuration
    validate_scripts
    validate_directories
    validate_database
    validate_systemd
    validate_web_interface
    validate_dependencies
    validate_permissions
    validate_network
    
    generate_summary
}

# Run main function
main "$@"