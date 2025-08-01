#!/bin/bash
# BitAxe V2.0.0 - Comprehensive Monitoring System Test Suite
# Tests all components of the monitoring and backup infrastructure

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"
TEST_LOG="/tmp/bitaxe_monitoring_test.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Logging setup
exec 1> >(tee -a "$TEST_LOG")
exec 2>&1

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] [$level] $message"
}

log_info() { log "INFO" "$1"; }
log_success() { log "SUCCESS" "${GREEN}✓${NC} $1"; }
log_error() { log "ERROR" "${RED}✗${NC} $1"; }
log_warning() { log "WARNING" "${YELLOW}⚠${NC} $1"; }
log_test() { log "TEST" "${BLUE}➤${NC} $1"; }

# Test execution framework
run_test() {
    local test_name="$1"
    local test_function="$2"
    local required="${3:-true}"
    
    ((TESTS_TOTAL++))
    log_test "Running test: $test_name"
    
    if $test_function; then
        ((TESTS_PASSED++))
        log_success "$test_name"
        return 0
    else
        if [[ "$required" == "true" ]]; then
            ((TESTS_FAILED++))
            log_error "$test_name"
            return 1
        else
            ((TESTS_SKIPPED++))
            log_warning "$test_name (optional - skipped)"
            return 0
        fi
    fi
}

# Test functions
test_configuration_file() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || return 1
        
        # Check required variables
        local required_vars=(
            "BACKUP_ENABLED"
            "HEALTH_CHECK_ENABLED"
            "SYSTEM_MONITORING_ENABLED"
            "LOG_DIR"
            "DATABASE_PATH"
        )
        
        for var in "${required_vars[@]}"; do
            if [[ -z "${!var:-}" ]]; then
                log_error "Required configuration variable $var is not set"
                return 1
            fi
        done
        
        return 0
    else
        log_error "Configuration file not found: $CONFIG_FILE"
        return 1
    fi
}

test_script_permissions() {
    local scripts=(
        "health_check.sh"
        "collect_metrics.sh"
        "backup_sync.sh"
        "maintenance.sh"
        "log_cleanup.sh"
        "system_update.sh"
        "setup_cron.sh"
        "setup_ssh_backup.sh"
        "control_timers.sh"
        "view_schedule.sh"
    )
    
    local failed=0
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        if [[ ! -x "$script_path" ]]; then
            log_error "Script $script is not executable"
            ((failed++))
        fi
    done
    
    return $failed
}

test_directory_structure() {
    local directories=(
        "$BASE_DIR/config"
        "$BASE_DIR/scripts"
        "$BASE_DIR/web"
        "$BASE_DIR/web/templates/monitoring"
        "${LOG_DIR:-/var/log/bitaxe}"
        "${BACKUP_LOCAL_DIR:-/opt/bitaxe-web/backups}"
    )
    
    local failed=0
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_error "Required directory does not exist: $dir"
            ((failed++))
        fi
    done
    
    return $failed
}

test_database_access() {
    local db_path="${DATABASE_PATH:-/opt/bitaxe-web/data/bitaxe_data.db}"
    
    if [[ ! -f "$db_path" ]]; then
        log_error "Database file not found: $db_path"
        return 1
    fi
    
    # Test database access
    if ! sqlite3 "$db_path" ".tables" >/dev/null 2>&1; then
        log_error "Cannot access database: $db_path"
        return 1
    fi
    
    # Test system_metrics table creation
    if ! sqlite3 "$db_path" "CREATE TABLE IF NOT EXISTS system_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        metric_type TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_unit TEXT,
        additional_data TEXT
    );" 2>/dev/null; then
        log_error "Cannot create system_metrics table"
        return 1
    fi
    
    return 0
}

test_health_check_script() {
    if [[ ! -x "$SCRIPT_DIR/health_check.sh" ]]; then
        log_error "Health check script is not executable"
        return 1
    fi
    
    # Test dry run (don't actually restart services)
    local temp_config="/tmp/test_monitoring.conf"
    cp "$CONFIG_FILE" "$temp_config"
    
    # Modify config for testing
    sed -i 's/AUTO_RESTART_ENABLED=true/AUTO_RESTART_ENABLED=false/' "$temp_config"
    
    # Run health check with test config
    if BASE_DIR="$BASE_DIR" CONFIG_FILE="$temp_config" timeout 30 "$SCRIPT_DIR/health_check.sh" >/dev/null 2>&1; then
        rm -f "$temp_config"
        return 0
    else
        rm -f "$temp_config"
        log_error "Health check script failed"
        return 1
    fi
}

test_metrics_collection_script() {
    if [[ ! -x "$SCRIPT_DIR/collect_metrics.sh" ]]; then
        log_error "Metrics collection script is not executable"
        return 1
    fi
    
    # Run metrics collection
    if timeout 30 "$SCRIPT_DIR/collect_metrics.sh" >/dev/null 2>&1; then
        # Check if metrics were inserted
        local db_path="${DATABASE_PATH:-/opt/bitaxe-web/data/bitaxe_data.db}"
        local count
        count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM system_metrics WHERE timestamp > datetime('now', '-1 minute');" 2>/dev/null || echo "0")
        
        if [[ $count -gt 0 ]]; then
            return 0
        else
            log_error "No metrics were collected"
            return 1
        fi
    else
        log_error "Metrics collection script failed"
        return 1
    fi
}

test_backup_script() {
    if [[ ! -x "$SCRIPT_DIR/backup_sync.sh" ]]; then
        log_error "Backup script is not executable"
        return 1
    fi
    
    # Create temporary config with laptop backup disabled
    local temp_config="/tmp/test_backup.conf"
    cp "$CONFIG_FILE" "$temp_config"
    sed -i 's/LAPTOP_BACKUP_ENABLED=true/LAPTOP_BACKUP_ENABLED=false/' "$temp_config"
    
    # Run backup script
    if BASE_DIR="$BASE_DIR" CONFIG_FILE="$temp_config" timeout 60 "$SCRIPT_DIR/backup_sync.sh" >/dev/null 2>&1; then
        # Check if backup files were created
        local backup_dir="${BACKUP_LOCAL_DIR:-/opt/bitaxe-web/backups}"
        local backup_count
        backup_count=$(find "$backup_dir" -name "bitaxe_data_*.db*" -mtime -1 | wc -l)
        
        rm -f "$temp_config"
        
        if [[ $backup_count -gt 0 ]]; then
            return 0
        else
            log_error "No backup files were created"
            return 1
        fi
    else
        rm -f "$temp_config"
        log_error "Backup script failed"
        return 1
    fi
}

test_maintenance_script() {
    if [[ ! -x "$SCRIPT_DIR/maintenance.sh" ]]; then
        log_error "Maintenance script is not executable"
        return 1
    fi
    
    # Run maintenance script
    if timeout 60 "$SCRIPT_DIR/maintenance.sh" >/dev/null 2>&1; then
        return 0
    else
        log_error "Maintenance script failed"
        return 1
    fi
}

test_systemd_timers() {
    if ! command -v systemctl >/dev/null 2>&1; then
        log_warning "systemctl not available, skipping timer tests"
        return 1
    fi
    
    local timers=(
        "bitaxe-health-check.timer"
        "bitaxe-metrics.timer"
        "bitaxe-backup.timer"
        "bitaxe-maintenance.timer"
        "bitaxe-log-cleanup.timer"
    )
    
    local installed=0
    local enabled=0
    
    for timer in "${timers[@]}"; do
        if systemctl list-unit-files "$timer" >/dev/null 2>&1; then
            ((installed++))
            if systemctl is-enabled "$timer" >/dev/null 2>&1; then
                ((enabled++))
            fi
        fi
    done
    
    if [[ $installed -eq 0 ]]; then
        log_warning "No systemd timers are installed"
        return 1
    fi
    
    log_info "Systemd timers: $installed installed, $enabled enabled"
    return 0
}

test_web_interface() {
    local web_files=(
        "$BASE_DIR/web/monitoring_routes.py"
        "$BASE_DIR/web/templates/monitoring/dashboard.html"
        "$BASE_DIR/web/templates/monitoring/health.html"
        "$BASE_DIR/web/templates/monitoring/metrics.html"
    )
    
    local failed=0
    for file in "${web_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Web interface file missing: $file"
            ((failed++))
        fi
    done
    
    return $failed
}

test_log_rotation() {
    if [[ ! -x "$SCRIPT_DIR/log_cleanup.sh" ]]; then
        log_error "Log cleanup script is not executable"
        return 1
    fi
    
    # Create test log file
    local test_log="${LOG_DIR:-/var/log/bitaxe}/test_rotation.log"
    mkdir -p "$(dirname "$test_log")"
    
    # Create a large test file (>10MB)
    dd if=/dev/zero of="$test_log" bs=1M count=11 2>/dev/null || {
        log_error "Cannot create test log file"
        return 1
    }
    
    # Run log cleanup
    if timeout 30 "$SCRIPT_DIR/log_cleanup.sh" >/dev/null 2>&1; then
        # Check if file was rotated (should be smaller or compressed)
        if [[ -f "$test_log.gz" ]] || [[ $(stat -f%z "$test_log" 2>/dev/null || stat -c%s "$test_log" 2>/dev/null || echo "0") -lt 1000000 ]]; then
            rm -f "$test_log" "$test_log.gz" 2>/dev/null || true
            return 0
        else
            rm -f "$test_log" 2>/dev/null || true
            log_error "Log rotation did not work as expected"
            return 1
        fi
    else
        rm -f "$test_log" 2>/dev/null || true
        log_error "Log cleanup script failed"
        return 1
    fi
}

test_ssh_backup_config() {
    # This is optional since SSH backup may not be configured
    if [[ "${LAPTOP_BACKUP_ENABLED:-false}" != "true" ]]; then
        log_info "SSH backup is disabled, skipping SSH tests"
        return 1
    fi
    
    local laptop_host="${LAPTOP_HOST:-}"
    local laptop_user="${LAPTOP_USER:-}"
    
    if [[ -z "$laptop_host" || -z "$laptop_user" ]]; then
        log_warning "SSH backup configuration incomplete"
        return 1
    fi
    
    # Test SSH connection (with timeout)
    if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes "$laptop_user@$laptop_host" "echo 'SSH test successful'" >/dev/null 2>&1; then
        return 0
    else
        log_warning "SSH connection to $laptop_user@$laptop_host failed"
        return 1
    fi
}

test_disk_space() {
    local backup_dir="${BACKUP_LOCAL_DIR:-/opt/bitaxe-web/backups}"
    local log_dir="${LOG_DIR:-/var/log/bitaxe}"
    
    # Check available disk space (need at least 1GB)
    local available_kb
    available_kb=$(df "$BASE_DIR" | awk 'NR==2 {print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [[ $available_gb -lt 1 ]]; then
        log_error "Insufficient disk space: ${available_gb}GB available (need at least 1GB)"
        return 1
    fi
    
    # Check if backup directory has write permissions
    if ! touch "$backup_dir/test_write" 2>/dev/null; then
        log_error "Cannot write to backup directory: $backup_dir"
        return 1
    fi
    rm -f "$backup_dir/test_write"
    
    # Check if log directory has write permissions
    if ! touch "$log_dir/test_write" 2>/dev/null; then
        log_error "Cannot write to log directory: $log_dir"
        return 1
    fi
    rm -f "$log_dir/test_write"
    
    return 0
}

test_performance_impact() {
    log_info "Testing performance impact (this may take a minute)..."
    
    # Get baseline CPU and memory usage
    local cpu_before
    local mem_before
    cpu_before=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    mem_before=$(free | awk 'NR==2{printf "%.1f", $3/$2*100}')
    
    # Run all monitoring scripts
    timeout 30 "$SCRIPT_DIR/health_check.sh" >/dev/null 2>&1 || true
    timeout 30 "$SCRIPT_DIR/collect_metrics.sh" >/dev/null 2>&1 || true
    
    # Wait a moment
    sleep 5
    
    # Get CPU and memory usage after
    local cpu_after
    local mem_after
    cpu_after=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    mem_after=$(free | awk 'NR==2{printf "%.1f", $3/$2*100}')
    
    log_info "CPU usage: ${cpu_before}% → ${cpu_after}%"
    log_info "Memory usage: ${mem_before}% → ${mem_after}%"
    
    # Performance impact should be minimal
    return 0
}

# Print test summary
print_summary() {
    echo
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}                    Test Summary${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo
    echo -e "Total tests run:    ${BLUE}$TESTS_TOTAL${NC}"
    echo -e "Tests passed:       ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed:       ${RED}$TESTS_FAILED${NC}"
    echo -e "Tests skipped:      ${YELLOW}$TESTS_SKIPPED${NC}"
    echo
    
    local success_rate=0
    if [[ $TESTS_TOTAL -gt 0 ]]; then
        success_rate=$(( (TESTS_PASSED * 100) / TESTS_TOTAL ))
    fi
    
    echo -e "Success rate:       ${success_rate}%"
    echo
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}✓ All critical tests passed! Monitoring system is ready.${NC}"
        echo
        echo "Next steps:"
        echo "1. Start monitoring: sudo ./scripts/control_timers.sh start"
        echo "2. View status: ./scripts/view_schedule.sh"
        echo "3. Access dashboard: http://your-server:5000/monitoring"
        return 0
    else
        echo -e "${RED}✗ Some tests failed. Please review and fix issues before deployment.${NC}"
        echo
        echo "Check the test log for details: $TEST_LOG"
        return 1
    fi
}

# Main test execution
main() {
    log_info "=== BitAxe Monitoring System Test Suite ==="
    log_info "Starting comprehensive testing..."
    echo
    
    # Initialize test log
    echo "BitAxe Monitoring System Test Suite" > "$TEST_LOG"
    echo "Started: $(date)" >> "$TEST_LOG"
    echo "=================================" >> "$TEST_LOG"
    
    # Load configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE" 2>/dev/null || true
    fi
    
    # Core system tests (required)
    run_test "Configuration File" test_configuration_file true
    run_test "Directory Structure" test_directory_structure true
    run_test "Script Permissions" test_script_permissions true
    run_test "Database Access" test_database_access true
    run_test "Disk Space Check" test_disk_space true
    
    # Component tests (required)
    run_test "Health Check Script" test_health_check_script true
    run_test "Metrics Collection Script" test_metrics_collection_script true
    run_test "Backup Script" test_backup_script true
    run_test "Maintenance Script" test_maintenance_script true
    run_test "Log Rotation" test_log_rotation true
    run_test "Web Interface Files" test_web_interface true
    
    # System integration tests (optional)
    run_test "Systemd Timers" test_systemd_timers false
    run_test "SSH Backup Configuration" test_ssh_backup_config false
    
    # Performance test
    run_test "Performance Impact" test_performance_impact false
    
    # Print final summary
    print_summary
}

# Handle interrupts
trap 'log_error "Test suite interrupted"; exit 1' INT TERM

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_info "Running as root - all tests will be executed"
else
    log_warning "Not running as root - some tests may be skipped"
fi

# Run main function
main "$@"