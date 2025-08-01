#!/bin/bash
# BitAxe V2.0.0 - Automated Backup & Sync Script
# Creates backups and syncs to laptop with retention policy

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"

# Source configuration
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
else
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Logging setup
LOG_FILE="$LOG_DIR/backup.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

# Create backup directories
create_directories() {
    mkdir -p "$BACKUP_LOCAL_DIR"
    mkdir -p "$LOG_DIR"
}

# Create database backup
backup_database() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$BACKUP_LOCAL_DIR/bitaxe_data_$timestamp.db"
    local csv_export_dir="$BACKUP_LOCAL_DIR/csv_$timestamp"
    
    log "Starting database backup..."
    
    # Create SQLite backup
    if [[ -f "$DATABASE_PATH" ]]; then
        sqlite3 "$DATABASE_PATH" ".backup $backup_file"
        
        if [[ "$BACKUP_COMPRESS" == "true" ]]; then
            gzip "$backup_file"
            backup_file="${backup_file}.gz"
        fi
        
        log "Database backup created: $(basename "$backup_file")"
    else
        log_error "Database file not found: $DATABASE_PATH"
        return 1
    fi
    
    # Create CSV exports for analysis
    mkdir -p "$csv_export_dir"
    
    # Export main tables to CSV
    sqlite3 "$DATABASE_PATH" <<EOF
.headers on
.mode csv
.output $csv_export_dir/miner_logs.csv
SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10000;

.output $csv_export_dir/benchmark_results.csv
SELECT * FROM benchmark_results ORDER BY timestamp DESC LIMIT 1000;

.output $csv_export_dir/system_events.csv
SELECT * FROM protocol ORDER BY timestamp DESC LIMIT 1000;

.output $csv_export_dir/efficiency_data.csv
SELECT * FROM efficiency_markers ORDER BY timestamp DESC LIMIT 5000;
EOF
    
    # Create summary statistics
    sqlite3 "$DATABASE_PATH" <<EOF > "$csv_export_dir/backup_summary.txt"
.headers on
SELECT 'Total Logs: ' || COUNT(*) FROM logs;
SELECT 'Total Benchmarks: ' || COUNT(*) FROM benchmark_results;
SELECT 'Total Events: ' || COUNT(*) FROM protocol;
SELECT 'Database Size: ' || ROUND(page_count * page_size / 1024.0 / 1024.0, 2) || ' MB' 
FROM pragma_page_count, pragma_page_size;
SELECT 'Backup Created: ' || datetime('now') || ' UTC';
EOF
    
    # Compress CSV exports
    tar -czf "$csv_export_dir.tar.gz" -C "$BACKUP_LOCAL_DIR" "$(basename "$csv_export_dir")"
    rm -rf "$csv_export_dir"
    
    log "CSV exports created: $(basename "$csv_export_dir.tar.gz")"
    
    echo "$backup_file|$csv_export_dir.tar.gz"
}

# Sync backup to laptop
sync_to_laptop() {
    local backup_files="$1"
    
    if [[ "$LAPTOP_BACKUP_ENABLED" != "true" ]]; then
        log "Laptop sync disabled"
        return 0
    fi
    
    local date_dir=$(date '+%Y/%m/%d')
    local laptop_target="$LAPTOP_USER@$LAPTOP_HOST:$LAPTOP_BACKUP_DIR/$date_dir/"
    
    log "Syncing backups to laptop..."
    
    # Test SSH connection
    if ! ssh -o ConnectTimeout=10 -q "$LAPTOP_USER@$LAPTOP_HOST" exit; then
        log_error "Cannot connect to laptop via SSH: $LAPTOP_HOST"
        return 1
    fi
    
    # Create directory on laptop
    ssh "$LAPTOP_USER@$LAPTOP_HOST" "mkdir -p $LAPTOP_BACKUP_DIR/$date_dir"
    
    # Transfer backup files
    local db_file=$(echo "$backup_files" | cut -d'|' -f1)
    local csv_file=$(echo "$backup_files" | cut -d'|' -f2)
    
    if scp "$db_file" "$csv_file" "$laptop_target"; then
        log "Backup files transferred successfully to laptop"
        
        # Verify transfer
        local db_basename=$(basename "$db_file")
        local csv_basename=$(basename "$csv_file")
        
        if ssh "$LAPTOP_USER@$LAPTOP_HOST" "test -f $LAPTOP_BACKUP_DIR/$date_dir/$db_basename && test -f $LAPTOP_BACKUP_DIR/$date_dir/$csv_basename"; then
            log "Backup verification successful"
            return 0
        else
            log_error "Backup verification failed"
            return 1
        fi
    else
        log_error "Failed to transfer backup files to laptop"
        return 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old local backups (keeping $BACKUP_RETENTION_DAYS days)..."
    
    find "$BACKUP_LOCAL_DIR" -name "bitaxe_data_*.db*" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_LOCAL_DIR" -name "csv_*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
    
    if [[ "$LAPTOP_BACKUP_ENABLED" == "true" ]]; then
        log "Cleaning up old laptop backups (keeping $LAPTOP_RETENTION_DAYS days)..."
        
        ssh "$LAPTOP_USER@$LAPTOP_HOST" "find $LAPTOP_BACKUP_DIR -type f -mtime +$LAPTOP_RETENTION_DAYS -delete 2>/dev/null || true"
        ssh "$LAPTOP_USER@$LAPTOP_HOST" "find $LAPTOP_BACKUP_DIR -type d -empty -delete 2>/dev/null || true"
    fi
}

# Generate backup report
generate_report() {
    local backup_status="$1"
    local sync_status="$2"
    
    cat > "$BACKUP_LOCAL_DIR/last_backup_report.txt" <<EOF
BitAxe Backup Report
===================
Date: $(date)
Backup Status: $backup_status
Laptop Sync Status: $sync_status

Local Backup Directory: $BACKUP_LOCAL_DIR
Laptop Backup Location: $LAPTOP_USER@$LAPTOP_HOST:$LAPTOP_BACKUP_DIR

Recent Backups:
$(ls -la "$BACKUP_LOCAL_DIR"/bitaxe_data_*.db* 2>/dev/null | tail -5 || echo "No backups found")

Disk Usage:
$(df -h "$BACKUP_LOCAL_DIR" | tail -1)
EOF
    
    log "Backup report generated: $BACKUP_LOCAL_DIR/last_backup_report.txt"
}

# Main execution
main() {
    log "=== BitAxe Backup Process Started ==="
    
    if [[ "$BACKUP_ENABLED" != "true" ]]; then
        log "Backup system disabled"
        exit 0
    fi
    
    local backup_status="FAILED"
    local sync_status="SKIPPED"
    
    create_directories
    
    # Create backup
    if backup_files=$(backup_database); then
        backup_status="SUCCESS"
        
        # Sync to laptop
        if sync_to_laptop "$backup_files"; then
            sync_status="SUCCESS"
        else
            sync_status="FAILED"
        fi
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Generate report
    generate_report "$backup_status" "$sync_status"
    
    log "=== Backup Process Completed: $backup_status / Sync: $sync_status ==="
    
    if [[ "$backup_status" == "FAILED" ]]; then
        exit 1
    fi
}

# Run main function
main "$@"