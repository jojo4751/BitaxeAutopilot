#!/bin/bash

# BitAxe Web Management System Backup Script
# Backs up database, models, and configuration

set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="bitaxe_backup_$TIMESTAMP"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
}

# Backup database
backup_database() {
    log "Backing up database..."
    
    # Get database credentials from environment or docker-compose
    local db_container="bitaxe-postgres"
    local db_name="bitaxe"
    local db_user="bitaxe"
    
    # Check if database container is running
    if ! docker ps | grep -q "$db_container"; then
        warn "Database container not running, skipping database backup"
        return 0
    fi
    
    # Create database dump
    docker exec "$db_container" pg_dump -U "$db_user" -d "$db_name" \
        > "$BACKUP_DIR/$BACKUP_NAME/database.sql"
    
    # Compress database dump
    gzip "$BACKUP_DIR/$BACKUP_NAME/database.sql"
    
    log "Database backup completed: database.sql.gz"
}

# Backup ML models
backup_models() {
    log "Backing up ML models..."
    
    local models_dir="$PROJECT_DIR/models"
    
    if [[ -d "$models_dir" && $(ls -A "$models_dir" 2>/dev/null) ]]; then
        cp -r "$models_dir" "$BACKUP_DIR/$BACKUP_NAME/"
        log "ML models backup completed"
    else
        warn "No ML models found to backup"
    fi
}

# Backup configuration
backup_configuration() {
    log "Backing up configuration..."
    
    # Create config backup directory
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME/config"
    
    # Backup main configuration files
    local config_files=(".env" "docker-compose.yml" "config.py")
    
    for file in "${config_files[@]}"; do
        if [[ -f "$PROJECT_DIR/$file" ]]; then
            cp "$PROJECT_DIR/$file" "$BACKUP_DIR/$BACKUP_NAME/config/"
            log "Backed up: $file"
        fi
    done
    
    # Backup docker configurations
    if [[ -d "$PROJECT_DIR/docker" ]]; then
        cp -r "$PROJECT_DIR/docker" "$BACKUP_DIR/$BACKUP_NAME/config/"
        log "Docker configuration backed up"
    fi
    
    log "Configuration backup completed"
}

# Backup application data
backup_data() {
    log "Backing up application data..."
    
    local data_dir="$PROJECT_DIR/data"
    
    if [[ -d "$data_dir" && $(ls -A "$data_dir" 2>/dev/null) ]]; then
        cp -r "$data_dir" "$BACKUP_DIR/$BACKUP_NAME/"
        log "Application data backup completed"
    else
        warn "No application data found to backup"
    fi
}

# Backup logs (recent only)
backup_logs() {
    log "Backing up recent logs..."
    
    local logs_dir="$PROJECT_DIR/logs"
    
    if [[ -d "$logs_dir" ]]; then
        mkdir -p "$BACKUP_DIR/$BACKUP_NAME/logs"
        
        # Only backup logs from last 7 days
        find "$logs_dir" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/$BACKUP_NAME/logs/" \;
        
        log "Recent logs backup completed"
    else
        warn "No logs directory found"
    fi
}

# Create backup metadata
create_metadata() {
    log "Creating backup metadata..."
    
    cat > "$BACKUP_DIR/$BACKUP_NAME/backup_info.txt" << EOF
BitAxe Web Management System Backup
Created: $(date)
Timestamp: $TIMESTAMP
Backup Name: $BACKUP_NAME
Hostname: $(hostname)
User: $(whoami)

Contents:
- Database dump (compressed)
- ML models
- Configuration files
- Application data
- Recent logs (last 7 days)

Restore Instructions:
1. Extract backup to project directory
2. Restore database: gunzip database.sql.gz && docker exec -i bitaxe-postgres psql -U bitaxe -d bitaxe < database.sql
3. Copy models to models/ directory
4. Copy configuration files
5. Restart services: docker-compose restart
EOF

    log "Backup metadata created"
}

# Compress backup
compress_backup() {
    log "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    
    # Remove uncompressed directory
    rm -rf "$BACKUP_NAME"
    
    log "Backup compressed: ${BACKUP_NAME}.tar.gz"
}

# Cleanup old backups
cleanup_old_backups() {
    local retention_days=${BACKUP_RETENTION_DAYS:-7}
    
    log "Cleaning up backups older than $retention_days days..."
    
    find "$BACKUP_DIR" -name "bitaxe_backup_*.tar.gz" -mtime +$retention_days -delete
    
    log "Old backups cleaned up"
}

# Verify backup
verify_backup() {
    log "Verifying backup..."
    
    local backup_file="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
    if [[ -f "$backup_file" ]]; then
        local size=$(du -h "$backup_file" | cut -f1)
        log "Backup verification successful"
        log "Backup file: $backup_file"
        log "Backup size: $size"
        return 0
    else
        error "Backup verification failed - file not found"
        return 1
    fi
}

# Main backup process
main() {
    log "Starting backup process..."
    
    create_backup_dir
    backup_database
    backup_models
    backup_configuration
    backup_data
    backup_logs
    create_metadata
    compress_backup
    cleanup_old_backups
    verify_backup
    
    log "Backup process completed successfully!"
    log "Backup location: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
}

# Print usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo
    echo "Environment Variables:"
    echo "  BACKUP_DIR              Backup directory (default: ./backups)"
    echo "  BACKUP_RETENTION_DAYS   Days to keep backups (default: 7)"
    echo
    echo "Examples:"
    echo "  $0                              - Create backup with default settings"
    echo "  BACKUP_DIR=/mnt/backups $0      - Create backup in custom directory"
}

# Handle command line arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Run main function
main