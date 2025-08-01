#!/bin/bash
# BitAxe V2.0.0 - SSH Setup Script for Laptop Backup Integration
# Sets up passwordless SSH authentication for automated backups

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$BASE_DIR/config/monitoring.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Interactive configuration setup
setup_laptop_config() {
    log "=== Laptop Backup Configuration Setup ==="
    echo
    
    read -p "Enter laptop username: " -i "$LAPTOP_USER" -e new_user
    read -p "Enter laptop IP address: " -i "$LAPTOP_HOST" -e new_host
    read -p "Enter backup directory on laptop: " -i "$LAPTOP_BACKUP_DIR" -e new_dir
    
    # Update configuration file
    sed -i "s/LAPTOP_USER=\".*\"/LAPTOP_USER=\"$new_user\"/" "$CONFIG_FILE"
    sed -i "s/LAPTOP_HOST=\".*\"/LAPTOP_HOST=\"$new_host\"/" "$CONFIG_FILE"
    sed -i "s|LAPTOP_BACKUP_DIR=\".*\"|LAPTOP_BACKUP_DIR=\"$new_dir\"|" "$CONFIG_FILE"
    
    LAPTOP_USER="$new_user"
    LAPTOP_HOST="$new_host"
    LAPTOP_BACKUP_DIR="$new_dir"
    
    log_success "Configuration updated"
}

# Generate SSH key pair
generate_ssh_key() {
    local ssh_dir="/home/$(whoami)/.ssh"
    local key_file="$ssh_dir/bitaxe_backup_rsa"
    
    log "Generating SSH key pair..."
    
    mkdir -p "$ssh_dir"
    chmod 700 "$ssh_dir"
    
    if [[ -f "$key_file" ]]; then
        log_warning "SSH key already exists: $key_file"
        read -p "Overwrite existing key? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Using existing SSH key"
            return 0
        fi
    fi
    
    ssh-keygen -t rsa -b 4096 -f "$key_file" -N "" -C "bitaxe-backup@$(hostname)"
    chmod 600 "$key_file"
    chmod 644 "$key_file.pub"
    
    log_success "SSH key pair generated: $key_file"
    
    # Add to SSH config
    local ssh_config="$ssh_dir/config"
    if ! grep -q "bitaxe-backup" "$ssh_config" 2>/dev/null; then
        cat >> "$ssh_config" <<EOF

# BitAxe Backup Configuration
Host bitaxe-backup
    HostName $LAPTOP_HOST
    User $LAPTOP_USER
    IdentityFile $key_file
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    ConnectTimeout 10
EOF
        log_success "SSH config updated"
    fi
}

# Copy public key to laptop
setup_laptop_ssh() {
    local ssh_dir="/home/$(whoami)/.ssh"
    local key_file="$ssh_dir/bitaxe_backup_rsa.pub"
    
    if [[ ! -f "$key_file" ]]; then
        log_error "Public key not found: $key_file"
        return 1
    fi
    
    log "Setting up SSH access to laptop..."
    
    # Test basic connectivity
    if ! ping -c 1 -W 5 "$LAPTOP_HOST" >/dev/null 2>&1; then
        log_error "Cannot reach laptop at $LAPTOP_HOST"
        return 1
    fi
    
    log "Copying public key to laptop (you may need to enter password)..."
    
    if ssh-copy-id -i "$key_file" "$LAPTOP_USER@$LAPTOP_HOST"; then
        log_success "Public key copied to laptop"
    else
        log_error "Failed to copy public key to laptop"
        log "You can manually copy the key with:"
        log "ssh-copy-id -i $key_file $LAPTOP_USER@$LAPTOP_HOST"
        return 1
    fi
}

# Test SSH connection
test_ssh_connection() {
    log "Testing SSH connection to laptop..."
    
    if ssh -o ConnectTimeout=10 -q "$LAPTOP_USER@$LAPTOP_HOST" exit; then
        log_success "SSH connection successful"
        
        # Test directory creation
        if ssh "$LAPTOP_USER@$LAPTOP_HOST" "mkdir -p $LAPTOP_BACKUP_DIR && echo 'BitAxe backup test' > $LAPTOP_BACKUP_DIR/test.txt"; then
            log_success "Backup directory created and accessible"
            ssh "$LAPTOP_USER@$LAPTOP_HOST" "rm -f $LAPTOP_BACKUP_DIR/test.txt"
        else
            log_error "Cannot create backup directory on laptop"
            return 1
        fi
    else
        log_error "SSH connection failed"
        return 1
    fi
}

# Create backup directory structure on laptop
setup_laptop_directories() {
    log "Setting up backup directory structure on laptop..."
    
    ssh "$LAPTOP_USER@$LAPTOP_HOST" "
        mkdir -p $LAPTOP_BACKUP_DIR/{$(date +%Y)/{01..12}/{01..31},scripts,logs}
        chmod 755 $LAPTOP_BACKUP_DIR
        
        # Create README file
        cat > $LAPTOP_BACKUP_DIR/README.txt <<'EOF'
BitAxe Backup Directory
======================

This directory contains automated backups from your BitAxe mining system.

Structure:
- YYYY/MM/DD/ - Daily backup directories
- Each day contains:
  - bitaxe_data_TIMESTAMP.db.gz - Compressed SQLite database backup
  - csv_TIMESTAMP.tar.gz - CSV exports for Excel analysis
  
Scripts:
- Use the CSV files for analysis in Excel or other tools
- SQLite files can be opened with DB Browser for SQLite

Retention: Files older than $LAPTOP_RETENTION_DAYS days are automatically deleted.

Last updated: $(date)
EOF
    "
    
    log_success "Laptop backup directory structure created"
}

# Display setup summary
show_summary() {
    log "=== SSH Backup Setup Summary ==="
    echo
    log_success "✓ SSH key pair generated"
    log_success "✓ Public key copied to laptop"
    log_success "✓ SSH connection tested successfully"
    log_success "✓ Backup directories created on laptop"
    echo
    log "Configuration:"
    log "  Laptop: $LAPTOP_USER@$LAPTOP_HOST"
    log "  Backup Directory: $LAPTOP_BACKUP_DIR"
    log "  SSH Key: /home/$(whoami)/.ssh/bitaxe_backup_rsa"
    echo
    log "Next steps:"
    log "1. Test backup script: sudo -u bitaxe $BASE_DIR/scripts/backup_sync.sh"
    log "2. Setup cron job for daily backups"
    echo
}

# Main execution
main() {
    log "=== BitAxe SSH Backup Setup ==="
    echo
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_error "Don't run this script as root. Run as the bitaxe user or regular user."
        exit 1
    fi
    
    load_config
    setup_laptop_config
    generate_ssh_key
    setup_laptop_ssh
    test_ssh_connection
    setup_laptop_directories
    show_summary
    
    log_success "SSH backup setup completed successfully!"
}

# Handle interrupts
trap 'log_error "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@"