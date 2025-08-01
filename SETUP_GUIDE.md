# BitAxe V2.0.0 - Monitoring Setup Guide

## Quick Start Installation

This guide will walk you through setting up the complete BitAxe monitoring and backup infrastructure in under 30 minutes.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] BitAxe V2.0.0 system running and accessible
- [ ] Ubuntu/Debian Linux system with root access
- [ ] Internet connection for package updates
- [ ] SSH access to a laptop/desktop for backups (optional)
- [ ] At least 2GB free disk space

## Step 1: Initial System Setup

### 1.1 Create Directory Structure

```bash
# Navigate to BitAxe installation
cd /opt/bitaxe-web

# Create monitoring directory
sudo mkdir -p BITAXE_V2.0.0
cd BITAXE_V2.0.0

# Create required directories
sudo mkdir -p {scripts,config,web,logs,backups}
sudo mkdir -p web/templates/monitoring
```

### 1.2 Set Permissions

```bash
# Set ownership
sudo chown -R root:root /opt/bitaxe-web/BITAXE_V2.0.0

# Create bitaxe user if it doesn't exist
sudo useradd -r -s /bin/bash -d /opt/bitaxe-web bitaxe || true

# Create log directory
sudo mkdir -p /var/log/bitaxe
sudo chown bitaxe:bitaxe /var/log/bitaxe
```

## Step 2: Configuration Setup

### 2.1 Create Main Configuration File

```bash
sudo nano config/monitoring.conf
```

Copy and customize this configuration:

```bash
# BitAxe V2.0.0 - Monitoring Configuration
# Customize these settings for your environment

# === BACKUP CONFIGURATION ===
BACKUP_ENABLED=true
BACKUP_LOCAL_DIR="/opt/bitaxe-web/backups"
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true

# Laptop backup settings (customize these)
LAPTOP_BACKUP_ENABLED=false  # Set to true after SSH setup
LAPTOP_USER="YOUR_USERNAME"
LAPTOP_HOST="192.168.1.XXX"  # Your laptop IP
LAPTOP_BACKUP_DIR="~/bitaxe_backups"
LAPTOP_RETENTION_DAYS=90

# Database settings
DATABASE_PATH="/opt/bitaxe-web/data/bitaxe_data.db"

# === HEALTH MONITORING ===
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=300  # 5 minutes
API_HEALTH_TIMEOUT=10

# Services to monitor (adjust based on your setup)
MONITORED_SERVICES="bitaxe-web bitaxe-autopilot bitaxe-logger"

# Auto-restart settings
AUTO_RESTART_ENABLED=true
MAX_RESTART_ATTEMPTS=3
RESTART_COOLDOWN=60

# === SYSTEM MONITORING ===
SYSTEM_MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=60

# Resource thresholds (adjust based on your hardware)
CPU_WARNING_THRESHOLD=80
CPU_CRITICAL_THRESHOLD=95
MEMORY_WARNING_THRESHOLD=80
MEMORY_CRITICAL_THRESHOLD=90
DISK_WARNING_THRESHOLD=80
DISK_CRITICAL_THRESHOLD=90
TEMP_WARNING_THRESHOLD=70
TEMP_CRITICAL_THRESHOLD=80

# === LOGGING ===
LOG_DIR="/var/log/bitaxe"
LOG_LEVEL="INFO"
LOG_RETENTION_DAYS=14
LOG_MAX_SIZE="100M"

# === MAINTENANCE ===
MAINTENANCE_ENABLED=true
AUTO_UPDATE_ENABLED=false  # Set to true for automatic updates
CLEANUP_TEMP_FILES=true
DATABASE_VACUUM_ENABLED=true
METRICS_HISTORY_DAYS=7
```

Save and exit (Ctrl+X, Y, Enter).

## Step 3: Install Monitoring Scripts

### 3.1 Copy All Scripts

You'll need to copy all the monitoring scripts to the `scripts/` directory. The key scripts are:

- `health_check.sh` - System health monitoring
- `collect_metrics.sh` - Performance metrics collection
- `backup_sync.sh` - Database backup and sync
- `maintenance.sh` - System maintenance tasks
- `log_cleanup.sh` - Log rotation and cleanup
- `system_update.sh` - System package updates
- `setup_cron.sh` - Scheduling setup
- `setup_ssh_backup.sh` - SSH backup configuration
- `control_timers.sh` - Timer management
- `view_schedule.sh` - Schedule viewer

### 3.2 Make Scripts Executable

```bash
sudo chmod +x scripts/*.sh
```

### 3.3 Test Configuration

```bash
# Test configuration loading
sudo bash -c 'source config/monitoring.conf && echo "Config loaded successfully"'

# Test script access
ls -la scripts/
```

## Step 4: Setup Laptop Backup (Optional)

If you want automated backups to sync to your laptop:

### 4.1 Configure SSH Access

```bash
# Run the SSH setup script
sudo ./scripts/setup_ssh_backup.sh
```

Follow the interactive prompts:
1. Enter your laptop username
2. Enter your laptop IP address
3. Enter backup directory path
4. Script will generate SSH keys and copy them to laptop

### 4.2 Test SSH Connection

```bash
# Test connection manually
ssh your-username@your-laptop-ip "echo 'SSH connection successful'"
```

### 4.3 Enable Laptop Backup

Edit configuration to enable laptop backups:

```bash
sudo nano config/monitoring.conf
```

Change:
```bash
LAPTOP_BACKUP_ENABLED=true
```

## Step 5: Install Scheduled Tasks

### 5.1 Run the Setup Script

```bash
sudo ./scripts/setup_cron.sh
```

This will:
- Create systemd timer services
- Enable and start all timers
- Create fallback cron jobs
- Set up management scripts

### 5.2 Verify Installation

```bash
# Check timer status
sudo systemctl list-timers 'bitaxe-*'

# Check if timers are active
sudo ./scripts/control_timers.sh status
```

You should see output showing all timers are active and enabled.

## Step 6: Web Dashboard Integration

### 6.1 Copy Web Templates

Copy the monitoring templates to your existing web application:

```bash
# If you have an existing Flask app, copy monitoring routes
sudo cp web/monitoring_routes.py /path/to/your/flask/app/

# Copy templates
sudo cp -r web/templates/monitoring /path/to/your/templates/
```

### 6.2 Update Flask Application

Add to your main Flask app:

```python
from monitoring_routes import monitoring_bp

# Register monitoring blueprint
app.register_blueprint(monitoring_bp)
```

## Step 7: Validation and Testing

### 7.1 Test All Components

```bash
# Test manual execution of each script
sudo ./scripts/health_check.sh
sudo ./scripts/collect_metrics.sh
sudo ./scripts/backup_sync.sh  # This will create a backup

# View results
ls -la /var/log/bitaxe/
ls -la /opt/bitaxe-web/backups/
```

### 7.2 Check Database

```bash
# Verify metrics are being collected
sqlite3 /opt/bitaxe-web/data/bitaxe_data.db "SELECT COUNT(*) FROM system_metrics;"

# Should show increasing numbers over time
```

### 7.3 Test Web Dashboard

Visit your web application and navigate to `/monitoring` to see the dashboard.

## Step 8: Monitoring and Maintenance

### 8.1 Daily Checks

Add these to your daily routine:

```bash
# Check overall system status
./scripts/view_schedule.sh

# Check recent logs
sudo journalctl -u 'bitaxe-*.service' --since "1 day ago"

# Check backup status
ls -la /opt/bitaxe-web/backups/ | tail -5
```

### 8.2 Weekly Maintenance

```bash
# Manual maintenance run
sudo ./scripts/maintenance.sh

# Check disk usage
df -h /opt/bitaxe-web
df -h /var/log
```

## Troubleshooting Common Issues

### Issue 1: Timers Not Starting

```bash
# Check systemd status
sudo systemctl daemon-reload
sudo systemctl list-unit-files 'bitaxe-*'

# Restart timers
sudo ./scripts/control_timers.sh restart
```

### Issue 2: Permission Errors

```bash
# Fix script permissions
sudo chmod +x scripts/*.sh
sudo chown -R root:root scripts/

# Fix log permissions
sudo mkdir -p /var/log/bitaxe
sudo chown bitaxe:bitaxe /var/log/bitaxe
```

### Issue 3: Database Access Issues

```bash
# Check database file
ls -la /opt/bitaxe-web/data/bitaxe_data.db

# Test database access
sqlite3 /opt/bitaxe-web/data/bitaxe_data.db ".tables"

# Fix permissions if needed
sudo chown bitaxe:bitaxe /opt/bitaxe-web/data/bitaxe_data.db
```

### Issue 4: SSH Backup Failing

```bash
# Test SSH connection
ssh -o ConnectTimeout=10 user@laptop-ip "echo 'SSH OK'"

# Check SSH key
ls -la ~/.ssh/bitaxe_backup_rsa*

# Re-run SSH setup
sudo ./scripts/setup_ssh_backup.sh
```

## Performance Optimization

### Reduce Resource Usage

If system performance is impacted:

1. **Increase collection intervals**:
   ```bash
   # Edit config/monitoring.conf
   HEALTH_CHECK_INTERVAL=600  # 10 minutes instead of 5
   METRICS_COLLECTION_INTERVAL=300  # 5 minutes instead of 1
   ```

2. **Reduce metrics retention**:
   ```bash
   METRICS_HISTORY_DAYS=3  # Instead of 7
   LOG_RETENTION_DAYS=7    # Instead of 14
   ```

3. **Disable optional features**:
   ```bash
   LAPTOP_BACKUP_ENABLED=false
   AUTO_UPDATE_ENABLED=false
   ```

## Security Hardening

### Secure SSH Access

```bash
# Use dedicated SSH key
ssh-keygen -t ed25519 -f ~/.ssh/bitaxe_backup_ed25519

# Restrict SSH key usage
echo 'command="rsync --server -t .",restrict' > ~/.ssh/authorized_keys_bitaxe
```

### Firewall Rules

```bash
# Allow only necessary access
sudo ufw allow from 192.168.1.0/24 to any port 22  # SSH from local network only
sudo ufw allow 5000  # Web dashboard
```

## Advanced Configuration

### Custom Metrics

Add custom metrics in `scripts/collect_metrics.sh`:

```bash
collect_custom_metrics() {
    # Example: Monitor specific process
    local process_count=$(pgrep -c python3)
    execute_sql "
    INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
    ('application', 'python_processes', $process_count, 'count');
    "
}
```

### Alert Integration

Add Slack/Discord notifications:

```bash
# In health_check.sh, add alert function
send_alert() {
    local message="$1"
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"BitAxe Alert: $message\"}" \
         YOUR_WEBHOOK_URL
}
```

## Conclusion

Your BitAxe monitoring system is now fully configured and operational. The system will:

- ✅ Monitor system health every 5 minutes
- ✅ Collect performance metrics every minute
- ✅ Create daily backups at 2:00 AM
- ✅ Perform weekly maintenance on Sundays
- ✅ Rotate logs daily
- ✅ Provide web dashboard access

For ongoing support, refer to the main `README_MONITORING.md` file and use the management commands:

```bash
# Quick status check
./scripts/view_schedule.sh

# Control all services
sudo ./scripts/control_timers.sh status

# View logs
sudo ./scripts/control_timers.sh logs health-check
```

The monitoring system will now run automatically and keep your BitAxe system healthy and backed up!