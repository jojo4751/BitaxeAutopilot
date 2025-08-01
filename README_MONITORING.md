# BitAxe V2.0.0 - Monitoring & Backup Infrastructure

## Overview

This comprehensive monitoring and backup system provides automated infrastructure management for BitAxe mining operations. The system includes health monitoring, performance metrics collection, automated backups with laptop synchronization, system maintenance, and a web-based dashboard.

## Features

### 🔍 Health Monitoring
- **Real-time health checks** every 5 minutes
- **Automatic service restart** with cooldown periods and attempt limits
- **System resource monitoring** (CPU, memory, disk, temperature)
- **API endpoint health verification**
- **Service status tracking** with detailed logging

### 📊 Performance Metrics
- **System metrics**: CPU usage, memory usage, disk usage, network traffic
- **Mining metrics**: hashrate, efficiency, temperature, active miners
- **Database metrics**: size, query performance, record counts
- **Application metrics**: API response times, service status
- **Historical data retention** with configurable cleanup

### 💾 Automated Backups
- **Daily database backups** with compression
- **CSV exports** for Excel analysis
- **Laptop synchronization** via SSH
- **Retention policies** for local and remote backups
- **Backup verification** and reporting

### 🔧 System Maintenance
- **Automated log rotation** and cleanup
- **Database optimization** (VACUUM, REINDEX)
- **Temporary file cleanup**
- **System package updates** (optional)
- **Performance optimization**

### 🌐 Web Dashboard
- **Real-time monitoring dashboard** with auto-refresh
- **Interactive charts** and visualizations
- **Health status overview** with alerts
- **System logs viewer**
- **Backup status and history**

## Installation

### Prerequisites

- Ubuntu/Debian-based Linux system
- Root/sudo access
- BitAxe V2.0.0 system already installed
- SSH access for laptop backup (optional)

### Quick Setup

1. **Clone the repository** (if not already done):
   ```bash
   cd /opt/bitaxe-web
   git clone <repository> BITAXE_V2.0.0
   cd BITAXE_V2.0.0
   ```

2. **Configure monitoring settings**:
   ```bash
   sudo nano config/monitoring.conf
   ```
   Update laptop backup settings, thresholds, and preferences.

3. **Setup SSH backup** (optional):
   ```bash
   chmod +x scripts/setup_ssh_backup.sh
   ./scripts/setup_ssh_backup.sh
   ```

4. **Install scheduled tasks**:
   ```bash
   chmod +x scripts/setup_cron.sh
   sudo ./scripts/setup_cron.sh
   ```

5. **Start monitoring**:
   ```bash
   sudo ./scripts/control_timers.sh start
   ```

## Configuration

### Main Configuration File

Edit `config/monitoring.conf` to customize the monitoring system:

```bash
# Backup Configuration
BACKUP_ENABLED=true
BACKUP_LOCAL_DIR="/opt/bitaxe-web/backups"
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESS=true

# Laptop Backup Settings
LAPTOP_BACKUP_ENABLED=true
LAPTOP_USER="your-username"
LAPTOP_HOST="192.168.1.100"
LAPTOP_BACKUP_DIR="~/bitaxe_backups"
LAPTOP_RETENTION_DAYS=90

# Health Monitoring
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=300  # 5 minutes
AUTO_RESTART_ENABLED=true
MAX_RESTART_ATTEMPTS=3
RESTART_COOLDOWN=60

# System Monitoring
SYSTEM_MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=60  # 1 minute

# Resource Thresholds
CPU_WARNING_THRESHOLD=80
CPU_CRITICAL_THRESHOLD=95
MEMORY_WARNING_THRESHOLD=80
MEMORY_CRITICAL_THRESHOLD=90
DISK_WARNING_THRESHOLD=80
DISK_CRITICAL_THRESHOLD=90
TEMP_WARNING_THRESHOLD=70
TEMP_CRITICAL_THRESHOLD=80

# Services to Monitor
MONITORED_SERVICES="bitaxe-web bitaxe-autopilot bitaxe-logger"

# Logging
LOG_DIR="/var/log/bitaxe"
LOG_RETENTION_DAYS=14
LOG_MAX_SIZE="100M"

# Maintenance
MAINTENANCE_ENABLED=true
AUTO_UPDATE_ENABLED=false
CLEANUP_TEMP_FILES=true
DATABASE_VACUUM_ENABLED=true
```

## System Components

### Core Scripts

| Script | Purpose | Schedule |
|--------|---------|----------|
| `health_check.sh` | System health monitoring | Every 5 minutes |
| `collect_metrics.sh` | Performance metrics collection | Every minute |
| `backup_sync.sh` | Database backup and laptop sync | Daily at 2:00 AM |
| `maintenance.sh` | System maintenance tasks | Weekly on Sunday |
| `log_cleanup.sh` | Log rotation and cleanup | Daily at midnight |
| `system_update.sh` | System package updates | Monthly (optional) |

### Management Scripts

| Script | Purpose |
|--------|---------|
| `setup_cron.sh` | Install/configure scheduled tasks |
| `setup_ssh_backup.sh` | Configure laptop backup via SSH |
| `control_timers.sh` | Start/stop/manage all timers |
| `view_schedule.sh` | View schedule and system status |

### Web Interface

| Route | Description |
|-------|-------------|
| `/monitoring` | Main monitoring dashboard |
| `/monitoring/health` | System health status |
| `/monitoring/metrics` | Performance metrics charts |
| `/monitoring/logs` | System logs viewer |
| `/monitoring/backups` | Backup status and history |

## Usage

### Starting the Monitoring System

```bash
# Start all monitoring timers
sudo ./scripts/control_timers.sh start

# Check status
sudo ./scripts/control_timers.sh status

# View schedule
./scripts/view_schedule.sh
```

### Manual Operations

```bash
# Run manual health check
sudo ./scripts/health_check.sh

# Collect metrics manually
sudo ./scripts/collect_metrics.sh

# Run backup manually
sudo ./scripts/backup_sync.sh

# Perform maintenance
sudo ./scripts/maintenance.sh

# View logs
sudo ./scripts/control_timers.sh logs health-check
sudo ./scripts/control_timers.sh logs backup
```

### Monitoring Commands

```bash
# View timer status
systemctl list-timers 'bitaxe-*'

# Check service logs
journalctl -u bitaxe-health-check.service -f

# Stop all monitoring
sudo ./scripts/control_timers.sh stop

# Restart monitoring
sudo ./scripts/control_timers.sh restart
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   
   # Check ownership
   chown -R root:root scripts/
   ```

2. **SSH Backup Issues**
   ```bash
   # Test SSH connection
   ssh user@laptop-ip
   
   # Re-run SSH setup
   ./scripts/setup_ssh_backup.sh
   ```

3. **Database Issues**
   ```bash
   # Check database integrity
   sqlite3 /opt/bitaxe-web/data/bitaxe_data.db "PRAGMA integrity_check;"
   
   # Manual backup
   sqlite3 /opt/bitaxe-web/data/bitaxe_data.db ".backup backup.db"
   ```

4. **Service Not Starting**
   ```bash
   # Check service status
   systemctl status bitaxe-health-check.service
   
   # View service logs
   journalctl -u bitaxe-health-check.service
   
   # Restart service
   systemctl restart bitaxe-health-check.service
   ```

### Log Locations

- **Health Check**: `/var/log/bitaxe/health_check.log`
- **Metrics**: `/var/log/bitaxe/metrics.log`
- **Backup**: `/var/log/bitaxe/backup.log`
- **Maintenance**: `/var/log/bitaxe/maintenance.log`
- **System Logs**: `journalctl -u bitaxe-*.service`

### Configuration Validation

```bash
# Test configuration
source config/monitoring.conf
echo "Backup enabled: $BACKUP_ENABLED"
echo "Health checks: $HEALTH_CHECK_ENABLED"

# Validate scripts
sudo ./scripts/control_timers.sh test
```

## Performance Impact

The monitoring system is designed to be lightweight:

- **Health checks**: ~1-2% CPU for 10-15 seconds every 5 minutes
- **Metrics collection**: ~0.5% CPU for 5-10 seconds every minute
- **Memory usage**: ~50-100MB additional RAM usage
- **Disk usage**: ~100MB/month for logs and metrics (with cleanup)
- **Network**: Minimal impact except during laptop backup sync

## Security Considerations

- **SSH keys**: Use dedicated SSH keys for backup access
- **File permissions**: Scripts run as root but with minimal privileges
- **Network access**: Laptop backup requires SSH access
- **Log rotation**: Automated cleanup prevents disk filling
- **Backup encryption**: Consider encrypting backup data

## Customization

### Adding Custom Metrics

1. **Edit** `scripts/collect_metrics.sh`
2. **Add collection function**:
   ```bash
   collect_custom_metrics() {
       local custom_value=$(your_command_here)
       execute_sql "
       INSERT INTO system_metrics (metric_type, metric_name, metric_value, metric_unit) VALUES
       ('custom', 'your_metric', $custom_value, 'unit');
       "
   }
   ```
3. **Add to main function**:
   ```bash
   collect_custom_metrics
   ```

### Custom Health Checks

1. **Edit** `scripts/health_check.sh`
2. **Add check function**:
   ```bash
   check_custom_health() {
       # Your health check logic
       if [[ condition ]]; then
           log_success "Custom check passed"
           return 0
       else
           log_error "Custom check failed"
           return 1
       fi
   }
   ```

### Additional Backup Destinations

1. **Edit** `scripts/backup_sync.sh`
2. **Add sync function**:
   ```bash
   sync_to_cloud() {
       # Your cloud sync logic
       rclone copy "$backup_file" "cloud:bitaxe-backups/"
   }
   ```

## API Integration

### Health Check API

```bash
# Get health status
curl http://localhost:5000/monitoring/api/health

# Get metrics
curl http://localhost:5000/monitoring/api/metrics?hours=24

# Get service status
curl http://localhost:5000/monitoring/api/services
```

### Example Response

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "overall_status": "OK",
  "system_metrics": {
    "cpu_usage": 25.5,
    "memory_usage": 67.2,
    "disk_usage": 45.8,
    "temperature": "N/A",
    "load_average": "0.75"
  },
  "services": {
    "bitaxe-web": "running",
    "bitaxe-autopilot": "running",
    "bitaxe-logger": "running"
  }
}
```

## Support

### Getting Help

1. **Check logs** for specific error messages
2. **Review configuration** for incorrect settings
3. **Test individual components** manually
4. **Verify dependencies** and permissions

### Reporting Issues

Include the following information:
- System information (`uname -a`)
- Service status (`systemctl status bitaxe-*.service`)
- Recent logs (`journalctl -u bitaxe-*.service --since "1 hour ago"`)
- Configuration file (`config/monitoring.conf`)

## License

This monitoring system is part of the BitAxe V2.0.0 project and follows the same licensing terms.