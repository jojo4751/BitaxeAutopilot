# BitAxe V2.0.0 - Raspberry Pi Deployment Guide

Complete step-by-step guide for deploying BitAxe Web Management System on Raspberry Pi 4.

## 🏁 Prerequisites

### Hardware Setup
1. **Raspberry Pi 4** (2GB RAM minimum, 4GB recommended)
2. **MicroSD Card** (16GB+, Class 10 or better)
3. **Power Supply** (Official Pi 4 power adapter recommended)
4. **Network Connection** (Ethernet or WiFi)

### Software Requirements
- **Raspberry Pi OS Lite (64-bit)** - Fresh installation
- **SSH Access** - Enabled during setup or via `raspi-config`

## 📋 Step-by-Step Installation

### Step 1: Initial System Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git curl wget nano htop tree unzip

# Reboot after updates
sudo reboot
```

### Step 2: Create BitAxe User and Directories

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash bitaxe
sudo passwd bitaxe

# Create application directories
sudo mkdir -p /opt/bitaxe-web
sudo mkdir -p /var/log/bitaxe
sudo chown -R bitaxe:bitaxe /opt/bitaxe-web /var/log/bitaxe
```

### Step 3: Download and Install BitAxe V2.0.0

```bash
# Switch to bitaxe user
sudo su - bitaxe

# Clone the repository
cd /opt/bitaxe-web
git clone https://github.com/jojo4751/BitaxeAutopilot.git .

# Checkout V2.0.0 tag
git checkout v2.0.0
```

### Step 4: Install Python Dependencies

```bash
# Still as bitaxe user
cd /opt/bitaxe-web

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Configure the System

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Essential .env Settings:**
```bash
FLASK_SECRET_KEY=your-secure-random-key-here
FLASK_ENV=production
DATABASE_PATH=/opt/bitaxe-web/data/bitaxe_data.db
LOG_LEVEL=INFO
```

**Configure Miner IPs:**
```bash
nano config/config.json
```

Update the `ips` array with your BitAxe device IP addresses:
```json
{
  "config": {
    "ips": [
      "192.168.1.100",
      "192.168.1.101"
    ]
  }
}
```

### Step 6: Initialize Database

```bash
# Create database schema
python scripts/init_database.py
```

### Step 7: Install System Services

```bash
# Exit bitaxe user back to admin user
exit

# Copy systemd service files
sudo cp /opt/bitaxe-web/deployment/systemd/*.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services for auto-start
sudo systemctl enable bitaxe-web
sudo systemctl enable bitaxe-logger  
sudo systemctl enable bitaxe-autopilot
```

### Step 8: Start Services

```bash
# Start all services
sudo systemctl start bitaxe-web
sudo systemctl start bitaxe-logger
sudo systemctl start bitaxe-autopilot

# Check service status
sudo systemctl status bitaxe-web
sudo systemctl status bitaxe-logger
sudo systemctl status bitaxe-autopilot
```

### Step 9: Configure Firewall (Optional)

```bash
# Install and configure UFW
sudo apt install -y ufw

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow web interface
sudo ufw allow 5000/tcp

# Enable firewall
sudo ufw enable
```

### Step 10: Set Static IP (Recommended)

```bash
# Edit network configuration
sudo nano /etc/dhcpcd.conf

# Add at the end (adjust IPs for your network):
interface eth0
static ip_address=192.168.1.50/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8

# Reboot to apply network changes
sudo reboot
```

## 🔍 Verification and Testing

### Test Web Interface
1. Open browser and navigate to: `http://[raspberry-pi-ip]:5000`
2. Verify dashboard loads without errors
3. Check that miners are detected and showing data

### Test Services
```bash
# Check all services are running
sudo systemctl status bitaxe-web bitaxe-logger bitaxe-autopilot

# View live logs
journalctl -u bitaxe-web -f
```

### Test Miner Communication
```bash
# Test each miner manually
curl http://192.168.1.100/api/system
curl http://192.168.1.101/api/system
```

## 📊 Monitoring and Maintenance

### View Logs
```bash
# Web application logs
journalctl -u bitaxe-web -f

# Autopilot logs
journalctl -u bitaxe-autopilot -f

# Data logger logs
journalctl -u bitaxe-logger -f

# System resource usage
htop
```

### Database Maintenance
```bash
# Manual database backup
sudo -u bitaxe cp /opt/bitaxe-web/data/bitaxe_data.db /opt/bitaxe-web/backups/backup_$(date +%Y%m%d_%H%M%S).db

# Check database size
sudo -u bitaxe du -h /opt/bitaxe-web/data/bitaxe_data.db
```

### Service Management
```bash
# Restart services
sudo systemctl restart bitaxe-web

# Stop services
sudo systemctl stop bitaxe-autopilot

# Disable service from auto-start
sudo systemctl disable bitaxe-logger

# View service configuration
sudo systemctl cat bitaxe-web
```

## 🔧 Configuration Tuning

### Performance Optimization
```bash
# Edit service configuration for more resources
sudo systemctl edit bitaxe-web

# Add override configuration:
[Service]
Environment=PYTHONPATH=/opt/bitaxe-web
LimitNOFILE=65536
```

### Temperature Thresholds
Edit `/opt/bitaxe-web/config/config.json`:
```json
{
  "settings": {
    "temp_limit": 73,
    "temp_overheat": 75,
    "autopilot_enabled": true,
    "safety_check_interval": 300
  }
}
```

### Logging Configuration
Edit `/opt/bitaxe-web/.env`:
```bash
LOG_LEVEL=INFO
LOG_FILE=/var/log/bitaxe/application.log
```

## 🚨 Troubleshooting

### Service Startup Issues
```bash
# Check service status and logs
sudo systemctl status bitaxe-web
journalctl -u bitaxe-web -e

# Manual service start for debugging
sudo -u bitaxe bash
cd /opt/bitaxe-web
source venv/bin/activate
python app.py
```

### Miner Connection Issues
```bash
# Verify network connectivity
ping 192.168.1.100

# Check miner API
curl -v http://192.168.1.100/api/system

# Test from Pi
sudo -u bitaxe bash
cd /opt/bitaxe-web
source venv/bin/activate
python -c "
import requests
r = requests.get('http://192.168.1.100/api/system')
print(r.status_code, r.text)
"
```

### Database Issues
```bash
# Check database file permissions
ls -la /opt/bitaxe-web/data/

# Test database connection
sudo -u bitaxe sqlite3 /opt/bitaxe-web/data/bitaxe_data.db ".tables"

# Rebuild database if needed
sudo -u bitaxe bash
cd /opt/bitaxe-web
source venv/bin/activate
python scripts/init_database.py
```

### Memory/Performance Issues
```bash
# Check system resources
free -h
df -h
htop

# Reduce log retention
sudo journalctl --vacuum-time=7d

# Monitor service resource usage
sudo systemctl status bitaxe-web -l
```

## 📈 Advanced Configuration

### Custom Port Configuration
Edit `/opt/bitaxe-web/.env`:
```bash
PORT=8080
HOST=0.0.0.0
```

Then restart the web service:
```bash
sudo systemctl restart bitaxe-web
```

### SSL/HTTPS Setup (Optional)
For production deployments, consider setting up a reverse proxy with SSL:

```bash
# Install nginx
sudo apt install -y nginx

# Configure proxy (example configuration)
sudo nano /etc/nginx/sites-available/bitaxe
```

### Backup Automation
Create automated backup script:
```bash
sudo nano /opt/bitaxe-web/scripts/backup_cron.sh
```

Add to crontab for daily backups:
```bash
sudo crontab -e
# Add line:
0 2 * * * /opt/bitaxe-web/scripts/backup_cron.sh
```

## ✅ Post-Installation Checklist

- [ ] Web interface accessible at `http://[pi-ip]:5000`
- [ ] All three services running and enabled
- [ ] Miners detected and communicating
- [ ] Database initialized and receiving data
- [ ] Logs showing normal operation
- [ ] Temperature monitoring active
- [ ] Autopilot mode functional (if enabled)
- [ ] Backup strategy implemented

## 🆘 Support

If you encounter issues:

1. **Check the logs first**: `journalctl -u bitaxe-web -e`
2. **Verify configuration**: Review config files for typos
3. **Test connectivity**: Ensure miners are reachable
4. **GitHub Issues**: Report bugs with full logs
5. **Community**: Join discussions for help

---

**BitAxe V2.0.0** - Professional deployment made simple! 🚀