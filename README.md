# BitAxe Web Management System V2.0.0

A comprehensive web-based management system for BitAxe ASIC Bitcoin mining devices, featuring real-time monitoring, automated optimization, and professional-grade logging.

## 🚀 Features

### Core Functionality
- **Real-time Monitoring**: Live dashboard with miner status, temperature, hashrate, and power consumption
- **Manual Control**: Web interface for adjusting frequency, voltage, and fan settings
- **Automated Benchmarking**: Performance testing with various frequency/voltage combinations
- **Autopilot Mode**: Intelligent optimization with temperature safety guards
- **Data Export**: CSV export of historical performance data

### Advanced Features
- **REST API**: Complete API for integration with external tools
- **Structured Logging**: JSON-formatted logs for monitoring and analysis
- **Multi-Miner Support**: Manage multiple BitAxe devices from a single interface
- **Safety Systems**: Temperature-based throttling and emergency shutdown
- **Historical Analytics**: Trend analysis and performance tracking

## 📋 Requirements

### Hardware
- Raspberry Pi 4 (2GB+ RAM recommended)
- MicroSD card (16GB+ Class 10)
- BitAxe ASIC mining device(s) on the same network

### Software
- Raspberry Pi OS Lite (64-bit)
- Python 3.9+
- SQLite 3
- Network access to BitAxe devices

## 🔧 Quick Installation

### Automated Installation (Recommended)
```bash
# Download and run the installation script
curl -sSL https://github.com/jojo4751/BitaxeAutopilot/raw/v2.0.0/deployment/install.sh | sudo bash
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/jojo4751/BitaxeAutopilot.git
cd BitaxeAutopilot

# Run installation script
sudo bash deployment/install.sh
```

## ⚙️ Configuration

### 1. Configure Miner IPs
Edit `/opt/bitaxe-web/config/config.json`:
```json
{
  "config": {
    "ips": [
      "192.168.1.100",
      "192.168.1.101",
      "192.168.1.102"
    ]
  },
  "settings": {
    "temp_limit": 73,
    "temp_overheat": 75,
    "autopilot_enabled": true
  }
}
```

### 2. Environment Variables
Copy and edit the environment file:
```bash
sudo cp /opt/bitaxe-web/.env.example /opt/bitaxe-web/.env
sudo nano /opt/bitaxe-web/.env
```

### 3. Start Services
```bash
sudo systemctl enable bitaxe-web bitaxe-logger bitaxe-autopilot
sudo systemctl start bitaxe-web bitaxe-logger bitaxe-autopilot
```

## 🌐 Access

Once installed, access the web interface at:
- **Local**: http://localhost:5000
- **Network**: http://[raspberry-pi-ip]:5000

## 📊 Service Architecture

The system consists of three main services:

### 1. Web Interface (`bitaxe-web`)
- Flask web application
- REST API endpoints
- Real-time dashboard
- Manual control interface

### 2. Data Logger (`bitaxe-logger`)
- Continuous miner monitoring
- SQLite database storage
- Performance metric collection
- Event logging

### 3. Autopilot (`bitaxe-autopilot`)
- Automated optimization
- Temperature safety monitoring
- Efficiency tracking
- Smart benchmarking

## 🛠️ Management Commands

### Service Control
```bash
# Check service status
sudo systemctl status bitaxe-web
sudo systemctl status bitaxe-logger
sudo systemctl status bitaxe-autopilot

# Restart services
sudo systemctl restart bitaxe-web

# View logs
journalctl -u bitaxe-web -f
journalctl -u bitaxe-autopilot -f
```

### Database Management
```bash
# Initialize database
cd /opt/bitaxe-web
python scripts/init_database.py

# Backup database
cp data/bitaxe_data.db backups/bitaxe_data_$(date +%Y%m%d).db
```

## 📈 Monitoring

### Log Files
- Application logs: `journalctl -u bitaxe-web -f`
- System logs: `/var/log/bitaxe/bitaxe.log`
- Database: `/opt/bitaxe-web/data/bitaxe_data.db`

### Health Check
Access the API health endpoint:
```bash
curl http://localhost:5000/api/v1/health
```

## 🔒 Security

### Network Security
- Web interface runs on port 5000 (configure firewall as needed)
- API endpoints support rate limiting
- No external internet access required for core functionality

### Data Security
- All miner data stored locally on Raspberry Pi
- SQLite database with file-based access control
- No cloud services or external data transmission

## 🆘 Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check logs for errors
journalctl -u bitaxe-web -e

# Verify configuration
cd /opt/bitaxe-web
python -c "from config.config_loader import load_config; print(load_config())"
```

**Miners not responding:**
```bash
# Test miner connectivity
ping 192.168.1.100
curl http://192.168.1.100/api/system
```

**Database issues:**
```bash
# Check database integrity
sqlite3 /opt/bitaxe-web/data/bitaxe_data.db ".schema"
```

### Support
- Issues: [GitHub Issues](https://github.com/jojo4751/BitaxeAutopilot/issues)
- Documentation: [Wiki](https://github.com/jojo4751/BitaxeAutopilot/wiki)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BitAxe Hardware Team for creating excellent mining hardware
- Flask and Python communities for robust frameworks
- Raspberry Pi Foundation for affordable computing platforms

---

**BitAxe V2.0.0** - Professional Mining Management Made Simple