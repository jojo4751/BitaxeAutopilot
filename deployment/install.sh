#!/bin/bash
# BitAxe V2.0.0 Production Installation Script for Raspberry Pi

set -e

echo "🚀 Installing BitAxe Web Management System V2.0.0"

# Configuration
INSTALL_DIR="/opt/bitaxe-web"
LOG_DIR="/var/log/bitaxe"
USER="bitaxe"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)" 
   exit 1
fi

echo "📦 Updating system packages..."
apt update && apt upgrade -y

echo "📦 Installing required system packages..."
apt install -y python3 python3-pip python3-venv python3-dev
apt install -y sqlite3 git curl wget nano htop tree
apt install -y systemd-sysv

echo "👤 Creating BitAxe user..."
if ! id "$USER" &>/dev/null; then
    useradd -m -s /bin/bash $USER
    echo "User $USER created"
else
    echo "User $USER already exists"
fi

echo "📁 Creating directories..."
mkdir -p $INSTALL_DIR
mkdir -p $LOG_DIR
mkdir -p $INSTALL_DIR/data
mkdir -p $INSTALL_DIR/backups

# Set permissions
chown -R $USER:$USER $INSTALL_DIR
chown -R $USER:$USER $LOG_DIR

echo "📋 Copying application files..."
# Assuming the script is run from the project directory
cp -r . $INSTALL_DIR/
chown -R $USER:$USER $INSTALL_DIR

echo "🐍 Setting up Python virtual environment..."
sudo -u $USER bash -c "cd $INSTALL_DIR && python3 -m venv venv"
sudo -u $USER bash -c "cd $INSTALL_DIR && source venv/bin/activate && pip install --upgrade pip"
sudo -u $USER bash -c "cd $INSTALL_DIR && source venv/bin/activate && pip install -r requirements.txt"

echo "📝 Installing systemd services..."
cp deployment/systemd/*.service /etc/systemd/system/
systemctl daemon-reload

echo "🗄️ Initializing database..."
sudo -u $USER bash -c "cd $INSTALL_DIR && source venv/bin/activate && python scripts/init_database.py"

echo "⚙️ Configuring environment..."
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp $INSTALL_DIR/.env.example $INSTALL_DIR/.env
    echo "Please edit $INSTALL_DIR/.env to configure your miner IPs and settings"
fi

echo "🔥 Enabling and starting services..."
systemctl enable bitaxe-web bitaxe-logger bitaxe-autopilot
systemctl start bitaxe-web bitaxe-logger bitaxe-autopilot

echo "🔍 Checking service status..."
systemctl status bitaxe-web --no-pager -l
systemctl status bitaxe-logger --no-pager -l
systemctl status bitaxe-autopilot --no-pager -l

echo ""
echo "✅ BitAxe V2.0.0 installation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit $INSTALL_DIR/config/config.json with your miner IP addresses"
echo "2. Edit $INSTALL_DIR/.env for additional configuration"
echo "3. Check logs: journalctl -u bitaxe-web -f"
echo "4. Access web interface: http://$(hostname -I | awk '{print $1}'):5000"
echo ""
echo "🔧 Useful commands:"
echo "   sudo systemctl restart bitaxe-web"
echo "   sudo systemctl status bitaxe-web"
echo "   journalctl -u bitaxe-web -f"
echo ""