#!/usr/bin/env python3
"""
BitAxe V2.0.0 - Database Initialization Script
Creates the SQLite database schema for the BitAxe system
"""

import os
import sqlite3
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import load_config


def create_database_schema(db_path):
    """Create the database schema for BitAxe system"""
    
    print(f"Creating database at: {db_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Main telemetry logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                hashRate REAL,
                temp REAL,
                power REAL,
                frequency INTEGER,
                coreVoltage INTEGER,
                shares_accepted_delta INTEGER,
                hostname TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Benchmark results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL,
                frequency INTEGER,
                voltage INTEGER,
                hashrate REAL,
                temp REAL,
                power REAL,
                efficiency REAL,
                duration INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        # System events and protocol log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protocol (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL,
                event_type TEXT,
                message TEXT,
                severity TEXT DEFAULT 'INFO',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Best settings per miner
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tuning_status (
                ip TEXT PRIMARY KEY,
                best_frequency INTEGER,
                best_voltage INTEGER,
                best_efficiency REAL,
                best_hashrate REAL,
                last_update DATETIME DEFAULT CURRENT_TIMESTAMP,
                benchmark_count INTEGER DEFAULT 0
            )
        ''')
        
        # Efficiency tracking markers
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS efficiency_markers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL,
                efficiency REAL,
                hashrate REAL,
                power REAL,
                temperature REAL,
                frequency INTEGER,
                voltage INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_ip_timestamp ON logs(ip, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_benchmark_ip ON benchmark_results(ip)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_protocol_ip_timestamp ON protocol(ip, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_efficiency_ip_timestamp ON efficiency_markers(ip, timestamp)')
        
        conn.commit()
        print("✅ Database schema created successfully")


def main():
    """Main initialization function"""
    print("🗄️ BitAxe V2.0.0 Database Initialization")
    
    try:
        # Load configuration to get database path
        config = load_config()
        db_path = config.get('paths', {}).get('database', 'data/bitaxe_data.db')
        
        # Check for environment variable override
        db_path = os.environ.get('DATABASE_PATH', db_path)
        
        # Convert to absolute path
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(__file__), '..', db_path)
            db_path = os.path.abspath(db_path)
        
        create_database_schema(db_path)
        
        print(f"🎉 Database initialization complete!")
        print(f"📍 Database location: {db_path}")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()