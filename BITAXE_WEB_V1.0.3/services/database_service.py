import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager


class DatabaseService:
    def __init__(self, config_service):
        self.config_service = config_service

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.config_service.database_path)
        try:
            yield conn
        finally:
            conn.close()

    def initialize_tables(self):
        """Create all necessary database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Logs table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_service.get("paths.log_table", "logs")} (
                    timestamp TEXT,
                    ip TEXT,
                    hostname TEXT,
                    temp REAL,
                    hashRate REAL,
                    power REAL,
                    voltage REAL,
                    frequency INTEGER,
                    coreVoltage INTEGER,
                    fanrpm INTEGER,
                    sharesAccepted INTEGER,
                    sharesRejected INTEGER,
                    uptime INTEGER,
                    version TEXT
                )
            ''')
            
            # Tuning status table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_service.get("paths.tuning_table", "tuning_status")} (
                    ip TEXT,
                    timestamp TEXT,
                    frequency INTEGER,
                    coreVoltage INTEGER,
                    avgHashRate REAL,
                    avgTemperature REAL,
                    avgEfficiency REAL,
                    duration INTEGER
                )
            ''')
            
            # Protocol/events table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_service.get("paths.protocol_table", "protocol")} (
                    timestamp TEXT,
                    ip TEXT,
                    event_type TEXT,
                    message TEXT
                )
            ''')
            
            # Efficiency markers table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_service.get("paths.efficiency_markers", "efficiency_markers")} (
                    timestamp TEXT,
                    ip TEXT,
                    efficiency REAL,
                    hashRate REAL,
                    power REAL,
                    temperature REAL,
                    frequency INTEGER,
                    coreVoltage INTEGER
                )
            ''')
            
            # Benchmark results table
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.config_service.get("paths.benchmark_results", "benchmark_results")} (
                    timestamp TEXT,
                    ip TEXT,
                    frequency INTEGER,
                    coreVoltage INTEGER,
                    averageHashRate REAL,
                    averageTemperature REAL,
                    efficiencyJTH REAL,
                    duration INTEGER
                )
            ''')
            
            conn.commit()

    def log_miner_data(self, ip: str, data: Dict[str, Any]) -> None:
        """Log miner telemetry data"""
        timestamp = datetime.now().isoformat()
        log_table = self.config_service.get("paths.log_table", "logs")
        
        row = (
            timestamp,
            ip,
            data.get("hostname"),
            round(data.get("temp", 0), 1),
            round(data.get("hashRate", 0), 2),
            round(data.get("power", 0), 2),
            round(data.get("voltage", 0), 2),
            data.get("frequency"),
            data.get("coreVoltage"),
            data.get("fanrpm"),
            data.get("sharesAccepted"),
            data.get("sharesRejected"),
            data.get("uptimeSeconds"),
            data.get("version")
        )

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {log_table} (
                    timestamp, ip, hostname, temp, hashRate, power, voltage,
                    frequency, coreVoltage, fanrpm, sharesAccepted, sharesRejected,
                    uptime, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', row)
            conn.commit()

    def log_event(self, ip: str, event_type: str, message: str) -> None:
        """Log system events"""
        timestamp = datetime.now().isoformat()
        protocol_table = self.config_service.get("paths.protocol_table", "protocol")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {protocol_table} (timestamp, ip, event_type, message)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, ip, event_type, message))
            conn.commit()

    def save_benchmark_result(self, ip: str, frequency: int, core_voltage: int, 
                            avg_hashrate: float, avg_temp: float, efficiency: float, 
                            duration: int) -> None:
        """Save benchmark results"""
        timestamp = datetime.now().isoformat()
        benchmark_table = self.config_service.get("paths.benchmark_results", "benchmark_results")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {benchmark_table} 
                (timestamp, ip, frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, ip, frequency, core_voltage, avg_hashrate, avg_temp, efficiency, duration))
            conn.commit()

    def log_efficiency_marker(self, ip: str, efficiency: float, hashrate: float, 
                            power: float, temperature: float, frequency: int, 
                            core_voltage: int) -> None:
        """Log efficiency marker"""
        timestamp = datetime.now().isoformat()
        efficiency_table = self.config_service.get("paths.efficiency_markers", "efficiency_markers")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {efficiency_table}
                (timestamp, ip, efficiency, hashRate, power, temperature, frequency, coreVoltage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, ip, efficiency, hashrate, power, temperature, frequency, core_voltage))
            conn.commit()

    def get_latest_status(self) -> List[Dict[str, Any]]:
        """Get latest status for all miners"""
        log_table = self.config_service.get("paths.log_table", "logs")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT ip, hostname, temp, hashRate, power, voltage, frequency, 
                       coreVoltage, fanrpm, sharesAccepted, sharesRejected, uptime, version, timestamp
                FROM {log_table}
                WHERE (ip, timestamp) IN (
                    SELECT ip, MAX(timestamp) FROM {log_table} GROUP BY ip
                )
                ORDER BY ip
            ''')
            
            columns = ['ip', 'hostname', 'temp', 'hashRate', 'power', 'voltage', 
                      'frequency', 'coreVoltage', 'fanrpm', 'sharesAccepted', 
                      'sharesRejected', 'uptime', 'version', 'timestamp']
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events"""
        protocol_table = self.config_service.get("paths.protocol_table", "protocol")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT timestamp, ip, event_type, message
                FROM {protocol_table}
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            columns = ['timestamp', 'ip', 'event_type', 'message']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_history_data(self, start: datetime, end: datetime) -> Dict[str, Dict[str, Any]]:
        """Get historical data for time range"""
        log_table = self.config_service.get("paths.log_table", "logs")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT ip, timestamp, temp, hashRate, power, frequency, coreVoltage
                FROM {log_table}
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start.isoformat(), end.isoformat()))
            
            data = {}
            for row in cursor.fetchall():
                ip = row[0]
                if ip not in data:
                    data[ip] = {'traces': []}
                
                data[ip]['traces'].append({
                    'timestamp': row[1],
                    'temp': row[2],
                    'hashRate': row[3],
                    'power': row[4],
                    'frequency': row[5],
                    'coreVoltage': row[6]
                })
            
            return data

    def get_top_settings(self) -> List[Dict[str, Any]]:
        """Get top performing settings"""
        benchmark_table = self.config_service.get("paths.benchmark_results", "benchmark_results")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT ip, frequency, coreVoltage, averageHashRate, efficiencyJTH
                FROM {benchmark_table}
                ORDER BY efficiencyJTH DESC
                LIMIT 10
            ''')
            
            columns = ['ip', 'frequency', 'coreVoltage', 'averageHashRate', 'efficiencyJTH']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_best_efficiency_for_ip(self, ip: str) -> Optional[float]:
        """Get best efficiency for specific IP"""
        tuning_table = self.config_service.get("paths.tuning_table", "tuning_status")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT MIN(avgEfficiency) FROM {tuning_table} WHERE ip = ?
            ''', (ip,))
            
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else None

    def get_benchmark_results_for_ip(self, ip: str, limit: int = 10) -> List[Tuple]:
        """Get recent benchmark results for specific IP"""
        benchmark_table = self.config_service.get("paths.benchmark_results", "benchmark_results")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, timestamp
                FROM {benchmark_table}
                WHERE ip = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (ip, limit))
            
            return cursor.fetchall()

    def get_benchmark_results(self, limit: int = 50) -> List[Tuple]:
        """Get all benchmark results ordered by performance"""
        benchmark_table = self.config_service.get("paths.benchmark_results", "benchmark_results")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT ip, frequency, coreVoltage, averageHashRate, averageTemperature, efficiencyJTH, timestamp, duration
                FROM {benchmark_table}
                ORDER BY averageHashRate DESC 
                LIMIT ?
            ''', (limit,))
            
            return cursor.fetchall()

    def get_efficiency_data_for_export(self, ip: str, start_str: str, end_str: str) -> Tuple[List, List]:
        """Get efficiency data for CSV export"""
        efficiency_table = self.config_service.get("paths.efficiency_markers", "efficiency_markers")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT * FROM {efficiency_table}
                WHERE ip = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (ip, start_str, end_str))
            
            rows = cursor.fetchall()
            headers = [d[0] for d in cursor.description]
            
            return headers, rows