"""
BitAxe V2.0.0 - Unified Database Manager
Consolidates all database operations into a single, robust class with comprehensive
error handling, connection pooling, and transaction management
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator, ContextManager
from datetime import datetime, timedelta
from contextlib import contextmanager
from threading import Lock
import os
from pathlib import Path

from .exceptions import DatabaseError
from .config_manager import ConfigManager


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Unified database manager for BitAxe system.
    
    This class provides centralized database management with features including:
    - Connection pooling and thread-safe operations
    - Automatic table creation and schema management
    - Comprehensive error handling and logging
    - Transaction support with context managers
    - Query optimization and result caching
    - Database maintenance and integrity checks
    
    Example:
        >>> db = DatabaseManager(config_manager)
        >>> with db.get_connection() as conn:
        ...     db.execute_query(conn, "SELECT * FROM miners")
        >>> 
        >>> # Using transaction context
        >>> with db.transaction() as tx:
        ...     db.log_event(tx, "MINER_001", "STATUS", "Miner online")
    """
    
    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize the database manager.
        
        Args:
            config_manager: Configuration manager instance
            
        Raises:
            DatabaseError: If database initialization fails
        """
        self.config_manager = config_manager
        self._lock = Lock()
        self._connection_pool: List[sqlite3.Connection] = []
        self._pool_size = 5
        self._initialized = False
        
        # Get database path from configuration
        self.database_path = self.config_manager.database_path
        
        # Initialize database
        self.initialize_database()
        
        logger.info(f"DatabaseManager initialized with database: {self.database_path}")
    
    def initialize_database(self) -> None:
        """Initialize database with required tables and indexes.
        
        Raises:
            DatabaseError: If database initialization fails
        """
        try:
            # Ensure database directory exists
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            # Create database connection and tables
            with self.get_connection() as conn:
                self._create_tables(conn)
                self._create_indexes(conn)
                self._enable_optimizations(conn)
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to initialize database: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """Create all required database tables.
        
        Args:
            conn: Database connection
        """
        # Main miner logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                hashRate REAL,
                temp REAL,
                fanSpeed INTEGER,
                power REAL,
                frequency INTEGER,
                coreVoltage INTEGER,
                efficiency REAL,
                bestDiff TEXT,
                freeHeap INTEGER,
                uptime INTEGER,
                wifiRSSI INTEGER,
                sharesAccepted INTEGER,
                sharesRejected INTEGER,
                asicCount INTEGER
            )
        """)
        
        # Benchmark results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                coreVoltage INTEGER NOT NULL,
                test_duration INTEGER NOT NULL,
                avg_hashrate REAL,
                min_hashrate REAL,
                max_hashrate REAL,
                avg_temp REAL,
                max_temp REAL,
                avg_power REAL,
                efficiency REAL,
                shares_accepted INTEGER,
                shares_rejected INTEGER,
                best_diff TEXT,
                stability_score REAL,
                notes TEXT
            )
        """)
        
        # System events and protocol messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS protocol (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT DEFAULT 'INFO',
                context TEXT
            )
        """)
        
        # Tuning status for optimal settings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tuning_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip TEXT NOT NULL UNIQUE,
                best_frequency INTEGER,
                best_voltage INTEGER,
                best_efficiency REAL,
                best_hashrate REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL DEFAULT 0.0,
                total_benchmarks INTEGER DEFAULT 0
            )
        """)
        
        # Efficiency markers for drift detection
        conn.execute("""
            CREATE TABLE IF NOT EXISTS efficiency_markers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                coreVoltage INTEGER NOT NULL,
                avg_efficiency REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                period_hours INTEGER DEFAULT 1
            )
        """)
        
        # System metrics for monitoring
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                additional_data TEXT,
                source_ip TEXT
            )
        """)
        
        # Configuration history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                config_key TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT NOT NULL,
                changed_by TEXT DEFAULT 'system',
                reason TEXT
            )
        """)
        
        # Analytics tables for comprehensive monitoring
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                period_type TEXT NOT NULL,  -- 'hourly', 'daily', 'weekly'
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                ip TEXT NOT NULL,
                avg_hashrate REAL,
                min_hashrate REAL,
                max_hashrate REAL,
                avg_temp REAL,
                min_temp REAL,
                max_temp REAL,
                avg_power REAL,
                avg_efficiency REAL,
                total_shares_accepted INTEGER,
                total_shares_rejected INTEGER,
                uptime_percentage REAL,
                data_points INTEGER,
                performance_score REAL
            )
        """)
        
        # Performance alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                alert_type TEXT NOT NULL,  -- 'temperature', 'efficiency', 'hashrate', 'connectivity'
                severity TEXT NOT NULL,   -- 'info', 'warning', 'critical'
                threshold_value REAL,
                actual_value REAL,
                message TEXT NOT NULL,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_by TEXT,
                acknowledged_at DATETIME,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at DATETIME
            )
        """)
        
        # Profitability tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profitability_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                hashrate REAL NOT NULL,
                power_consumption REAL NOT NULL,
                electricity_cost_kwh REAL DEFAULT 0.12,
                btc_price_usd REAL,
                network_difficulty REAL,
                pool_fee_percentage REAL DEFAULT 1.0,
                estimated_daily_btc REAL,
                estimated_daily_usd REAL,
                daily_power_cost_usd REAL,
                estimated_daily_profit_usd REAL,
                profit_margin_percentage REAL
            )
        """)
        
        # Miner health tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS miner_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                health_score REAL NOT NULL,  -- 0-100 overall health score
                temp_health REAL,           -- 0-100 temperature health
                efficiency_health REAL,     -- 0-100 efficiency health
                stability_health REAL,      -- 0-100 stability health
                connectivity_health REAL,   -- 0-100 connectivity health
                last_restart DATETIME,
                consecutive_errors INTEGER DEFAULT 0,
                maintenance_due BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
        """)
        
        # Data quality metrics
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip TEXT NOT NULL,
                collection_timestamp DATETIME NOT NULL,
                data_completeness REAL,     -- percentage of expected fields present
                data_validity REAL,         -- percentage of valid data points
                anomaly_score REAL,         -- 0-1 anomaly detection score
                missing_fields TEXT,        -- JSON array of missing field names
                invalid_fields TEXT,        -- JSON array of invalid field names
                quality_flags TEXT          -- JSON array of quality issue flags
            )
        """)
        
        logger.debug("Database tables created/verified")
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance optimization.
        
        Args:
            conn: Database connection
        """
        indexes = [
            # Main logs table indexes
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_ip ON logs(ip)",
            "CREATE INDEX IF NOT EXISTS idx_logs_ip_timestamp ON logs(ip, timestamp)",
            
            # Benchmark results indexes
            "CREATE INDEX IF NOT EXISTS idx_benchmark_timestamp ON benchmark_results(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_ip ON benchmark_results(ip)",
            "CREATE INDEX IF NOT EXISTS idx_benchmark_efficiency ON benchmark_results(efficiency DESC)",
            
            # Protocol/events indexes
            "CREATE INDEX IF NOT EXISTS idx_protocol_timestamp ON protocol(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_protocol_ip ON protocol(ip)",
            "CREATE INDEX IF NOT EXISTS idx_protocol_event_type ON protocol(event_type)",
            
            # Efficiency markers indexes
            "CREATE INDEX IF NOT EXISTS idx_efficiency_timestamp ON efficiency_markers(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_efficiency_ip ON efficiency_markers(ip)",
            
            # System metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type_name ON system_metrics(metric_type, metric_name)",
            
            # Configuration history indexes
            "CREATE INDEX IF NOT EXISTS idx_config_history_timestamp ON config_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_config_history_key ON config_history(config_key)",
            
            # Analytics summary indexes
            "CREATE INDEX IF NOT EXISTS idx_analytics_summary_timestamp ON analytics_summary(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_summary_ip ON analytics_summary(ip)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_summary_period ON analytics_summary(period_type, period_start)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_summary_performance ON analytics_summary(performance_score DESC)",
            
            # Performance alerts indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_alerts_timestamp ON performance_alerts(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_alerts_ip ON performance_alerts(ip)",
            "CREATE INDEX IF NOT EXISTS idx_performance_alerts_severity ON performance_alerts(severity)",
            "CREATE INDEX IF NOT EXISTS idx_performance_alerts_unresolved ON performance_alerts(resolved, acknowledged)",
            
            # Profitability data indexes
            "CREATE INDEX IF NOT EXISTS idx_profitability_timestamp ON profitability_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_profitability_ip ON profitability_data(ip)",
            "CREATE INDEX IF NOT EXISTS idx_profitability_profit ON profitability_data(estimated_daily_profit_usd DESC)",
            
            # Miner health indexes
            "CREATE INDEX IF NOT EXISTS idx_miner_health_timestamp ON miner_health(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_miner_health_ip ON miner_health(ip)",
            "CREATE INDEX IF NOT EXISTS idx_miner_health_score ON miner_health(health_score DESC)",
            
            # Data quality indexes
            "CREATE INDEX IF NOT EXISTS idx_data_quality_timestamp ON data_quality(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_data_quality_ip ON data_quality(ip)",
            "CREATE INDEX IF NOT EXISTS idx_data_quality_score ON data_quality(anomaly_score DESC)"
        ]
        
        for index_sql in indexes:
            try:
                conn.execute(index_sql)
            except sqlite3.Error as e:
                logger.warning(f"Failed to create index: {e}")
        
        logger.debug("Database indexes created/verified")
    
    def _enable_optimizations(self, conn: sqlite3.Connection) -> None:
        """Enable SQLite optimizations for better performance.
        
        Args:
            conn: Database connection
        """
        optimizations = [
            "PRAGMA journal_mode=WAL",  # Write-Ahead Logging for better concurrency
            "PRAGMA synchronous=NORMAL",  # Balance between safety and performance
            "PRAGMA cache_size=10000",  # Increase cache size
            "PRAGMA temp_store=MEMORY",  # Store temp tables in memory
            "PRAGMA mmap_size=268435456",  # Enable memory mapping (256MB)
            "PRAGMA optimize"  # Analyze and optimize
        ]
        
        for pragma in optimizations:
            try:
                conn.execute(pragma)
            except sqlite3.Error as e:
                logger.warning(f"Failed to apply optimization '{pragma}': {e}")
        
        logger.debug("Database optimizations applied")
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """Get a database connection with automatic resource management.
        
        Yields:
            sqlite3.Connection: Database connection
            
        Raises:
            DatabaseError: If connection cannot be established
        """
        conn = None
        try:
            conn = sqlite3.connect(
                self.database_path,
                timeout=30.0,  # 30 second timeout
                check_same_thread=False
            )
            
            # Configure connection
            conn.row_factory = sqlite3.Row  # Enable dict-like row access
            conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints
            
            yield conn
            
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseError(
                f"Database connection error: {e}",
                database_path=self.database_path,
                sqlite_error=str(e)
            )
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def transaction(self) -> ContextManager[sqlite3.Connection]:
        """Get a database connection with automatic transaction management.
        
        Yields:
            sqlite3.Connection: Database connection in transaction context
            
        Raises:
            DatabaseError: If transaction fails
        """
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN TRANSACTION")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise DatabaseError(
                    f"Transaction failed: {e}",
                    database_path=self.database_path,
                    original_error=str(e)
                )
    
    def execute_query(
        self,
        conn: sqlite3.Connection,
        query: str,
        params: Optional[Union[Tuple, Dict]] = None
    ) -> sqlite3.Cursor:
        """Execute a database query with error handling.
        
        Args:
            conn: Database connection
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            sqlite3.Cursor: Query result cursor
            
        Raises:
            DatabaseError: If query execution fails
        """
        try:
            if params:
                cursor = conn.execute(query, params)
            else:
                cursor = conn.execute(query)
            
            logger.debug(f"Query executed: {query[:100]}...")
            return cursor
            
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Query execution failed: {e}",
                database_path=self.database_path,
                query=query,
                params=str(params) if params else None,
                sqlite_error=str(e)
            )
    
    def log_miner_data(
        self,
        conn: sqlite3.Connection,
        ip: str,
        data: Dict[str, Any]
    ) -> None:
        """Log miner telemetry data to the database.
        
        Args:
            conn: Database connection
            ip: Miner IP address
            data: Miner telemetry data
            
        Raises:
            DatabaseError: If logging fails
        """
        try:
            # Calculate efficiency if possible
            efficiency = None
            if data.get('hashRate') and data.get('power'):
                efficiency = data['hashRate'] / data['power']
            
            query = """
                INSERT INTO logs (
                    ip, hashRate, temp, fanSpeed, power, frequency, coreVoltage,
                    efficiency, bestDiff, freeHeap, uptime, wifiRSSI,
                    sharesAccepted, sharesRejected, asicCount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                ip,
                data.get('hashRate'),
                data.get('temp'),
                data.get('fanSpeed'),
                data.get('power'),
                data.get('frequency'),
                data.get('coreVoltage'),
                efficiency,
                data.get('bestDiff'),
                data.get('freeHeap'),
                data.get('uptime'),
                data.get('wifiRSSI'),
                data.get('sharesAccepted'),
                data.get('sharesRejected'),
                data.get('asicCount')
            )
            
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log miner data for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def log_event(
        self,
        conn: sqlite3.Connection,
        ip: str,
        event_type: str,
        message: str,
        severity: str = 'INFO',
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a system event to the database.
        
        Args:
            conn: Database connection
            ip: Source IP address
            event_type: Type of event
            message: Event message
            severity: Event severity (INFO, WARN, ERROR, CRITICAL)
            context: Additional context data
            
        Raises:
            DatabaseError: If logging fails
        """
        try:
            query = """
                INSERT INTO protocol (ip, event_type, message, severity, context)
                VALUES (?, ?, ?, ?, ?)
            """
            
            context_json = None
            if context:
                import json
                context_json = json.dumps(context)
            
            params = (ip, event_type, message, severity, context_json)
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log event for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def save_benchmark_result(
        self,
        conn: sqlite3.Connection,
        result: Dict[str, Any]
    ) -> None:
        """Save benchmark result to the database.
        
        Args:
            conn: Database connection
            result: Benchmark result data
            
        Raises:
            DatabaseError: If save fails
        """
        try:
            query = """
                INSERT INTO benchmark_results (
                    ip, frequency, coreVoltage, test_duration,
                    avg_hashrate, min_hashrate, max_hashrate,
                    avg_temp, max_temp, avg_power, efficiency,
                    shares_accepted, shares_rejected, best_diff,
                    stability_score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                result['ip'],
                result['frequency'],
                result['coreVoltage'],
                result['test_duration'],
                result.get('avg_hashrate'),
                result.get('min_hashrate'),
                result.get('max_hashrate'),
                result.get('avg_temp'),
                result.get('max_temp'),
                result.get('avg_power'),
                result.get('efficiency'),
                result.get('shares_accepted'),
                result.get('shares_rejected'),
                result.get('best_diff'),
                result.get('stability_score'),
                result.get('notes')
            )
            
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to save benchmark result: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_latest_status(self) -> List[Dict[str, Any]]:
        """Get the latest status for all miners.
        
        Returns:
            List of latest miner status dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT l1.*
                    FROM logs l1
                    INNER JOIN (
                        SELECT ip, MAX(timestamp) as max_timestamp
                        FROM logs
                        GROUP BY ip
                    ) l2 ON l1.ip = l2.ip AND l1.timestamp = l2.max_timestamp
                    ORDER BY l1.ip
                """
                
                cursor = self.execute_query(conn, query)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get latest status: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_history_data(
        self,
        start: datetime,
        end: datetime,
        ip_filter: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical data for specified time range.
        
        Args:
            start: Start datetime
            end: End datetime
            ip_filter: Optional IP filter
            
        Returns:
            Dictionary mapping IP addresses to historical data lists
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM logs
                    WHERE timestamp BETWEEN ? AND ?
                """
                params = [start.isoformat(), end.isoformat()]
                
                if ip_filter:
                    query += " AND ip = ?"
                    params.append(ip_filter)
                
                query += " ORDER BY timestamp"
                
                cursor = self.execute_query(conn, query, params)
                rows = [dict(row) for row in cursor.fetchall()]
                
                # Group by IP
                result = {}
                for row in rows:
                    ip = row['ip']
                    if ip not in result:
                        result[ip] = []
                    result[ip].append(row)
                
                return result
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get history data: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_benchmark_results(
        self,
        limit: int = 50,
        ip_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent benchmark results.
        
        Args:
            limit: Maximum number of results to return
            ip_filter: Optional IP filter
            
        Returns:
            List of benchmark result dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM benchmark_results
                """
                params = []
                
                if ip_filter:
                    query += " WHERE ip = ?"
                    params.append(ip_filter)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = self.execute_query(conn, query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get benchmark results: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_event_log(
        self,
        limit: int = 100,
        severity_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent system events.
        
        Args:
            limit: Maximum number of events to return
            severity_filter: Optional severity filter
            
        Returns:
            List of event dictionaries
            
        Raises:
            DatabaseError: If query fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM protocol
                """
                params = []
                
                if severity_filter:
                    query += " WHERE severity = ?"
                    params.append(severity_filter)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = self.execute_query(conn, query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get event log: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from the database.
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Dictionary with counts of deleted records by table
            
        Raises:
            DatabaseError: If cleanup fails
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_counts = {}
            
            with self.transaction() as conn:
                # Tables to clean up
                tables = ['logs', 'protocol', 'system_metrics']
                
                for table in tables:
                    query = f"DELETE FROM {table} WHERE timestamp < ?"
                    cursor = self.execute_query(conn, query, (cutoff_date.isoformat(),))
                    deleted_counts[table] = cursor.rowcount
            
            logger.info(f"Cleaned up old data: {deleted_counts}")
            return deleted_counts
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to cleanup old data: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def vacuum_database(self) -> None:
        """Perform database maintenance (VACUUM operation).
        
        Raises:
            DatabaseError: If vacuum fails
        """
        try:
            with self.get_connection() as conn:
                # VACUUM cannot be run in a transaction
                conn.isolation_level = None
                conn.execute("VACUUM")
                conn.isolation_level = ""  # Restore auto-commit mode
                
            logger.info("Database vacuum completed successfully")
            
        except Exception as e:
            raise DatabaseError(
                f"Database vacuum failed: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and information.
        
        Returns:
            Dictionary containing database statistics
            
        Raises:
            DatabaseError: If stats collection fails
        """
        try:
            stats = {}
            
            with self.get_connection() as conn:
                # Database file size
                stats['file_size_mb'] = os.path.getsize(self.database_path) / (1024 * 1024)
                
                # Table row counts
                tables = ['logs', 'benchmark_results', 'protocol', 'tuning_status', 
                         'efficiency_markers', 'system_metrics']
                
                for table in tables:
                    try:
                        cursor = self.execute_query(conn, f"SELECT COUNT(*) FROM {table}")
                        stats[f'{table}_count'] = cursor.fetchone()[0]
                    except sqlite3.Error:
                        stats[f'{table}_count'] = 0
                
                # Database info
                cursor = self.execute_query(conn, "PRAGMA user_version")
                stats['user_version'] = cursor.fetchone()[0]
                
                cursor = self.execute_query(conn, "PRAGMA journal_mode")
                stats['journal_mode'] = cursor.fetchone()[0]
                
                cursor = self.execute_query(conn, "PRAGMA page_size")
                stats['page_size'] = cursor.fetchone()[0]
                
                cursor = self.execute_query(conn, "PRAGMA page_count")
                stats['page_count'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to get database stats: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database.
        
        Returns:
            Dictionary containing health check results
            
        Raises:
            DatabaseError: If health check fails
        """
        try:
            health_status = {
                'status': 'healthy',
                'checks': {},
                'timestamp': datetime.now().isoformat()
            }
            
            with self.get_connection() as conn:
                # Basic connectivity test
                cursor = self.execute_query(conn, "SELECT 1")
                health_status['checks']['connectivity'] = cursor.fetchone()[0] == 1
                
                # Table existence check
                cursor = self.execute_query(conn, 
                    "SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                required_tables = ['logs', 'benchmark_results', 'protocol']
                
                health_status['checks']['tables'] = all(
                    table in tables for table in required_tables
                )
                
                # Recent data check
                cursor = self.execute_query(conn,
                    "SELECT COUNT(*) FROM logs WHERE timestamp > datetime('now', '-1 hour')")
                recent_logs = cursor.fetchone()[0]
                health_status['checks']['recent_data'] = recent_logs > 0
                
                # Database integrity check
                cursor = self.execute_query(conn, "PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                health_status['checks']['integrity'] = integrity_result == 'ok'
            
            # Overall health status
            if not all(health_status['checks'].values()):
                health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            raise DatabaseError(
                f"Database health check failed: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def __str__(self) -> str:
        """String representation of the database manager."""
        return f"DatabaseManager(path='{self.database_path}', initialized={self._initialized})"
    
    def log_analytics_summary(
        self,
        conn: sqlite3.Connection,
        period_type: str,
        period_start: datetime,
        period_end: datetime,
        ip: str,
        summary_data: Dict[str, Any]
    ) -> None:
        """Log analytics summary data for efficient querying.
        
        Args:
            conn: Database connection
            period_type: Type of period ('hourly', 'daily', 'weekly')
            period_start: Start of the period
            period_end: End of the period
            ip: Miner IP address
            summary_data: Aggregated summary data
        """
        try:
            query = """
                INSERT OR REPLACE INTO analytics_summary (
                    period_type, period_start, period_end, ip,
                    avg_hashrate, min_hashrate, max_hashrate,
                    avg_temp, min_temp, max_temp, avg_power, avg_efficiency,
                    total_shares_accepted, total_shares_rejected,
                    uptime_percentage, data_points, performance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                period_type,
                period_start.isoformat(),
                period_end.isoformat(),
                ip,
                summary_data.get('avg_hashrate'),
                summary_data.get('min_hashrate'),
                summary_data.get('max_hashrate'),
                summary_data.get('avg_temp'),
                summary_data.get('min_temp'),
                summary_data.get('max_temp'),
                summary_data.get('avg_power'),
                summary_data.get('avg_efficiency'),
                summary_data.get('total_shares_accepted'),
                summary_data.get('total_shares_rejected'),
                summary_data.get('uptime_percentage'),
                summary_data.get('data_points'),
                summary_data.get('performance_score')
            )
            
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log analytics summary for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def log_performance_alert(
        self,
        conn: sqlite3.Connection,
        ip: str,
        alert_type: str,
        severity: str,
        threshold_value: float,
        actual_value: float,
        message: str
    ) -> None:
        """Log a performance alert.
        
        Args:
            conn: Database connection
            ip: Miner IP address
            alert_type: Type of alert
            severity: Alert severity
            threshold_value: Threshold that was breached
            actual_value: Actual measured value
            message: Alert message
        """
        try:
            query = """
                INSERT INTO performance_alerts (
                    ip, alert_type, severity, threshold_value, actual_value, message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (ip, alert_type, severity, threshold_value, actual_value, message)
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log performance alert for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def log_profitability_data(
        self,
        conn: sqlite3.Connection,
        ip: str,
        profitability_data: Dict[str, Any]
    ) -> None:
        """Log profitability calculation data.
        
        Args:
            conn: Database connection
            ip: Miner IP address
            profitability_data: Profitability metrics
        """
        try:
            query = """
                INSERT INTO profitability_data (
                    ip, hashrate, power_consumption, electricity_cost_kwh,
                    btc_price_usd, network_difficulty, pool_fee_percentage,
                    estimated_daily_btc, estimated_daily_usd, daily_power_cost_usd,
                    estimated_daily_profit_usd, profit_margin_percentage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                ip,
                profitability_data.get('hashrate'),
                profitability_data.get('power_consumption'),
                profitability_data.get('electricity_cost_kwh', 0.12),
                profitability_data.get('btc_price_usd'),
                profitability_data.get('network_difficulty'),
                profitability_data.get('pool_fee_percentage', 1.0),
                profitability_data.get('estimated_daily_btc'),
                profitability_data.get('estimated_daily_usd'),
                profitability_data.get('daily_power_cost_usd'),
                profitability_data.get('estimated_daily_profit_usd'),
                profitability_data.get('profit_margin_percentage')
            )
            
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log profitability data for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def log_miner_health(
        self,
        conn: sqlite3.Connection,
        ip: str,
        health_data: Dict[str, Any]
    ) -> None:
        """Log miner health assessment.
        
        Args:
            conn: Database connection
            ip: Miner IP address
            health_data: Health assessment data
        """
        try:
            query = """
                INSERT OR REPLACE INTO miner_health (
                    ip, health_score, temp_health, efficiency_health,
                    stability_health, connectivity_health, last_restart,
                    consecutive_errors, maintenance_due, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                ip,
                health_data.get('health_score'),
                health_data.get('temp_health'),
                health_data.get('efficiency_health'),
                health_data.get('stability_health'),
                health_data.get('connectivity_health'),
                health_data.get('last_restart'),
                health_data.get('consecutive_errors', 0),
                health_data.get('maintenance_due', False),
                health_data.get('notes')
            )
            
            self.execute_query(conn, query, params)
            
        except Exception as e:
            raise DatabaseError(
                f"Failed to log miner health for {ip}: {e}",
                database_path=self.database_path,
                miner_ip=ip,
                original_error=str(e)
            )
    
    def get_analytics_summary(
        self,
        period_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ip_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get analytics summary data.
        
        Args:
            period_type: Type of period to retrieve
            start_time: Start time filter
            end_time: End time filter
            ip_filter: Optional IP filter
            
        Returns:
            List of analytics summary records
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM analytics_summary
                    WHERE period_type = ?
                """
                params = [period_type]
                
                if start_time:
                    query += " AND period_start >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND period_end <= ?"
                    params.append(end_time.isoformat())
                
                if ip_filter:
                    query += " AND ip = ?"
                    params.append(ip_filter)
                
                query += " ORDER BY period_start DESC"
                
                cursor = self.execute_query(conn, query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get analytics summary: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_active_alerts(
        self,
        severity_filter: Optional[str] = None,
        ip_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get active performance alerts.
        
        Args:
            severity_filter: Optional severity filter
            ip_filter: Optional IP filter
            
        Returns:
            List of active alert records
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT * FROM performance_alerts
                    WHERE resolved = FALSE
                """
                params = []
                
                if severity_filter:
                    query += " AND severity = ?"
                    params.append(severity_filter)
                
                if ip_filter:
                    query += " AND ip = ?"
                    params.append(ip_filter)
                
                query += " ORDER BY timestamp DESC"
                
                cursor = self.execute_query(conn, query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get active alerts: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_profitability_data(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        ip_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get profitability data.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            ip_filter: Optional IP filter
            
        Returns:
            List of profitability records
        """
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM profitability_data WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                if ip_filter:
                    query += " AND ip = ?"
                    params.append(ip_filter)
                
                query += " ORDER BY timestamp DESC"
                
                cursor = self.execute_query(conn, query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get profitability data: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def get_miner_health_scores(self) -> List[Dict[str, Any]]:
        """Get latest health scores for all miners.
        
        Returns:
            List of miner health records
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT h1.*
                    FROM miner_health h1
                    INNER JOIN (
                        SELECT ip, MAX(timestamp) as max_timestamp
                        FROM miner_health
                        GROUP BY ip
                    ) h2 ON h1.ip = h2.ip AND h1.timestamp = h2.max_timestamp
                    ORDER BY h1.health_score DESC
                """
                
                cursor = self.execute_query(conn, query)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            raise DatabaseError(
                f"Failed to get miner health scores: {e}",
                database_path=self.database_path,
                original_error=str(e)
            )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"DatabaseManager(database_path='{self.database_path}', "
                f"initialized={self._initialized}, pool_size={self._pool_size})")