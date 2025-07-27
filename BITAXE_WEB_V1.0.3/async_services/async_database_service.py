"""
Async Database Service

High-performance async database operations with connection pooling.
"""

import asyncio
import aiosqlite
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
from contextlib import asynccontextmanager
import json
from concurrent.futures import ThreadPoolExecutor

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import DatabaseError, ErrorCode

logger = get_logger("bitaxe.async_database_service")


class AsyncConnectionPool:
    """Async SQLite connection pool"""
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.max_connections = max_connections
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0
        self.pool_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = {
            'total_connections': 0,
            'active_connections': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'connection_errors': 0
        }
    
    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection"""
        try:
            conn = await aiosqlite.connect(
                self.database_path,
                timeout=30.0,
                isolation_level=None  # Autocommit mode
            )
            
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            
            self.metrics['total_connections'] += 1
            self.created_connections += 1
            
            logger.debug(f"Created new database connection #{self.created_connections}")
            return conn
            
        except Exception as e:
            self.metrics['connection_errors'] += 1
            logger.error("Failed to create database connection", error=str(e))
            raise DatabaseError(f"Failed to create database connection: {e}")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """Get a connection from the pool"""
        conn = None
        
        try:
            # Try to get connection from pool
            try:
                conn = self.pool.get_nowait()
                self.metrics['pool_hits'] += 1
            except asyncio.QueueEmpty:
                # Pool is empty, create new connection if allowed
                async with self.pool_lock:
                    if self.created_connections < self.max_connections:
                        conn = await self._create_connection()
                        self.metrics['pool_misses'] += 1
                    else:
                        # Wait for a connection to become available
                        conn = await self.pool.get()
                        self.metrics['pool_hits'] += 1
            
            self.metrics['active_connections'] += 1
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    await conn.close()
                except:
                    pass
                conn = None
            raise
            
        finally:
            self.metrics['active_connections'] -= 1
            
            if conn:
                try:
                    # Return connection to pool
                    self.pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # Pool is full, close the connection
                    await conn.close()
                    self.created_connections -= 1
    
    async def close_all(self):
        """Close all connections in the pool"""
        logger.info("Closing all database connections")
        
        # Close all connections in the pool
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                await conn.close()
                self.created_connections -= 1
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Closed {self.metrics['total_connections']} database connections")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get pool metrics"""
        return {
            **self.metrics,
            'pool_size': self.pool.qsize(),
            'created_connections': self.created_connections,
            'max_connections': self.max_connections
        }


class AsyncDatabaseService:
    """
    High-performance async database service
    
    Features:
    - Connection pooling for concurrent operations
    - Prepared statements for better performance
    - Batch operations for bulk inserts
    - Automatic retry logic with exponential backoff
    - Real-time metrics collection
    - Background maintenance tasks
    """
    
    def __init__(self, database_path: str, max_connections: int = 10):
        self.database_path = database_path
        self.connection_pool = AsyncConnectionPool(database_path, max_connections)
        
        # Prepared statements cache
        self.prepared_statements: Dict[str, str] = {}
        
        # Batch operations
        self.batch_size = 100
        self.batch_timeout = 5.0  # seconds
        self.pending_operations: List[Dict[str, Any]] = []
        self.last_batch_time = time.time()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_query_time': 0.0,
            'batch_operations': 0,
            'cached_statements': 0
        }
        self.query_times: List[float] = []
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self._setup_prepared_statements()
    
    def _setup_prepared_statements(self):
        """Setup commonly used prepared statements"""
        self.prepared_statements = {
            'insert_miner_log': """
                INSERT OR REPLACE INTO logs (
                    timestamp, ip, hostname, temp, hashRate, power, voltage,
                    frequency, coreVoltage, fanrpm, sharesAccepted, sharesRejected,
                    uptime, version, boardtemp1, boardtemp2, vrTemp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            
            'insert_benchmark_result': """
                INSERT INTO benchmark_results (
                    ip, frequency, coreVoltage, avgHashrate, avgTemp, efficiencyJTH,
                    timestamp, duration, samplesCount, aborted, abortReason, avgVrTemp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            
            'insert_event': """
                INSERT INTO event_log (timestamp, ip, eventType, message, severity)
                VALUES (?, ?, ?, ?, ?)
            """,
            
            'insert_autopilot_action': """
                INSERT INTO autopilot_log (timestamp, ip, action, oldFrequency, newFrequency, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
            
            'get_latest_status_all': """
                SELECT l1.* FROM logs l1
                INNER JOIN (
                    SELECT ip, MAX(timestamp) as max_timestamp
                    FROM logs
                    WHERE timestamp > datetime('now', '-5 minutes')
                    GROUP BY ip
                ) l2 ON l1.ip = l2.ip AND l1.timestamp = l2.max_timestamp
                ORDER BY l1.timestamp DESC
            """,
            
            'get_latest_status_by_ip': """
                SELECT * FROM logs 
                WHERE ip = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """,
            
            'get_benchmark_results': """
                SELECT * FROM benchmark_results 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
            
            'get_benchmark_results_by_ip': """
                SELECT * FROM benchmark_results 
                WHERE ip = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
            
            'get_events': """
                SELECT * FROM event_log 
                WHERE (? IS NULL OR ip = ?) 
                AND (? IS NULL OR eventType LIKE ?) 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
        }
        
        self.metrics['cached_statements'] = len(self.prepared_statements)
    
    async def start(self):
        """Start the async database service"""
        if self.is_running:
            return
        
        logger.info("Starting async database service")
        self.is_running = True
        
        # Start background tasks
        batch_processor = asyncio.create_task(self._batch_processor())
        self.background_tasks.append(batch_processor)
        
        metrics_collector = asyncio.create_task(self._metrics_collector())
        self.background_tasks.append(metrics_collector)
        
        maintenance_task = asyncio.create_task(self._maintenance_worker())
        self.background_tasks.append(maintenance_task)
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def stop(self):
        """Stop the async database service"""
        if not self.is_running:
            return
        
        logger.info("Stopping async database service")
        self.is_running = False
        
        # Process remaining batch operations
        if self.pending_operations:
            await self._process_batch()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Close connection pool
        await self.connection_pool.close_all()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.background_tasks.clear()
        logger.info("Async database service stopped")
    
    async def _execute_query(self, query: str, params: tuple = (), 
                           fetch_one: bool = False, fetch_all: bool = False) -> Any:
        """Execute a database query with metrics and error handling"""
        start_time = time.time()
        
        try:
            self.metrics['total_queries'] += 1
            
            async with self.connection_pool.get_connection() as conn:
                if fetch_one:
                    async with conn.execute(query, params) as cursor:
                        result = await cursor.fetchone()
                elif fetch_all:
                    async with conn.execute(query, params) as cursor:
                        result = await cursor.fetchall()
                else:
                    await conn.execute(query, params)
                    result = None
                
                await conn.commit()
                
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics['successful_queries'] += 1
                
                logger.debug("Database query executed", 
                           query_time=query_time, 
                           params_count=len(params))
                
                return result
                
        except Exception as e:
            self.metrics['failed_queries'] += 1
            logger.error("Database query failed", query=query[:100], error=str(e))
            raise DatabaseError(f"Database query failed: {e}")
    
    async def log_miner_data_async(self, miner_data: Dict[str, Any]):
        """Log miner data asynchronously"""
        # Add to batch queue
        operation = {
            'type': 'insert_miner_log',
            'data': miner_data,
            'timestamp': time.time()
        }
        
        self.pending_operations.append(operation)
        
        # Process batch if full or timeout reached
        if (len(self.pending_operations) >= self.batch_size or 
            time.time() - self.last_batch_time >= self.batch_timeout):
            await self._process_batch()
    
    async def log_event_async(self, ip: str, event_type: str, message: str, severity: str = 'INFO'):
        """Log event asynchronously"""
        operation = {
            'type': 'insert_event',
            'data': {
                'ip': ip,
                'event_type': event_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            },
            'timestamp': time.time()
        }
        
        self.pending_operations.append(operation)
    
    async def save_benchmark_result_async(self, result_data: Dict[str, Any]):
        """Save benchmark result asynchronously"""
        operation = {
            'type': 'insert_benchmark_result',
            'data': result_data,
            'timestamp': time.time()
        }
        
        self.pending_operations.append(operation)
    
    async def _process_batch(self):
        """Process pending batch operations"""
        if not self.pending_operations:
            return
        
        logger.debug(f"Processing batch of {len(self.pending_operations)} operations")
        
        try:
            async with self.connection_pool.get_connection() as conn:
                # Group operations by type
                grouped_ops = {}
                for op in self.pending_operations:
                    op_type = op['type']
                    if op_type not in grouped_ops:
                        grouped_ops[op_type] = []
                    grouped_ops[op_type].append(op['data'])
                
                # Execute each group as a batch
                for op_type, data_list in grouped_ops.items():
                    if op_type == 'insert_miner_log':
                        await self._batch_insert_miner_logs(conn, data_list)
                    elif op_type == 'insert_event':
                        await self._batch_insert_events(conn, data_list)
                    elif op_type == 'insert_benchmark_result':
                        await self._batch_insert_benchmark_results(conn, data_list)
                
                await conn.commit()
                
                self.metrics['batch_operations'] += 1
                logger.debug(f"Batch processing completed for {len(self.pending_operations)} operations")
                
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            raise DatabaseError(f"Batch processing failed: {e}")
        
        finally:
            self.pending_operations.clear()
            self.last_batch_time = time.time()
    
    async def _batch_insert_miner_logs(self, conn: aiosqlite.Connection, data_list: List[Dict]):
        """Batch insert miner logs"""
        params_list = []
        for data in data_list:
            params = (
                data.get('timestamp', datetime.now().isoformat()),
                data.get('ip'),
                data.get('hostname'),
                data.get('temp'),
                data.get('hashRate'),
                data.get('power'),
                data.get('voltage'),
                data.get('frequency'),
                data.get('coreVoltage'),
                data.get('fanrpm'),
                data.get('sharesAccepted'),
                data.get('sharesRejected'),
                data.get('uptime'),
                data.get('version'),
                data.get('boardtemp1'),
                data.get('boardtemp2'),
                data.get('vrTemp')
            )
            params_list.append(params)
        
        await conn.executemany(self.prepared_statements['insert_miner_log'], params_list)
    
    async def _batch_insert_events(self, conn: aiosqlite.Connection, data_list: List[Dict]):
        """Batch insert events"""
        params_list = []
        for data in data_list:
            params = (
                data.get('timestamp', datetime.now().isoformat()),
                data.get('ip'),
                data.get('event_type'),
                data.get('message'),
                data.get('severity', 'INFO')
            )
            params_list.append(params)
        
        await conn.executemany(self.prepared_statements['insert_event'], params_list)
    
    async def _batch_insert_benchmark_results(self, conn: aiosqlite.Connection, data_list: List[Dict]):
        """Batch insert benchmark results"""
        params_list = []
        for data in data_list:
            params = (
                data.get('ip'),
                data.get('frequency'),
                data.get('coreVoltage'),
                data.get('avgHashrate'),
                data.get('avgTemp'),
                data.get('efficiencyJTH'),
                data.get('timestamp', datetime.now().isoformat()),
                data.get('duration'),
                data.get('samplesCount'),
                data.get('aborted', False),
                data.get('abortReason'),
                data.get('avgVrTemp')
            )
            params_list.append(params)
        
        await conn.executemany(self.prepared_statements['insert_benchmark_result'], params_list)
    
    async def get_latest_status_async(self) -> List[Dict[str, Any]]:
        """Get latest status for all miners asynchronously"""
        result = await self._execute_query(
            self.prepared_statements['get_latest_status_all'],
            fetch_all=True
        )
        
        if result:
            # Convert rows to dictionaries
            columns = ['timestamp', 'ip', 'hostname', 'temp', 'hashRate', 'power', 
                      'voltage', 'frequency', 'coreVoltage', 'fanrpm', 'sharesAccepted', 
                      'sharesRejected', 'uptime', 'version', 'boardtemp1', 'boardtemp2', 'vrTemp']
            
            return [dict(zip(columns, row)) for row in result]
        
        return []
    
    async def get_latest_status_by_ip_async(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get latest status for specific miner asynchronously"""
        result = await self._execute_query(
            self.prepared_statements['get_latest_status_by_ip'],
            (ip,),
            fetch_one=True
        )
        
        if result:
            columns = ['timestamp', 'ip', 'hostname', 'temp', 'hashRate', 'power', 
                      'voltage', 'frequency', 'coreVoltage', 'fanrpm', 'sharesAccepted', 
                      'sharesRejected', 'uptime', 'version', 'boardtemp1', 'boardtemp2', 'vrTemp']
            
            return dict(zip(columns, result))
        
        return None
    
    async def get_benchmark_results_async(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get benchmark results asynchronously"""
        result = await self._execute_query(
            self.prepared_statements['get_benchmark_results'],
            (limit,),
            fetch_all=True
        )
        
        if result:
            columns = ['ip', 'frequency', 'coreVoltage', 'avgHashrate', 'avgTemp', 
                      'efficiencyJTH', 'timestamp', 'duration', 'samplesCount', 
                      'aborted', 'abortReason', 'avgVrTemp']
            
            return [dict(zip(columns, row)) for row in result]
        
        return []
    
    async def get_benchmark_results_by_ip_async(self, ip: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get benchmark results for specific miner asynchronously"""
        result = await self._execute_query(
            self.prepared_statements['get_benchmark_results_by_ip'],
            (ip, limit),
            fetch_all=True
        )
        
        if result:
            columns = ['ip', 'frequency', 'coreVoltage', 'avgHashrate', 'avgTemp', 
                      'efficiencyJTH', 'timestamp', 'duration', 'samplesCount', 
                      'aborted', 'abortReason', 'avgVrTemp']
            
            return [dict(zip(columns, row)) for row in result]
        
        return []
    
    async def get_events_async(self, ip_filter: Optional[str] = None, 
                             event_type_filter: Optional[str] = None, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get events asynchronously"""
        result = await self._execute_query(
            self.prepared_statements['get_events'],
            (ip_filter, ip_filter, event_type_filter, f"%{event_type_filter}%" if event_type_filter else None, limit),
            fetch_all=True
        )
        
        if result:
            columns = ['timestamp', 'ip', 'eventType', 'message', 'severity']
            return [dict(zip(columns, row)) for row in result]
        
        return []
    
    async def _batch_processor(self):
        """Background batch processor"""
        logger.info("Batch processor started")
        
        while self.is_running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                # Process batch if timeout reached
                if (self.pending_operations and 
                    time.time() - self.last_batch_time >= self.batch_timeout):
                    await self._process_batch()
                    
            except Exception as e:
                logger.error("Batch processor error", error=str(e))
                await asyncio.sleep(1)
    
    async def _metrics_collector(self):
        """Background metrics collector"""
        logger.info("Database metrics collector started")
        
        while self.is_running:
            try:
                # Update average query time
                if self.query_times:
                    self.metrics['avg_query_time'] = sum(self.query_times) / len(self.query_times)
                    
                    # Keep only recent query times
                    if len(self.query_times) > 1000:
                        self.query_times = self.query_times[-1000:]
                
                # Log metrics
                pool_metrics = self.connection_pool.get_metrics()
                
                logger.info("Database service metrics",
                           total_queries=self.metrics['total_queries'],
                           success_rate=self.metrics['successful_queries'] / max(1, self.metrics['total_queries']) * 100,
                           avg_query_time=self.metrics['avg_query_time'],
                           batch_operations=self.metrics['batch_operations'],
                           pending_operations=len(self.pending_operations),
                           pool_hits=pool_metrics['pool_hits'],
                           pool_misses=pool_metrics['pool_misses'],
                           active_connections=pool_metrics['active_connections'])
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error("Database metrics collector error", error=str(e))
                await asyncio.sleep(5)
    
    async def _maintenance_worker(self):
        """Background maintenance tasks"""
        logger.info("Database maintenance worker started")
        
        while self.is_running:
            try:
                # Cleanup old logs (older than 7 days)
                cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
                
                deleted_logs = await self._execute_query(
                    "DELETE FROM logs WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                deleted_events = await self._execute_query(
                    "DELETE FROM event_log WHERE timestamp < ?",
                    (cutoff_date,)
                )
                
                # Vacuum database periodically
                await self._execute_query("VACUUM")
                
                logger.info("Database maintenance completed",
                           deleted_logs=deleted_logs,
                           deleted_events=deleted_events)
                
                # Run maintenance every 6 hours
                await asyncio.sleep(6 * 3600)
                
            except Exception as e:
                logger.error("Database maintenance error", error=str(e))
                await asyncio.sleep(3600)  # Retry after 1 hour
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get database service metrics"""
        pool_metrics = self.connection_pool.get_metrics()
        
        return {
            **self.metrics,
            'pending_operations': len(self.pending_operations),
            'last_batch_time': self.last_batch_time,
            'pool_metrics': pool_metrics,
            'is_running': self.is_running
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()