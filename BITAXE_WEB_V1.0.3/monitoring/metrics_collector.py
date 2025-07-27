"""
Comprehensive Metrics Collection System

Real-time metrics collection, aggregation, and monitoring for the BitAxe system.
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics
from concurrent.futures import ThreadPoolExecutor

from logging.structured_logger import get_logger

logger = get_logger("bitaxe.metrics_collector")


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series of metric points"""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)
    retention: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    def add_point(self, value: Union[int, float], timestamp: datetime = None, 
                  tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a new metric point"""
        if timestamp is None:
            timestamp = datetime.now()
        
        point_tags = {**self.tags, **(tags or {})}
        
        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp,
            tags=point_tags,
            metadata=metadata or {}
        )
        
        self.points.append(point)
        self._cleanup_old_points()
    
    def _cleanup_old_points(self):
        """Remove points older than retention period"""
        cutoff_time = datetime.now() - self.retention
        while self.points and self.points[0].timestamp < cutoff_time:
            self.points.popleft()
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent value"""
        return self.points[-1].value if self.points else None
    
    def get_values_in_range(self, start_time: datetime, end_time: datetime) -> List[float]:
        """Get values within a time range"""
        return [
            point.value for point in self.points
            if start_time <= point.timestamp <= end_time
        ]
    
    def calculate_stats(self, duration: timedelta = None) -> Dict[str, float]:
        """Calculate statistics for the series"""
        if duration:
            cutoff_time = datetime.now() - duration
            values = [p.value for p in self.points if p.timestamp >= cutoff_time]
        else:
            values = [p.value for p in self.points]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'latest': values[-1] if values else 0
        }


class SystemMetricsCollector:
    """Collect system-level metrics"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.last_cpu_time = 0
        self.last_measurement = time.time()
    
    def collect_cpu_metrics(self) -> Dict[str, float]:
        """Collect CPU metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'load_average_1m': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'load_average_5m': psutil.getloadavg()[1] if hasattr(psutil, 'getloadavg') else 0,
            'load_average_15m': psutil.getloadavg()[2] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def collect_memory_metrics(self) -> Dict[str, float]:
        """Collect memory metrics"""
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        process_memory = self.process.memory_info()
        
        return {
            'memory_total': virtual_memory.total,
            'memory_available': virtual_memory.available,
            'memory_used': virtual_memory.used,
            'memory_percent': virtual_memory.percent,
            'swap_total': swap_memory.total,
            'swap_used': swap_memory.used,
            'swap_percent': swap_memory.percent,
            'process_memory_rss': process_memory.rss,
            'process_memory_vms': process_memory.vms,
            'process_memory_percent': self.process.memory_percent()
        }
    
    def collect_disk_metrics(self) -> Dict[str, float]:
        """Collect disk metrics"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'disk_total': disk_usage.total,
            'disk_used': disk_usage.used,
            'disk_free': disk_usage.free,
            'disk_percent': disk_usage.percent
        }
        
        if disk_io:
            metrics.update({
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count,
                'disk_read_bytes': disk_io.read_bytes,
                'disk_write_bytes': disk_io.write_bytes,
                'disk_read_time': disk_io.read_time,
                'disk_write_time': disk_io.write_time
            })
        
        return metrics
    
    def collect_network_metrics(self) -> Dict[str, float]:
        """Collect network metrics"""
        network_io = psutil.net_io_counters()
        
        if not network_io:
            return {}
        
        return {
            'network_bytes_sent': network_io.bytes_sent,
            'network_bytes_recv': network_io.bytes_recv,
            'network_packets_sent': network_io.packets_sent,
            'network_packets_recv': network_io.packets_recv,
            'network_errin': network_io.errin,
            'network_errout': network_io.errout,
            'network_dropin': network_io.dropin,
            'network_dropout': network_io.dropout
        }
    
    def collect_process_metrics(self) -> Dict[str, float]:
        """Collect process-specific metrics"""
        try:
            cpu_times = self.process.cpu_times()
            current_time = time.time()
            
            # Calculate CPU usage rate
            cpu_usage_rate = 0
            if self.last_cpu_time > 0:
                time_delta = current_time - self.last_measurement
                cpu_delta = (cpu_times.user + cpu_times.system) - self.last_cpu_time
                if time_delta > 0:
                    cpu_usage_rate = (cpu_delta / time_delta) * 100
            
            self.last_cpu_time = cpu_times.user + cpu_times.system
            self.last_measurement = current_time
            
            return {
                'process_cpu_percent': self.process.cpu_percent(),
                'process_cpu_usage_rate': cpu_usage_rate,
                'process_num_threads': self.process.num_threads(),
                'process_num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                'process_create_time': self.process.create_time(),
                'process_status': hash(self.process.status())  # Convert to numeric
            }
        except psutil.NoSuchProcess:
            return {}


class ApplicationMetricsCollector:
    """Collect application-specific metrics"""
    
    def __init__(self):
        self.custom_metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Request metrics
        self.request_count = 0
        self.request_duration_sum = 0.0
        self.request_durations: deque = deque(maxlen=1000)
        
        # Error metrics
        self.error_count = 0
        self.error_rate_window: deque = deque(maxlen=100)
        
        # Response time percentiles
        self.response_times: deque = deque(maxlen=1000)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer duration"""
        key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        self.timers[key].append(duration)
        
        # Keep only recent values
        if len(self.timers[key]) > 1000:
            self.timers[key] = self.timers[key][-1000:]
    
    def record_request(self, duration: float, status_code: int, endpoint: str = ""):
        """Record HTTP request metrics"""
        self.request_count += 1
        self.request_duration_sum += duration
        self.request_durations.append(duration)
        self.response_times.append(duration)
        
        # Track errors (4xx and 5xx)
        is_error = status_code >= 400
        self.error_rate_window.append(1 if is_error else 0)
        
        if is_error:
            self.error_count += 1
        
        # Record by endpoint
        self.increment_counter('http_requests_total', tags={'endpoint': endpoint, 'status': str(status_code)})
        self.record_timer('http_request_duration', duration, tags={'endpoint': endpoint})
    
    def get_error_rate(self, window_size: int = 100) -> float:
        """Calculate error rate over recent requests"""
        if not self.error_rate_window:
            return 0.0
        
        recent_errors = list(self.error_rate_window)[-window_size:]
        return sum(recent_errors) / len(recent_errors) * 100
    
    def get_response_time_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not self.response_times:
            return {}
        
        sorted_times = sorted(self.response_times)
        
        def percentile(p):
            k = (len(sorted_times) - 1) * p / 100
            f = int(k)
            c = k - f
            if f == len(sorted_times) - 1:
                return sorted_times[f]
            return sorted_times[f] * (1 - c) + sorted_times[f + 1] * c
        
        return {
            'p50': percentile(50),
            'p95': percentile(95),
            'p99': percentile(99),
            'p99_9': percentile(99.9)
        }
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get all application metrics"""
        metrics = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.get_error_rate(),
            'avg_response_time': self.request_duration_sum / max(1, self.request_count),
            'response_time_percentiles': self.get_response_time_percentiles()
        }
        
        # Add histogram stats
        histogram_stats = {}
        for name, values in self.histograms.items():
            if values:
                histogram_stats[f"{name}_count"] = len(values)
                histogram_stats[f"{name}_mean"] = statistics.mean(values)
                histogram_stats[f"{name}_median"] = statistics.median(values)
                histogram_stats[f"{name}_min"] = min(values)
                histogram_stats[f"{name}_max"] = max(values)
        
        metrics['histogram_stats'] = histogram_stats
        
        # Add timer stats
        timer_stats = {}
        for name, durations in self.timers.items():
            if durations:
                timer_stats[f"{name}_count"] = len(durations)
                timer_stats[f"{name}_mean"] = statistics.mean(durations)
                timer_stats[f"{name}_median"] = statistics.median(durations)
                timer_stats[f"{name}_min"] = min(durations)
                timer_stats[f"{name}_max"] = max(durations)
        
        metrics['timer_stats'] = timer_stats
        
        return metrics


class MetricsCollector:
    """
    Comprehensive metrics collection system
    
    Features:
    - System metrics (CPU, memory, disk, network)
    - Application metrics (counters, gauges, histograms, timers)
    - Custom metric series with retention
    - Real-time aggregation and statistics
    - Background collection and cleanup
    - Export capabilities
    """
    
    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        
        # Metric collectors
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        
        # Metric series storage
        self.metric_series: Dict[str, MetricSeries] = {}
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks for metric updates
        self.metric_callbacks: List[Callable[[str, MetricPoint], None]] = []
        
        # Aggregation cache
        self.aggregation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 60  # seconds
        self.last_cache_update = 0
    
    async def start(self):
        """Start the metrics collector"""
        if self.is_running:
            return
        
        logger.info("Starting metrics collector")
        self.is_running = True
        
        # Start background collection
        self.collection_task = asyncio.create_task(self._collection_worker())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info(f"Metrics collector started with {self.collection_interval}s interval")
    
    async def stop(self):
        """Stop the metrics collector"""
        if not self.is_running:
            return
        
        logger.info("Stopping metrics collector")
        self.is_running = False
        
        # Cancel background tasks
        if self.collection_task:
            self.collection_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to finish
        tasks = [t for t in [self.collection_task, self.cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Metrics collector stopped")
    
    def add_metric_callback(self, callback: Callable[[str, MetricPoint], None]):
        """Add a callback for metric updates"""
        self.metric_callbacks.append(callback)
    
    def record_metric(self, name: str, value: Union[int, float], 
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a custom metric"""
        series_key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        
        if series_key not in self.metric_series:
            self.metric_series[series_key] = MetricSeries(name=name, tags=tags or {})
        
        series = self.metric_series[series_key]
        timestamp = datetime.now()
        
        series.add_point(value, timestamp, tags, metadata)
        
        # Notify callbacks
        point = MetricPoint(name, value, timestamp, tags or {}, metadata or {})
        for callback in self.metric_callbacks:
            try:
                callback(name, point)
            except Exception as e:
                logger.error("Metric callback error", callback=str(callback), error=str(e))
        
        logger.debug(f"Recorded metric: {name} = {value}", tags=tags)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.app_collector.increment_counter(name, value, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        self.app_collector.set_gauge(name, value, tags)
        self.record_metric(name, value, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        self.app_collector.record_histogram(name, value, tags)
        self.record_metric(name, value, tags)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer duration"""
        self.app_collector.record_timer(name, duration, tags)
        self.record_metric(f"{name}_duration", duration, tags)
    
    def record_request(self, duration: float, status_code: int, endpoint: str = ""):
        """Record HTTP request metrics"""
        self.app_collector.record_request(duration, status_code, endpoint)
        
        # Record as metric series
        self.record_metric('http_request_duration', duration, 
                          tags={'endpoint': endpoint, 'status': str(status_code)})
    
    def get_metric_series(self, name: str, tags: Dict[str, str] = None) -> Optional[MetricSeries]:
        """Get a metric series"""
        series_key = f"{name}_{hash(str(sorted((tags or {}).items())))}"
        return self.metric_series.get(series_key)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics"""
        current_time = time.time()
        
        # Use cache if recent
        if (current_time - self.last_cache_update) < self.cache_ttl and self.aggregation_cache:
            return self.aggregation_cache
        
        metrics = {}
        
        # System metrics
        try:
            system_metrics = {}
            system_metrics.update(self.system_collector.collect_cpu_metrics())
            system_metrics.update(self.system_collector.collect_memory_metrics())
            system_metrics.update(self.system_collector.collect_disk_metrics())
            system_metrics.update(self.system_collector.collect_network_metrics())
            system_metrics.update(self.system_collector.collect_process_metrics())
            
            metrics['system'] = system_metrics
        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))
            metrics['system'] = {}
        
        # Application metrics
        try:
            metrics['application'] = self.app_collector.get_application_metrics()
        except Exception as e:
            logger.error("Error collecting application metrics", error=str(e))
            metrics['application'] = {}
        
        # Custom metric series
        try:
            series_metrics = {}
            for series_key, series in self.metric_series.items():
                latest_value = series.get_latest_value()
                if latest_value is not None:
                    series_metrics[series_key] = {
                        'latest_value': latest_value,
                        'stats': series.calculate_stats(timedelta(minutes=5))
                    }
            
            metrics['custom_series'] = series_metrics
        except Exception as e:
            logger.error("Error collecting custom series metrics", error=str(e))
            metrics['custom_series'] = {}
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['collection_interval'] = self.collection_interval
        
        # Update cache
        self.aggregation_cache = metrics
        self.last_cache_update = current_time
        
        return metrics
    
    def get_metric_stats(self, name: str, duration: timedelta = None, 
                        tags: Dict[str, str] = None) -> Dict[str, float]:
        """Get statistics for a specific metric"""
        series = self.get_metric_series(name, tags)
        if not series:
            return {}
        
        return series.calculate_stats(duration)
    
    def export_metrics(self, format_type: str = 'json') -> str:
        """Export metrics in specified format"""
        metrics = self.get_latest_metrics()
        
        if format_type == 'json':
            return json.dumps(metrics, indent=2, default=str)
        elif format_type == 'prometheus':
            return self._export_prometheus_format(metrics)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_prometheus_format(self, metrics: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # System metrics
        for name, value in metrics.get('system', {}).items():
            if isinstance(value, (int, float)):
                lines.append(f"system_{name} {value}")
        
        # Application metrics
        app_metrics = metrics.get('application', {})
        
        # Counters
        for name, value in app_metrics.get('counters', {}).items():
            lines.append(f"app_{name}_total {value}")
        
        # Gauges
        for name, value in app_metrics.get('gauges', {}).items():
            lines.append(f"app_{name} {value}")
        
        # Request metrics
        if 'request_count' in app_metrics:
            lines.append(f"http_requests_total {app_metrics['request_count']}")
        if 'error_count' in app_metrics:
            lines.append(f"http_errors_total {app_metrics['error_count']}")
        if 'avg_response_time' in app_metrics:
            lines.append(f"http_response_time_avg {app_metrics['avg_response_time']}")
        
        # Response time percentiles
        percentiles = app_metrics.get('response_time_percentiles', {})
        for percentile, value in percentiles.items():
            lines.append(f"http_response_time_percentile{{percentile=\"{percentile}\"}} {value}")
        
        return '\n'.join(lines) + '\n'
    
    async def _collection_worker(self):
        """Background worker for periodic metric collection"""
        logger.debug("Metrics collection worker started")
        
        while self.is_running:
            try:
                # Collect system metrics in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Record system metrics as series
                system_metrics = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: {
                        **self.system_collector.collect_cpu_metrics(),
                        **self.system_collector.collect_memory_metrics(),
                        **self.system_collector.collect_process_metrics()
                    }
                )
                
                timestamp = datetime.now()
                for name, value in system_metrics.items():
                    if isinstance(value, (int, float)):
                        self.record_metric(f"system_{name}", value, tags={'source': 'system'})
                
                # Clear cache to force refresh
                self.last_cache_update = 0
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection worker error", error=str(e))
                await asyncio.sleep(5)
        
        logger.debug("Metrics collection worker stopped")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        logger.debug("Metrics cleanup worker started")
        
        while self.is_running:
            try:
                # Clean up old metric series points
                cleanup_count = 0
                for series in self.metric_series.values():
                    old_count = len(series.points)
                    series._cleanup_old_points()
                    cleanup_count += old_count - len(series.points)
                
                if cleanup_count > 0:
                    logger.debug(f"Cleaned up {cleanup_count} old metric points")
                
                # Reset application counters periodically (optional)
                # This could be configurable based on requirements
                
                await asyncio.sleep(300)  # Run cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics cleanup worker error", error=str(e))
                await asyncio.sleep(60)
        
        logger.debug("Metrics cleanup worker stopped")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_metric(name: str, value: Union[int, float], tags: Dict[str, str] = None):
    """Convenience function to record a metric"""
    collector = get_metrics_collector()
    collector.record_metric(name, value, tags)


def increment_counter(name: str, value: int = 1, tags: Dict[str, str] = None):
    """Convenience function to increment a counter"""
    collector = get_metrics_collector()
    collector.increment_counter(name, value, tags)


def set_gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Convenience function to set a gauge"""
    collector = get_metrics_collector()
    collector.set_gauge(name, value, tags)


def record_timer(name: str, duration: float, tags: Dict[str, str] = None):
    """Convenience function to record a timer"""
    collector = get_metrics_collector()
    collector.record_timer(name, duration, tags)


class MetricsContext:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str, tags: Dict[str, str] = None):
        self.metric_name = metric_name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            record_timer(self.metric_name, duration, self.tags)


def time_metric(metric_name: str, tags: Dict[str, str] = None):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MetricsContext(metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


async def time_metric_async(metric_name: str, tags: Dict[str, str] = None):
    """Async decorator to time function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                record_timer(metric_name, duration, tags)
        return wrapper
    return decorator