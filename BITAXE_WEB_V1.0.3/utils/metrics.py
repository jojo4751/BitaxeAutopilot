import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from logging.structured_logger import get_logger

logger = get_logger("bitaxe.metrics")


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


@dataclass
class MetricValue:
    """Individual metric value"""
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


class Metric:
    """Base metric class"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.values = deque(maxlen=1000)  # Keep last 1000 values
        self.lock = threading.Lock()
    
    def record(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self.lock:
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self.values.append(metric_value)
            
            logger.debug("Metric recorded",
                        metric_name=self.name,
                        metric_type=self.type.value,
                        value=value,
                        tags=tags)
    
    def get_current_value(self) -> Optional[float]:
        """Get the most recent value"""
        with self.lock:
            return self.values[-1].value if self.values else None
    
    def get_values(self, since: Optional[datetime] = None) -> List[MetricValue]:
        """Get values since specified time"""
        with self.lock:
            if since is None:
                return list(self.values)
            
            return [v for v in self.values if v.timestamp >= since]
    
    def get_stats(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get statistical summary of values"""
        values = self.get_values(since)
        
        if not values:
            return {}
        
        numeric_values = [v.value for v in values]
        
        return {
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "avg": sum(numeric_values) / len(numeric_values),
            "latest": numeric_values[-1],
            "first_timestamp": values[0].timestamp.isoformat(),
            "last_timestamp": values[-1].timestamp.isoformat()
        }


class Counter(Metric):
    """Counter metric - monotonically increasing"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.COUNTER, description)
        self._current_value = 0.0
    
    def increment(self, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment counter"""
        self._current_value += value
        self.record(self._current_value, tags)
    
    def reset(self):
        """Reset counter to zero"""
        self._current_value = 0.0
        self.record(0.0)


class Gauge(Metric):
    """Gauge metric - can go up and down"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, MetricType.GAUGE, description)
    
    def set(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        self.record(value, tags)
    
    def increment(self, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment gauge value"""
        current = self.get_current_value() or 0.0
        self.record(current + value, tags)
    
    def decrement(self, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Decrement gauge value"""
        current = self.get_current_value() or 0.0
        self.record(current - value, tags)


class Histogram(Metric):
    """Histogram metric - distribution of values"""
    
    def __init__(self, name: str, description: str = "", 
                 buckets: Optional[List[float]] = None):
        super().__init__(name, MetricType.HISTOGRAM, description)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
        self.bucket_counts = defaultdict(int)
    
    def observe(self, value: float, tags: Optional[Dict[str, str]] = None):
        """Observe a value"""
        self.record(value, tags)
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
    
    def get_percentile(self, percentile: float, since: Optional[datetime] = None) -> Optional[float]:
        """Calculate percentile value"""
        values = [v.value for v in self.get_values(since)]
        if not values:
            return None
        
        values.sort()
        index = int((percentile / 100.0) * len(values))
        return values[min(index, len(values) - 1)]


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metric: Histogram, tags: Optional[Dict[str, str]] = None):
        self.metric = metric
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metric.observe(duration * 1000, self.tags)  # Convert to milliseconds


class MetricsCollector:
    """Central metrics collection and management"""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()
        
        # Initialize system metrics
        self._initialize_system_metrics()
    
    def _initialize_system_metrics(self):
        """Initialize common system metrics"""
        
        # Request metrics
        self.register_counter("http_requests_total", "Total HTTP requests")
        self.register_histogram("http_request_duration_ms", "HTTP request duration in milliseconds")
        self.register_counter("http_errors_total", "Total HTTP errors")
        
        # Miner metrics
        self.register_gauge("miners_online", "Number of online miners")
        self.register_gauge("total_hashrate", "Total hashrate across all miners")
        self.register_gauge("total_power", "Total power consumption")
        self.register_gauge("average_temperature", "Average temperature across all miners")
        self.register_gauge("total_efficiency", "Total efficiency (GH/W)")
        
        # Benchmark metrics
        self.register_counter("benchmarks_started", "Total benchmarks started")
        self.register_counter("benchmarks_completed", "Total benchmarks completed")
        self.register_counter("benchmarks_failed", "Total benchmarks failed")
        self.register_gauge("active_benchmarks", "Number of active benchmarks")
        
        # Database metrics
        self.register_histogram("db_query_duration_ms", "Database query duration")
        self.register_counter("db_queries_total", "Total database queries")
        self.register_counter("db_errors_total", "Database errors")
        
        # Health check metrics
        self.register_gauge("health_check_success", "Health check success (1=success, 0=failure)")
        self.register_histogram("health_check_duration_ms", "Health check duration")
    
    def register_counter(self, name: str, description: str = "") -> Counter:
        """Register a new counter metric"""
        with self.lock:
            if name in self.metrics:
                raise ValueError(f"Metric {name} already exists")
            
            counter = Counter(name, description)
            self.metrics[name] = counter
            return counter
    
    def register_gauge(self, name: str, description: str = "") -> Gauge:
        """Register a new gauge metric"""
        with self.lock:
            if name in self.metrics:
                raise ValueError(f"Metric {name} already exists")
            
            gauge = Gauge(name, description)
            self.metrics[name] = gauge
            return gauge
    
    def register_histogram(self, name: str, description: str = "", 
                         buckets: Optional[List[float]] = None) -> Histogram:
        """Register a new histogram metric"""
        with self.lock:
            if name in self.metrics:
                raise ValueError(f"Metric {name} already exists")
            
            histogram = Histogram(name, description, buckets)
            self.metrics[name] = histogram
            return histogram
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name"""
        return self.metrics.get(name)
    
    def get_counter(self, name: str) -> Optional[Counter]:
        """Get counter metric by name"""
        metric = self.get_metric(name)
        return metric if isinstance(metric, Counter) else None
    
    def get_gauge(self, name: str) -> Optional[Gauge]:
        """Get gauge metric by name"""
        metric = self.get_metric(name)
        return metric if isinstance(metric, Gauge) else None
    
    def get_histogram(self, name: str) -> Optional[Histogram]:
        """Get histogram metric by name"""
        metric = self.get_metric(name)
        return metric if isinstance(metric, Histogram) else None
    
    def time_operation(self, metric_name: str, tags: Optional[Dict[str, str]] = None) -> Timer:
        """Create timer for operation timing"""
        histogram = self.get_histogram(metric_name)
        if not histogram:
            histogram = self.register_histogram(metric_name, f"Duration of {metric_name}")
        
        return Timer(histogram, tags)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics with their current stats"""
        result = {}
        
        with self.lock:
            for name, metric in self.metrics.items():
                result[name] = {
                    "type": metric.type.value,
                    "description": metric.description,
                    "current_value": metric.get_current_value(),
                    "stats": metric.get_stats()
                }
        
        return result
    
    def get_metrics_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get summary of all metrics"""
        if since is None:
            since = datetime.now() - timedelta(hours=1)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "period_start": since.isoformat(),
            "metrics": {}
        }
        
        with self.lock:
            for name, metric in self.metrics.items():
                stats = metric.get_stats(since)
                if stats:
                    summary["metrics"][name] = {
                        "type": metric.type.value,
                        "description": metric.description,
                        **stats
                    }
        
        return summary
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        with self.lock:
            for name, metric in self.metrics.items():
                # Add metric help
                if metric.description:
                    lines.append(f"# HELP {name} {metric.description}")
                
                # Add metric type
                type_mapping = {
                    MetricType.COUNTER: "counter",
                    MetricType.GAUGE: "gauge",
                    MetricType.HISTOGRAM: "histogram"
                }
                lines.append(f"# TYPE {name} {type_mapping.get(metric.type, 'gauge')}")
                
                # Add current value
                current_value = metric.get_current_value()
                if current_value is not None:
                    lines.append(f"{name} {current_value}")
                
                lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def increment_counter(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
    """Convenience function to increment counter"""
    collector = get_metrics_collector()
    counter = collector.get_counter(name)
    if counter:
        counter.increment(value, tags)


def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function to set gauge"""
    collector = get_metrics_collector()
    gauge = collector.get_gauge(name)
    if gauge:
        gauge.set(value, tags)


def observe_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Convenience function to observe histogram"""
    collector = get_metrics_collector()
    histogram = collector.get_histogram(name)
    if histogram:
        histogram.observe(value, tags)


def time_operation(name: str, tags: Optional[Dict[str, str]] = None) -> Timer:
    """Convenience function for timing operations"""
    collector = get_metrics_collector()
    return collector.time_operation(name, tags)