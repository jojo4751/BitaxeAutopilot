"""
ML Model Monitoring and Validation System

Comprehensive monitoring for model performance, drift detection, and validation.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector

logger = get_logger("bitaxe.ml.model_metrics")


@dataclass
class ModelPerformanceMetric:
    """Single model performance measurement"""
    model_id: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    data_window: str  # e.g., "1h", "24h", "7d"
    sample_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DriftDetectionResult:
    """Result of data/model drift detection"""
    model_id: str
    drift_type: str  # "data_drift", "concept_drift", "prediction_drift"
    drift_detected: bool
    drift_score: float
    threshold: float
    timestamp: datetime
    affected_features: List[str] = None
    drift_magnitude: float = 0.0
    recommendation: str = ""
    
    def __post_init__(self):
        if self.affected_features is None:
            self.affected_features = []


@dataclass
class ModelValidationResult:
    """Result of model validation"""
    model_id: str
    validation_type: str
    passed: bool
    score: float
    threshold: float
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ModelPerformanceTracker:
    """
    Track model performance metrics over time
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Performance data storage
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Configuration
        self.monitoring_windows = self.config.get('monitoring_windows', ["1h", "24h", "7d"])
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'accuracy_drop': 0.1,  # 10% drop from baseline
            'error_rate_increase': 0.05,  # 5% increase
            'latency_increase': 2.0  # 2x increase
        })
        
        logger.info("Model performance tracker initialized")
    
    async def record_prediction_metrics(self, 
                                      model_id: str,
                                      predictions: np.ndarray,
                                      actuals: Optional[np.ndarray] = None,
                                      prediction_time: float = 0.0,
                                      metadata: Dict[str, Any] = None):
        """Record metrics from model predictions"""
        try:
            timestamp = datetime.now()
            
            # Basic prediction metrics
            metrics = {
                'prediction_count': len(predictions),
                'prediction_time_ms': prediction_time * 1000,
                'prediction_mean': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'prediction_min': np.min(predictions),
                'prediction_max': np.max(predictions)
            }
            
            # Accuracy metrics if actuals are provided
            if actuals is not None and len(actuals) == len(predictions):
                mae = np.mean(np.abs(predictions - actuals))
                mse = np.mean((predictions - actuals) ** 2)
                rmse = np.sqrt(mse)
                
                # R² score
                ss_res = np.sum((actuals - predictions) ** 2)
                ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics.update({
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy_available': True
                })
            else:
                metrics['accuracy_available'] = False
            
            # Store metrics
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric = ModelPerformanceMetric(
                        model_id=model_id,
                        metric_name=metric_name,
                        metric_value=value,
                        timestamp=timestamp,
                        data_window="real_time",
                        sample_count=len(predictions),
                        metadata=metadata or {}
                    )
                    
                    self.performance_history[f"{model_id}_{metric_name}"].append(metric)
                    self.current_metrics[model_id][metric_name] = value
            
            # Record in global metrics collector
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and metric_name != 'accuracy_available':
                    self.metrics_collector.record_metric(
                        f'model_{metric_name}', value,
                        tags={'model_id': model_id}
                    )
            
            self.metrics_collector.increment_counter('model_predictions_recorded_total',
                                                   tags={'model_id': model_id})
            
        except Exception as e:
            logger.error("Failed to record prediction metrics", error=str(e))
    
    async def calculate_window_metrics(self, 
                                     model_id: str, 
                                     window: str = "1h") -> Dict[str, float]:
        """Calculate aggregated metrics for a time window"""
        try:
            # Parse window
            window_seconds = self._parse_window(window)
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            
            window_metrics = {}
            
            for metric_key, history in self.performance_history.items():
                if not metric_key.startswith(f"{model_id}_"):
                    continue
                
                metric_name = metric_key.replace(f"{model_id}_", "")
                
                # Filter to window
                window_data = [
                    m for m in history 
                    if m.timestamp >= cutoff_time
                ]
                
                if not window_data:
                    continue
                
                # Calculate aggregations
                values = [m.metric_value for m in window_data]
                
                window_metrics[f"{metric_name}_mean"] = np.mean(values)
                window_metrics[f"{metric_name}_std"] = np.std(values)
                window_metrics[f"{metric_name}_min"] = np.min(values)
                window_metrics[f"{metric_name}_max"] = np.max(values)
                window_metrics[f"{metric_name}_count"] = len(values)
                
                # Trend calculation
                if len(values) > 1:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    window_metrics[f"{metric_name}_trend"] = trend
            
            return window_metrics
            
        except Exception as e:
            logger.error("Window metrics calculation failed", error=str(e))
            return {}
    
    def _parse_window(self, window: str) -> int:
        """Parse window string to seconds"""
        if window.endswith('h'):
            return int(window[:-1]) * 3600
        elif window.endswith('d'):
            return int(window[:-1]) * 86400
        elif window.endswith('m'):
            return int(window[:-1]) * 60
        else:
            return 3600  # Default 1 hour
    
    async def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        try:
            summary = {
                'model_id': model_id,
                'current_metrics': self.current_metrics.get(model_id, {}),
                'window_metrics': {},
                'health_status': 'unknown',
                'last_prediction': None
            }
            
            # Calculate window metrics
            for window in self.monitoring_windows:
                window_metrics = await self.calculate_window_metrics(model_id, window)
                summary['window_metrics'][window] = window_metrics
            
            # Determine health status
            summary['health_status'] = await self._assess_model_health(model_id)
            
            # Find last prediction time
            for metric_key, history in self.performance_history.items():
                if metric_key.startswith(f"{model_id}_") and history:
                    last_metric = history[-1]
                    if summary['last_prediction'] is None or last_metric.timestamp > summary['last_prediction']:
                        summary['last_prediction'] = last_metric.timestamp.isoformat()
            
            return summary
            
        except Exception as e:
            logger.error("Model summary generation failed", error=str(e))
            return {'model_id': model_id, 'error': str(e)}
    
    async def _assess_model_health(self, model_id: str) -> str:
        """Assess overall model health"""
        try:
            current = self.current_metrics.get(model_id, {})
            
            # Check if model is active
            if not current:
                return 'inactive'
            
            # Check performance thresholds
            if 'r2' in current:
                if current['r2'] < 0.5:
                    return 'poor'
                elif current['r2'] < 0.7:
                    return 'degraded'
            
            if 'mae' in current:
                # This would compare against baseline or historical performance
                pass
            
            # Check prediction latency
            if 'prediction_time_ms' in current:
                if current['prediction_time_ms'] > 1000:  # 1 second
                    return 'slow'
            
            return 'healthy'
            
        except Exception as e:
            logger.error("Model health assessment failed", error=str(e))
            return 'unknown'


class DriftDetector:
    """
    Detect data drift and concept drift in ML models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Drift detection configuration
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.reference_window_size = self.config.get('reference_window_size', 1000)
        self.detection_window_size = self.config.get('detection_window_size', 200)
        
        # Data storage for drift detection
        self.reference_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.reference_window_size))
        self.recent_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.detection_window_size))
        
        logger.info("Drift detector initialized")
    
    async def update_reference_data(self, 
                                  model_id: str, 
                                  features: np.ndarray,
                                  predictions: np.ndarray):
        """Update reference data for drift detection"""
        try:
            timestamp = datetime.now()
            
            # Store feature statistics
            for i, feature_values in enumerate(features.T):
                key = f"{model_id}_feature_{i}"
                stats = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'timestamp': timestamp
                }
                self.reference_data[key].append(stats)
            
            # Store prediction statistics
            pred_stats = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'timestamp': timestamp
            }
            self.reference_data[f"{model_id}_predictions"].append(pred_stats)
            
        except Exception as e:
            logger.error("Failed to update reference data", error=str(e))
    
    async def detect_drift(self, 
                          model_id: str,
                          features: np.ndarray,
                          predictions: np.ndarray) -> List[DriftDetectionResult]:
        """Detect various types of drift"""
        try:
            results = []
            
            # Data drift detection
            data_drift = await self._detect_data_drift(model_id, features)
            if data_drift:
                results.append(data_drift)
            
            # Prediction drift detection
            pred_drift = await self._detect_prediction_drift(model_id, predictions)
            if pred_drift:
                results.append(pred_drift)
            
            # Update recent data for ongoing monitoring
            await self._update_recent_data(model_id, features, predictions)
            
            # Record drift detection metrics
            for result in results:
                if result.drift_detected:
                    self.metrics_collector.increment_counter('model_drift_detected_total',
                                                           tags={
                                                               'model_id': model_id,
                                                               'drift_type': result.drift_type
                                                           })
                    self.metrics_collector.record_metric('drift_score', result.drift_score,
                                                        tags={'model_id': model_id})
            
            return results
            
        except Exception as e:
            logger.error("Drift detection failed", error=str(e))
            return []
    
    async def _detect_data_drift(self, 
                               model_id: str, 
                               features: np.ndarray) -> Optional[DriftDetectionResult]:
        """Detect data drift using statistical tests"""
        try:
            affected_features = []
            drift_scores = []
            
            for i, feature_values in enumerate(features.T):
                ref_key = f"{model_id}_feature_{i}"
                
                if len(self.reference_data[ref_key]) < 10:
                    continue  # Not enough reference data
                
                # Get reference statistics
                ref_stats = list(self.reference_data[ref_key])
                ref_means = [s['mean'] for s in ref_stats]
                ref_stds = [s['std'] for s in ref_stats]
                
                # Calculate current statistics
                current_mean = np.mean(feature_values)
                current_std = np.std(feature_values)
                
                # Statistical drift detection (simplified)
                ref_mean_avg = np.mean(ref_means)
                ref_std_avg = np.mean(ref_stds)
                
                # Z-score based drift detection
                mean_drift = abs(current_mean - ref_mean_avg) / (ref_std_avg + 1e-8)
                std_drift = abs(current_std - ref_std_avg) / (ref_std_avg + 1e-8)
                
                feature_drift_score = max(mean_drift, std_drift)
                drift_scores.append(feature_drift_score)
                
                if feature_drift_score > self.drift_threshold:
                    affected_features.append(f"feature_{i}")
            
            if not drift_scores:
                return None
            
            overall_drift_score = np.mean(drift_scores)
            drift_detected = overall_drift_score > self.drift_threshold
            
            return DriftDetectionResult(
                model_id=model_id,
                drift_type="data_drift",
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                threshold=self.drift_threshold,
                timestamp=datetime.now(),
                affected_features=affected_features,
                drift_magnitude=overall_drift_score,
                recommendation="Retrain model if data drift persists" if drift_detected else ""
            )
            
        except Exception as e:
            logger.error("Data drift detection failed", error=str(e))
            return None
    
    async def _detect_prediction_drift(self, 
                                     model_id: str, 
                                     predictions: np.ndarray) -> Optional[DriftDetectionResult]:
        """Detect prediction drift"""
        try:
            ref_key = f"{model_id}_predictions"
            
            if len(self.reference_data[ref_key]) < 10:
                return None
            
            # Get reference prediction statistics
            ref_stats = list(self.reference_data[ref_key])
            ref_means = [s['mean'] for s in ref_stats]
            ref_stds = [s['std'] for s in ref_stats]
            
            # Calculate current prediction statistics
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            
            # Compare with reference
            ref_mean_avg = np.mean(ref_means)
            ref_std_avg = np.mean(ref_stds)
            
            mean_drift = abs(current_mean - ref_mean_avg) / (ref_std_avg + 1e-8)
            std_drift = abs(current_std - ref_std_avg) / (ref_std_avg + 1e-8)
            
            drift_score = max(mean_drift, std_drift)
            drift_detected = drift_score > self.drift_threshold
            
            return DriftDetectionResult(
                model_id=model_id,
                drift_type="prediction_drift",
                drift_detected=drift_detected,
                drift_score=drift_score,
                threshold=self.drift_threshold,
                timestamp=datetime.now(),
                drift_magnitude=drift_score,
                recommendation="Investigate model behavior changes" if drift_detected else ""
            )
            
        except Exception as e:
            logger.error("Prediction drift detection failed", error=str(e))
            return None
    
    async def _update_recent_data(self, 
                                model_id: str,
                                features: np.ndarray,
                                predictions: np.ndarray):
        """Update recent data for ongoing monitoring"""
        try:
            timestamp = datetime.now()
            
            # Store recent feature statistics
            for i, feature_values in enumerate(features.T):
                key = f"{model_id}_recent_feature_{i}"
                stats = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'timestamp': timestamp
                }
                self.recent_data[key].append(stats)
            
            # Store recent prediction statistics
            pred_stats = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'timestamp': timestamp
            }
            self.recent_data[f"{model_id}_recent_predictions"].append(pred_stats)
            
        except Exception as e:
            logger.error("Failed to update recent data", error=str(e))


class ModelValidator:
    """
    Validate model performance and behavior
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Validation thresholds
        self.validation_thresholds = self.config.get('validation_thresholds', {
            'min_accuracy': 0.7,
            'max_error_rate': 0.2,
            'max_prediction_time': 1000,  # ms
            'min_prediction_variance': 0.01
        })
        
        logger.info("Model validator initialized")
    
    async def validate_model_performance(self, 
                                       model_id: str,
                                       predictions: np.ndarray,
                                       actuals: Optional[np.ndarray] = None,
                                       prediction_time: float = 0.0) -> List[ModelValidationResult]:
        """Comprehensive model performance validation"""
        try:
            results = []
            
            # Accuracy validation
            if actuals is not None:
                accuracy_result = await self._validate_accuracy(model_id, predictions, actuals)
                if accuracy_result:
                    results.append(accuracy_result)
            
            # Performance validation
            perf_result = await self._validate_performance(model_id, prediction_time)
            if perf_result:
                results.append(perf_result)
            
            # Prediction quality validation
            quality_result = await self._validate_prediction_quality(model_id, predictions)
            if quality_result:
                results.append(quality_result)
            
            # Record validation metrics
            for result in results:
                self.metrics_collector.record_metric('model_validation_score', result.score,
                                                    tags={
                                                        'model_id': model_id,
                                                        'validation_type': result.validation_type
                                                    })
                
                if not result.passed:
                    self.metrics_collector.increment_counter('model_validation_failures_total',
                                                           tags={
                                                               'model_id': model_id,
                                                               'validation_type': result.validation_type
                                                           })
            
            return results
            
        except Exception as e:
            logger.error("Model validation failed", error=str(e))
            return []
    
    async def _validate_accuracy(self, 
                               model_id: str,
                               predictions: np.ndarray,
                               actuals: np.ndarray) -> Optional[ModelValidationResult]:
        """Validate model accuracy"""
        try:
            # Calculate R² score
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            threshold = self.validation_thresholds['min_accuracy']
            passed = r2 >= threshold
            
            return ModelValidationResult(
                model_id=model_id,
                validation_type="accuracy",
                passed=passed,
                score=r2,
                threshold=threshold,
                timestamp=datetime.now(),
                details={
                    'r2_score': r2,
                    'mae': np.mean(np.abs(predictions - actuals)),
                    'rmse': np.sqrt(np.mean((predictions - actuals) ** 2))
                }
            )
            
        except Exception as e:
            logger.error("Accuracy validation failed", error=str(e))
            return None
    
    async def _validate_performance(self, 
                                  model_id: str,
                                  prediction_time: float) -> Optional[ModelValidationResult]:
        """Validate model performance (speed)"""
        try:
            prediction_time_ms = prediction_time * 1000
            threshold = self.validation_thresholds['max_prediction_time']
            passed = prediction_time_ms <= threshold
            
            # Score based on how much under threshold (inverted)
            score = max(0, 1 - (prediction_time_ms / threshold))
            
            return ModelValidationResult(
                model_id=model_id,
                validation_type="performance",
                passed=passed,
                score=score,
                threshold=threshold,
                timestamp=datetime.now(),
                details={
                    'prediction_time_ms': prediction_time_ms,
                    'threshold_ms': threshold
                }
            )
            
        except Exception as e:
            logger.error("Performance validation failed", error=str(e))
            return None
    
    async def _validate_prediction_quality(self, 
                                         model_id: str,
                                         predictions: np.ndarray) -> Optional[ModelValidationResult]:
        """Validate prediction quality (variance, outliers, etc.)"""
        try:
            # Check prediction variance
            pred_variance = np.var(predictions)
            min_variance = self.validation_thresholds['min_prediction_variance']
            
            # Check for outliers (simple IQR method)
            q1, q3 = np.percentile(predictions, [25, 75])
            iqr = q3 - q1
            outlier_threshold = 1.5 * iqr
            outliers = np.sum((predictions < (q1 - outlier_threshold)) | 
                            (predictions > (q3 + outlier_threshold)))
            outlier_rate = outliers / len(predictions)
            
            # Overall quality score
            variance_score = min(1.0, pred_variance / min_variance) if min_variance > 0 else 1.0
            outlier_score = max(0.0, 1.0 - outlier_rate)
            quality_score = (variance_score + outlier_score) / 2
            
            passed = quality_score >= 0.7  # 70% quality threshold
            
            return ModelValidationResult(
                model_id=model_id,
                validation_type="prediction_quality",
                passed=passed,
                score=quality_score,
                threshold=0.7,
                timestamp=datetime.now(),
                details={
                    'prediction_variance': pred_variance,
                    'outlier_rate': outlier_rate,
                    'outlier_count': outliers,
                    'prediction_range': (np.min(predictions), np.max(predictions))
                }
            )
            
        except Exception as e:
            logger.error("Prediction quality validation failed", error=str(e))
            return None


class MLMonitoringService:
    """
    Comprehensive ML monitoring and validation service
    
    Integrates performance tracking, drift detection, and model validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Initialize components
        self.performance_tracker = ModelPerformanceTracker(
            self.config.get('performance_tracking', {})
        )
        self.drift_detector = DriftDetector(
            self.config.get('drift_detection', {})
        )
        self.model_validator = ModelValidator(
            self.config.get('validation', {})
        )
        
        # Monitoring configuration
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.alert_enabled = self.config.get('alert_enabled', True)
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("ML monitoring service initialized")
    
    async def start(self):
        """Start ML monitoring service"""
        if self.is_running:
            return
        
        logger.info("Starting ML monitoring service")
        self.is_running = True
        
        # Start background monitoring
        if self.monitoring_enabled:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.metrics_collector.increment_counter('ml_monitoring_service_starts_total')
        logger.info("ML monitoring service started")
    
    async def stop(self):
        """Stop ML monitoring service"""
        if not self.is_running:
            return
        
        logger.info("Stopping ML monitoring service")
        self.is_running = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ML monitoring service stopped")
    
    async def monitor_prediction(self, 
                               model_id: str,
                               features: np.ndarray,
                               predictions: np.ndarray,
                               actuals: Optional[np.ndarray] = None,
                               prediction_time: float = 0.0,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive monitoring of a model prediction
        
        Returns monitoring report with performance, drift, and validation results
        """
        try:
            start_time = datetime.now()
            
            # Record performance metrics
            await self.performance_tracker.record_prediction_metrics(
                model_id, predictions, actuals, prediction_time, metadata
            )
            
            # Detect drift
            drift_results = await self.drift_detector.detect_drift(
                model_id, features, predictions
            )
            
            # Validate model
            validation_results = await self.model_validator.validate_model_performance(
                model_id, predictions, actuals, prediction_time
            )
            
            # Generate monitoring report
            report = {
                'model_id': model_id,
                'timestamp': start_time.isoformat(),
                'monitoring_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'drift_detected': any(r.drift_detected for r in drift_results),
                'validation_passed': all(r.passed for r in validation_results),
                'drift_results': [asdict(r) for r in drift_results],
                'validation_results': [asdict(r) for r in validation_results],
                'performance_summary': await self.performance_tracker.get_model_summary(model_id)
            }
            
            # Generate alerts if needed
            if self.alert_enabled:
                alerts = await self._generate_alerts(model_id, drift_results, validation_results)
                report['alerts'] = alerts
            
            # Record monitoring metrics
            self.metrics_collector.increment_counter('ml_predictions_monitored_total',
                                                   tags={'model_id': model_id})
            
            if report['drift_detected']:
                self.metrics_collector.increment_counter('ml_drift_alerts_total',
                                                       tags={'model_id': model_id})
            
            if not report['validation_passed']:
                self.metrics_collector.increment_counter('ml_validation_alerts_total',
                                                       tags={'model_id': model_id})
            
            return report
            
        except Exception as e:
            logger.error("Prediction monitoring failed", error=str(e))
            return {
                'model_id': model_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_alerts(self, 
                             model_id: str,
                             drift_results: List[DriftDetectionResult],
                             validation_results: List[ModelValidationResult]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring results"""
        alerts = []
        
        try:
            # Drift alerts
            for drift_result in drift_results:
                if drift_result.drift_detected:
                    alerts.append({
                        'type': 'drift_alert',
                        'severity': 'high' if drift_result.drift_score > 0.3 else 'medium',
                        'model_id': model_id,
                        'message': f"{drift_result.drift_type} detected with score {drift_result.drift_score:.3f}",
                        'drift_type': drift_result.drift_type,
                        'drift_score': drift_result.drift_score,
                        'recommendation': drift_result.recommendation,
                        'timestamp': drift_result.timestamp.isoformat()
                    })
            
            # Validation alerts
            for validation_result in validation_results:
                if not validation_result.passed:
                    alerts.append({
                        'type': 'validation_alert',
                        'severity': 'high' if validation_result.score < 0.5 else 'medium',
                        'model_id': model_id,
                        'message': f"{validation_result.validation_type} validation failed with score {validation_result.score:.3f}",
                        'validation_type': validation_result.validation_type,
                        'score': validation_result.score,
                        'threshold': validation_result.threshold,
                        'timestamp': validation_result.timestamp.isoformat()
                    })
            
            return alerts
            
        except Exception as e:
            logger.error("Alert generation failed", error=str(e))
            return []
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("ML monitoring loop started")
        
        while self.is_running:
            try:
                # Periodic monitoring tasks
                # This could include:
                # - Aggregate performance reports
                # - Clean up old monitoring data
                # - Generate summary reports
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("ML monitoring loop stopped")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring service status"""
        return {
            'is_running': self.is_running,
            'monitoring_enabled': self.monitoring_enabled,
            'alert_enabled': self.alert_enabled,
            'components': {
                'performance_tracker': True,
                'drift_detector': True,
                'model_validator': True
            },
            'tracked_models': len(self.performance_tracker.current_metrics)
        }


async def create_ml_monitoring_service(config: Dict[str, Any] = None) -> MLMonitoringService:
    """Factory function to create ML monitoring service"""
    service = MLMonitoringService(config)
    await service.start()
    return service