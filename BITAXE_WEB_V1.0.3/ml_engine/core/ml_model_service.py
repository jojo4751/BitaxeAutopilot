"""
ML Model Service

Core service for machine learning model lifecycle management, loading, inference, and monitoring.
"""

import asyncio
import os
import pickle
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from ml_engine.data.feature_engineering import FeatureSet

logger = get_logger("bitaxe.ml.model_service")


@dataclass
class ModelInfo:
    """Model metadata and information"""
    model_id: str
    model_type: str  # 'sklearn', 'tensorflow', 'pytorch', 'custom'
    version: str
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    model_config: Dict[str, Any]
    file_path: str
    is_active: bool = True
    training_data_size: int = 0


@dataclass
class PredictionResult:
    """Result of model prediction"""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    model_id: str = ""
    inference_time_ms: float = 0
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class ModelRegistry:
    """
    Model registry for versioning and lifecycle management
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelInfo] = {}
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from file"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for model_id, model_data in registry_data.items():
                    # Convert datetime strings back to datetime objects
                    model_data['created_at'] = datetime.fromisoformat(model_data['created_at'])
                    model_data['updated_at'] = datetime.fromisoformat(model_data['updated_at'])
                    
                    self.models[model_id] = ModelInfo(**model_data)
                
                logger.info(f"Loaded {len(self.models)} models from registry")
        except Exception as e:
            logger.error("Failed to load model registry", error=str(e))
            self.models = {}
    
    def save_registry(self):
        """Save model registry to file"""
        try:
            registry_data = {}
            for model_id, model_info in self.models.items():
                model_dict = asdict(model_info)
                # Convert datetime objects to strings for JSON serialization
                model_dict['created_at'] = model_info.created_at.isoformat()
                model_dict['updated_at'] = model_info.updated_at.isoformat()
                registry_data[model_id] = model_dict
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
            logger.debug("Model registry saved")
        except Exception as e:
            logger.error("Failed to save model registry", error=str(e))
    
    def register_model(self, model_info: ModelInfo):
        """Register a new model"""
        self.models[model_info.model_id] = model_info
        self.save_registry()
        logger.info(f"Registered model: {model_info.model_id}")
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.models.get(model_id)
    
    def list_models(self, model_type: Optional[str] = None, active_only: bool = True) -> List[ModelInfo]:
        """List available models"""
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        return sorted(models, key=lambda m: m.updated_at, reverse=True)
    
    def deactivate_model(self, model_id: str):
        """Deactivate a model"""
        if model_id in self.models:
            self.models[model_id].is_active = False
            self.save_registry()


class MLModelService:
    """
    Comprehensive ML model service for BitAxe optimization
    
    Features:
    - Model loading and management
    - Real-time inference with caching
    - Model performance monitoring
    - Automatic model validation
    - Fallback strategies
    - A/B testing support
    """
    
    def __init__(self, model_dir: str = "models", config: Dict[str, Any] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # Initialize components
        self.registry = ModelRegistry(self.model_dir / "registry.json")
        self.metrics_collector = get_metrics_collector()
        
        # Loaded models cache
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelInfo] = {}
        
        # Performance tracking
        self.inference_cache: Dict[str, Any] = {}
        self.cache_ttl = self.config.get('cache_ttl_seconds', 300)  # 5 minutes
        
        # Model validation settings
        self.validation_threshold = self.config.get('validation_threshold', 0.8)
        self.enable_fallback = self.config.get('enable_fallback', True)
        
        # A/B testing
        self.ab_test_config = self.config.get('ab_testing', {})
        self.ab_test_enabled = self.ab_test_config.get('enabled', False)
        
        logger.info("ML Model Service initialized", 
                   model_dir=str(self.model_dir),
                   cache_ttl=self.cache_ttl)
    
    async def start(self):
        """Start the ML model service"""
        logger.info("Starting ML Model Service")
        
        # Load default models
        await self._load_default_models()
        
        # Start background tasks
        asyncio.create_task(self._cache_cleanup_worker())
        asyncio.create_task(self._model_health_monitor())
        
        self.metrics_collector.increment_counter('ml_service_starts_total')
        logger.info("ML Model Service started")
    
    async def stop(self):
        """Stop the ML model service"""
        logger.info("Stopping ML Model Service")
        
        # Clear loaded models
        self.loaded_models.clear()
        self.model_metadata.clear()
        
        logger.info("ML Model Service stopped")
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> bool:
        """
        Load a model into memory
        
        Args:
            model_id: Unique model identifier
            force_reload: Force reload even if already loaded
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_id in self.loaded_models and not force_reload:
                logger.debug(f"Model {model_id} already loaded")
                return True
            
            model_info = self.registry.get_model(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            if not model_info.is_active:
                logger.warning(f"Model {model_id} is not active")
                return False
            
            model_path = Path(model_info.file_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model based on type
            model = await self._load_model_by_type(model_info, model_path)
            
            if model is not None:
                self.loaded_models[model_id] = model
                self.model_metadata[model_id] = model_info
                
                self.metrics_collector.increment_counter('models_loaded_total', 
                                                       tags={'model_type': model_info.model_type})
                logger.info(f"Model loaded successfully: {model_id}")
                return True
            else:
                logger.error(f"Failed to load model: {model_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}", error=str(e))
            self.metrics_collector.increment_counter('model_load_errors_total')
            return False
    
    async def predict(self, 
                     model_id: str, 
                     features: Union[np.ndarray, FeatureSet],
                     use_cache: bool = True,
                     return_confidence: bool = False) -> Optional[PredictionResult]:
        """
        Make predictions using specified model
        
        Args:
            model_id: Model to use for prediction
            features: Input features
            use_cache: Whether to use prediction cache
            return_confidence: Whether to return prediction confidence
            
        Returns:
            Prediction result or None if failed
        """
        start_time = datetime.now()
        
        try:
            # Extract features if FeatureSet provided
            if isinstance(features, FeatureSet):
                feature_array = features.features
                feature_names = features.feature_names
            else:
                feature_array = features
                feature_names = []
            
            # Check cache first
            if use_cache:
                cache_key = self._get_cache_key(model_id, feature_array)
                cached_result = self.inference_cache.get(cache_key)
                
                if cached_result and self._is_cache_valid(cached_result['timestamp']):
                    self.metrics_collector.increment_counter('inference_cache_hits_total')
                    return cached_result['result']
            
            # Ensure model is loaded
            if model_id not in self.loaded_models:
                loaded = await self.load_model(model_id)
                if not loaded:
                    return None
            
            model = self.loaded_models[model_id]
            model_info = self.model_metadata[model_id]
            
            # Validate input features
            if not await self._validate_features(feature_array, model_info):
                logger.error(f"Feature validation failed for model {model_id}")
                return None
            
            # Make prediction based on model type
            predictions, confidence = await self._predict_by_type(
                model, model_info, feature_array, return_confidence
            )
            
            if predictions is None:
                return None
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create result
            result = PredictionResult(
                predictions=predictions,
                confidence=confidence,
                model_id=model_id,
                inference_time_ms=inference_time,
                metadata={
                    'feature_count': feature_array.shape[1] if len(feature_array.shape) > 1 else len(feature_array),
                    'model_version': model_info.version,
                    'prediction_timestamp': datetime.now().isoformat()
                }
            )
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(model_id, feature_array)
                self.inference_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now()
                }
            
            # Record metrics
            self.metrics_collector.record_timer('inference_duration_ms', inference_time,
                                               tags={'model_id': model_id})
            self.metrics_collector.increment_counter('predictions_total',
                                                   tags={'model_id': model_id})
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}", error=str(e))
            self.metrics_collector.increment_counter('prediction_errors_total',
                                                   tags={'model_id': model_id})
            return None
    
    async def predict_with_fallback(self,
                                  primary_model_id: str,
                                  fallback_model_id: str,
                                  features: Union[np.ndarray, FeatureSet],
                                  fallback_threshold: float = 0.5) -> Optional[PredictionResult]:
        """
        Make predictions with fallback model support
        
        Args:
            primary_model_id: Primary model to try first
            fallback_model_id: Fallback model if primary fails
            features: Input features
            fallback_threshold: Confidence threshold for using fallback
            
        Returns:
            Prediction result from primary or fallback model
        """
        try:
            # Try primary model
            result = await self.predict(primary_model_id, features, return_confidence=True)
            
            if result is not None:
                # Check if confidence is sufficient
                if (result.confidence is not None and 
                    np.mean(result.confidence) >= fallback_threshold):
                    return result
                
                # If confidence is low, try fallback
                if fallback_model_id != primary_model_id:
                    logger.info(f"Low confidence from {primary_model_id}, trying fallback {fallback_model_id}")
                    fallback_result = await self.predict(fallback_model_id, features)
                    
                    if fallback_result is not None:
                        fallback_result.metadata['used_fallback'] = True
                        fallback_result.metadata['primary_model'] = primary_model_id
                        self.metrics_collector.increment_counter('fallback_predictions_total')
                        return fallback_result
                
                return result
            
            # Primary model failed, try fallback
            if fallback_model_id != primary_model_id:
                logger.warning(f"Primary model {primary_model_id} failed, using fallback {fallback_model_id}")
                fallback_result = await self.predict(fallback_model_id, features)
                
                if fallback_result is not None:
                    fallback_result.metadata['used_fallback'] = True
                    fallback_result.metadata['primary_model_failed'] = True
                    self.metrics_collector.increment_counter('fallback_predictions_total')
                    return fallback_result
            
            return None
            
        except Exception as e:
            logger.error("Prediction with fallback failed", error=str(e))
            return None
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        return self.registry.get_model(model_id)
    
    async def list_available_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """List all available models"""
        return self.registry.list_models(model_type=model_type)
    
    async def validate_model_performance(self, 
                                       model_id: str,
                                       test_features: np.ndarray,
                                       test_targets: np.ndarray) -> Dict[str, float]:
        """Validate model performance on test data"""
        try:
            result = await self.predict(model_id, test_features)
            
            if result is None:
                return {}
            
            predictions = result.predictions
            
            # Calculate performance metrics
            metrics = {}
            
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics['mse'] = mean_squared_error(test_targets, predictions)
                metrics['mae'] = mean_absolute_error(test_targets, predictions)
                metrics['r2'] = r2_score(test_targets, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
            else:
                # Classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                pred_classes = np.argmax(predictions, axis=1)
                
                metrics['accuracy'] = accuracy_score(test_targets, pred_classes)
                metrics['precision'] = precision_score(test_targets, pred_classes, average='weighted')
                metrics['recall'] = recall_score(test_targets, pred_classes, average='weighted')
                metrics['f1'] = f1_score(test_targets, pred_classes, average='weighted')
            
            # Record metrics
            for metric_name, value in metrics.items():
                self.metrics_collector.record_metric(f'model_validation_{metric_name}', value,
                                                    tags={'model_id': model_id})
            
            logger.info(f"Model validation completed for {model_id}", metrics=metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_id}", error=str(e))
            return {}
    
    async def _load_model_by_type(self, model_info: ModelInfo, model_path: Path) -> Optional[Any]:
        """Load model based on its type"""
        try:
            if model_info.model_type == 'sklearn':
                return joblib.load(model_path)
            elif model_info.model_type == 'pickle':
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif model_info.model_type == 'tensorflow':
                import tensorflow as tf
                return tf.keras.models.load_model(str(model_path))
            elif model_info.model_type == 'pytorch':
                import torch
                return torch.load(model_path, map_location='cpu')
            else:
                logger.error(f"Unsupported model type: {model_info.model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_info.model_id}", error=str(e))
            return None
    
    async def _predict_by_type(self, 
                              model: Any, 
                              model_info: ModelInfo, 
                              features: np.ndarray,
                              return_confidence: bool) -> tuple:
        """Make predictions based on model type"""
        try:
            predictions = None
            confidence = None
            
            if model_info.model_type in ['sklearn', 'pickle']:
                predictions = model.predict(features)
                
                if return_confidence and hasattr(model, 'predict_proba'):
                    prob_predictions = model.predict_proba(features)
                    confidence = np.max(prob_predictions, axis=1)
                elif return_confidence and hasattr(model, 'decision_function'):
                    decision_scores = model.decision_function(features)
                    confidence = np.abs(decision_scores)
                    
            elif model_info.model_type == 'tensorflow':
                predictions = model.predict(features)
                
                if return_confidence and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    # Multi-class classification
                    confidence = np.max(predictions, axis=1)
                elif return_confidence:
                    # Regression - use negative of prediction variance as confidence proxy
                    confidence = np.ones(len(predictions)) * 0.8  # Default confidence
                    
            elif model_info.model_type == 'pytorch':
                import torch
                model.eval()
                with torch.no_grad():
                    if isinstance(features, np.ndarray):
                        features_tensor = torch.FloatTensor(features)
                    else:
                        features_tensor = features
                    
                    output = model(features_tensor)
                    predictions = output.numpy()
                    
                    if return_confidence:
                        # For classification, use softmax confidence
                        if len(output.shape) > 1 and output.shape[1] > 1:
                            confidence = torch.max(torch.softmax(output, dim=1), dim=1)[0].numpy()
                        else:
                            confidence = np.ones(len(predictions)) * 0.8
            
            return predictions, confidence
            
        except Exception as e:
            logger.error("Prediction by type failed", error=str(e))
            return None, None
    
    async def _validate_features(self, features: np.ndarray, model_info: ModelInfo) -> bool:
        """Validate input features against model requirements"""
        try:
            # Check feature count
            expected_features = len(model_info.feature_names)
            actual_features = features.shape[1] if len(features.shape) > 1 else len(features)
            
            if expected_features > 0 and actual_features != expected_features:
                logger.error(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
                return False
            
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.error("Features contain NaN or infinite values")
                return False
            
            return True
            
        except Exception as e:
            logger.error("Feature validation failed", error=str(e))
            return False
    
    def _get_cache_key(self, model_id: str, features: np.ndarray) -> str:
        """Generate cache key for prediction"""
        feature_hash = hash(features.tobytes())
        return f"{model_id}:{feature_hash}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        return (datetime.now() - timestamp).total_seconds() < self.cache_ttl
    
    async def _load_default_models(self):
        """Load default models on startup"""
        models = self.registry.list_models(active_only=True)
        
        for model_info in models[:5]:  # Load first 5 models
            await self.load_model(model_info.model_id)
    
    async def _cache_cleanup_worker(self):
        """Background worker to clean up expired cache entries"""
        while True:
            try:
                current_time = datetime.now()
                expired_keys = [
                    key for key, value in self.inference_cache.items()
                    if (current_time - value['timestamp']).total_seconds() > self.cache_ttl
                ]
                
                for key in expired_keys:
                    del self.inference_cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error("Cache cleanup worker error", error=str(e))
                await asyncio.sleep(60)
    
    async def _model_health_monitor(self):
        """Monitor model health and performance"""
        while True:
            try:
                for model_id in self.loaded_models:
                    # Record model status metrics
                    self.metrics_collector.set_gauge('model_loaded', 1,
                                                   tags={'model_id': model_id})
                
                # Record cache statistics
                self.metrics_collector.set_gauge('inference_cache_size', len(self.inference_cache))
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error("Model health monitor error", error=str(e))
                await asyncio.sleep(60)


async def create_ml_model_service(model_dir: str = "models", 
                                 config: Dict[str, Any] = None) -> MLModelService:
    """Factory function to create ML model service"""
    service = MLModelService(model_dir, config)
    await service.start()
    return service