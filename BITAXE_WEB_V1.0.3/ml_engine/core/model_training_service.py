"""
Model Training Service

Automated model training, validation, and deployment pipeline for continuous learning.
"""

import asyncio
import os
import shutil
import json
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from ml_engine.data.feature_engineering import FeatureEngineer, FeatureSet
from ml_engine.core.ml_model_service import ModelInfo, ModelRegistry

logger = get_logger("bitaxe.ml.model_training")


@dataclass
class TrainingJob:
    """Training job configuration and metadata"""
    job_id: str
    model_type: str
    model_name: str
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    metrics: Dict[str, float] = None
    model_path: Optional[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class TrainingData:
    """Training data container"""
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]
    validation_split: float = 0.2
    
    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets"""
        split_idx = int(len(self.features) * (1 - self.validation_split))
        
        X_train = self.features[:split_idx]
        X_val = self.features[split_idx:]
        y_train = self.targets[:split_idx]
        y_val = self.targets[split_idx:]
        
        return X_train, X_val, y_train, y_val


class ModelTrainer:
    """Base class for model trainers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
    
    async def train(self, training_data: TrainingData) -> Tuple[Any, Dict[str, float]]:
        """
        Train a model
        
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        raise NotImplementedError
    
    async def validate(self, model: Any, validation_data: TrainingData) -> Dict[str, float]:
        """Validate a trained model"""
        raise NotImplementedError
    
    async def save_model(self, model: Any, filepath: str) -> bool:
        """Save trained model to file"""
        raise NotImplementedError


class SklearnTrainer(ModelTrainer):
    """Trainer for scikit-learn models"""
    
    async def train(self, training_data: TrainingData) -> Tuple[Any, Dict[str, float]]:
        """Train sklearn model"""
        try:
            model_type = self.config.get('model_type', 'random_forest')
            
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    max_depth=self.config.get('max_depth', 10),
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=self.config.get('n_estimators', 100),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    random_state=42
                )
            elif model_type == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                model = IsolationForest(
                    contamination=self.config.get('contamination', 0.1),
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported sklearn model type: {model_type}")
            
            # Split data
            X_train, X_val, y_train, y_val = training_data.get_train_test_split()
            
            # Train model
            start_time = datetime.now()
            
            if model_type == 'isolation_forest':
                # Unsupervised training
                model.fit(X_train)
            else:
                # Supervised training
                model.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            metrics = {'training_time': training_time}
            
            if model_type != 'isolation_forest':
                # Regression metrics
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                metrics.update({
                    'train_mse': mean_squared_error(y_train, train_pred),
                    'train_r2': r2_score(y_train, train_pred),
                    'train_mae': mean_absolute_error(y_train, train_pred),
                    'val_mse': mean_squared_error(y_val, val_pred),
                    'val_r2': r2_score(y_val, val_pred),
                    'val_mae': mean_absolute_error(y_val, val_pred)
                })
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(training_data.feature_names, model.feature_importances_))
                    metrics['feature_importance'] = importance_dict
            
            return model, metrics
            
        except Exception as e:
            logger.error("Sklearn model training failed", error=str(e))
            raise
    
    async def validate(self, model: Any, validation_data: TrainingData) -> Dict[str, float]:
        """Validate sklearn model"""
        try:
            X_val = validation_data.features
            y_val = validation_data.targets
            
            predictions = model.predict(X_val)
            
            # Calculate validation metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            metrics = {
                'validation_mse': mean_squared_error(y_val, predictions),
                'validation_r2': r2_score(y_val, predictions),
                'validation_mae': mean_absolute_error(y_val, predictions),
                'validation_samples': len(X_val)
            }
            
            return metrics
            
        except Exception as e:
            logger.error("Sklearn model validation failed", error=str(e))
            return {}
    
    async def save_model(self, model: Any, filepath: str) -> bool:
        """Save sklearn model"""
        try:
            joblib.dump(model, filepath)
            return True
        except Exception as e:
            logger.error("Failed to save sklearn model", error=str(e))
            return False


class TensorFlowTrainer(ModelTrainer):
    """Trainer for TensorFlow/Keras models"""
    
    async def train(self, training_data: TrainingData) -> Tuple[Any, Dict[str, float]]:
        """Train TensorFlow model"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Build model architecture
            model = self._build_model(training_data)
            
            # Compile model
            model.compile(
                optimizer=self.config.get('optimizer', 'adam'),
                loss=self.config.get('loss', 'mse'),
                metrics=['mae']
            )
            
            # Split data
            X_train, X_val, y_train, y_val = training_data.get_train_test_split()
            
            # Train model
            start_time = datetime.now()
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                verbose=0
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Extract training metrics
            metrics = {
                'training_time': training_time,
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1],
                'final_train_mae': history.history['mae'][-1],
                'final_val_mae': history.history['val_mae'][-1],
                'epochs_trained': len(history.history['loss'])
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error("TensorFlow model training failed", error=str(e))
            raise
    
    def _build_model(self, training_data: TrainingData):
        """Build TensorFlow model architecture"""
        import tensorflow as tf
        from tensorflow import keras
        
        input_dim = training_data.features.shape[1]
        
        # Simple neural network for temperature prediction
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        
        return model
    
    async def validate(self, model: Any, validation_data: TrainingData) -> Dict[str, float]:
        """Validate TensorFlow model"""
        try:
            X_val = validation_data.features
            y_val = validation_data.targets
            
            # Evaluate model
            loss, mae = model.evaluate(X_val, y_val, verbose=0)
            
            # Make predictions for additional metrics
            predictions = model.predict(X_val, verbose=0)
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(y_val, predictions)
            
            metrics = {
                'validation_loss': loss,
                'validation_mae': mae,
                'validation_r2': r2,
                'validation_samples': len(X_val)
            }
            
            return metrics
            
        except Exception as e:
            logger.error("TensorFlow model validation failed", error=str(e))
            return {}
    
    async def save_model(self, model: Any, filepath: str) -> bool:
        """Save TensorFlow model"""
        try:
            model.save(filepath)
            return True
        except Exception as e:
            logger.error("Failed to save TensorFlow model", error=str(e))
            return False


class ModelTrainingService:
    """
    Comprehensive model training and deployment service
    
    Features:
    - Automated model training pipeline
    - Multiple model types (sklearn, TensorFlow)
    - Model validation and performance tracking
    - Automated model deployment
    - Training job management
    - Data preprocessing and feature engineering
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Configuration
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.training_data_dir = Path(self.config.get('training_data_dir', 'training_data'))
        self.max_concurrent_jobs = self.config.get('max_concurrent_jobs', 2)
        self.auto_deployment_threshold = self.config.get('auto_deployment_threshold', 0.85)
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model_registry = ModelRegistry(self.models_dir / "registry.json")
        
        # Training job management
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Trainers
        self.trainers = {
            'sklearn': SklearnTrainer,
            'tensorflow': TensorFlowTrainer
        }
        
        # Background tasks
        self.training_scheduler_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Model training service initialized",
                   models_dir=str(self.models_dir),
                   max_concurrent_jobs=self.max_concurrent_jobs)
    
    async def start(self):
        """Start the model training service"""
        if self.is_running:
            return
        
        logger.info("Starting model training service")
        self.is_running = True
        
        # Initialize feature engineer
        feature_config = self.config.get('feature_engineering', {})
        self.feature_engineer = FeatureEngineer(feature_config)
        
        # Start training scheduler
        self.training_scheduler_task = asyncio.create_task(self._training_scheduler())
        
        self.metrics_collector.increment_counter('model_training_service_starts_total')
        logger.info("Model training service started")
    
    async def stop(self):
        """Stop the model training service"""
        if not self.is_running:
            return
        
        logger.info("Stopping model training service")
        self.is_running = False
        
        # Cancel training scheduler
        if self.training_scheduler_task:
            self.training_scheduler_task.cancel()
            try:
                await self.training_scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active training jobs
        for job_id, task in self.active_jobs.items():
            logger.info(f"Cancelling training job: {job_id}")
            task.cancel()
        
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)
        
        logger.info("Model training service stopped")
    
    async def submit_training_job(self, 
                                 model_type: str,
                                 model_name: str,
                                 training_config: Dict[str, Any],
                                 data_config: Dict[str, Any]) -> str:
        """
        Submit a new training job
        
        Args:
            model_type: Type of model (sklearn, tensorflow)
            model_name: Name for the trained model
            training_config: Model-specific training configuration
            data_config: Data preparation configuration
            
        Returns:
            Job ID
        """
        try:
            job_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            job = TrainingJob(
                job_id=job_id,
                model_type=model_type,
                model_name=model_name,
                training_config=training_config,
                data_config=data_config,
                created_at=datetime.now()
            )
            
            self.training_jobs[job_id] = job
            
            logger.info(f"Training job submitted: {job_id}",
                       model_type=model_type,
                       model_name=model_name)
            
            self.metrics_collector.increment_counter('training_jobs_submitted_total',
                                                   tags={'model_type': model_type})
            
            return job_id
            
        except Exception as e:
            logger.error("Failed to submit training job", error=str(e))
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        job = self.training_jobs.get(job_id)
        if not job:
            return None
        
        status = asdict(job)
        
        # Add runtime information
        if job.started_at:
            if job.completed_at:
                status['duration_seconds'] = (job.completed_at - job.started_at).total_seconds()
            else:
                status['running_time_seconds'] = (datetime.now() - job.started_at).total_seconds()
        
        return status
    
    async def list_training_jobs(self, 
                                status_filter: Optional[str] = None,
                                limit: int = 50) -> List[Dict[str, Any]]:
        """List training jobs with optional filtering"""
        jobs = list(self.training_jobs.values())
        
        if status_filter:
            jobs = [job for job in jobs if job.status == status_filter]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        # Limit results
        jobs = jobs[:limit]
        
        return [asdict(job) for job in jobs]
    
    async def prepare_training_data(self, 
                                  historical_data: List[Dict[str, Any]],
                                  data_config: Dict[str, Any]) -> Optional[TrainingData]:
        """
        Prepare training data from historical telemetry
        
        Args:
            historical_data: Raw historical miner data
            data_config: Data preparation configuration
            
        Returns:
            Prepared training data or None if failed
        """
        try:
            if not self.feature_engineer:
                logger.error("Feature engineer not initialized")
                return None
            
            # Engineer features
            target_column = data_config.get('target_column', 'temp')
            feature_set = await self.feature_engineer.engineer_features(
                historical_data, 
                target_column=target_column
            )
            
            if feature_set.features.size == 0:
                logger.error("No features generated from historical data")
                return None
            
            # Scale features if requested
            if data_config.get('scale_features', True):
                feature_set = await self.feature_engineer.scale_features(
                    feature_set, 
                    fit_scalers=True
                )
            
            # Create training data
            training_data = TrainingData(
                features=feature_set.features,
                targets=feature_set.target,
                feature_names=feature_set.feature_names,
                metadata=feature_set.metadata or {},
                validation_split=data_config.get('validation_split', 0.2)
            )
            
            logger.info("Training data prepared",
                       samples=len(training_data.features),
                       features=len(training_data.feature_names),
                       target=target_column)
            
            return training_data
            
        except Exception as e:
            logger.error("Training data preparation failed", error=str(e))
            return None
    
    async def train_model(self, job_id: str) -> bool:
        """Execute a training job"""
        try:
            job = self.training_jobs.get(job_id)
            if not job:
                logger.error(f"Training job not found: {job_id}")
                return False
            
            logger.info(f"Starting training job: {job_id}")
            job.status = "running"
            job.started_at = datetime.now()
            
            # Load historical data for training
            training_data = await self._load_training_data(job.data_config)
            if not training_data:
                job.status = "failed"
                job.error_message = "Failed to load training data"
                return False
            
            # Get trainer
            trainer_class = self.trainers.get(job.model_type)
            if not trainer_class:
                job.status = "failed"
                job.error_message = f"Unsupported model type: {job.model_type}"
                return False
            
            trainer = trainer_class(job.training_config)
            
            # Train model
            trained_model, training_metrics = await trainer.train(training_data)
            
            # Validate model
            validation_metrics = await trainer.validate(trained_model, training_data)
            
            # Combine metrics
            all_metrics = {**training_metrics, **validation_metrics}
            job.metrics = all_metrics
            
            # Save model
            model_filename = f"{job.model_name}_{job.job_id}.pkl"
            model_path = self.models_dir / model_filename
            
            save_success = await trainer.save_model(trained_model, str(model_path))
            if not save_success:
                job.status = "failed"
                job.error_message = "Failed to save trained model"
                return False
            
            job.model_path = str(model_path)
            
            # Register model
            await self._register_trained_model(job, training_data)
            
            # Check for auto-deployment
            if self._should_auto_deploy(all_metrics):
                await self._deploy_model(job)
            
            job.status = "completed"
            job.completed_at = datetime.now()
            
            # Record metrics
            training_duration = (job.completed_at - job.started_at).total_seconds()
            self.metrics_collector.record_timer('model_training_duration', training_duration)
            self.metrics_collector.increment_counter('models_trained_total',
                                                   tags={'model_type': job.model_type})
            
            for metric_name, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_collector.record_metric(f'training_{metric_name}', value,
                                                        tags={'model_name': job.model_name})
            
            logger.info(f"Training job completed successfully: {job_id}",
                       training_duration=training_duration,
                       metrics=all_metrics)
            
            return True
            
        except Exception as e:
            logger.error(f"Training job failed: {job_id}", error=str(e))
            
            job = self.training_jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error_message = str(e)
                job.completed_at = datetime.now()
            
            self.metrics_collector.increment_counter('model_training_failures_total')
            return False
    
    async def _load_training_data(self, data_config: Dict[str, Any]) -> Optional[TrainingData]:
        """Load training data based on configuration"""
        try:
            # This would typically load from a database or file
            # For now, we'll generate mock training data
            
            data_source = data_config.get('source', 'mock')
            
            if data_source == 'mock':
                # Generate mock training data
                n_samples = data_config.get('n_samples', 1000)
                n_features = data_config.get('n_features', 10)
                
                # Generate realistic mining data
                np.random.seed(42)
                
                # Base features: temp, power, frequency, voltage, hashrate, etc.
                features = np.random.randn(n_samples, n_features)
                
                # Realistic relationships for temperature prediction
                # temp = f(power, frequency, voltage, ambient_temp, ...)
                targets = (features[:, 1] * 20 +  # power effect
                          features[:, 2] * 10 +  # frequency effect
                          features[:, 3] * 5 +   # voltage effect
                          np.random.normal(50, 5, n_samples))  # base temp + noise
                
                feature_names = [f'feature_{i}' for i in range(n_features)]
                
                return TrainingData(
                    features=features,
                    targets=targets,
                    feature_names=feature_names,
                    metadata={'source': 'mock', 'generated_at': datetime.now().isoformat()}
                )
            
            else:
                logger.error(f"Unsupported data source: {data_source}")
                return None
            
        except Exception as e:
            logger.error("Failed to load training data", error=str(e))
            return None
    
    async def _register_trained_model(self, job: TrainingJob, training_data: TrainingData):
        """Register trained model in the model registry"""
        try:
            model_info = ModelInfo(
                model_id=job.job_id,
                model_type=job.model_type,
                version="1.0.0",
                created_at=job.created_at,
                updated_at=job.completed_at or datetime.now(),
                performance_metrics=job.metrics,
                feature_names=training_data.feature_names,
                model_config=job.training_config,
                file_path=job.model_path,
                training_data_size=len(training_data.features)
            )
            
            self.model_registry.register_model(model_info)
            logger.info(f"Model registered: {job.job_id}")
            
        except Exception as e:
            logger.error("Failed to register trained model", error=str(e))
    
    def _should_auto_deploy(self, metrics: Dict[str, float]) -> bool:
        """Check if model should be automatically deployed"""
        try:
            # Check validation R² score
            val_r2 = metrics.get('val_r2', metrics.get('validation_r2', 0))
            
            return val_r2 >= self.auto_deployment_threshold
            
        except Exception as e:
            logger.error("Auto-deployment check failed", error=str(e))
            return False
    
    async def _deploy_model(self, job: TrainingJob):
        """Deploy trained model (activate in registry)"""
        try:
            # Mark model as active
            model_info = self.model_registry.get_model(job.job_id)
            if model_info:
                model_info.is_active = True
                self.model_registry.save_registry()
                
                logger.info(f"Model auto-deployed: {job.job_id}")
                self.metrics_collector.increment_counter('models_auto_deployed_total')
        
        except Exception as e:
            logger.error("Model deployment failed", error=str(e))
    
    async def _training_scheduler(self):
        """Background task to manage training job queue"""
        logger.info("Training scheduler started")
        
        while self.is_running:
            try:
                # Find pending jobs
                pending_jobs = [
                    job for job in self.training_jobs.values()
                    if job.status == "pending"
                ]
                
                # Start jobs up to concurrency limit
                running_count = len(self.active_jobs)
                available_slots = self.max_concurrent_jobs - running_count
                
                for job in pending_jobs[:available_slots]:
                    task = asyncio.create_task(self.train_model(job.job_id))
                    self.active_jobs[job.job_id] = task
                    
                    logger.info(f"Started training job: {job.job_id}")
                
                # Clean up completed tasks
                completed_jobs = []
                for job_id, task in self.active_jobs.items():
                    if task.done():
                        completed_jobs.append(job_id)
                
                for job_id in completed_jobs:
                    del self.active_jobs[job_id]
                
                # Update metrics
                self.metrics_collector.set_gauge('training_jobs_pending', len(pending_jobs))
                self.metrics_collector.set_gauge('training_jobs_running', len(self.active_jobs))
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Training scheduler error", error=str(e))
                await asyncio.sleep(30)
        
        logger.info("Training scheduler stopped")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get training service status"""
        pending_jobs = [j for j in self.training_jobs.values() if j.status == "pending"]
        running_jobs = [j for j in self.training_jobs.values() if j.status == "running"]
        completed_jobs = [j for j in self.training_jobs.values() if j.status == "completed"]
        failed_jobs = [j for j in self.training_jobs.values() if j.status == "failed"]
        
        return {
            'is_running': self.is_running,
            'job_counts': {
                'total': len(self.training_jobs),
                'pending': len(pending_jobs),
                'running': len(running_jobs),
                'completed': len(completed_jobs),
                'failed': len(failed_jobs)
            },
            'active_jobs': len(self.active_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'models_dir': str(self.models_dir),
            'registered_models': len(self.model_registry.models),
            'auto_deployment_threshold': self.auto_deployment_threshold
        }


async def create_model_training_service(config: Dict[str, Any] = None) -> ModelTrainingService:
    """Factory function to create model training service"""
    service = ModelTrainingService(config)
    await service.start()
    return service