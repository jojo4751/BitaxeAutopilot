"""
Predictive Analytics Models

Advanced machine learning models for temperature prediction, efficiency forecasting,
failure detection, and performance optimization.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import pickle
from pathlib import Path

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from ml_engine.data.feature_engineering import FeatureSet

logger = get_logger("bitaxe.ml.predictive_models")


@dataclass
class PredictionResult:
    """Result of a prediction model"""
    predictions: np.ndarray
    confidence: Optional[np.ndarray] = None
    prediction_horizon: int = 1  # minutes
    model_type: str = ""
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None


class TemperaturePredictor:
    """
    LSTM-based neural network for temperature forecasting
    
    Predicts temperature 30 minutes ahead using historical telemetry,
    environmental conditions, and operational parameters.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Model configuration
        self.sequence_length = self.config.get('sequence_length', 30)  # 30 data points
        self.prediction_horizon = self.config.get('prediction_horizon', 30)  # 30 minutes
        self.hidden_units = self.config.get('hidden_units', 64)
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Training configuration
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.validation_split = self.config.get('validation_split', 0.2)
        
        # Model state
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_columns = [
            'temp', 'power', 'frequency', 'voltage', 'hashRate',
            'weather_temp', 'weather_humidity', 'hour_sin', 'hour_cos'
        ]
        
        # Performance tracking
        self.last_accuracy = 0.0
        self.training_history = []
        
        logger.info("Temperature predictor initialized",
                   sequence_length=self.sequence_length,
                   prediction_horizon=self.prediction_horizon)
    
    async def train(self, historical_data: List[Dict[str, Any]]) -> bool:
        """
        Train the temperature prediction model
        
        Args:
            historical_data: Historical miner telemetry data
            
        Returns:
            True if training successful
        """
        try:
            logger.info("Starting temperature predictor training")
            start_time = datetime.now()
            
            # Prepare training data
            X, y = await self._prepare_training_data(historical_data)
            
            if X is None or len(X) < 100:
                logger.error("Insufficient training data")
                return False
            
            # Initialize model (simplified LSTM-like approach)
            await self._initialize_model()
            
            # Train model (simplified implementation)
            training_loss = await self._train_model(X, y)
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics_collector.record_metric('temp_predictor_training_time', training_time)
            self.metrics_collector.record_metric('temp_predictor_training_loss', training_loss)
            self.metrics_collector.increment_counter('temp_predictor_training_runs_total')
            
            logger.info("Temperature predictor training completed",
                       training_samples=len(X),
                       training_loss=training_loss,
                       training_time_sec=training_time)
            
            return True
            
        except Exception as e:
            logger.error("Temperature predictor training failed", error=str(e))
            self.metrics_collector.increment_counter('temp_predictor_training_errors_total')
            return False
    
    async def predict(self, recent_data: List[Dict[str, Any]], 
                     weather_data: Optional[Dict[str, Any]] = None) -> Optional[PredictionResult]:
        """
        Predict temperature for next 30 minutes
        
        Args:
            recent_data: Recent miner telemetry (at least sequence_length points)
            weather_data: Current weather conditions
            
        Returns:
            Temperature prediction result
        """
        try:
            if not self.is_trained:
                logger.warning("Temperature predictor not trained")
                return None
            
            # Prepare input sequence
            input_sequence = await self._prepare_prediction_input(recent_data, weather_data)
            
            if input_sequence is None:
                return None
            
            start_time = datetime.now()
            
            # Make prediction (simplified)
            predictions = await self._predict_temperature(input_sequence)
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate confidence based on recent accuracy
            confidence = np.ones(len(predictions)) * max(0.5, self.last_accuracy)
            
            result = PredictionResult(
                predictions=predictions,
                confidence=confidence,
                prediction_horizon=self.prediction_horizon,
                model_type="temperature_lstm",
                metadata={
                    'inference_time_ms': inference_time,
                    'sequence_length': len(recent_data),
                    'model_accuracy': self.last_accuracy
                }
            )
            
            # Record metrics
            self.metrics_collector.record_timer('temp_predictor_inference_ms', inference_time)
            self.metrics_collector.increment_counter('temp_predictions_total')
            
            return result
            
        except Exception as e:
            logger.error("Temperature prediction failed", error=str(e))
            self.metrics_collector.increment_counter('temp_prediction_errors_total')
            return None
    
    async def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from historical telemetry"""
        try:
            df = pd.DataFrame(data)
            
            if len(df) < self.sequence_length + self.prediction_horizon:
                return None, None
            
            # Ensure timestamp sorting
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Add time features
            if 'timestamp' in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
            else:
                df['hour_sin'] = 0
                df['hour_cos'] = 1
            
            # Add weather features (mock if not available)
            if 'weather_temp' not in df.columns:
                df['weather_temp'] = df.get('temp', 25) - 10  # Assume ambient is cooler
            if 'weather_humidity' not in df.columns:
                df['weather_humidity'] = 50.0
            
            # Select and clean features
            feature_data = df[self.feature_columns].fillna(method='ffill').fillna(0)
            
            # Create sequences
            X, y = [], []
            
            for i in range(len(feature_data) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                sequence = feature_data.iloc[i:i + self.sequence_length].values
                
                # Target (temperature after prediction_horizon)
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target = feature_data.iloc[target_idx]['temp']
                
                X.append(sequence)
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error("Training data preparation failed", error=str(e))
            return None, None
    
    async def _initialize_model(self):
        """Initialize the LSTM model (simplified implementation)"""
        # This is a simplified model - in production use TensorFlow/PyTorch
        self.model = {
            'weights': np.random.normal(0, 0.1, (len(self.feature_columns), self.hidden_units)),
            'bias': np.zeros(self.hidden_units),
            'output_weights': np.random.normal(0, 0.1, (self.hidden_units, 1)),
            'output_bias': 0.0
        }
        
        # Initialize scaler for normalization
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
    
    async def _train_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the model (simplified implementation)"""
        try:
            # Normalize data
            X_reshaped = X.reshape(-1, X.shape[-1])
            self.scaler.fit(X_reshaped)
            X_normalized = self.scaler.transform(X_reshaped).reshape(X.shape)
            
            # Simple training loop (this would be much more complex in practice)
            learning_rate = self.learning_rate
            training_loss = 0.0
            
            for epoch in range(min(self.epochs, 10)):  # Simplified training
                epoch_loss = 0.0
                
                for i in range(len(X)):
                    # Forward pass (simplified)
                    sequence = X_normalized[i]
                    
                    # Average pooling over sequence (simplified LSTM)
                    pooled_features = np.mean(sequence, axis=0)
                    
                    # Hidden layer
                    hidden = np.maximum(0, np.dot(pooled_features, self.model['weights']) + self.model['bias'])
                    
                    # Output
                    prediction = np.dot(hidden, self.model['output_weights']) + self.model['output_bias']
                    
                    # Loss (MSE)
                    loss = (prediction[0] - y[i]) ** 2
                    epoch_loss += loss
                    
                    # Simple gradient update (very simplified)
                    error = prediction[0] - y[i]
                    
                    # Update output weights
                    self.model['output_weights'] -= learning_rate * error * hidden.reshape(-1, 1)
                    self.model['output_bias'] -= learning_rate * error
                
                training_loss = epoch_loss / len(X)
                
                if epoch % 2 == 0:
                    logger.debug(f"Training epoch {epoch}, loss: {training_loss:.4f}")
            
            # Calculate accuracy on training data (simplified)
            predictions = []
            for i in range(len(X)):
                sequence = X_normalized[i]
                pooled_features = np.mean(sequence, axis=0)
                hidden = np.maximum(0, np.dot(pooled_features, self.model['weights']) + self.model['bias'])
                prediction = np.dot(hidden, self.model['output_weights']) + self.model['output_bias']
                predictions.append(prediction[0])
            
            predictions = np.array(predictions)
            mae = np.mean(np.abs(predictions - y))
            
            # Convert MAE to accuracy (rough approximation)
            self.last_accuracy = max(0.0, 1.0 - mae / np.std(y))
            
            return training_loss
            
        except Exception as e:
            logger.error("Model training failed", error=str(e))
            return float('inf')
    
    async def _prepare_prediction_input(self, recent_data: List[Dict[str, Any]], 
                                      weather_data: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare input for prediction"""
        try:
            if len(recent_data) < self.sequence_length:
                logger.warning(f"Insufficient recent data: {len(recent_data)} < {self.sequence_length}")
                return None
            
            # Take last sequence_length points
            df = pd.DataFrame(recent_data[-self.sequence_length:])
            
            # Add time features
            now = datetime.now()
            df['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
            
            # Add weather features
            if weather_data:
                df['weather_temp'] = weather_data.get('temperature', 25)
                df['weather_humidity'] = weather_data.get('humidity', 50)
            else:
                df['weather_temp'] = df.get('temp', 25) - 10
                df['weather_humidity'] = 50.0
            
            # Select features
            feature_data = df[self.feature_columns].fillna(method='ffill').fillna(0)
            
            # Normalize
            if self.scaler:
                normalized_data = self.scaler.transform(feature_data.values)
            else:
                normalized_data = feature_data.values
            
            return normalized_data.reshape(1, self.sequence_length, -1)
            
        except Exception as e:
            logger.error("Prediction input preparation failed", error=str(e))
            return None
    
    async def _predict_temperature(self, input_sequence: np.ndarray) -> np.ndarray:
        """Make temperature prediction"""
        try:
            # Simple prediction (simplified LSTM forward pass)
            sequence = input_sequence[0]  # Remove batch dimension
            
            # Average pooling over sequence
            pooled_features = np.mean(sequence, axis=0)
            
            # Forward pass
            hidden = np.maximum(0, np.dot(pooled_features, self.model['weights']) + self.model['bias'])
            prediction = np.dot(hidden, self.model['output_weights']) + self.model['output_bias']
            
            # Return array with single prediction
            return np.array([prediction[0]])
            
        except Exception as e:
            logger.error("Temperature prediction failed", error=str(e))
            return np.array([50.0])  # Safe default


class EfficiencyForecaster:
    """
    Random Forest model for efficiency trend prediction
    
    Forecasts mining efficiency trends and identifies optimization opportunities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Model configuration
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 60)  # minutes
        
        # Model state
        self.model = None
        self.feature_importance = {}
        self.is_trained = False
        self.last_accuracy = 0.0
        
        logger.info("Efficiency forecaster initialized")
    
    async def train(self, historical_data: List[Dict[str, Any]]) -> bool:
        """Train efficiency forecasting model"""
        try:
            logger.info("Starting efficiency forecaster training")
            start_time = datetime.now()
            
            # Prepare features and targets
            X, y = await self._prepare_efficiency_data(historical_data)
            
            if X is None or len(X) < 50:
                logger.error("Insufficient training data for efficiency forecaster")
                return False
            
            # Train Random Forest model
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_absolute_error
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            self.last_accuracy = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Get feature importance
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics_collector.record_metric('efficiency_forecaster_training_time', training_time)
            self.metrics_collector.record_metric('efficiency_forecaster_accuracy', self.last_accuracy)
            self.metrics_collector.record_metric('efficiency_forecaster_mae', mae)
            
            logger.info("Efficiency forecaster training completed",
                       accuracy=self.last_accuracy,
                       mae=mae,
                       training_time_sec=training_time)
            
            return True
            
        except Exception as e:
            logger.error("Efficiency forecaster training failed", error=str(e))
            return False
    
    async def predict(self, recent_data: List[Dict[str, Any]]) -> Optional[PredictionResult]:
        """Predict efficiency trends"""
        try:
            if not self.is_trained:
                logger.warning("Efficiency forecaster not trained")
                return None
            
            # Prepare features
            features = await self._prepare_forecast_features(recent_data)
            if features is None:
                return None
            
            start_time = datetime.now()
            
            # Make prediction
            efficiency_forecast = self.model.predict(features.reshape(1, -1))
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate confidence
            confidence = np.array([max(0.5, self.last_accuracy)])
            
            result = PredictionResult(
                predictions=efficiency_forecast,
                confidence=confidence,
                prediction_horizon=self.forecast_horizon,
                model_type="efficiency_random_forest",
                feature_importance=self.feature_importance,
                metadata={
                    'inference_time_ms': inference_time,
                    'model_accuracy': self.last_accuracy
                }
            )
            
            self.metrics_collector.record_timer('efficiency_forecaster_inference_ms', inference_time)
            self.metrics_collector.increment_counter('efficiency_predictions_total')
            
            return result
            
        except Exception as e:
            logger.error("Efficiency prediction failed", error=str(e))
            return None
    
    async def _prepare_efficiency_data(self, data: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare efficiency training data"""
        try:
            df = pd.DataFrame(data)
            
            if len(df) < 100:
                return None, None
            
            # Calculate efficiency
            df['efficiency'] = df['hashRate'] / df['power'].replace(0, np.nan)
            df['efficiency'] = df['efficiency'].fillna(df['efficiency'].median())
            
            # Create features
            features = []
            targets = []
            
            window_size = 20
            
            for i in range(window_size, len(df)):
                # Features from recent window
                window_data = df.iloc[i-window_size:i]
                
                feature_vector = [
                    # Current values
                    df.iloc[i]['temp'],
                    df.iloc[i]['power'],
                    df.iloc[i]['frequency'],
                    df.iloc[i]['voltage'],
                    
                    # Statistical features over window
                    window_data['efficiency'].mean(),
                    window_data['efficiency'].std(),
                    window_data['temp'].mean(),
                    window_data['temp'].std(),
                    window_data['power'].mean(),
                    
                    # Trend features
                    np.polyfit(range(len(window_data)), window_data['efficiency'], 1)[0],
                    np.polyfit(range(len(window_data)), window_data['temp'], 1)[0],
                ]
                
                features.append(feature_vector)
                targets.append(df.iloc[i]['efficiency'])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error("Efficiency data preparation failed", error=str(e))
            return None, None
    
    async def _prepare_forecast_features(self, recent_data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare features for efficiency forecasting"""
        try:
            if len(recent_data) < 20:
                return None
            
            df = pd.DataFrame(recent_data)
            df['efficiency'] = df['hashRate'] / df['power'].replace(0, np.nan)
            df['efficiency'] = df['efficiency'].fillna(df['efficiency'].median())
            
            # Get latest values and statistics
            latest = df.iloc[-1]
            window_data = df.iloc[-20:]
            
            feature_vector = [
                # Current values
                latest['temp'],
                latest['power'],
                latest['frequency'],
                latest['voltage'],
                
                # Statistical features
                window_data['efficiency'].mean(),
                window_data['efficiency'].std(),
                window_data['temp'].mean(),
                window_data['temp'].std(),
                window_data['power'].mean(),
                
                # Trend features
                np.polyfit(range(len(window_data)), window_data['efficiency'], 1)[0],
                np.polyfit(range(len(window_data)), window_data['temp'], 1)[0],
            ]
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error("Forecast feature preparation failed", error=str(e))
            return None


class FailureDetector:
    """
    Isolation Forest model for anomaly detection and failure prediction
    
    Detects unusual patterns that may indicate hardware issues or performance degradation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Model configuration
        self.contamination = self.config.get('contamination', 0.1)  # Expected anomaly rate
        self.n_estimators = self.config.get('n_estimators', 100)
        
        # Model state
        self.model = None
        self.is_trained = False
        self.anomaly_threshold = -0.5
        
        logger.info("Failure detector initialized")
    
    async def train(self, historical_data: List[Dict[str, Any]]) -> bool:
        """Train anomaly detection model"""
        try:
            logger.info("Starting failure detector training")
            start_time = datetime.now()
            
            # Prepare features
            X = await self._prepare_anomaly_features(historical_data)
            
            if X is None or len(X) < 100:
                logger.error("Insufficient training data for failure detector")
                return False
            
            # Train Isolation Forest
            from sklearn.ensemble import IsolationForest
            
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            
            self.model.fit(X)
            
            # Set anomaly threshold based on training data
            scores = self.model.decision_function(X)
            self.anomaly_threshold = np.percentile(scores, self.contamination * 100)
            
            self.is_trained = True
            training_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics_collector.record_metric('failure_detector_training_time', training_time)
            self.metrics_collector.increment_counter('failure_detector_training_runs_total')
            
            logger.info("Failure detector training completed",
                       training_samples=len(X),
                       anomaly_threshold=self.anomaly_threshold,
                       training_time_sec=training_time)
            
            return True
            
        except Exception as e:
            logger.error("Failure detector training failed", error=str(e))
            return False
    
    async def detect_anomalies(self, recent_data: List[Dict[str, Any]]) -> Optional[PredictionResult]:
        """Detect anomalies in recent data"""
        try:
            if not self.is_trained:
                logger.warning("Failure detector not trained")
                return None
            
            # Prepare features
            features = await self._prepare_detection_features(recent_data)
            if features is None:
                return None
            
            start_time = datetime.now()
            
            # Detect anomalies
            anomaly_scores = self.model.decision_function(features)
            anomaly_predictions = (anomaly_scores < self.anomaly_threshold).astype(int)
            
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate confidence (inverse of anomaly score)
            confidence = np.clip((-anomaly_scores + 1) / 2, 0, 1)
            
            result = PredictionResult(
                predictions=anomaly_predictions,
                confidence=confidence,
                model_type="isolation_forest_anomaly",
                metadata={
                    'inference_time_ms': inference_time,
                    'anomaly_scores': anomaly_scores.tolist(),
                    'anomaly_threshold': self.anomaly_threshold
                }
            )
            
            self.metrics_collector.record_timer('failure_detector_inference_ms', inference_time)
            self.metrics_collector.increment_counter('anomaly_detections_total')
            
            if np.any(anomaly_predictions):
                self.metrics_collector.increment_counter('anomalies_detected_total')
                logger.warning(f"Anomalies detected: {np.sum(anomaly_predictions)} out of {len(anomaly_predictions)}")
            
            return result
            
        except Exception as e:
            logger.error("Anomaly detection failed", error=str(e))
            return None
    
    async def _prepare_anomaly_features(self, data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare features for anomaly detection training"""
        try:
            df = pd.DataFrame(data)
            
            if len(df) < 100:
                return None
            
            # Calculate derived metrics
            df['efficiency'] = df['hashRate'] / df['power'].replace(0, np.nan)
            df['temp_power_ratio'] = df['temp'] / df['power'].replace(0, np.nan)
            
            # Select features for anomaly detection
            features = [
                'temp', 'hashRate', 'power', 'voltage', 'frequency',
                'efficiency', 'temp_power_ratio'
            ]
            
            feature_data = df[features].fillna(method='ffill').fillna(0)
            
            return feature_data.values
            
        except Exception as e:
            logger.error("Anomaly feature preparation failed", error=str(e))
            return None
    
    async def _prepare_detection_features(self, recent_data: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Prepare features for anomaly detection"""
        try:
            df = pd.DataFrame(recent_data)
            
            if len(df) == 0:
                return None
            
            # Calculate derived metrics
            df['efficiency'] = df['hashRate'] / df['power'].replace(0, np.nan)
            df['temp_power_ratio'] = df['temp'] / df['power'].replace(0, np.nan)
            
            # Select features
            features = [
                'temp', 'hashRate', 'power', 'voltage', 'frequency',
                'efficiency', 'temp_power_ratio'
            ]
            
            feature_data = df[features].fillna(method='ffill').fillna(0)
            
            return feature_data.values
            
        except Exception as e:
            logger.error("Detection feature preparation failed", error=str(e))
            return None


class PerformanceOptimizer:
    """
    Genetic Algorithm for parameter tuning and performance optimization
    
    Evolves optimal mining parameters through genetic optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # GA parameters
        self.population_size = self.config.get('population_size', 50)
        self.generations = self.config.get('generations', 20)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        
        # Parameter bounds
        self.parameter_bounds = {
            'frequency': (400, 800),
            'voltage': (10.0, 14.0)
        }
        
        # Optimization state
        self.current_population = []
        self.best_individual = None
        self.fitness_history = []
        
        logger.info("Performance optimizer initialized")
    
    async def optimize(self, 
                      objective_function: callable,
                      initial_params: Dict[str, float]) -> Dict[str, float]:
        """
        Optimize parameters using genetic algorithm
        
        Args:
            objective_function: Function to evaluate parameter fitness
            initial_params: Starting parameter values
            
        Returns:
            Optimized parameters
        """
        try:
            logger.info("Starting genetic algorithm optimization")
            start_time = datetime.now()
            
            # Initialize population
            await self._initialize_population(initial_params)
            
            best_fitness = float('-inf')
            
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in self.current_population:
                    fitness = await objective_function(individual)
                    fitness_scores.append(fitness)
                
                # Track best individual
                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > best_fitness:
                    best_fitness = fitness_scores[best_idx]
                    self.best_individual = self.current_population[best_idx].copy()
                
                self.fitness_history.append(best_fitness)
                
                # Selection, crossover, and mutation
                new_population = await self._evolve_population(fitness_scores)
                self.current_population = new_population
                
                if generation % 5 == 0:
                    logger.debug(f"Generation {generation}, best fitness: {best_fitness:.4f}")
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics_collector.record_metric('ga_optimization_time', optimization_time)
            self.metrics_collector.record_metric('ga_best_fitness', best_fitness)
            self.metrics_collector.increment_counter('ga_optimizations_total')
            
            logger.info("Genetic algorithm optimization completed",
                       best_fitness=best_fitness,
                       optimization_time_sec=optimization_time)
            
            return self.best_individual if self.best_individual else initial_params
            
        except Exception as e:
            logger.error("Genetic algorithm optimization failed", error=str(e))
            return initial_params
    
    async def _initialize_population(self, initial_params: Dict[str, float]):
        """Initialize random population around initial parameters"""
        self.current_population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, value in initial_params.items():
                if param in self.parameter_bounds:
                    min_val, max_val = self.parameter_bounds[param]
                    # Add random variation around initial value
                    variation = np.random.uniform(-0.1, 0.1) * (max_val - min_val)
                    new_value = np.clip(value + variation, min_val, max_val)
                    individual[param] = new_value
                else:
                    individual[param] = value
            
            self.current_population.append(individual)
    
    async def _evolve_population(self, fitness_scores: List[float]) -> List[Dict[str, float]]:
        """Evolve population through selection, crossover, and mutation"""
        fitness_array = np.array(fitness_scores)
        
        # Tournament selection
        new_population = []
        
        for _ in range(self.population_size):
            if np.random.random() < self.crossover_rate and len(new_population) > 0:
                # Crossover
                parent1 = self._tournament_selection(fitness_array)
                parent2 = self._tournament_selection(fitness_array)
                child = self._crossover(parent1, parent2)
            else:
                # Selection only
                child = self._tournament_selection(fitness_array)
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, fitness_scores: np.ndarray) -> Dict[str, float]:
        """Select individual using tournament selection"""
        tournament_size = 3
        tournament_indices = np.random.choice(len(fitness_scores), tournament_size, replace=False)
        best_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
        return self.current_population[best_idx].copy()
    
    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Create offspring through crossover"""
        child = {}
        for param in parent1:
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def _mutate(self, individual: Dict[str, float]) -> Dict[str, float]:
        """Mutate individual parameters"""
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                mutation_strength = 0.05 * (max_val - min_val)
                mutation = np.random.normal(0, mutation_strength)
                mutated[param] = np.clip(value + mutation, min_val, max_val)
        
        return mutated


async def create_predictive_models(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Factory function to create all predictive models"""
    models = {
        'temperature_predictor': TemperaturePredictor(config.get('temperature_predictor', {}) if config else {}),
        'efficiency_forecaster': EfficiencyForecaster(config.get('efficiency_forecaster', {}) if config else {}),
        'failure_detector': FailureDetector(config.get('failure_detector', {}) if config else {}),
        'performance_optimizer': PerformanceOptimizer(config.get('performance_optimizer', {}) if config else {})
    }
    
    logger.info("Predictive models created", models=list(models.keys()))
    return models