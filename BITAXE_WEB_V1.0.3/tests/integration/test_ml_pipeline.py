"""
Integration tests for ML Pipeline

End-to-end testing of the machine learning optimization pipeline including
data ingestion, feature engineering, model inference, and optimization decisions.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from ml_engine.core.optimization_engine import OptimizationEngine
from ml_engine.core.ml_model_service import MLModelService
from ml_engine.data.feature_engineering import FeatureEngineer
from ml_engine.models.rl_agent import FrequencyVoltageOptimizer
from ml_engine.core.weather_service import WeatherService, WeatherData
from ml_engine.utils.ml_config import MLConfigManager
from services.miner_service import MinerService
from monitoring.metrics_collector import MetricsCollector


class TestMLPipelineIntegration:
    """Test complete ML pipeline integration"""
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_complete_optimization_pipeline(self, sample_miner_telemetry, sample_weather_data, tmp_path):
        """Test complete end-to-end optimization pipeline"""
        # Setup components
        ml_config = {
            'enable_rl_optimization': True,
            'enable_predictive_analytics': True,
            'optimization_interval': 60,
            'target_efficiency': 120.0,
            'target_temperature': 70.0,
            'max_temperature': 85.0,
            'models_dir': str(tmp_path / 'models')
        }
        
        # Create services
        feature_engineer = FeatureEngineer({
            'window_sizes': [5, 15, 30],
            'statistical_features': ['mean', 'std', 'min', 'max']
        })
        
        model_service = MLModelService(model_dir=str(tmp_path / 'models'))
        await model_service.start()
        
        optimization_engine = OptimizationEngine(ml_config)
        
        # Mock weather service
        weather_service = Mock(spec=WeatherService)
        weather_service.get_current_weather = AsyncMock(return_value=WeatherData(
            temperature=sample_weather_data['temperature'],
            humidity=sample_weather_data['humidity'],
            pressure=sample_weather_data['pressure'],
            wind_speed=sample_weather_data['wind_speed'],
            wind_direction=sample_weather_data['wind_direction'],
            cloud_cover=sample_weather_data['cloud_cover'],
            visibility=sample_weather_data['visibility'],
            uv_index=sample_weather_data['uv_index'],
            timestamp=datetime.now(),
            location="Test Location"
        ))
        
        # Step 1: Feature Engineering
        feature_set = await feature_engineer.engineer_features(
            sample_miner_telemetry,
            weather_data=[sample_weather_data],
            target_column='temp'
        )
        
        assert feature_set.features.shape[0] > 0
        assert len(feature_set.feature_names) > 0
        
        # Step 2: Model Prediction (mock successful prediction)
        with patch.object(model_service, 'predict') as mock_predict:
            mock_predict.return_value = Mock(
                predictions=np.array([72.5]),  # Predicted temperature
                confidence=np.array([0.85]),
                model_id="temp_predictor",
                inference_time_ms=15.0
            )
            
            # Step 3: Optimization Decision
            miner_context = {
                'telemetry': sample_miner_telemetry[-1],  # Latest telemetry
                'miner_ip': '192.168.1.100',
                'timestamp': datetime.now(),
                'weather': sample_weather_data
            }
            
            optimization_result = await optimization_engine.optimize_miner(
                '192.168.1.100',
                miner_context
            )
            
            # Verify optimization result
            assert optimization_result is not None
            assert 'status' in optimization_result
            assert 'optimizations' in optimization_result
            
            if optimization_result['status'] == 'success':
                optimizations = optimization_result['optimizations']
                assert len(optimizations) > 0
                
                # Check optimization structure
                opt = optimizations[0]
                assert 'type' in opt
                assert 'adjustments' in opt
                assert 'confidence' in opt
        
        await model_service.stop()
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_rl_optimization_workflow(self, sample_miner_telemetry, sample_weather_data):
        """Test reinforcement learning optimization workflow"""
        from ml_engine.models.rl_agent import RLState, RLAction
        
        # Setup RL optimizer
        config = {
            'agent_config': {
                'learning_rate': 0.001,
                'clip_epsilon': 0.2
            },
            'safety_enabled': True,
            'max_temp_threshold': 85.0
        }
        
        rl_optimizer = FrequencyVoltageOptimizer(config)
        
        # Test state creation from real telemetry
        latest_telemetry = sample_miner_telemetry[-1]
        state = RLState.from_miner_data(latest_telemetry, sample_weather_data)
        
        assert isinstance(state, RLState)
        assert state.temperature > 0
        assert state.hashrate >= 0
        
        # Test action selection
        action = await rl_optimizer.agent.select_action(state, exploration=True)
        assert isinstance(action, RLAction)
        
        # Test safety validation
        is_safe = rl_optimizer._safety_check(state)
        if state.temperature > 85.0:
            assert is_safe is False
        else:
            assert is_safe is True
        
        # Test action application
        current_miner_config = {
            'frequency': latest_telemetry['frequency'],
            'voltage': latest_telemetry['voltage']
        }
        
        new_config = action.apply_to_miner_config(current_miner_config)
        
        # Verify safety bounds
        assert 400 <= new_config['frequency'] <= 800
        assert 10.0 <= new_config['voltage'] <= 14.0
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_weather_adaptation_integration(self, sample_miner_telemetry):
        """Test weather adaptation integration with optimization"""
        from ml_engine.core.weather_service import WeatherService
        
        config = {
            'location': 'Test,US',
            'enable_local_sensors': True,
            'openweathermap_api_key': 'test_key'
        }
        
        weather_service = WeatherService(config)
        
        # Test different weather conditions
        hot_weather = WeatherData(
            temperature=35.0,  # Hot day
            humidity=80.0,     # High humidity
            pressure=1013.25,
            wind_speed=2.0,    # Low wind
            wind_direction=180.0,
            cloud_cover=60.0,
            visibility=8.0,
            uv_index=8.0,
            timestamp=datetime.now(),
            location="Test Location"
        )
        
        weather_service.current_weather = hot_weather
        
        # Test cooling strategy for hot conditions
        strategy = await weather_service.get_cooling_strategy(current_temp=75.0)
        
        assert strategy['strategy'] in ['hot_humid', 'aggressive_cooling']
        assert strategy['adjustments'].get('reduce_frequency', False) is True
        
        # Test optimal weather conditions
        optimal_weather = WeatherData(
            temperature=20.0,  # Cool day
            humidity=40.0,     # Low humidity
            pressure=1013.25,
            wind_speed=5.0,    # Good breeze
            wind_direction=180.0,
            cloud_cover=20.0,
            visibility=10.0,
            uv_index=3.0,
            timestamp=datetime.now(),
            location="Test Location"
        )
        
        weather_service.current_weather = optimal_weather
        
        optimal_strategy = await weather_service.get_cooling_strategy(current_temp=65.0)
        
        assert optimal_strategy['strategy'] == 'optimal'
        assert not optimal_strategy['adjustments'].get('reduce_frequency', True)
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_predictive_analytics_pipeline(self, sample_miner_telemetry, tmp_path):
        """Test predictive analytics pipeline integration"""
        # Setup feature engineering
        feature_engineer = FeatureEngineer({
            'window_sizes': [5, 15, 30],
            'statistical_features': ['mean', 'std', 'min', 'max'],
            'scaling_method': 'standard'
        })
        
        # Process telemetry data
        feature_set = await feature_engineer.engineer_features(
            sample_miner_telemetry,
            target_column='temp'
        )
        
        assert feature_set.features.shape[0] > 0
        
        # Test feature scaling
        scaled_set = await feature_engineer.scale_features(feature_set, fit_scalers=True)
        
        # Verify scaling worked
        feature_means = np.mean(scaled_set.features, axis=0)
        feature_stds = np.std(scaled_set.features, axis=0)
        
        # Features should be roughly normalized
        assert np.all(np.abs(feature_means) < 2.0)
        assert np.all(feature_stds > 0.1)
        
        # Test temporal features
        temporal_features = await feature_engineer.generate_temporal_features(
            sample_miner_telemetry
        )
        
        assert 'hour_of_day' in temporal_features
        assert 'day_of_week' in temporal_features
        assert len(temporal_features['hour_of_day']) == len(sample_miner_telemetry)
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_optimization_validation_pipeline(self, sample_miner_telemetry):
        """Test optimization validation and safety pipeline"""
        config = {
            'safety': {
                'max_temperature': 85.0,
                'min_efficiency': 50.0,
                'max_power': 200.0,
                'max_frequency_change': 100,  # MHz
                'max_voltage_change': 1.0     # V
            },
            'validation': {
                'enabled': True,
                'confidence_threshold': 0.7
            }
        }
        
        optimization_engine = OptimizationEngine(config)
        
        # Test with safe telemetry
        safe_telemetry = {
            'temp': 70.0,
            'power': 12.0,
            'hashRate': 480.0,
            'voltage': 12.0,
            'frequency': 600
        }
        
        is_safe = await optimization_engine._safety_check('192.168.1.100', safe_telemetry)
        assert is_safe is True
        
        # Test optimization validation  
        current_config = {'frequency': 600, 'voltage': 12.0}
        
        # Valid optimization (small changes)
        valid_optimized = {'frequency': 650, 'voltage': 12.1}
        is_valid = await optimization_engine._validate_optimization(
            current_config, valid_optimized, safe_telemetry
        )
        assert is_valid is True
        
        # Invalid optimization (large changes)
        invalid_optimized = {'frequency': 750, 'voltage': 13.5}  # Too large
        is_valid = await optimization_engine._validate_optimization(
            current_config, invalid_optimized, safe_telemetry
        )
        assert is_valid is False
        
        # Test with unsafe telemetry
        unsafe_telemetry = {
            'temp': 90.0,  # Too hot
            'power': 12.0,
            'hashRate': 480.0,
            'voltage': 12.0,
            'frequency': 600
        }
        
        is_safe = await optimization_engine._safety_check('192.168.1.100', unsafe_telemetry)
        assert is_safe is False
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_multi_miner_optimization_coordination(self, sample_miner_telemetry):
        """Test coordination of optimization across multiple miners"""
        config = {
            'coordination': {
                'enabled': True,
                'global_power_limit': 100.0,  # Watts
                'load_balancing': True
            },
            'optimization_interval': 60
        }
        
        optimization_engine = OptimizationEngine(config)
        
        # Simulate multiple miners
        miners = {
            '192.168.1.100': {
                'telemetry': {
                    'temp': 65.0,
                    'power': 12.0,
                    'hashRate': 480.0,
                    'voltage': 12.0,
                    'frequency': 600
                }
            },
            '192.168.1.101': {
                'telemetry': {
                    'temp': 75.0,  # Running hotter
                    'power': 15.0,  # Using more power
                    'hashRate': 520.0,
                    'voltage': 12.5,
                    'frequency': 650
                }
            },
            '192.168.1.102': {
                'telemetry': {
                    'temp': 60.0,  # Running cool
                    'power': 10.0,  # Using less power
                    'hashRate': 400.0,
                    'voltage': 11.5,
                    'frequency': 550
                }
            }
        }
        
        # Mock global optimization method
        with patch.object(optimization_engine, '_coordinate_global_optimization') as mock_coord:
            mock_coord.return_value = {
                '192.168.1.100': {'frequency': 620, 'voltage': 12.1},  # Slight increase
                '192.168.1.101': {'frequency': 600, 'voltage': 12.0},  # Reduce to cool down
                '192.168.1.102': {'frequency': 580, 'voltage': 11.8}   # Increase to balance
            }
            
            # Test coordinated optimization
            results = {}
            for miner_ip, miner_data in miners.items():
                context = {
                    'telemetry': miner_data['telemetry'],
                    'miner_ip': miner_ip,
                    'timestamp': datetime.now(),
                    'global_context': miners
                }
                
                result = await optimization_engine.optimize_miner(miner_ip, context)
                results[miner_ip] = result
            
            # Verify coordination was called
            mock_coord.assert_called()
            
            # Verify all miners got optimization results
            assert len(results) == 3
            for miner_ip, result in results.items():
                assert result is not None
                assert 'status' in result


class TestMLDataPipeline:
    """Test ML data processing pipeline"""
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_data_ingestion_pipeline(self, sample_miner_telemetry, sample_weather_data):
        """Test data ingestion and preprocessing pipeline"""
        from ml_engine.data.data_collector import DataCollector
        from ml_engine.data.data_preprocessor import DataPreprocessor
        
        # Mock data collector
        data_collector = Mock(spec=DataCollector)
        data_collector.collect_miner_data = AsyncMock(return_value=sample_miner_telemetry)
        data_collector.collect_weather_data = AsyncMock(return_value=[sample_weather_data])
        
        # Test data collection
        miner_data = await data_collector.collect_miner_data(['192.168.1.100'])
        weather_data = await data_collector.collect_weather_data()
        
        assert len(miner_data) > 0
        assert len(weather_data) > 0
        
        # Test data preprocessing
        preprocessor = DataPreprocessor({
            'outlier_detection': True,
            'missing_value_strategy': 'interpolate',
            'validation': True
        })
        
        # Mock preprocessing methods
        with patch.object(preprocessor, 'clean_data') as mock_clean, \
             patch.object(preprocessor, 'validate_data') as mock_validate:
            
            mock_clean.return_value = miner_data
            mock_validate.return_value = True
            
            cleaned_data = await preprocessor.clean_data(miner_data)
            is_valid = await preprocessor.validate_data(cleaned_data)
            
            assert cleaned_data is not None
            assert is_valid is True
            mock_clean.assert_called_once()
            mock_validate.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.database
    async def test_data_persistence_pipeline(self, sample_miner_telemetry, test_database):
        """Test data persistence and retrieval pipeline"""
        from ml_engine.data.data_storage import MLDataStorage
        
        # Mock storage service
        storage = Mock(spec=MLDataStorage)
        storage.store_training_data = AsyncMock(return_value=True)
        storage.retrieve_training_data = AsyncMock(return_value=sample_miner_telemetry)
        storage.store_model_predictions = AsyncMock(return_value=True)
        
        # Test storing training data
        success = await storage.store_training_data(
            data_type='miner_telemetry',
            data=sample_miner_telemetry,
            metadata={'source': 'test', 'timestamp': datetime.now()}
        )
        assert success is True
        
        # Test retrieving training data
        retrieved_data = await storage.retrieve_training_data(
            data_type='miner_telemetry',
            time_range={'start': datetime.now() - timedelta(hours=2), 'end': datetime.now()}
        )
        assert retrieved_data is not None
        assert len(retrieved_data) > 0
        
        # Test storing predictions
        predictions = {
            '192.168.1.100': {
                'temperature_prediction': 72.5,
                'confidence': 0.85,
                'timestamp': datetime.now()
            }
        }
        
        success = await storage.store_model_predictions(predictions)
        assert success is True
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_real_time_data_streaming(self, sample_miner_telemetry):
        """Test real-time data streaming and processing"""
        from ml_engine.data.stream_processor import StreamProcessor
        
        # Mock stream processor
        processor = Mock(spec=StreamProcessor)
        processor.start_streaming = AsyncMock()
        processor.stop_streaming = AsyncMock()
        processor.process_stream_batch = AsyncMock()
        
        # Simulate streaming data
        stream_queue = asyncio.Queue()
        
        # Add data to stream
        for data_point in sample_miner_telemetry[:10]:  # First 10 points
            await stream_queue.put(data_point)
        
        # Mock processing
        batch_data = []
        while not stream_queue.empty():
            data = await stream_queue.get()
            batch_data.append(data)
        
        await processor.process_stream_batch(batch_data)
        
        # Verify processing
        processor.process_stream_batch.assert_called_once_with(batch_data)
        assert len(batch_data) == 10


class TestMLModelIntegration:
    """Test ML model integration and lifecycle"""
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_model_training_pipeline(self, sample_training_data, tmp_path):
        """Test model training and deployment pipeline"""
        from ml_engine.training.model_trainer import ModelTrainer
        from ml_engine.core.ml_model_service import MLModelService
        
        X, y = sample_training_data
        
        # Mock model trainer
        trainer = Mock(spec=ModelTrainer)
        trainer.train_model = AsyncMock(return_value={
            'model_id': 'test_model_v1',
            'performance': {'mse': 0.15, 'r2': 0.85},
            'model_path': str(tmp_path / 'test_model.pkl')
        })
        
        # Test model training
        training_result = await trainer.train_model(
            model_type='temperature_predictor',
            features=X,
            targets=y,
            config={'algorithm': 'random_forest', 'n_estimators': 100}
        )
        
        assert training_result['model_id'] == 'test_model_v1'
        assert training_result['performance']['r2'] > 0.8
        
        # Test model deployment
        model_service = MLModelService(model_dir=str(tmp_path))
        await model_service.start()
        
        # Mock model loading
        with patch.object(model_service, 'load_model') as mock_load:
            mock_load.return_value = True
            
            success = await model_service.load_model('test_model_v1')
            assert success is True
            mock_load.assert_called_once_with('test_model_v1')
        
        await model_service.stop()
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_model_validation_pipeline(self, sample_training_data, tmp_path):
        """Test model validation and performance monitoring"""
        from ml_engine.validation.model_validator import ModelValidator
        
        X, y = sample_training_data
        
        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Mock model validator
        validator = Mock(spec=ModelValidator)
        validator.validate_model = AsyncMock(return_value={
            'metrics': {
                'mse': 0.12,
                'mae': 0.25,
                'r2': 0.88,
                'mape': 5.2
            },
            'validation_passed': True,
            'quality_score': 0.88
        })
        
        # Test model validation
        validation_result = await validator.validate_model(
            model_id='test_model_v1',
            validation_data=(X_val, y_val),
            metrics=['mse', 'mae', 'r2', 'mape']
        )
        
        assert validation_result['validation_passed'] is True
        assert validation_result['quality_score'] > 0.8
        assert 'mse' in validation_result['metrics']
    
    @pytest.mark.integration
    @pytest.mark.ml
    async def test_model_monitoring_pipeline(self, sample_miner_telemetry, tmp_path):
        """Test model performance monitoring and drift detection"""
        from ml_engine.monitoring.model_monitor import ModelMonitor
        
        # Mock model monitor
        monitor = Mock(spec=ModelMonitor)
        monitor.detect_drift = AsyncMock(return_value={
            'drift_detected': False,
            'drift_score': 0.15,
            'threshold': 0.3,
            'drift_type': None
        })
        
        monitor.track_prediction_quality = AsyncMock(return_value={
            'accuracy_score': 0.87,
            'prediction_latency_ms': 23.5,
            'error_rate': 0.03
        })
        
        # Test drift detection
        recent_data = sample_miner_telemetry[-50:]  # Recent 50 samples
        drift_result = await monitor.detect_drift(
            model_id='test_model_v1',
            reference_data=sample_miner_telemetry[:-50],
            current_data=recent_data
        )
        
        assert 'drift_detected' in drift_result
        assert 'drift_score' in drift_result
        
        # Test prediction quality tracking
        quality_result = await monitor.track_prediction_quality(
            model_id='test_model_v1',
            time_window='1h'
        )
        
        assert 'accuracy_score' in quality_result
        assert 'prediction_latency_ms' in quality_result


class TestMLPerformanceIntegration:
    """Test ML pipeline performance and scalability"""
    
    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.performance
    async def test_pipeline_throughput(self, performance_test_data, performance_metrics):
        """Test ML pipeline throughput with large dataset"""
        from ml_engine.data.feature_engineering import FeatureEngineer
        
        feature_engineer = FeatureEngineer({
            'window_sizes': [5, 15],  # Smaller windows for performance
            'statistical_features': ['mean', 'std']
        })
        
        # Test processing throughput
        performance_metrics.start_timer('pipeline_throughput')
        
        # Process data in batches
        batch_size = 1000
        total_processed = 0
        
        for i in range(0, len(performance_test_data), batch_size):
            batch = performance_test_data[i:i+batch_size]
            
            feature_set = await feature_engineer.engineer_features(
                batch,
                target_column='temp'
            )
            
            total_processed += len(batch)
            
            # Verify batch processing
            assert feature_set.features.shape[0] == len(batch)
        
        performance_metrics.end_timer('pipeline_throughput')
        
        # Should process 10k samples in reasonable time (< 30 seconds)
        performance_metrics.assert_performance('pipeline_throughput', 30.0)
        
        assert total_processed == len(performance_test_data)
    
    @pytest.mark.integration
    @pytest.mark.ml
    @pytest.mark.performance
    async def test_concurrent_optimization(self, performance_metrics):
        """Test concurrent optimization of multiple miners"""
        from ml_engine.core.optimization_engine import OptimizationEngine
        
        config = {
            'concurrent_miners': 10,
            'optimization_timeout': 5.0
        }
        
        optimization_engine = OptimizationEngine(config)
        
        # Create multiple miner contexts
        miner_contexts = {}
        for i in range(10):
            miner_ip = f'192.168.1.{100 + i}'
            miner_contexts[miner_ip] = {
                'telemetry': {
                    'temp': 65 + i * 2,
                    'power': 12 + i * 0.5,
                    'hashRate': 480 + i * 10,
                    'voltage': 12.0,
                    'frequency': 600 + i * 10
                },
                'miner_ip': miner_ip,
                'timestamp': datetime.now()
            }
        
        # Mock optimization method
        with patch.object(optimization_engine, 'optimize_miner') as mock_optimize:
            mock_optimize.return_value = {
                'status': 'success', 
                'optimizations': [{'type': 'frequency', 'adjustments': {'frequency': 650}}]
            }
            
            # Test concurrent optimization
            performance_metrics.start_timer('concurrent_optimization')
            
            tasks = []
            for miner_ip, context in miner_contexts.items():
                task = asyncio.create_task(
                    optimization_engine.optimize_miner(miner_ip, context)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            performance_metrics.end_timer('concurrent_optimization')
            
            # Should complete within timeout
            performance_metrics.assert_performance('concurrent_optimization', 10.0)
            
            # Verify all optimizations completed
            assert len(results) == 10
            for result in results:
                assert not isinstance(result, Exception)
                assert result['status'] == 'success'