"""
Unit tests for ML Engine components

Comprehensive testing of machine learning optimization components including
feature engineering, model services, and optimization algorithms.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ml_engine.data.feature_engineering import FeatureEngineer, FeatureSet
from ml_engine.core.ml_model_service import MLModelService, ModelInfo, PredictionResult
from ml_engine.core.optimization_engine import OptimizationEngine, OptimizationResult
from ml_engine.models.rl_agent import FrequencyVoltageOptimizer, RLState, RLAction, RewardFunction
from ml_engine.core.weather_service import WeatherService, WeatherData
from ml_engine.utils.ml_config import MLConfigManager, MLEngineConfig


class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_feature_engineering_basic(self, feature_engineer, sample_miner_telemetry):
        """Test basic feature engineering"""
        # Test feature engineering with sample data
        feature_set = await feature_engineer.engineer_features(
            sample_miner_telemetry, 
            target_column='temp'
        )
        
        assert isinstance(feature_set, FeatureSet)
        assert feature_set.features.shape[0] > 0
        assert len(feature_set.feature_names) > 0
        assert feature_set.target is not None
        assert len(feature_set.target) == feature_set.features.shape[0]
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_feature_scaling(self, feature_engineer, sample_miner_telemetry):
        """Test feature scaling functionality"""
        # Generate features
        feature_set = await feature_engineer.engineer_features(
            sample_miner_telemetry,
            target_column='temp'
        )
        
        # Scale features
        scaled_set = await feature_engineer.scale_features(feature_set, fit_scalers=True)
        
        assert scaled_set.features.shape == feature_set.features.shape
        assert scaled_set.metadata.get('scaled') is True
        
        # Check that features are roughly normalized (mean ~0, std ~1)
        feature_means = np.mean(scaled_set.features, axis=0)
        feature_stds = np.std(scaled_set.features, axis=0)
        
        # Allow some tolerance for small datasets
        assert np.all(np.abs(feature_means) < 2.0)
        assert np.all(feature_stds > 0.1)  # Not all zeros
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_empty_data_handling(self, feature_engineer):
        """Test handling of empty data"""
        empty_data = []
        
        feature_set = await feature_engineer.engineer_features(empty_data)
        
        assert feature_set.features.size == 0
        assert len(feature_set.feature_names) == 0
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_weather_features(self, feature_engineer, sample_miner_telemetry, sample_weather_data):
        """Test weather feature integration"""
        feature_set = await feature_engineer.engineer_features(
            sample_miner_telemetry,
            weather_data=[sample_weather_data],
            target_column='temp'
        )
        
        # Check that weather features are included
        weather_feature_names = [name for name in feature_set.feature_names if 'weather' in name]
        assert len(weather_feature_names) > 0
        
        # Check for heat index feature
        heat_index_features = [name for name in feature_set.feature_names if 'heat_index' in name]
        assert len(heat_index_features) > 0


class TestMLModelService:
    """Test ML model service functionality"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_model_service_initialization(self, tmp_path):
        """Test ML model service initialization"""
        service = MLModelService(model_dir=str(tmp_path))
        await service.start()
        
        assert service.model_dir == tmp_path
        assert service.registry is not None
        assert service.is_running is False  # Not actually running in test
        
        await service.stop()
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_model_loading_nonexistent(self, tmp_path):
        """Test loading non-existent model"""
        service = MLModelService(model_dir=str(tmp_path))
        await service.start()
        
        # Try to load non-existent model
        success = await service.load_model("nonexistent_model")
        assert success is False
        
        await service.stop()
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_prediction_without_model(self, tmp_path):
        """Test prediction without loaded model"""
        service = MLModelService(model_dir=str(tmp_path))
        await service.start()
        
        # Try prediction without loaded model
        features = np.random.randn(10, 5)
        result = await service.predict("nonexistent_model", features)
        
        assert result is None
        
        await service.stop()
    
    @pytest.mark.unit
    @pytest.mark.ml  
    async def test_model_validation(self, tmp_path):
        """Test model validation functionality"""
        service = MLModelService(model_dir=str(tmp_path))
        await service.start()
        
        # Test feature validation
        model_info = ModelInfo(
            model_id="test_model",
            model_type="sklearn",
            version="1.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            performance_metrics={},
            feature_names=['feature_0', 'feature_1', 'feature_2'],
            model_config={},
            file_path="test.pkl"
        )
        
        # Valid features
        valid_features = np.random.randn(10, 3)
        is_valid = await service._validate_features(valid_features, model_info)
        assert is_valid is True
        
        # Invalid feature count
        invalid_features = np.random.randn(10, 5)  # Wrong number of features
        is_valid = await service._validate_features(invalid_features, model_info)
        assert is_valid is False
        
        # Features with NaN
        nan_features = np.random.randn(10, 3)
        nan_features[0, 0] = np.nan
        is_valid = await service._validate_features(nan_features, model_info)
        assert is_valid is False
        
        await service.stop()


class TestRLAgent:
    """Test reinforcement learning agent"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_rl_state_creation(self, sample_miner_telemetry, sample_weather_data):
        """Test RL state creation from miner data"""
        miner_data = sample_miner_telemetry[0]
        
        state = RLState.from_miner_data(miner_data, sample_weather_data)
        
        assert isinstance(state, RLState)
        assert state.temperature > 0
        assert state.hashrate >= 0
        assert state.power > 0
        assert state.efficiency >= 0
        assert state.weather_temp == sample_weather_data['temperature']
        assert state.weather_humidity == sample_weather_data['humidity']
        
        # Test array conversion
        state_array = state.to_array()
        assert isinstance(state_array, np.ndarray)
        assert len(state_array) == 10  # Expected number of state features
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_rl_action_creation(self):
        """Test RL action creation and application"""
        action = RLAction(frequency_delta=50.0, voltage_delta=0.1)
        
        assert action.frequency_delta == 50.0
        assert action.voltage_delta == 0.1
        
        # Test array conversion
        action_array = action.to_array()
        assert isinstance(action_array, np.ndarray)
        assert len(action_array) == 2
        
        # Test from array
        recreated_action = RLAction.from_array(action_array)
        assert recreated_action.frequency_delta == action.frequency_delta
        assert recreated_action.voltage_delta == action.voltage_delta
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_action_application(self):
        """Test applying action to miner configuration"""
        action = RLAction(frequency_delta=50.0, voltage_delta=0.1)
        current_config = {
            'frequency': 600,
            'voltage': 12.0,
            'other_setting': 'unchanged'
        }
        
        new_config = action.apply_to_miner_config(current_config)
        
        assert new_config['frequency'] == 650  # 600 + 50
        assert new_config['voltage'] == 12.1  # 12.0 + 0.1
        assert new_config['other_setting'] == 'unchanged'
        
        # Test safety limits
        extreme_action = RLAction(frequency_delta=1000.0, voltage_delta=10.0)
        safe_config = extreme_action.apply_to_miner_config(current_config)
        
        assert 400 <= safe_config['frequency'] <= 800  # Safety bounds
        assert 10.0 <= safe_config['voltage'] <= 14.0  # Safety bounds
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_reward_function(self):
        """Test reward function calculation"""
        reward_func = RewardFunction()
        
        # Create test states
        prev_state = RLState(
            temperature=70.0,
            hashrate=480.0,
            power=12.0,
            efficiency=40.0,
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        # Improved state (better efficiency, lower temperature)
        improved_state = RLState(
            temperature=65.0,  # Lower temperature
            hashrate=500.0,   # Higher hashrate
            power=12.0,       # Same power
            efficiency=41.7,  # Better efficiency
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        action = RLAction(frequency_delta=0.0, voltage_delta=0.0)
        
        reward = reward_func.calculate_reward(prev_state, action, improved_state)
        
        # Should be positive reward for improvement
        assert reward > 0
        
        # Test with degraded state
        degraded_state = RLState(
            temperature=85.0,  # Higher temperature  
            hashrate=400.0,   # Lower hashrate
            power=15.0,       # Higher power
            efficiency=26.7,  # Worse efficiency
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=0.5
        )
        
        penalty_reward = reward_func.calculate_reward(prev_state, action, degraded_state)
        
        # Should be negative reward for degradation
        assert penalty_reward < 0
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_rl_optimizer_initialization(self):
        """Test RL optimizer initialization"""
        config = {
            'agent_config': {
                'learning_rate': 0.001,
                'clip_epsilon': 0.2
            },
            'safety_enabled': True,
            'max_temp_threshold': 85.0
        }
        
        optimizer = FrequencyVoltageOptimizer(config)
        
        assert optimizer.agent is not None
        assert optimizer.safety_enabled is True
        assert optimizer.max_temp_threshold == 85.0
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_rl_optimizer_safety_check(self):
        """Test RL optimizer safety checks"""
        optimizer = FrequencyVoltageOptimizer({'safety_enabled': True})
        
        # Safe state
        safe_state = RLState(
            temperature=65.0,
            hashrate=480.0,
            power=12.0,
            efficiency=40.0,
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        assert optimizer._safety_check(safe_state) is True
        
        # Unsafe state (high temperature)
        unsafe_state = RLState(
            temperature=95.0,  # Too high
            hashrate=480.0,
            power=12.0,
            efficiency=40.0,
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        assert optimizer._safety_check(unsafe_state) is False
        
        # Unsafe state (low efficiency)
        low_eff_state = RLState(
            temperature=65.0,
            hashrate=480.0,
            power=12.0,
            efficiency=30.0,  # Too low
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        assert optimizer._safety_check(low_eff_state) is False


class TestWeatherService:
    """Test weather service functionality"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_weather_data_creation(self, mock_openweathermap_response):
        """Test weather data creation from API response"""
        weather_data = WeatherData.from_openweather_response(
            mock_openweathermap_response, 
            "New York"
        )
        
        assert isinstance(weather_data, WeatherData)
        assert weather_data.temperature == mock_openweathermap_response['main']['temp']
        assert weather_data.humidity == mock_openweathermap_response['main']['humidity']
        assert weather_data.pressure == mock_openweathermap_response['main']['pressure']
        assert weather_data.location == "New York"
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_weather_service_initialization(self):
        """Test weather service initialization"""
        config = {
            'location': 'New York,US',
            'openweathermap_api_key': 'test_key',
            'update_interval': 300
        }
        
        service = WeatherService(config)
        
        assert service.location == 'New York,US'
        assert service.update_interval == 300
        assert len(service.clients) > 0  # Should have local sensor client at minimum
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_cooling_strategy_selection(self):
        """Test cooling strategy selection logic"""
        config = {'enable_local_sensors': True}
        service = WeatherService(config)
        
        # Test optimal conditions
        service.current_weather = WeatherData(
            temperature=20.0,  # Cool
            humidity=40.0,     # Low humidity
            pressure=1013.25,
            wind_speed=5.0,
            wind_direction=180.0,
            cloud_cover=20.0,
            visibility=10.0,
            uv_index=3.0,
            timestamp=datetime.now(),
            location="Test"
        )
        
        strategy = await service.get_cooling_strategy(current_temp=65.0)
        
        assert strategy['strategy'] == 'optimal'
        assert not strategy['adjustments'].get('reduce_frequency', True)
        
        # Test hot and humid conditions
        service.current_weather = WeatherData(
            temperature=35.0,  # Hot
            humidity=80.0,     # High humidity
            pressure=1013.25,
            wind_speed=2.0,
            wind_direction=180.0,
            cloud_cover=60.0,
            visibility=8.0,
            uv_index=8.0,
            timestamp=datetime.now(),
            location="Test"
        )
        
        hot_strategy = await service.get_cooling_strategy(current_temp=75.0)
        
        assert hot_strategy['strategy'] == 'hot_humid'
        assert hot_strategy['adjustments'].get('reduce_frequency', False) is True
        assert hot_strategy['adjustments'].get('frequency_reduction', 0) > 0


class TestOptimizationEngine:
    """Test optimization engine functionality"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_optimization_engine_initialization(self, ml_config):
        """Test optimization engine initialization"""
        config = {
            'optimization_interval': 300,
            'enable_rl': True,
            'enable_predictive': True,
            'safety': {
                'max_temperature': 85.0,
                'min_efficiency': 50.0
            }
        }
        
        engine = OptimizationEngine(config)
        
        assert engine.optimization_interval == 300
        assert engine.enable_rl_optimization is True
        assert engine.enable_predictive_optimization is True
        assert engine.max_temperature == 85.0
    
    @pytest.mark.unit
    @pytest.mark.ml 
    async def test_safety_checks(self):
        """Test optimization engine safety checks"""
        engine = OptimizationEngine({
            'safety': {
                'max_temperature': 85.0,
                'min_efficiency': 50.0,
                'max_power': 200.0
            }
        })
        
        # Safe telemetry
        safe_telemetry = {
            'temp': 70.0,
            'power': 12.0,
            'hashRate': 480.0,
            'voltage': 12.0,
            'frequency': 600
        }
        
        is_safe = await engine._safety_check('192.168.1.100', safe_telemetry)
        assert is_safe is True
        
        # Unsafe temperature
        hot_telemetry = {
            'temp': 90.0,  # Too hot
            'power': 12.0,
            'hashRate': 480.0,
            'voltage': 12.0,
            'frequency': 600
        }
        
        is_safe = await engine._safety_check('192.168.1.100', hot_telemetry)
        assert is_safe is False
        
        # Low efficiency
        inefficient_telemetry = {
            'temp': 70.0,
            'power': 20.0,    # High power
            'hashRate': 200.0, # Low hashrate -> low efficiency
            'voltage': 12.0,
            'frequency': 600
        }
        
        is_safe = await engine._safety_check('192.168.1.100', inefficient_telemetry)
        assert is_safe is False
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_optimization_strategy_generation(self):
        """Test optimization strategy generation"""
        engine = OptimizationEngine({
            'target_temperature': 70.0,
            'target_efficiency': 120.0
        })
        
        # Context with high temperature
        hot_context = {
            'telemetry': {
                'temp': 85.0,  # Above target
                'power': 15.0,
                'hashRate': 450.0,
                'frequency': 700,
                'voltage': 12.5
            },
            'miner_ip': '192.168.1.100',
            'timestamp': datetime.now()
        }
        
        strategies = await engine._generate_optimization_strategies(hot_context)
        
        # Should generate temperature reduction strategy
        temp_strategies = [s for s in strategies if 'temperature' in s['name']]
        assert len(temp_strategies) > 0
        
        temp_strategy = temp_strategies[0]
        assert temp_strategy['priority'] == 'high'
        assert 'frequency_reduction' in temp_strategy['adjustments']
        assert temp_strategy['adjustments']['frequency_reduction'] > 0
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_optimization_validation(self):
        """Test optimization validation"""
        engine = OptimizationEngine({})
        
        current_config = {
            'frequency': 600,
            'voltage': 12.0
        }
        
        # Valid optimization (small changes)
        valid_optimized = {
            'frequency': 650,  # +50 MHz
            'voltage': 12.1    # +0.1V
        }
        
        is_valid = await engine._validate_optimization(
            current_config, valid_optimized, {}
        )
        assert is_valid is True
        
        # Invalid optimization (large frequency change)
        invalid_optimized = {
            'frequency': 800,  # +200 MHz (too large)
            'voltage': 12.0
        }
        
        is_valid = await engine._validate_optimization(
            current_config, invalid_optimized, {}
        )
        assert is_valid is False
        
        # Invalid optimization (out of bounds)
        out_of_bounds = {
            'frequency': 1000,  # Too high
            'voltage': 16.0     # Too high
        }
        
        is_valid = await engine._validate_optimization(
            current_config, out_of_bounds, {}
        )
        assert is_valid is False


class TestMLConfig:
    """Test ML configuration management"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_ml_config_creation(self):
        """Test ML configuration creation"""
        config = MLEngineConfig(
            enable_rl_optimization=True,
            optimization_interval=300,
            target_efficiency=120.0,
            max_temperature=85.0
        )
        
        assert config.enable_rl_optimization is True
        assert config.optimization_interval == 300
        assert config.target_efficiency == 120.0
        assert config.max_temperature == 85.0
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_config_serialization(self):
        """Test configuration serialization"""
        config = MLEngineConfig(
            enable_rl_optimization=True,
            target_efficiency=120.0
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['enable_rl_optimization'] is True
        assert config_dict['target_efficiency'] == 120.0
        
        # Test from_dict
        recreated_config = MLEngineConfig.from_dict(config_dict)
        assert recreated_config.enable_rl_optimization == config.enable_rl_optimization
        assert recreated_config.target_efficiency == config.target_efficiency
    
    @pytest.mark.unit
    @pytest.mark.ml
    def test_config_manager_service_configs(self, ml_config_manager):
        """Test service-specific configuration generation"""
        # Test RL optimizer config
        rl_config = ml_config_manager.get_service_config('rl_optimizer')
        
        assert 'enabled' in rl_config
        assert 'safety' in rl_config
        assert 'targets' in rl_config
        assert 'constraints' in rl_config
        
        # Test weather service config
        weather_config = ml_config_manager.get_service_config('weather_service')
        
        assert 'enabled' in weather_config
        assert 'location' in weather_config
        assert 'openweathermap_api_key' in weather_config
        
        # Test monitoring config
        monitoring_config = ml_config_manager.get_service_config('monitoring')
        
        assert 'enabled' in monitoring_config
        assert 'performance_tracking' in monitoring_config
        assert 'drift_detection' in monitoring_config
        assert 'validation' in monitoring_config


# Performance and stress tests
class TestMLPerformance:
    """Test ML component performance"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    @pytest.mark.performance
    async def test_feature_engineering_performance(self, feature_engineer, performance_test_data, performance_metrics):
        """Test feature engineering performance with large dataset"""
        performance_metrics.start_timer('feature_engineering')
        
        feature_set = await feature_engineer.engineer_features(
            performance_test_data,
            target_column='temp'
        )
        
        performance_metrics.end_timer('feature_engineering')
        
        # Should handle 10k samples reasonably quickly (< 10 seconds)
        performance_metrics.assert_performance('feature_engineering', 10.0)
        
        assert feature_set.features.shape[0] == len(performance_test_data)
        assert len(feature_set.feature_names) > 0
    
    @pytest.mark.unit
    @pytest.mark.ml
    @pytest.mark.performance
    async def test_rl_action_selection_performance(self, performance_metrics):
        """Test RL action selection performance"""
        optimizer = FrequencyVoltageOptimizer({})
        
        state = RLState(
            temperature=70.0,
            hashrate=480.0,
            power=12.0,
            efficiency=40.0,
            voltage=12.0,
            frequency=600,
            weather_temp=25.0,
            weather_humidity=50.0,
            time_of_day=0.5,
            stability_score=1.0
        )
        
        performance_metrics.start_timer('rl_action_selection')
        
        # Test multiple action selections
        for _ in range(100):
            action = await optimizer.agent.select_action(state, exploration=True)
            assert isinstance(action, RLAction)
        
        performance_metrics.end_timer('rl_action_selection')
        
        # Should be very fast (< 1 second for 100 selections)
        performance_metrics.assert_performance('rl_action_selection', 1.0)


# Error handling and edge cases
class TestMLErrorHandling:
    """Test ML component error handling"""
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_feature_engineering_invalid_data(self, feature_engineer):
        """Test feature engineering with invalid data"""
        # Test with malformed data
        invalid_data = [
            {'invalid': 'data'},
            {'temp': 'not_a_number'},
            {'temp': None, 'power': 'invalid'}
        ]
        
        # Should handle gracefully without crashing
        feature_set = await feature_engineer.engineer_features(invalid_data)
        
        # Should return empty or minimal feature set
        assert feature_set.features.size == 0 or feature_set.features.shape[0] == 0
    
    @pytest.mark.unit
    @pytest.mark.ml
    async def test_rl_optimizer_extreme_states(self):
        """Test RL optimizer with extreme states"""
        optimizer = FrequencyVoltageOptimizer({'safety_enabled': True})
        
        # Extreme state values
        extreme_state = RLState(
            temperature=150.0,  # Impossibly high
            hashrate=-100.0,    # Negative
            power=1000.0,       # Very high
            efficiency=-50.0,   # Negative
            voltage=50.0,       # Too high
            frequency=2000,     # Too high
            weather_temp=-50.0, # Extreme cold
            weather_humidity=150.0, # > 100%
            time_of_day=5.0,    # > 1.0
            stability_score=-1.0 # Negative
        )
        
        # Safety check should fail
        assert optimizer._safety_check(extreme_state) is False
        
        # Action selection should still work (with safety constraints)
        action = await optimizer.agent.select_action(extreme_state, exploration=False)
        assert isinstance(action, RLAction)
        
        # Applied action should be within safety bounds
        test_config = {'frequency': 600, 'voltage': 12.0}
        new_config = action.apply_to_miner_config(test_config)
        
        assert 400 <= new_config['frequency'] <= 800
        assert 10.0 <= new_config['voltage'] <= 14.0