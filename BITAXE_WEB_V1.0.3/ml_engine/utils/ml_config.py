"""
ML Configuration Management

Centralized configuration management for the ML engine components.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from logging.structured_logger import get_logger

logger = get_logger("bitaxe.ml.config")


@dataclass
class MLEngineConfig:
    """Main ML engine configuration"""
    
    # Core services
    enable_rl_optimization: bool = True
    enable_predictive_analytics: bool = True
    enable_weather_adaptation: bool = True
    enable_model_training: bool = True
    enable_monitoring: bool = True
    
    # Optimization settings
    optimization_interval: int = 300  # seconds
    min_optimization_gap: int = 900  # seconds
    target_efficiency: float = 120.0  # GH/W
    target_temperature: float = 70.0  # °C
    
    # Safety constraints
    max_temperature: float = 85.0  # °C
    min_efficiency: float = 50.0  # GH/W
    max_power: float = 200.0  # W
    max_frequency_change: float = 100.0  # MHz per optimization
    max_voltage_change: float = 0.5  # V per optimization
    
    # Paths
    models_dir: str = "models"
    training_data_dir: str = "training_data"
    logs_dir: str = "logs"
    
    # External APIs
    openweathermap_api_key: Optional[str] = None
    weather_location: str = "New York,US"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MLEngineConfig':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class MLConfigManager:
    """ML configuration manager with environment variable support"""
    
    def __init__(self, config_file: str = "ml_config.json"):
        self.config_file = Path(config_file)
        self.config = MLEngineConfig()
        
        # Load configuration
        self._load_config()
        self._load_env_overrides()
        
        logger.info("ML config manager initialized", config_file=str(self.config_file))
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                self.config = MLEngineConfig.from_dict(config_data)
                logger.info("Configuration loaded from file")
            else:
                logger.info("Configuration file not found, using defaults")
                self._save_config()
        
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            logger.info("Using default configuration")
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        try:
            # Environment variable mappings
            env_mappings = {
                'ML_OPTIMIZATION_INTERVAL': ('optimization_interval', int),
                'ML_TARGET_EFFICIENCY': ('target_efficiency', float),
                'ML_TARGET_TEMPERATURE': ('target_temperature', float),
                'ML_MAX_TEMPERATURE': ('max_temperature', float),
                'ML_ENABLE_RL': ('enable_rl_optimization', lambda x: x.lower() == 'true'),
                'ML_ENABLE_PREDICTIVE': ('enable_predictive_analytics', lambda x: x.lower() == 'true'),
                'ML_ENABLE_WEATHER': ('enable_weather_adaptation', lambda x: x.lower() == 'true'),
                'OPENWEATHERMAP_API_KEY': ('openweathermap_api_key', str),
                'WEATHER_LOCATION': ('weather_location', str),
                'ML_MODELS_DIR': ('models_dir', str),
            }
            
            for env_var, (config_attr, converter) in env_mappings.items():
                env_value = os.environ.get(env_var)
                if env_value is not None:
                    try:
                        converted_value = converter(env_value)
                        setattr(self.config, config_attr, converted_value)
                        logger.debug(f"Environment override: {config_attr} = {converted_value}")
                    except Exception as e:
                        logger.warning(f"Failed to parse environment variable {env_var}: {e}")
        
        except Exception as e:
            logger.error("Failed to load environment overrides", error=str(e))
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.debug("Configuration saved to file")
        
        except Exception as e:
            logger.error("Failed to save configuration", error=str(e))
    
    def get_config(self) -> MLEngineConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.debug(f"Configuration updated: {key} = {value}")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            self._save_config()
        
        except Exception as e:
            logger.error("Failed to update configuration", error=str(e))
    
    def get_service_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a specific ML service"""
        base_config = self.config.to_dict()
        
        if service_name == "rl_optimizer":
            return {
                'enabled': base_config['enable_rl_optimization'],
                'optimization_interval': base_config['optimization_interval'],
                'safety': {
                    'max_temperature': base_config['max_temperature'],
                    'min_efficiency': base_config['min_efficiency'],
                    'max_power': base_config['max_power']
                },
                'targets': {
                    'efficiency': base_config['target_efficiency'],
                    'temperature': base_config['target_temperature']
                },
                'constraints': {
                    'max_frequency_change': base_config['max_frequency_change'],
                    'max_voltage_change': base_config['max_voltage_change']
                }
            }
        
        elif service_name == "weather_service":
            return {
                'enabled': base_config['enable_weather_adaptation'],
                'openweathermap_api_key': base_config['openweathermap_api_key'],
                'location': base_config['weather_location'],
                'update_interval': 300,  # 5 minutes
                'enable_local_sensors': True,
                'local_sensor_config': {'mock_mode': True}
            }
        
        elif service_name == "model_training":
            return {
                'enabled': base_config['enable_model_training'],
                'models_dir': base_config['models_dir'],
                'training_data_dir': base_config['training_data_dir'],
                'max_concurrent_jobs': 2,
                'auto_deployment_threshold': 0.85
            }
        
        elif service_name == "monitoring":
            return {
                'enabled': base_config['enable_monitoring'],
                'monitoring_enabled': True,
                'alert_enabled': True,
                'performance_tracking': {
                    'monitoring_windows': ["1h", "24h", "7d"],
                    'alert_thresholds': {
                        'accuracy_drop': 0.1,
                        'error_rate_increase': 0.05,
                        'latency_increase': 2.0
                    }
                },
                'drift_detection': {
                    'drift_threshold': 0.1,
                    'reference_window_size': 1000,
                    'detection_window_size': 200
                },
                'validation': {
                    'validation_thresholds': {
                        'min_accuracy': 0.7,
                        'max_error_rate': 0.2,
                        'max_prediction_time': 1000,
                        'min_prediction_variance': 0.01
                    }
                }
            }
        
        elif service_name == "optimization_engine":
            return {
                'optimization_interval': base_config['optimization_interval'],
                'min_optimization_gap': base_config['min_optimization_gap'],
                'enable_predictive': base_config['enable_predictive_analytics'],
                'enable_weather': base_config['enable_weather_adaptation'],
                'enable_rl': base_config['enable_rl_optimization'],
                'safety': {
                    'max_temperature': base_config['max_temperature'],
                    'min_efficiency': base_config['min_efficiency'],
                    'max_power': base_config['max_power']
                },
                'target_efficiency': base_config['target_efficiency'],
                'target_temperature': base_config['target_temperature']
            }
        
        else:
            logger.warning(f"Unknown service: {service_name}")
            return {}


# Global config manager instance
_config_manager: Optional[MLConfigManager] = None


def get_ml_config_manager() -> MLConfigManager:
    """Get global ML config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = MLConfigManager()
    return _config_manager


def get_ml_config() -> MLEngineConfig:
    """Get current ML configuration"""
    return get_ml_config_manager().get_config()


def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get configuration for a specific service"""
    return get_ml_config_manager().get_service_config(service_name)