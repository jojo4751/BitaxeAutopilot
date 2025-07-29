import json
import os
from typing import Dict, Any, Optional


class ConfigService:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join("config", "config.json")
        self._config: Optional[Dict[str, Any]] = None
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            return self._config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from file"""
        return self.load_config()

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        if self._config is None:
            self.load_config()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot notation (e.g., 'config.ips')"""
        config = self.get_config()
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot notation and save to file"""
        config = self.get_config()
        keys = key.split('.')
        target = config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        
        target[keys[-1]] = value
        self.save_config()

    def save_config(self) -> None:
        """Save current configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    @property
    def ips(self) -> list:
        """Get list of miner IP addresses"""
        return self.get('config.ips', [])

    @property
    def database_path(self) -> str:
        """Get database file path"""
        return self.get('paths.database', 'data/bitaxe_data.db')

    @property
    def log_interval(self) -> int:
        """Get logging interval in seconds"""
        return self.get('config.log_interval_sec', 30)

    @property
    def temp_limit(self) -> float:
        """Get temperature limit in Celsius"""
        return self.get('settings.temp_limit', 73)

    @property
    def temp_overheat(self) -> float:
        """Get overheat temperature in Celsius"""
        return self.get('settings.temp_overheat', 75)

    @property
    def benchmark_interval(self) -> int:
        """Get benchmark interval in seconds"""
        return self.get('settings.benchmark_interval_sec', 86400)

    @property
    def freq_list(self) -> list:
        """Get available frequency list"""
        return self.get('settings.freq_list', [750, 775, 800, 825, 850, 875, 900, 925])

    @property
    def volt_list(self) -> list:
        """Get available voltage list"""
        return self.get('settings.volt_list', [1150, 1175, 1200, 1225, 1250, 1275])

    @property
    def fallback_settings(self) -> Dict[str, int]:
        """Get fallback frequency and voltage settings"""
        return self.get('settings.fallback', {'frequency': 750, 'coreVoltage': 1150})

    def get_miner_color(self, ip: str) -> str:
        """Get color for specific miner IP"""
        return self.get(f'visual.colors.{ip}', '#3498db')

    def get_profile(self, profile_name: str) -> Dict[str, int]:
        """Get settings for a specific profile (max/eco)"""
        return self.get(f'profiles.{profile_name}', {})