"""
BitAxe V2.0.0 - Unified Configuration Manager
Consolidates all configuration management into a single, robust class
"""

import json
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging
from threading import Lock

from .exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager for BitAxe system.
    
    This class provides centralized configuration management with thread-safe
    operations, automatic reloading, validation, and comprehensive error handling.
    
    Features:
        - Thread-safe configuration access and modification
        - Automatic configuration validation
        - Support for dot-notation key access (e.g., 'config.ips')
        - Configuration change callbacks
        - Environment variable override support
        - Comprehensive error reporting
        
    Example:
        >>> config = ConfigManager('config/config.json')
        >>> ips = config.get('config.ips', [])
        >>> config.set('settings.temp_limit', 75)
        >>> config.save()
    """
    
    _instance: Optional['ConfigManager'] = None
    _lock = Lock()
    
    def __new__(cls, config_path: Optional[str] = None) -> 'ConfigManager':
        """Singleton pattern implementation for global config access."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
            
        Raises:
            ConfigurationError: If configuration file cannot be loaded or is invalid
        """
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
            
        self.config_path = self._resolve_config_path(config_path)
        self._config: Dict[str, Any] = {}
        self._callbacks: List[callable] = []
        self._lock = Lock()
        self._initialized = True
        
        # Load initial configuration
        self.load_config()
        
        logger.info(f"ConfigManager initialized with config: {self.config_path}")
    
    def _resolve_config_path(self, config_path: Optional[str]) -> str:
        """Resolve the configuration file path with environment variable support.
        
        Args:
            config_path: Provided configuration path
            
        Returns:
            Resolved absolute path to configuration file
        """
        if config_path:
            path = config_path
        else:
            # Check environment variable first, then use default
            path = os.environ.get("BITAXE_CONFIG", "config/config.json")
        
        # Convert to absolute path
        if not os.path.isabs(path):
            # Resolve relative to project root
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path
        
        return str(path)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with comprehensive error handling.
        
        Returns:
            The loaded configuration dictionary
            
        Raises:
            ConfigurationError: If file cannot be read or contains invalid JSON
        """
        try:
            with self._lock:
                logger.debug(f"Loading configuration from: {self.config_path}")
                
                if not os.path.exists(self.config_path):
                    raise ConfigurationError(
                        f"Configuration file not found: {self.config_path}",
                        config_path=self.config_path
                    )
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
                
                # Validate configuration structure
                self._validate_config()
                
                logger.info(f"Configuration loaded successfully: {len(self._config)} sections")
                
                # Notify callbacks
                self._notify_callbacks('loaded')
                
                return self._config.copy()
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {e}",
                config_path=self.config_path,
                line_number=e.lineno,
                column=e.colno
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
                config_path=self.config_path,
                original_error=str(e)
            )
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration structure.
        
        Raises:
            ConfigurationError: If configuration is missing required sections
        """
        required_sections = ['paths', 'config', 'settings']
        missing_sections = []
        
        for section in required_sections:
            if section not in self._config:
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigurationError(
                f"Configuration missing required sections: {missing_sections}",
                config_path=self.config_path,
                missing_sections=missing_sections
            )
        
        # Validate specific required fields
        required_fields = {
            'config.ips': list,
            'paths.database': str,
            'settings.temp_limit': (int, float),
            'settings.temp_overheat': (int, float)
        }
        
        for field_path, expected_type in required_fields.items():
            value = self._get_nested_value(field_path)
            if value is None:
                raise ConfigurationError(
                    f"Required configuration field missing: {field_path}",
                    config_path=self.config_path,
                    config_key=field_path
                )
            
            if not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Configuration field has wrong type: {field_path} "
                    f"(expected {expected_type}, got {type(value)})",
                    config_path=self.config_path,
                    config_key=field_path,
                    expected_type=str(expected_type),
                    actual_type=str(type(value))
                )
    
    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from file and notify callbacks.
        
        Returns:
            The reloaded configuration dictionary
        """
        logger.info("Reloading configuration from file")
        config = self.load_config()
        self._notify_callbacks('reloaded')
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get a copy of the current configuration.
        
        Returns:
            Complete configuration dictionary (copy for thread safety)
        """
        with self._lock:
            return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'config.ips')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('config.ips', [])
            ['192.168.1.100', '192.168.1.101']
            >>> config.get('settings.temp_limit', 75)
            73
        """
        with self._lock:
            value = self._get_nested_value(key)
            return value if value is not None else default
    
    def _get_nested_value(self, key: str) -> Any:
        """Get nested value from configuration using dot notation.
        
        Args:
            key: Dot-separated key path
            
        Returns:
            Value at the specified path or None if not found
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return None
    
    def set(self, key: str, value: Any, save: bool = False) -> None:
        """Set a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
            save: Whether to immediately save to file
            
        Raises:
            ConfigurationError: If key path is invalid
            
        Example:
            >>> config.set('settings.temp_limit', 75)
            >>> config.set('config.ips', ['192.168.1.100'], save=True)
        """
        with self._lock:
            keys = key.split('.')
            target = self._config
            
            # Navigate to parent of target key
            try:
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                    
                    if not isinstance(target, dict):
                        raise ConfigurationError(
                            f"Cannot set nested key '{key}': intermediate key '{k}' is not a dictionary",
                            config_path=self.config_path,
                            config_key=key
                        )
                
                # Set the final value
                target[keys[-1]] = value
                
                logger.debug(f"Configuration updated: {key} = {value}")
                
                if save:
                    self.save_config()
                
                # Notify callbacks
                self._notify_callbacks('updated', key=key, value=value)
                
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to set configuration key '{key}': {e}",
                    config_path=self.config_path,
                    config_key=key,
                    original_error=str(e)
                )
    
    def save_config(self) -> None:
        """Save current configuration to file.
        
        Raises:
            ConfigurationError: If file cannot be written
        """
        try:
            with self._lock:
                # Create backup of existing config
                backup_path = f"{self.config_path}.backup"
                if os.path.exists(self.config_path):
                    import shutil
                    shutil.copy2(self.config_path, backup_path)
                    logger.debug(f"Configuration backup created: {backup_path}")
                
                # Write new configuration
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Configuration saved to: {self.config_path}")
                
                # Notify callbacks
                self._notify_callbacks('saved')
                
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                config_path=self.config_path,
                original_error=str(e)
            )
    
    def add_callback(self, callback: callable) -> None:
        """Add a callback function to be called on configuration changes.
        
        Args:
            callback: Function to call with (event_type, **kwargs) signature
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.debug(f"Configuration callback added: {callback.__name__}")
    
    def remove_callback(self, callback: callable) -> None:
        """Remove a configuration change callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.debug(f"Configuration callback removed: {callback.__name__}")
    
    def _notify_callbacks(self, event_type: str, **kwargs) -> None:
        """Notify all registered callbacks of configuration changes.
        
        Args:
            event_type: Type of event ('loaded', 'reloaded', 'updated', 'saved')
            **kwargs: Additional event data
        """
        for callback in self._callbacks:
            try:
                callback(event_type, **kwargs)
            except Exception as e:
                logger.warning(f"Configuration callback failed: {callback.__name__}: {e}")
    
    # Convenience properties for commonly accessed values
    @property
    def ips(self) -> List[str]:
        """Get list of miner IP addresses."""
        return self.get('config.ips', [])
    
    @property
    def database_path(self) -> str:
        """Get database file path."""
        return self.get('paths.database', 'data/bitaxe_data.db')
    
    @property
    def log_interval(self) -> int:
        """Get logging interval in seconds."""
        return self.get('config.log_interval_sec', 30)
    
    @property
    def temp_limit(self) -> float:
        """Get temperature limit in Celsius."""
        return self.get('settings.temp_limit', 73.0)
    
    @property
    def temp_overheat(self) -> float:
        """Get overheat temperature in Celsius."""
        return self.get('settings.temp_overheat', 75.0)
    
    @property
    def benchmark_interval(self) -> int:
        """Get benchmark interval in seconds."""
        return self.get('settings.benchmark_interval_sec', 86400)
    
    @property
    def freq_list(self) -> List[int]:
        """Get available frequency list."""
        return self.get('settings.freq_list', [750, 775, 800, 825, 850, 875, 900, 925])
    
    @property
    def volt_list(self) -> List[int]:
        """Get available voltage list."""
        return self.get('settings.volt_list', [1150, 1175, 1200, 1225, 1250, 1275])
    
    @property
    def fallback_settings(self) -> Dict[str, int]:
        """Get fallback frequency and voltage settings."""
        return self.get('settings.fallback', {'frequency': 750, 'coreVoltage': 1150})
    
    def get_miner_color(self, ip: str) -> str:
        """Get color for specific miner IP.
        
        Args:
            ip: Miner IP address
            
        Returns:
            Color code for the miner (hex format)
        """
        return self.get(f'visual.colors.{ip}', '#3498db')
    
    def get_profile(self, profile_name: str) -> Dict[str, int]:
        """Get settings for a specific profile.
        
        Args:
            profile_name: Profile name ('max', 'eco', etc.)
            
        Returns:
            Profile settings dictionary
        """
        return self.get(f'profiles.{profile_name}', {})
    
    def validate_miner_ip(self, ip: str) -> bool:
        """Validate that a miner IP is configured.
        
        Args:
            ip: IP address to validate
            
        Returns:
            True if IP is in configured miner list
        """
        return ip in self.ips
    
    def get_all_colors(self) -> Dict[str, str]:
        """Get all configured miner colors.
        
        Returns:
            Dictionary mapping IP addresses to color codes
        """
        return self.get('visual.colors', {})
    
    def get_all_profiles(self) -> Dict[str, Dict[str, int]]:
        """Get all configured profiles.
        
        Returns:
            Dictionary of all available profiles
        """
        return self.get('profiles', {})
    
    def export_config(self, export_path: Optional[str] = None) -> str:
        """Export current configuration to a file.
        
        Args:
            export_path: Path for exported config. If None, creates timestamped export.
            
        Returns:
            Path to exported configuration file
        """
        if export_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = f"{self.config_path}.export_{timestamp}"
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to: {export_path}")
            return export_path
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to export configuration: {e}",
                config_path=export_path,
                original_error=str(e)
            )
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigManager(path='{self.config_path}', sections={list(self._config.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"ConfigManager(config_path='{self.config_path}', "
                f"loaded={len(self._config)} sections, "
                f"callbacks={len(self._callbacks)})")


# Global configuration instance for backward compatibility
_global_config: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get the global configuration manager instance.
    
    Args:
        config_path: Path to configuration file (only used for first initialization)
        
    Returns:
        Global ConfigManager instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    return _global_config


def load_config() -> Dict[str, Any]:
    """Legacy function for backward compatibility.
    
    Returns:
        Current configuration dictionary
    """
    return get_config_manager().get_config()


def reload_config() -> Dict[str, Any]:
    """Legacy function for backward compatibility.
    
    Returns:
        Reloaded configuration dictionary
    """
    return get_config_manager().reload_config()