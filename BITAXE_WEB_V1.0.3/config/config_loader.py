# config_loader.py

import json
import os

_config_cache = None  # interne Cache-Variable

def load_config():
    global _config_cache

    if _config_cache is None:
        config_path = os.environ.get("BITAXE_CONFIG", "config/config.json")
        with open(config_path, "r") as f:
            _config_cache = json.load(f)

    return _config_cache

def reload_config():
    """Optional: erzwinge ein Neuladen, z.B. wenn du das JSON während Laufzeit änderst"""
    global _config_cache
    config_path = os.environ.get("BITAXE_CONFIG", "config/config.json")
    with open(config_path, "r") as f:
        _config_cache = json.load(f)
    return _config_cache
