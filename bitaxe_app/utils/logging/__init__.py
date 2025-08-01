"""
BitAxe V2.0.0 - Logging Module
Unified logging configuration and utilities
"""

from .setup import setup_logging, get_logger
from .structured_logger import StructuredLogger

__all__ = ['setup_logging', 'get_logger', 'StructuredLogger']