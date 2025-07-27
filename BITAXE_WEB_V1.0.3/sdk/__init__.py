"""
BitAxe API Client SDK

A comprehensive Python SDK for interacting with the BitAxe Web Management API.
"""

from .client import BitAxeClient, create_client, BitAxeAPIError, AuthenticationError, AuthorizationError, RateLimitError, ValidationError
from .async_client import AsyncBitAxeClient

__version__ = "1.0.0"
__author__ = "BitAxe Team"
__email__ = "support@bitaxe.org"

__all__ = [
    'BitAxeClient',
    'AsyncBitAxeClient', 
    'create_client',
    'BitAxeAPIError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'ValidationError'
]