"""
Pytest configuration and fixtures for BitAxe Web Management tests
"""

import os
import pytest
import tempfile
from datetime import datetime
from flask import Flask

from app import app as flask_app
from services.service_container_v2 import get_container_v2
from auth.auth_service import AuthService
from models.miner_models import Base
from database import DatabaseManager


@pytest.fixture(scope='session')
def app():
    """Create application for testing"""
    # Use a temporary database for testing
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    flask_app.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
        'DATABASE_PATH': db_path,
        'SECRET_KEY': 'test-secret-key'
    })
    
    # Initialize test database
    db_manager = DatabaseManager(db_path)
    db_manager.init_database()
    
    # Create tables
    Base.metadata.create_all(db_manager.engine)
    
    yield flask_app
    
    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create test CLI runner"""
    return app.test_cli_runner()


@pytest.fixture
def auth_service():
    """Create auth service for testing"""
    return AuthService(secret_key='test-secret-key')


@pytest.fixture
def admin_token(auth_service):
    """Get admin authentication token"""
    user = auth_service.authenticate_user('admin', 'admin123')
    if user:
        token_data = auth_service.generate_token(user)
        return token_data['access_token']
    return None


@pytest.fixture
def readonly_token(auth_service):
    """Get readonly authentication token"""
    user = auth_service.authenticate_user('readonly', 'readonly123')
    if user:
        token_data = auth_service.generate_token(user)
        return token_data['access_token']
    return None


@pytest.fixture
def operator_token(auth_service):
    """Get operator authentication token"""
    user = auth_service.authenticate_user('operator', 'operator123')
    if user:
        token_data = auth_service.generate_token(user)
        return token_data['access_token']
    return None


@pytest.fixture
def auth_headers(admin_token):
    """Get authorization headers for admin user"""
    return {'Authorization': f'Bearer {admin_token}'}


@pytest.fixture
def readonly_headers(readonly_token):
    """Get authorization headers for readonly user"""
    return {'Authorization': f'Bearer {readonly_token}'}


@pytest.fixture
def operator_headers(operator_token):
    """Get authorization headers for operator user"""
    return {'Authorization': f'Bearer {operator_token}'}


@pytest.fixture
def sample_miner_data():
    """Sample miner data for testing"""
    return {
        'ip': '192.168.1.100',
        'hostname': 'test-miner-001',
        'temp': 65.5,
        'hashRate': 485.2,
        'power': 12.8,
        'voltage': 5.1,
        'frequency': 800,
        'coreVoltage': 1200,
        'fanrpm': 3500,
        'sharesAccepted': 1250,
        'sharesRejected': 5,
        'uptime': 86400,
        'version': '2.0.4',
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data for testing"""
    return {
        'ip': '192.168.1.100',
        'frequency': 800,
        'core_voltage': 1200,
        'duration': 600
    }


@pytest.fixture
def container():
    """Get service container for testing"""
    return get_container_v2()


@pytest.fixture
def database_service(container):
    """Get database service for testing"""
    return container.get_database_service()


@pytest.fixture
def miner_service(container):
    """Get miner service for testing"""
    return container.get_miner_service()


@pytest.fixture
def benchmark_service(container):
    """Get benchmark service for testing"""
    return container.get_benchmark_service()


@pytest.fixture
def config_service(container):
    """Get config service for testing"""
    return container.get_config_service()


# Test data fixtures
@pytest.fixture
def mock_miner_response():
    """Mock successful miner API response"""
    return {
        "power": 12.8,
        "voltage": 5.1,
        "current": 2.5,
        "temp": 65.5,
        "vrTemp": 58.2,
        "hashRate": 485.2,
        "bestDiff": "1.2K",
        "bestSession": "456",
        "freeHeap": 123456,
        "coreVoltage": 1200,
        "frequency": 800,
        "version": "2.0.4",
        "boardtemp1": 62.3,
        "boardtemp2": 63.1,
        "hostname": "test-miner-001",
        "fanrpm": 3500,
        "uptimeSeconds": 86400,
        "ASICModel": "BM1397",
        "MACAddr": "aa:bb:cc:dd:ee:ff",
        "stratumURL": "stratum+tcp://pool.example.com:4334",
        "stratumPort": 4334,
        "stratumUser": "test.worker",
        "wifiSSID": "TestNetwork",
        "wifiStatus": "Connected",
        "sharesAccepted": 1250,
        "sharesRejected": 5,
        "jobsPerSecond": 2.5
    }


@pytest.fixture
def mock_system_info():
    """Mock system info response"""
    return {
        "power": 12.8,
        "voltage": 5.1,
        "current": 2.5,
        "temp": 65.5,
        "vrTemp": 58.2,
        "hashRate": 485.2,
        "bestDiff": "1.2K",
        "bestSession": "456",
        "freeHeap": 123456,
        "coreVoltage": 1200,
        "frequency": 800,
        "version": "2.0.4",
        "boardtemp1": 62.3,
        "boardtemp2": 63.1,
        "hostname": "test-miner-001",
        "fanrpm": 3500,
        "uptimeSeconds": 86400
    }


# Utility functions for tests
def assert_json_response(response, expected_status=200):
    """Assert JSON response format and status"""
    assert response.status_code == expected_status
    assert response.is_json
    return response.get_json()


def assert_api_success(response_json, expected_status=True):
    """Assert API response success format"""
    assert 'success' in response_json
    assert response_json['success'] == expected_status
    if expected_status:
        assert 'data' in response_json or 'message' in response_json
    else:
        assert 'error' in response_json


def assert_error_response(response_json, expected_code=None):
    """Assert API error response format"""
    assert_api_success(response_json, False)
    assert 'error' in response_json
    error = response_json['error']
    assert 'code' in error
    assert 'message' in error
    if expected_code:
        assert error['code'] == expected_code