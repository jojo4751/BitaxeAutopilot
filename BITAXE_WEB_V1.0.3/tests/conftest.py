"""
Pytest configuration and fixtures for BitAxe Web Management tests

Comprehensive test fixtures for unit, integration, and end-to-end testing
including ML components, async services, and external integrations.
"""

import os
import pytest
import tempfile
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import json

# Flask and Quart imports
from flask import Flask
from quart import Quart

# Application imports
from app import app as flask_app
from async_app import AsyncBitAxeApp
from services.service_container_v2 import get_container_v2
from auth.auth_service import AuthService
from models.miner_models import Base
from database import DatabaseManager

# ML Engine imports
from ml_engine.utils.ml_config import MLConfigManager, MLEngineConfig
from ml_engine.core.ml_model_service import MLModelService
from ml_engine.core.optimization_engine import OptimizationEngine
from ml_engine.models.rl_agent import FrequencyVoltageOptimizer
from ml_engine.data.feature_engineering import FeatureEngineer
from monitoring.metrics_collector import MetricsCollector


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


# =============================================================================
# ML and Async Test Fixtures
# =============================================================================

@pytest.fixture(scope='session')
def event_loop():
    """Create event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_app():
    """Create async application for testing"""
    app = AsyncBitAxeApp()
    
    # Configure for testing
    app.config_service = Mock()
    app.config_service.ips = ['192.168.1.100', '192.168.1.101']
    app.config_service.database_path = ':memory:'
    
    await app.initialize()
    yield app
    await app.stop()


@pytest.fixture
def ml_config():
    """Create ML configuration for testing"""
    return MLEngineConfig(
        enable_rl_optimization=True,
        enable_predictive_analytics=True,
        enable_weather_adaptation=True,
        optimization_interval=60,
        target_efficiency=120.0,
        target_temperature=70.0,
        max_temperature=85.0,
        openweathermap_api_key="test_api_key",
        models_dir="test_models"
    )


@pytest.fixture
def ml_config_manager(ml_config, tmp_path):
    """Create ML config manager for testing"""
    config_file = tmp_path / "test_ml_config.json"
    
    with patch('ml_engine.utils.ml_config.MLConfigManager.__init__') as mock_init:
        mock_init.return_value = None
        manager = MLConfigManager.__new__(MLConfigManager)
        manager.config_file = Path(config_file)
        manager.config = ml_config
        return manager


@pytest.fixture
def mock_metrics_collector():
    """Create mock metrics collector"""
    collector = Mock(spec=MetricsCollector)
    collector.record_metric = Mock()
    collector.increment_counter = Mock()
    collector.set_gauge = Mock()
    collector.record_timer = Mock()
    collector.get_latest_metrics = Mock(return_value={})
    return collector


@pytest.fixture
async def feature_engineer():
    """Create feature engineer for testing"""
    config = {
        'window_sizes': [5, 15, 30],
        'statistical_features': ['mean', 'std', 'min', 'max'],
        'scaling_method': 'standard'
    }
    return FeatureEngineer(config)


@pytest.fixture
def sample_miner_telemetry():
    """Generate sample miner telemetry data for ML testing"""
    np.random.seed(42)  # For reproducible tests
    
    data = []
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(120):  # 2 hours of data at 1-minute intervals
        timestamp = base_time + timedelta(minutes=i)
        
        # Generate realistic mining data with some variation
        temp = 65 + np.random.normal(0, 5)
        power = 12.0 + np.random.normal(0, 1)
        frequency = 600 + np.random.normal(0, 50)
        voltage = 12.0 + np.random.normal(0, 0.5)
        hashRate = frequency * 0.8 + np.random.normal(0, 20)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'temp': max(30, min(90, temp)),
            'power': max(5, min(20, power)),
            'frequency': max(400, min(800, frequency)),
            'voltage': max(10, min(14, voltage)),
            'hashRate': max(0, hashRate),
            'ip': '192.168.1.100',
            'hostname': 'test-miner'
        })
    
    return data


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for testing"""
    return {
        'temperature': 25.5,
        'humidity': 60.0,
        'pressure': 1013.25,
        'wind_speed': 5.2,
        'wind_direction': 180.0,
        'cloud_cover': 20.0,
        'visibility': 10.0,
        'uv_index': 3.0
    }


@pytest.fixture
def mock_rl_optimizer():
    """Create mock RL optimizer"""
    optimizer = Mock(spec=FrequencyVoltageOptimizer)
    optimizer.optimize_miner = AsyncMock(return_value={
        'frequency': 650,
        'voltage': 12.2
    })
    optimizer.start_optimization = AsyncMock()
    optimizer.stop_optimization = AsyncMock()
    optimizer.get_optimization_status = AsyncMock(return_value={
        'active': True,
        'exploration_rate': 0.1
    })
    return optimizer


@pytest.fixture
def mock_weather_service():
    """Create mock weather service"""
    from ml_engine.core.weather_service import WeatherData
    
    service = Mock()
    service.get_current_weather = AsyncMock(return_value=WeatherData(
        temperature=25.0,
        humidity=60.0,
        pressure=1013.25,
        wind_speed=5.0,
        wind_direction=180.0,
        cloud_cover=20.0,
        visibility=10.0,
        uv_index=3.0,
        timestamp=datetime.now(),
        location="Test Location"
    ))
    service.get_cooling_strategy = AsyncMock(return_value={
        'strategy': 'optimal',
        'adjustments': {},
        'recommendation_strength': 0.5
    })
    service.start = AsyncMock()
    service.stop = AsyncMock()
    return service


@pytest.fixture
def mock_ml_model_service():
    """Create mock ML model service"""
    from ml_engine.core.ml_model_service import PredictionResult
    
    service = Mock(spec=MLModelService)
    service.predict = AsyncMock(return_value=PredictionResult(
        predictions=np.array([70.5]),  # Temperature prediction
        confidence=np.array([0.85]),
        model_id="test_model",
        inference_time_ms=25.0
    ))
    service.load_model = AsyncMock(return_value=True)
    service.list_available_models = AsyncMock(return_value=[])
    service.start = AsyncMock()
    service.stop = AsyncMock()
    return service


@pytest.fixture
def sample_training_data():
    """Generate sample training data for ML models"""
    np.random.seed(42)
    
    # Generate features (temperature, power, frequency, voltage, hashrate, etc.)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    
    # Generate targets (temperature prediction)
    # Temperature = f(power, frequency, voltage, ...) + noise
    y = (X[:, 1] * 20 +  # power effect
         X[:, 2] * 10 +  # frequency effect  
         X[:, 3] * 5 +   # voltage effect
         np.random.normal(50, 5, n_samples))  # base temp + noise
    
    return X, y


@pytest.fixture
def performance_test_data():
    """Generate large dataset for performance testing"""
    np.random.seed(42)
    
    # Generate 10,000 samples for performance testing
    n_samples = 10000
    data = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(n_samples):
        timestamp = base_time + timedelta(minutes=i)
        
        data.append({
            'timestamp': timestamp.isoformat(),
            'temp': 60 + np.random.normal(0, 10),
            'power': 12 + np.random.normal(0, 2),
            'frequency': 600 + np.random.normal(0, 100),
            'voltage': 12 + np.random.normal(0, 1),
            'hashRate': 480 + np.random.normal(0, 50),
            'ip': f'192.168.1.{100 + (i % 10)}',
            'hostname': f'miner-{i % 10:03d}'
        })
    
    return data


# =============================================================================
# Test Database Fixtures  
# =============================================================================

@pytest.fixture
def test_database():
    """Create test database with sample data"""
    # Create temporary database
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    try:
        # Initialize database
        db_manager = DatabaseManager(db_path)
        db_manager.init_database()
        Base.metadata.create_all(db_manager.engine)
        
        # Insert sample data
        from models.miner_models import MinerStatus, BenchmarkResult
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=db_manager.engine)
        session = Session()
        
        # Add sample miner status
        status = MinerStatus(
            ip='192.168.1.100',
            hostname='test-miner',
            temp=65.5,
            hashRate=485.2,
            power=12.8,
            voltage=5.1,
            frequency=800,
            timestamp=datetime.now()
        )
        session.add(status)
        
        # Add sample benchmark
        benchmark = BenchmarkResult(
            ip='192.168.1.100',
            frequency=800,
            core_voltage=1200,
            duration=600,
            avg_hashrate=485.2,
            avg_temperature=65.5,
            avg_power=12.8,
            efficiency=37.9,
            timestamp=datetime.now()
        )
        session.add(benchmark)
        
        session.commit()
        session.close()
        
        yield db_manager
        
    finally:
        os.unlink(db_path)


# =============================================================================
# Mock External Services
# =============================================================================

@pytest.fixture
def mock_miner_api_responses():
    """Mock miner API responses for different scenarios"""
    return {
        'success': {
            "power": 12.8,
            "voltage": 5.1,
            "temp": 65.5,
            "hashRate": 485.2,
            "frequency": 800,
            "coreVoltage": 1200,
            "hostname": "miner-001",
            "version": "2.0.4"
        },
        'high_temp': {
            "power": 15.2,
            "voltage": 5.3,
            "temp": 89.5,  # High temperature
            "hashRate": 520.1,
            "frequency": 850,
            "coreVoltage": 1300,
            "hostname": "miner-002",
            "version": "2.0.4"
        },
        'low_efficiency': {
            "power": 18.5,  # High power consumption
            "voltage": 5.8,
            "temp": 72.1,
            "hashRate": 350.2,  # Low hashrate
            "frequency": 650,
            "coreVoltage": 1100,
            "hostname": "miner-003",
            "version": "2.0.4"
        },
        'offline': None,  # Miner not responding
        'error': {'error': 'Internal server error'}
    }


@pytest.fixture
def mock_openweathermap_response():
    """Mock OpenWeatherMap API response"""
    return {
        "coord": {"lon": -74.006, "lat": 40.7143},
        "weather": [{"id": 800, "main": "Clear", "description": "clear sky"}],
        "base": "stations",
        "main": {
            "temp": 25.5,
            "feels_like": 27.2,
            "temp_min": 23.1,
            "temp_max": 28.3,
            "pressure": 1013,
            "humidity": 60
        },
        "visibility": 10000,
        "wind": {"speed": 5.2, "deg": 180},
        "clouds": {"all": 20},
        "dt": 1640995200,
        "sys": {
            "type": 2,
            "id": 2039034,
            "country": "US",
            "sunrise": 1640955847,
            "sunset": 1640989382
        },
        "timezone": -18000,
        "id": 5128581,
        "name": "New York",
        "cod": 200
    }


# =============================================================================
# Security Testing Fixtures
# =============================================================================

@pytest.fixture
def malicious_payloads():
    """Common malicious payloads for security testing"""
    return {
        'sql_injection': [
            "'; DROP TABLE miners; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; SELECT * FROM users; --"
        ],
        'xss': [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ],
        'command_injection': [
            "; ls -la",
            "& dir",
            "| cat /etc/passwd",
            "`id`"
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "....//....//....//etc/passwd"
        ]
    }


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_metrics():
    """Performance metrics collection for benchmarking"""
    class PerformanceMetrics:
        def __init__(self):
            self.metrics = {}
            
        def start_timer(self, name):
            self.metrics[name] = {'start': datetime.now()}
            
        def end_timer(self, name):
            if name in self.metrics:
                self.metrics[name]['end'] = datetime.now()
                self.metrics[name]['duration'] = (
                    self.metrics[name]['end'] - self.metrics[name]['start']
                ).total_seconds()
                
        def get_duration(self, name):
            return self.metrics.get(name, {}).get('duration', 0)
            
        def assert_performance(self, name, max_duration):
            duration = self.get_duration(name)
            assert duration <= max_duration, f"{name} took {duration}s, expected <= {max_duration}s"
    
    return PerformanceMetrics()


# =============================================================================
# Cleanup and Utilities
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically cleanup test files after each test"""
    yield
    
    # Clean up any test files or temporary directories
    test_dirs = ['test_models', 'test_data', 'test_logs']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)