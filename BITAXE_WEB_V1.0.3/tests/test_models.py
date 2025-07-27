"""
Tests for data models and validation
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from api.models import (
    MinerStatus, MinerSettings, BenchmarkRequest, MultiBenchmarkRequest,
    Event, EventSeverity, ComponentHealth, HealthStatus,
    LoginRequest, ConfigUpdate, PaginationParams, PaginatedResponse
)


class TestMinerModels:
    """Test miner-related models"""
    
    def test_miner_status_valid(self):
        """Test valid miner status model"""
        data = {
            'ip': '192.168.1.100',
            'hostname': 'test-miner',
            'temperature': 65.5,
            'hash_rate': 485.2,
            'power': 12.8,
            'voltage': 5.1,
            'frequency': 800,
            'core_voltage': 1200,
            'fan_rpm': 3500,
            'shares_accepted': 1250,
            'shares_rejected': 5,
            'uptime': 86400,
            'version': '2.0.4',
            'efficiency': 37.9,
            'last_seen': datetime.now()
        }
        
        miner = MinerStatus(**data)
        assert miner.ip == '192.168.1.100'
        assert miner.temperature == 65.5
        assert miner.hash_rate == 485.2
    
    def test_miner_status_optional_fields(self):
        """Test miner status with minimal required fields"""
        data = {
            'ip': '192.168.1.100',
            'last_seen': datetime.now()
        }
        
        miner = MinerStatus(**data)
        assert miner.ip == '192.168.1.100'
        assert miner.hostname is None
        assert miner.temperature is None
    
    def test_miner_status_validation_temperature(self):
        """Test temperature validation"""
        with pytest.raises(ValidationError):
            MinerStatus(
                ip='192.168.1.100',
                temperature=200,  # Too high
                last_seen=datetime.now()
            )
        
        with pytest.raises(ValidationError):
            MinerStatus(
                ip='192.168.1.100',
                temperature=-10,  # Too low
                last_seen=datetime.now()
            )
    
    def test_miner_settings_valid(self):
        """Test valid miner settings"""
        settings = MinerSettings(
            frequency=800,
            core_voltage=1200,
            autofanspeed=True
        )
        
        assert settings.frequency == 800
        assert settings.core_voltage == 1200
        assert settings.autofanspeed is True
    
    def test_miner_settings_validation(self):
        """Test miner settings validation"""
        # Frequency out of range
        with pytest.raises(ValidationError):
            MinerSettings(frequency=5000, core_voltage=1200)
        
        # Core voltage out of range
        with pytest.raises(ValidationError):
            MinerSettings(frequency=800, core_voltage=500)
    
    def test_miner_settings_fanspeed_validation(self):
        """Test fanspeed validation with autofanspeed disabled"""
        # Should require fanspeed when autofanspeed is False
        with pytest.raises(ValidationError):
            MinerSettings(
                frequency=800,
                core_voltage=1200,
                autofanspeed=False
                # Missing fanspeed
            )
        
        # Should work with fanspeed provided
        settings = MinerSettings(
            frequency=800,
            core_voltage=1200,
            autofanspeed=False,
            fanspeed=75
        )
        assert settings.fanspeed == 75


class TestBenchmarkModels:
    """Test benchmark-related models"""
    
    def test_benchmark_request_valid(self):
        """Test valid benchmark request"""
        request = BenchmarkRequest(
            ip='192.168.1.100',
            frequency=800,
            core_voltage=1200,
            duration=600
        )
        
        assert request.ip == '192.168.1.100'
        assert request.frequency == 800
        assert request.duration == 600
    
    def test_benchmark_request_defaults(self):
        """Test benchmark request with defaults"""
        request = BenchmarkRequest(
            ip='192.168.1.100',
            frequency=800,
            core_voltage=1200
            # duration should default to 600
        )
        
        assert request.duration == 600
    
    def test_benchmark_request_ip_validation(self):
        """Test IP address validation in benchmark request"""
        with pytest.raises(ValidationError):
            BenchmarkRequest(
                ip='invalid-ip',  # Invalid IP format
                frequency=800,
                core_voltage=1200
            )
    
    def test_multi_benchmark_request(self):
        """Test multi-benchmark request"""
        request = MultiBenchmarkRequest(
            ips=['192.168.1.100', '192.168.1.101'],
            frequency=800,
            core_voltage=1200,
            duration=600
        )
        
        assert len(request.ips) == 2
        assert '192.168.1.100' in request.ips
    
    def test_multi_benchmark_ip_validation(self):
        """Test IP validation in multi-benchmark request"""
        with pytest.raises(ValidationError):
            MultiBenchmarkRequest(
                ips=['192.168.1.100', 'invalid-ip'],
                frequency=800,
                core_voltage=1200
            )
    
    def test_benchmark_result_model(self):
        """Test benchmark result model"""
        from api.models import BenchmarkResult
        
        result = BenchmarkResult(
            id=123,
            ip='192.168.1.100',
            frequency=800,
            core_voltage=1200,
            average_hashrate=485.2,
            average_temperature=65.5,
            efficiency_jth=26.3,
            duration=600,
            timestamp=datetime.now()
        )
        
        assert result.id == 123
        assert result.ip == '192.168.1.100'
        assert result.average_hashrate == 485.2


class TestEventModels:
    """Test event-related models"""
    
    def test_event_valid(self):
        """Test valid event model"""
        event = Event(
            id=1,
            timestamp=datetime.now(),
            ip='192.168.1.100',
            event_type='BENCHMARK_COMPLETED',
            message='Benchmark completed successfully',
            severity=EventSeverity.INFO
        )
        
        assert event.id == 1
        assert event.event_type == 'BENCHMARK_COMPLETED'
        assert event.severity == EventSeverity.INFO
    
    def test_event_severity_enum(self):
        """Test event severity enumeration"""
        severities = [
            EventSeverity.INFO,
            EventSeverity.WARNING,
            EventSeverity.ERROR,
            EventSeverity.CRITICAL
        ]
        
        for severity in severities:
            event = Event(
                id=1,
                timestamp=datetime.now(),
                ip='SYSTEM',
                event_type='TEST',
                message='Test message',
                severity=severity
            )
            assert event.severity == severity


class TestHealthModels:
    """Test health monitoring models"""
    
    def test_component_health(self):
        """Test component health model"""
        health = ComponentHealth(
            component='database',
            status=HealthStatus.HEALTHY,
            message='Database is accessible',
            details={'connection_info': {'pool_size': 10}},
            timestamp=datetime.now(),
            duration_ms=15.3
        )
        
        assert health.component == 'database'
        assert health.status == HealthStatus.HEALTHY
        assert health.duration_ms == 15.3
    
    def test_health_status_enum(self):
        """Test health status enumeration"""
        statuses = [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN
        ]
        
        for status in statuses:
            health = ComponentHealth(
                component='test',
                status=status,
                message='Test message',
                timestamp=datetime.now(),
                duration_ms=10.0
            )
            assert health.status == status


class TestAuthModels:
    """Test authentication models"""
    
    def test_login_request_valid(self):
        """Test valid login request"""
        login = LoginRequest(
            username='admin',
            password='password123'
        )
        
        assert login.username == 'admin'
        assert login.password == 'password123'
    
    def test_login_request_validation(self):
        """Test login request validation"""
        # Empty username
        with pytest.raises(ValidationError):
            LoginRequest(username='', password='password')
        
        # Empty password
        with pytest.raises(ValidationError):
            LoginRequest(username='admin', password='')


class TestConfigModels:
    """Test configuration models"""
    
    def test_config_update_valid(self):
        """Test valid config update"""
        update = ConfigUpdate(
            key='settings.temp_limit',
            value='80'
        )
        
        assert update.key == 'settings.temp_limit'
        assert update.value == '80'


class TestPaginationModels:
    """Test pagination models"""
    
    def test_pagination_params_defaults(self):
        """Test pagination parameters with defaults"""
        params = PaginationParams()
        
        assert params.page == 1
        assert params.page_size == 50
        assert params.offset == 0
        assert params.limit == 50
    
    def test_pagination_params_custom(self):
        """Test pagination parameters with custom values"""
        params = PaginationParams(page=3, page_size=25)
        
        assert params.page == 3
        assert params.page_size == 25
        assert params.offset == 50  # (3-1) * 25
        assert params.limit == 25
    
    def test_pagination_params_validation(self):
        """Test pagination parameters validation"""
        # Page must be >= 1
        with pytest.raises(ValidationError):
            PaginationParams(page=0)
        
        # Page size must be >= 1
        with pytest.raises(ValidationError):
            PaginationParams(page_size=0)
        
        # Page size must be <= 1000
        with pytest.raises(ValidationError):
            PaginationParams(page_size=2000)
    
    def test_paginated_response_creation(self):
        """Test paginated response creation"""
        data = [{'id': 1}, {'id': 2}, {'id': 3}]
        params = PaginationParams(page=1, page_size=2)
        
        response = PaginatedResponse.create(
            data=data[:2],  # First 2 items
            params=params,
            total_count=3
        )
        
        assert response.success is True
        assert len(response.data) == 2
        
        pagination = response.pagination
        assert pagination['page'] == 1
        assert pagination['page_size'] == 2
        assert pagination['total_count'] == 3
        assert pagination['total_pages'] == 2
        assert pagination['has_next'] is True
        assert pagination['has_previous'] is False


class TestModelSerialization:
    """Test model serialization and deserialization"""
    
    def test_miner_status_dict_conversion(self):
        """Test miner status to dict conversion"""
        miner = MinerStatus(
            ip='192.168.1.100',
            hostname='test-miner',
            temperature=65.5,
            last_seen=datetime.now()
        )
        
        miner_dict = miner.dict()
        
        assert miner_dict['ip'] == '192.168.1.100'
        assert miner_dict['hostname'] == 'test-miner'
        assert miner_dict['temperature'] == 65.5
        assert 'last_seen' in miner_dict
    
    def test_model_json_serialization(self):
        """Test model JSON serialization"""
        settings = MinerSettings(
            frequency=800,
            core_voltage=1200,
            autofanspeed=True
        )
        
        json_str = settings.json()
        assert '800' in json_str
        assert '1200' in json_str
        assert 'true' in json_str.lower()
    
    def test_model_parsing_from_dict(self):
        """Test creating models from dictionary data"""
        data = {
            'username': 'testuser',
            'password': 'testpass'
        }
        
        login = LoginRequest.parse_obj(data)
        assert login.username == 'testuser'
        assert login.password == 'testpass'
    
    def test_model_validation_errors(self):
        """Test model validation error details"""
        try:
            MinerSettings(
                frequency=5000,  # Invalid
                core_voltage=1200,
                autofanspeed=True
            )
        except ValidationError as e:
            errors = e.errors()
            assert len(errors) > 0
            assert 'frequency' in str(errors[0])


class TestModelExamples:
    """Test that model examples are valid"""
    
    def test_miner_status_example(self):
        """Test that MinerStatus example is valid"""
        example_data = {
            "ip": "192.168.1.100",
            "hostname": "bitaxe-001",
            "temperature": 65.5,
            "hash_rate": 485.2,
            "power": 12.8,
            "voltage": 5.1,
            "frequency": 800,
            "core_voltage": 1200,
            "fan_rpm": 3500,
            "shares_accepted": 1250,
            "shares_rejected": 5,
            "uptime": 86400,
            "version": "2.0.4",
            "efficiency": 37.9,
            "last_seen": datetime.now()
        }
        
        miner = MinerStatus(**example_data)
        assert miner.ip == "192.168.1.100"
    
    def test_benchmark_request_example(self):
        """Test that BenchmarkRequest example is valid"""
        example_data = {
            "ip": "192.168.1.100",
            "frequency": 800,
            "core_voltage": 1200,
            "duration": 600
        }
        
        request = BenchmarkRequest(**example_data)
        assert request.ip == "192.168.1.100"
    
    def test_event_example(self):
        """Test that Event example is valid"""
        example_data = {
            "id": 456,
            "timestamp": datetime.now(),
            "ip": "192.168.1.100",
            "event_type": "BENCHMARK_COMPLETED",
            "message": "Benchmark completed successfully",
            "severity": "INFO"
        }
        
        event = Event(**example_data)
        assert event.event_type == "BENCHMARK_COMPLETED"