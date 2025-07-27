from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class APIResponse(BaseModel):
    """Base API response model"""
    success: bool = Field(..., description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class ErrorResponse(APIResponse):
    """Error response model"""
    success: bool = Field(False, description="Always false for error responses")
    error: Dict[str, Any] = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Validation failed",
                "timestamp": "2024-07-27T10:30:45.123Z",
                "request_id": "req_1690456245123",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "details": {
                        "field": "frequency",
                        "message": "Value must be between 400 and 1200"
                    }
                }
            }
        }


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(50, ge=1, le=1000, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        return self.page_size


class PaginatedResponse(APIResponse):
    """Paginated response model"""
    data: List[Any] = Field(..., description="Response data items")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    
    @classmethod
    def create(cls, data: List[Any], params: PaginationParams, total_count: int):
        """Create paginated response"""
        total_pages = (total_count + params.page_size - 1) // params.page_size
        
        return cls(
            success=True,
            data=data,
            pagination={
                "page": params.page,
                "page_size": params.page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": params.page < total_pages,
                "has_previous": params.page > 1
            }
        )


# Miner Models
class MinerStatus(BaseModel):
    """Miner status model"""
    ip: str = Field(..., description="Miner IP address")
    hostname: Optional[str] = Field(None, description="Miner hostname")
    temperature: Optional[float] = Field(None, ge=0, le=150, description="Temperature in Celsius")
    hash_rate: Optional[float] = Field(None, ge=0, description="Hash rate in GH/s")
    power: Optional[float] = Field(None, ge=0, description="Power consumption in watts")
    voltage: Optional[float] = Field(None, ge=0, description="Input voltage")
    frequency: Optional[int] = Field(None, ge=100, le=2000, description="Frequency in MHz")
    core_voltage: Optional[int] = Field(None, ge=500, le=2000, description="Core voltage in mV")
    fan_rpm: Optional[int] = Field(None, ge=0, description="Fan RPM")
    shares_accepted: Optional[int] = Field(None, ge=0, description="Accepted shares")
    shares_rejected: Optional[int] = Field(None, ge=0, description="Rejected shares")
    uptime: Optional[int] = Field(None, ge=0, description="Uptime in seconds")
    version: Optional[str] = Field(None, description="Firmware version")
    efficiency: Optional[float] = Field(None, ge=0, description="Efficiency in GH/W")
    last_seen: datetime = Field(..., description="Last data update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
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
                "last_seen": "2024-07-27T10:30:45.123Z"
            }
        }


class MinerSettings(BaseModel):
    """Miner settings model"""
    frequency: int = Field(..., ge=400, le=1200, description="Frequency in MHz")
    core_voltage: int = Field(..., ge=800, le=1500, description="Core voltage in mV")
    autofanspeed: bool = Field(True, description="Enable automatic fan speed control")
    fanspeed: Optional[int] = Field(None, ge=0, le=100, description="Manual fan speed percentage")
    
    @validator('fanspeed')
    def validate_fanspeed(cls, v, values):
        if not values.get('autofanspeed') and v is None:
            raise ValueError('Manual fanspeed required when autofanspeed is disabled')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "frequency": 800,
                "core_voltage": 1200,
                "autofanspeed": True,
                "fanspeed": None
            }
        }


class MinerSettingsResponse(APIResponse):
    """Response for miner settings update"""
    data: Dict[str, Any] = Field(..., description="Updated settings confirmation")


class MinersListResponse(APIResponse):
    """Response for miners list"""
    data: List[MinerStatus] = Field(..., description="List of miner statuses")


class MinersSummary(BaseModel):
    """Miners summary statistics"""
    total_miners: int = Field(..., description="Total configured miners")
    online_miners: int = Field(..., description="Number of online miners")
    offline_miners: int = Field(..., description="Number of offline miners")
    total_hashrate: float = Field(..., description="Combined hash rate in GH/s")
    total_power: float = Field(..., description="Combined power consumption in watts")
    total_efficiency: float = Field(..., description="Overall efficiency in GH/W")
    average_temperature: float = Field(..., description="Average temperature across all miners")
    timestamp: datetime = Field(..., description="Summary timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "total_miners": 5,
                "online_miners": 4,
                "offline_miners": 1,
                "total_hashrate": 1942.8,
                "total_power": 64.2,
                "total_efficiency": 30.3,
                "average_temperature": 67.2,
                "timestamp": "2024-07-27T10:30:45.123Z"
            }
        }


class MinersSummaryResponse(APIResponse):
    """Response for miners summary"""
    data: MinersSummary = Field(..., description="Miners summary statistics")


# Benchmark Models
class BenchmarkRequest(BaseModel):
    """Benchmark request model"""
    ip: str = Field(..., description="Miner IP address", regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    frequency: int = Field(..., ge=400, le=1200, description="Frequency in MHz")
    core_voltage: int = Field(..., ge=800, le=1500, description="Core voltage in mV")
    duration: int = Field(600, ge=60, le=7200, description="Benchmark duration in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "ip": "192.168.1.100",
                "frequency": 800,
                "core_voltage": 1200,
                "duration": 600
            }
        }


class MultiBenchmarkRequest(BaseModel):
    """Multi-miner benchmark request model"""
    ips: List[str] = Field(..., min_items=1, description="List of miner IP addresses")
    frequency: int = Field(..., ge=400, le=1200, description="Frequency in MHz")
    core_voltage: int = Field(..., ge=800, le=1500, description="Core voltage in mV")
    duration: int = Field(600, ge=60, le=7200, description="Benchmark duration in seconds")
    
    @validator('ips')
    def validate_ips(cls, v):
        import re
        ip_pattern = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
        for ip in v:
            if not ip_pattern.match(ip):
                raise ValueError(f'Invalid IP address: {ip}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "ips": ["192.168.1.100", "192.168.1.101"],
                "frequency": 800,
                "core_voltage": 1200,
                "duration": 600
            }
        }


class BenchmarkResult(BaseModel):
    """Benchmark result model"""
    id: int = Field(..., description="Benchmark result ID")
    ip: str = Field(..., description="Miner IP address")
    frequency: int = Field(..., description="Frequency in MHz")
    core_voltage: int = Field(..., description="Core voltage in mV")
    average_hashrate: Optional[float] = Field(None, description="Average hash rate in GH/s")
    average_temperature: Optional[float] = Field(None, description="Average temperature in Celsius")
    efficiency_jth: Optional[float] = Field(None, description="Efficiency in J/TH")
    average_vr_temp: Optional[float] = Field(None, description="Average VR temperature")
    duration: int = Field(..., description="Actual benchmark duration")
    samples_count: int = Field(0, description="Number of samples collected")
    aborted: bool = Field(False, description="Whether benchmark was aborted")
    abort_reason: Optional[str] = Field(None, description="Reason for abort if applicable")
    timestamp: datetime = Field(..., description="Benchmark completion timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 123,
                "ip": "192.168.1.100",
                "frequency": 800,
                "core_voltage": 1200,
                "average_hashrate": 485.2,
                "average_temperature": 65.5,
                "efficiency_jth": 26.3,
                "average_vr_temp": 58.2,
                "duration": 600,
                "samples_count": 40,
                "aborted": False,
                "abort_reason": None,
                "timestamp": "2024-07-27T10:30:45.123Z"
            }
        }


class BenchmarkStartResponse(APIResponse):
    """Response for benchmark start"""
    data: Dict[str, Any] = Field(..., description="Benchmark start confirmation")


class BenchmarkResultsResponse(APIResponse):
    """Response for benchmark results"""
    data: List[BenchmarkResult] = Field(..., description="List of benchmark results")


class BenchmarkStatus(BaseModel):
    """Benchmark status model"""
    active_benchmarks: List[str] = Field(..., description="List of IPs with active benchmarks")
    total_active: int = Field(..., description="Number of active benchmarks")
    
    class Config:
        schema_extra = {
            "example": {
                "active_benchmarks": ["192.168.1.100", "192.168.1.101"],
                "total_active": 2
            }
        }


class BenchmarkStatusResponse(APIResponse):
    """Response for benchmark status"""
    data: BenchmarkStatus = Field(..., description="Current benchmark status")


# Event Models
class EventSeverity(str, Enum):
    """Event severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Event(BaseModel):
    """Event model"""
    id: int = Field(..., description="Event ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    ip: str = Field(..., description="Related IP address or SYSTEM")
    event_type: str = Field(..., description="Event type")
    message: str = Field(..., description="Event message")
    severity: EventSeverity = Field(..., description="Event severity")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 456,
                "timestamp": "2024-07-27T10:30:45.123Z",
                "ip": "192.168.1.100",
                "event_type": "BENCHMARK_COMPLETED",
                "message": "Benchmark completed successfully",
                "severity": "INFO"
            }
        }


class EventsResponse(APIResponse):
    """Response for events list"""
    data: List[Event] = Field(..., description="List of events")


class EventsQuery(BaseModel):
    """Query parameters for events"""
    ip: Optional[str] = Field(None, description="Filter by IP address")
    event_type: Optional[str] = Field(None, description="Filter by event type")
    severity: Optional[EventSeverity] = Field(None, description="Filter by severity")
    since: Optional[datetime] = Field(None, description="Events since timestamp")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of events")


# Health Models
class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Component health model"""
    component: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: str = Field(..., description="Health message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")
    timestamp: datetime = Field(..., description="Health check timestamp")
    duration_ms: float = Field(..., description="Health check duration in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "component": "database",
                "status": "healthy",
                "message": "Database is accessible",
                "details": {
                    "connection_info": {
                        "pool_size": 10,
                        "checked_out": 2
                    }
                },
                "timestamp": "2024-07-27T10:30:45.123Z",
                "duration_ms": 15.3
            }
        }


class SystemHealth(BaseModel):
    """System health model"""
    overall_status: HealthStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    total_checks: int = Field(..., description="Total number of health checks")
    status_counts: Dict[str, int] = Field(..., description="Count of each status type")
    checks: Dict[str, ComponentHealth] = Field(..., description="Individual component health")
    
    class Config:
        schema_extra = {
            "example": {
                "overall_status": "healthy",
                "timestamp": "2024-07-27T10:30:45.123Z",
                "total_checks": 8,
                "status_counts": {
                    "healthy": 7,
                    "degraded": 1,
                    "unhealthy": 0,
                    "unknown": 0
                },
                "checks": {
                    "database": {
                        "component": "database",
                        "status": "healthy",
                        "message": "Database is accessible",
                        "details": {},
                        "timestamp": "2024-07-27T10:30:45.123Z",
                        "duration_ms": 15.3
                    }
                }
            }
        }


class HealthResponse(APIResponse):
    """Response for health checks"""
    data: Union[SystemHealth, ComponentHealth] = Field(..., description="Health check data")


# Configuration Models
class ConfigUpdate(BaseModel):
    """Configuration update model"""
    key: str = Field(..., description="Configuration key")
    value: str = Field(..., description="Configuration value")
    
    class Config:
        schema_extra = {
            "example": {
                "key": "settings.benchmark_interval_sec",
                "value": "3600"
            }
        }


class ConfigResponse(APIResponse):
    """Response for configuration operations"""
    data: Dict[str, Any] = Field(..., description="Configuration data")


# Authentication Models
class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "password": "password123"
            }
        }


class TokenResponse(APIResponse):
    """Token response model"""
    data: Dict[str, str] = Field(..., description="Authentication token data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {
                    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                    "token_type": "bearer",
                    "expires_in": 3600
                }
            }
        }


class UserInfo(BaseModel):
    """User information model"""
    username: str = Field(..., description="Username")
    roles: List[str] = Field(..., description="User roles")
    permissions: List[str] = Field(..., description="User permissions")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "admin",
                "roles": ["admin", "operator"],
                "permissions": ["read", "write", "control"]
            }
        }


class UserInfoResponse(APIResponse):
    """Response for user info"""
    data: UserInfo = Field(..., description="User information")