from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, validator, IPvAnyAddress
from enum import Enum


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MinerDataSchema(BaseModel):
    """Validation schema for miner telemetry data"""
    hostname: Optional[str] = Field(None, max_length=50)
    temp: Optional[float] = Field(None, ge=0, le=150)  # 0-150°C
    hashRate: Optional[float] = Field(None, ge=0, le=10000)  # 0-10TH/s
    power: Optional[float] = Field(None, ge=0, le=5000)  # 0-5000W
    voltage: Optional[float] = Field(None, ge=0, le=15)  # 0-15V
    frequency: Optional[int] = Field(None, ge=100, le=2000)  # 100-2000 MHz
    coreVoltage: Optional[int] = Field(None, ge=500, le=2000)  # 500-2000 mV
    fanrpm: Optional[int] = Field(None, ge=0, le=10000)  # 0-10000 RPM
    sharesAccepted: Optional[int] = Field(None, ge=0)
    sharesRejected: Optional[int] = Field(None, ge=0)
    uptimeSeconds: Optional[int] = Field(None, ge=0)
    version: Optional[str] = Field(None, max_length=20)
    
    @validator('temp')
    def validate_temperature(cls, v):
        if v is not None and v > 100:
            raise ValueError('Temperature seems unusually high')
        return v
    
    @validator('hashRate')
    def validate_hashrate(cls, v):
        if v is not None and v < 0:
            raise ValueError('Hash rate cannot be negative')
        return v


class BenchmarkConfigSchema(BaseModel):
    """Validation schema for benchmark configuration"""
    ip: str = Field(..., regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    frequency: int = Field(..., ge=400, le=1200)  # 400-1200 MHz
    coreVoltage: int = Field(..., ge=800, le=1500)  # 800-1500 mV
    duration: int = Field(600, ge=60, le=7200)  # 1 minute to 2 hours
    
    @validator('frequency')
    def validate_frequency_range(cls, v):
        # Common frequency values
        valid_frequencies = list(range(400, 1201, 25))
        if v not in valid_frequencies:
            raise ValueError(f'Frequency must be in range 400-1200 MHz in 25MHz steps')
        return v
    
    @validator('coreVoltage')
    def validate_voltage_range(cls, v):
        # Common voltage values
        valid_voltages = list(range(800, 1501, 25))
        if v not in valid_voltages:
            raise ValueError(f'Core voltage must be in range 800-1500 mV in 25mV steps')
        return v


class MinerSettingsSchema(BaseModel):
    """Validation schema for miner settings"""
    frequency: int = Field(..., ge=400, le=1200)
    coreVoltage: int = Field(..., ge=800, le=1500)
    autofanspeed: bool = Field(True)
    fanspeed: Optional[int] = Field(None, ge=0, le=100)
    
    @validator('fanspeed')
    def validate_fanspeed_with_auto(cls, v, values):
        if not values.get('autofanspeed') and v is None:
            raise ValueError('Manual fanspeed required when autofanspeed is disabled')
        return v


class EventLogSchema(BaseModel):
    """Validation schema for event logs"""
    ip: str = Field(..., regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^SYSTEM$')
    event_type: str = Field(..., min_length=1, max_length=50)
    message: str = Field(..., min_length=1, max_length=1000)
    severity: EventSeverity = Field(EventSeverity.INFO)


class MinerConfigurationSchema(BaseModel):
    """Validation schema for miner configuration"""
    ip: str = Field(..., regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    hostname: Optional[str] = Field(None, max_length=50)
    alias: Optional[str] = Field(None, max_length=50)
    color: str = Field('#3498db', regex=r'^#[0-9A-Fa-f]{6}$')
    is_active: bool = Field(True)
    location: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = Field(None, max_length=500)
    
    target_frequency: Optional[int] = Field(None, ge=400, le=1200)
    target_voltage: Optional[int] = Field(None, ge=800, le=1500)
    temp_limit: float = Field(73.0, ge=40, le=90)
    temp_overheat: float = Field(75.0, ge=45, le=95)
    
    autopilot_enabled: bool = Field(True)
    benchmark_interval: int = Field(86400, ge=3600, le=604800)  # 1 hour to 1 week
    
    @validator('temp_overheat')
    def validate_overheat_higher_than_limit(cls, v, values):
        temp_limit = values.get('temp_limit', 73.0)
        if v <= temp_limit:
            raise ValueError('Overheat temperature must be higher than temp limit')
        return v


class BenchmarkResultSchema(BaseModel):
    """Validation schema for benchmark results"""
    ip: str = Field(..., regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    frequency: int = Field(..., ge=400, le=1200)
    coreVoltage: int = Field(..., ge=800, le=1500)
    averageHashRate: Optional[float] = Field(None, ge=0, le=10000)
    averageTemperature: Optional[float] = Field(None, ge=0, le=150)
    efficiencyJTH: Optional[float] = Field(None, ge=0, le=1000)
    averageVRTemp: Optional[float] = Field(None, ge=0, le=150)
    duration: int = Field(..., ge=60, le=7200)
    samples_count: int = Field(0, ge=0)
    aborted: bool = Field(False)
    abort_reason: Optional[str] = Field(None, max_length=100)


class EfficiencyMarkerSchema(BaseModel):
    """Validation schema for efficiency markers"""
    ip: str = Field(..., regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    efficiency: float = Field(..., ge=0, le=1000)
    hashRate: Optional[float] = Field(None, ge=0, le=10000)
    power: Optional[float] = Field(None, ge=0, le=5000)
    temperature: Optional[float] = Field(None, ge=0, le=150)
    frequency: Optional[int] = Field(None, ge=400, le=1200)
    coreVoltage: Optional[int] = Field(None, ge=800, le=1500)


class QueryParametersSchema(BaseModel):
    """Validation schema for query parameters"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=10000)
    offset: int = Field(0, ge=0)
    ip_filter: Optional[str] = Field(None, regex=r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        start_time = values.get('start_time')
        if start_time and v and v <= start_time:
            raise ValueError('End time must be after start time')
        return v


class SystemConfigSchema(BaseModel):
    """Validation schema for system configuration"""
    key: str = Field(..., min_length=1, max_length=100, regex=r'^[a-zA-Z0-9._-]+$')
    value: str = Field(..., max_length=1000)
    value_type: str = Field('string', regex=r'^(string|int|float|bool|json)$')
    description: Optional[str] = Field(None, max_length=500)
    category: str = Field('general', max_length=50)
    
    @validator('value')
    def validate_value_type(cls, v, values):
        value_type = values.get('value_type', 'string')
        
        if value_type == 'int':
            try:
                int(v)
            except ValueError:
                raise ValueError('Value must be a valid integer')
        elif value_type == 'float':
            try:
                float(v)
            except ValueError:
                raise ValueError('Value must be a valid float')
        elif value_type == 'bool':
            if v.lower() not in ['true', 'false', '1', '0']:
                raise ValueError('Value must be a valid boolean')
        elif value_type == 'json':
            import json
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError('Value must be valid JSON')
        
        return v