from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.sql import func
from .base import Base, BaseModel, TimestampMixin


class MinerLog(Base, TimestampMixin):
    """Model for miner telemetry logs"""
    __tablename__ = 'logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ip = Column(String(15), nullable=False, index=True)
    hostname = Column(String(50))
    temp = Column(Float)
    hashRate = Column(Float)
    power = Column(Float)
    voltage = Column(Float)
    frequency = Column(Integer)
    coreVoltage = Column(Integer)
    fanrpm = Column(Integer)
    sharesAccepted = Column(Integer)
    sharesRejected = Column(Integer)
    uptime = Column(Integer)
    version = Column(String(20))
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_ip_timestamp', 'ip', 'timestamp'),
        Index('idx_timestamp_desc', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MinerLog(ip={self.ip}, timestamp={self.timestamp})>"


class TuningStatus(Base, TimestampMixin):
    """Model for tuning status and best settings"""
    __tablename__ = 'tuning_status'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ip = Column(String(15), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    frequency = Column(Integer, nullable=False)
    coreVoltage = Column(Integer, nullable=False)
    avgHashRate = Column(Float)
    avgTemperature = Column(Float)
    avgEfficiency = Column(Float)
    duration = Column(Integer)
    failed = Column(Boolean, default=False)
    
    # Composite index for finding best settings per IP
    __table_args__ = (
        Index('idx_ip_efficiency', 'ip', 'avgEfficiency'),
    )
    
    def __repr__(self):
        return f"<TuningStatus(ip={self.ip}, efficiency={self.avgEfficiency})>"


class ProtocolEvent(Base, TimestampMixin):
    """Model for system events and protocol messages"""
    __tablename__ = 'protocol'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ip = Column(String(15), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    message = Column(Text)
    severity = Column(String(20), default='INFO')  # INFO, WARNING, ERROR, CRITICAL
    
    # Index for querying recent events
    __table_args__ = (
        Index('idx_timestamp_desc_events', 'timestamp'),
        Index('idx_ip_event_type', 'ip', 'event_type'),
    )
    
    def __repr__(self):
        return f"<ProtocolEvent(ip={self.ip}, event_type={self.event_type})>"


class EfficiencyMarker(Base, TimestampMixin):
    """Model for efficiency tracking markers"""
    __tablename__ = 'efficiency_markers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ip = Column(String(15), nullable=False, index=True)
    efficiency = Column(Float, nullable=False)
    hashRate = Column(Float)
    power = Column(Float)
    temperature = Column(Float)
    frequency = Column(Integer)
    coreVoltage = Column(Integer)
    
    # Index for time-series queries
    __table_args__ = (
        Index('idx_ip_timestamp_eff', 'ip', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<EfficiencyMarker(ip={self.ip}, efficiency={self.efficiency})>"


class BenchmarkResult(Base, TimestampMixin):
    """Model for benchmark results"""
    __tablename__ = 'benchmark_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ip = Column(String(15), nullable=False, index=True)
    frequency = Column(Integer, nullable=False)
    coreVoltage = Column(Integer, nullable=False)
    averageHashRate = Column(Float)
    averageTemperature = Column(Float)
    efficiencyJTH = Column(Float)  # J/TH efficiency
    averageVRTemp = Column(Float)
    duration = Column(Integer)
    samples_count = Column(Integer, default=0)
    aborted = Column(Boolean, default=False)
    abort_reason = Column(String(100))
    
    # Indexes for performance queries
    __table_args__ = (
        Index('idx_ip_efficiency_desc', 'ip', 'efficiencyJTH'),
        Index('idx_hashrate_desc', 'averageHashRate'),
        Index('idx_frequency_voltage', 'frequency', 'coreVoltage'),
    )
    
    def __repr__(self):
        return f"<BenchmarkResult(ip={self.ip}, efficiency={self.efficiencyJTH})>"


class MinerConfiguration(Base, TimestampMixin):
    """Model for miner configuration and metadata"""
    __tablename__ = 'miner_configurations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ip = Column(String(15), nullable=False, unique=True, index=True)
    hostname = Column(String(50))
    alias = Column(String(50))  # User-friendly name
    color = Column(String(7), default='#3498db')  # Hex color for UI
    is_active = Column(Boolean, default=True)
    location = Column(String(100))
    notes = Column(Text)
    
    # Current settings
    target_frequency = Column(Integer)
    target_voltage = Column(Integer)
    temp_limit = Column(Float, default=73.0)
    temp_overheat = Column(Float, default=75.0)
    
    # Autopilot settings
    autopilot_enabled = Column(Boolean, default=True)
    benchmark_interval = Column(Integer, default=86400)  # seconds
    
    def __repr__(self):
        return f"<MinerConfiguration(ip={self.ip}, alias={self.alias})>"


class SystemConfiguration(Base, TimestampMixin):
    """Model for system-wide configuration"""
    __tablename__ = 'system_configuration'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text)
    value_type = Column(String(20), default='string')  # string, int, float, bool, json
    description = Column(Text)
    category = Column(String(50), default='general')
    
    def __repr__(self):
        return f"<SystemConfiguration(key={self.key})>"