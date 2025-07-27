from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from models.miner_models import (
    MinerLog, TuningStatus, ProtocolEvent, EfficiencyMarker, 
    BenchmarkResult, MinerConfiguration, SystemConfiguration
)
from .base_repository import BaseRepository, TimestampRepository


class MinerLogRepository(TimestampRepository[MinerLog]):
    """Repository for miner telemetry logs"""
    
    def __init__(self, session: Session):
        super().__init__(session, MinerLog)
    
    def get_latest_status_by_ip(self, ip: str) -> Optional[MinerLog]:
        """Get latest status for specific IP"""
        return self.session.query(MinerLog).filter(
            MinerLog.ip == ip
        ).order_by(desc(MinerLog.timestamp)).first()
    
    def get_latest_status_all(self) -> List[MinerLog]:
        """Get latest status for all miners"""
        # Use window function to get latest record per IP
        subquery = self.session.query(
            MinerLog,
            func.row_number().over(
                partition_by=MinerLog.ip,
                order_by=desc(MinerLog.timestamp)
            ).label('rn')
        ).subquery()
        
        return self.session.query(MinerLog).select_from(subquery).filter(
            subquery.c.rn == 1
        ).all()
    
    def get_history_data(self, start_time: datetime, end_time: datetime, 
                        ips: List[str] = None) -> Dict[str, List[MinerLog]]:
        """Get historical data grouped by IP"""
        query = self.session.query(MinerLog).filter(
            and_(
                MinerLog.timestamp >= start_time,
                MinerLog.timestamp <= end_time
            )
        )
        
        if ips:
            query = query.filter(MinerLog.ip.in_(ips))
        
        records = query.order_by(MinerLog.timestamp).all()
        
        # Group by IP
        result = {}
        for record in records:
            if record.ip not in result:
                result[record.ip] = []
            result[record.ip].append(record)
        
        return result
    
    def get_efficiency_stats(self, ip: str, hours: int = 24) -> Dict[str, float]:
        """Get efficiency statistics for an IP"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        result = self.session.query(
            func.avg(MinerLog.hashRate / MinerLog.power).label('avg_efficiency'),
            func.min(MinerLog.hashRate / MinerLog.power).label('min_efficiency'),
            func.max(MinerLog.hashRate / MinerLog.power).label('max_efficiency'),
            func.avg(MinerLog.temp).label('avg_temp'),
            func.avg(MinerLog.hashRate).label('avg_hashrate'),
            func.avg(MinerLog.power).label('avg_power')
        ).filter(
            and_(
                MinerLog.ip == ip,
                MinerLog.timestamp >= cutoff_time,
                MinerLog.power > 0  # Avoid division by zero
            )
        ).first()
        
        return {
            'avg_efficiency': result.avg_efficiency or 0,
            'min_efficiency': result.min_efficiency or 0,
            'max_efficiency': result.max_efficiency or 0,
            'avg_temp': result.avg_temp or 0,
            'avg_hashrate': result.avg_hashrate or 0,
            'avg_power': result.avg_power or 0
        }


class TuningStatusRepository(BaseRepository[TuningStatus]):
    """Repository for tuning status and best settings"""
    
    def __init__(self, session: Session):
        super().__init__(session, TuningStatus)
    
    def get_best_efficiency_by_ip(self, ip: str) -> Optional[TuningStatus]:
        """Get best efficiency settings for specific IP"""
        return self.session.query(TuningStatus).filter(
            and_(
                TuningStatus.ip == ip,
                TuningStatus.failed == False
            )
        ).order_by(asc(TuningStatus.avgEfficiency)).first()
    
    def get_top_performers(self, limit: int = 10) -> List[TuningStatus]:
        """Get top performing settings across all miners"""
        return self.session.query(TuningStatus).filter(
            TuningStatus.failed == False
        ).order_by(asc(TuningStatus.avgEfficiency)).limit(limit).all()
    
    def update_best_settings(self, ip: str, frequency: int, core_voltage: int, 
                           avg_efficiency: float) -> TuningStatus:
        """Update or insert best settings for IP"""
        existing = self.find_one_by(ip=ip)
        
        if existing:
            if avg_efficiency < existing.avgEfficiency:
                # Better efficiency found, update
                return self.update(existing.id, 
                                 frequency=frequency,
                                 coreVoltage=core_voltage,
                                 avgEfficiency=avg_efficiency,
                                 timestamp=datetime.utcnow())
            return existing
        else:
            # Create new record
            return self.create(
                ip=ip,
                frequency=frequency,
                coreVoltage=core_voltage,
                avgEfficiency=avg_efficiency,
                timestamp=datetime.utcnow()
            )


class ProtocolEventRepository(TimestampRepository[ProtocolEvent]):
    """Repository for protocol events and system logs"""
    
    def __init__(self, session: Session):
        super().__init__(session, ProtocolEvent)
    
    def log_event(self, ip: str, event_type: str, message: str, 
                  severity: str = 'INFO') -> ProtocolEvent:
        """Log a new event"""
        return self.create(
            ip=ip,
            event_type=event_type,
            message=message,
            severity=severity,
            timestamp=datetime.utcnow()
        )
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[ProtocolEvent]:
        """Get events by type"""
        return self.find_by(
            event_type=event_type,
            limit=limit,
            order_by='timestamp',
            desc_order=True
        )
    
    def get_events_by_ip(self, ip: str, limit: int = 100) -> List[ProtocolEvent]:
        """Get events for specific IP"""
        return self.find_by(
            ip=ip,
            limit=limit,
            order_by='timestamp',
            desc_order=True
        )
    
    def get_error_events(self, hours: int = 24) -> List[ProtocolEvent]:
        """Get error events from last N hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.find_by(
            severity=['ERROR', 'CRITICAL'],
            timestamp={'operator': 'gte', 'value': cutoff_time},
            order_by='timestamp',
            desc_order=True
        )


class EfficiencyMarkerRepository(TimestampRepository[EfficiencyMarker]):
    """Repository for efficiency tracking markers"""
    
    def __init__(self, session: Session):
        super().__init__(session, EfficiencyMarker)
    
    def log_efficiency(self, ip: str, efficiency: float, hashrate: float, 
                      power: float, temperature: float, frequency: int, 
                      core_voltage: int) -> EfficiencyMarker:
        """Log efficiency marker"""
        return self.create(
            ip=ip,
            efficiency=efficiency,
            hashRate=hashrate,
            power=power,
            temperature=temperature,
            frequency=frequency,
            coreVoltage=core_voltage,
            timestamp=datetime.utcnow()
        )
    
    def get_efficiency_trend(self, ip: str, hours: int = 24) -> List[EfficiencyMarker]:
        """Get efficiency trend for IP"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.find_by(
            ip=ip,
            timestamp={'operator': 'gte', 'value': cutoff_time},
            order_by='timestamp',
            desc_order=False
        )
    
    def get_export_data(self, ip: str, start_time: datetime, 
                       end_time: datetime) -> List[EfficiencyMarker]:
        """Get efficiency data for CSV export"""
        return self.get_by_time_range(start_time, end_time)


class BenchmarkResultRepository(TimestampRepository[BenchmarkResult]):
    """Repository for benchmark results"""
    
    def __init__(self, session: Session):
        super().__init__(session, BenchmarkResult)
    
    def save_result(self, ip: str, frequency: int, core_voltage: int, 
                   avg_hashrate: float, avg_temp: float, efficiency: float, 
                   duration: int, **kwargs) -> BenchmarkResult:
        """Save benchmark result"""
        return self.create(
            ip=ip,
            frequency=frequency,
            coreVoltage=core_voltage,
            averageHashRate=avg_hashrate,
            averageTemperature=avg_temp,
            efficiencyJTH=efficiency,
            duration=duration,
            timestamp=datetime.utcnow(),
            **kwargs
        )
    
    def get_results_by_ip(self, ip: str, limit: int = 10) -> List[BenchmarkResult]:
        """Get benchmark results for specific IP"""
        return self.find_by(
            ip=ip,
            limit=limit,
            order_by='timestamp',
            desc_order=True
        )
    
    def get_best_results(self, limit: int = 50) -> List[BenchmarkResult]:
        """Get best benchmark results by hash rate"""
        return self.find_by(
            aborted=False,
            limit=limit,
            order_by='averageHashRate',
            desc_order=True
        )
    
    def get_best_efficiency_results(self, limit: int = 50) -> List[BenchmarkResult]:
        """Get best benchmark results by efficiency"""
        return self.session.query(BenchmarkResult).filter(
            and_(
                BenchmarkResult.aborted == False,
                BenchmarkResult.efficiencyJTH.isnot(None)
            )
        ).order_by(asc(BenchmarkResult.efficiencyJTH)).limit(limit).all()
    
    def get_frequency_voltage_performance(self, frequency: int, 
                                        core_voltage: int) -> Dict[str, Any]:
        """Get performance stats for specific frequency/voltage combination"""
        result = self.session.query(
            func.avg(BenchmarkResult.averageHashRate).label('avg_hashrate'),
            func.avg(BenchmarkResult.averageTemperature).label('avg_temp'),
            func.avg(BenchmarkResult.efficiencyJTH).label('avg_efficiency'),
            func.count(BenchmarkResult.id).label('sample_count')
        ).filter(
            and_(
                BenchmarkResult.frequency == frequency,
                BenchmarkResult.coreVoltage == core_voltage,
                BenchmarkResult.aborted == False
            )
        ).first()
        
        return {
            'avg_hashrate': result.avg_hashrate or 0,
            'avg_temp': result.avg_temp or 0,
            'avg_efficiency': result.avg_efficiency or 0,
            'sample_count': result.sample_count or 0
        }


class MinerConfigurationRepository(BaseRepository[MinerConfiguration]):
    """Repository for miner configurations"""
    
    def __init__(self, session: Session):
        super().__init__(session, MinerConfiguration)
    
    def get_by_ip(self, ip: str) -> Optional[MinerConfiguration]:
        """Get configuration by IP"""
        return self.find_one_by(ip=ip)
    
    def get_active_miners(self) -> List[MinerConfiguration]:
        """Get all active miner configurations"""
        return self.find_by(is_active=True)
    
    def update_or_create_config(self, ip: str, **kwargs) -> MinerConfiguration:
        """Update existing config or create new one"""
        existing = self.get_by_ip(ip)
        
        if existing:
            return self.update(existing.id, **kwargs)
        else:
            return self.create(ip=ip, **kwargs)


class SystemConfigurationRepository(BaseRepository[SystemConfiguration]):
    """Repository for system configuration"""
    
    def __init__(self, session: Session):
        super().__init__(session, SystemConfiguration)
    
    def get_by_key(self, key: str) -> Optional[SystemConfiguration]:
        """Get configuration by key"""
        return self.find_one_by(key=key)
    
    def get_by_category(self, category: str) -> List[SystemConfiguration]:
        """Get all configurations in category"""
        return self.find_by(category=category)
    
    def set_config(self, key: str, value: str, value_type: str = 'string', 
                   description: str = None, category: str = 'general') -> SystemConfiguration:
        """Set configuration value"""
        existing = self.get_by_key(key)
        
        if existing:
            return self.update(existing.id, 
                             value=value, 
                             value_type=value_type,
                             description=description,
                             category=category)
        else:
            return self.create(
                key=key,
                value=value,
                value_type=value_type,
                description=description,
                category=category
            )
    
    def get_typed_value(self, key: str, default=None):
        """Get configuration value with proper type conversion"""
        config = self.get_by_key(key)
        if not config:
            return default
        
        value = config.value
        value_type = config.value_type
        
        try:
            if value_type == 'int':
                return int(value)
            elif value_type == 'float':
                return float(value)
            elif value_type == 'bool':
                return value.lower() in ['true', '1', 'yes', 'on']
            elif value_type == 'json':
                import json
                return json.loads(value)
            else:
                return value
        except (ValueError, TypeError):
            return default