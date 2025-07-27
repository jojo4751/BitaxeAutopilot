from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from pydantic import ValidationError

from database import get_database_manager
from repositories.repository_factory import RepositoryFactory
from models.validation_schemas import (
    MinerDataSchema, BenchmarkResultSchema, EventLogSchema,
    EfficiencyMarkerSchema, QueryParametersSchema
)


class DatabaseServiceV2:
    """Enhanced database service using repositories and validation"""
    
    def __init__(self, config_service=None):
        self.config_service = config_service
        self.db_manager = get_database_manager()
    
    @contextmanager
    def get_repository_factory(self):
        """Get repository factory with session management"""
        with self.db_manager.get_session() as session:
            yield RepositoryFactory(session)
    
    def validate_and_log_miner_data(self, ip: str, data: Dict[str, Any]) -> bool:
        """Validate and log miner telemetry data"""
        try:
            # Validate data
            validated_data = MinerDataSchema(**data)
            
            with self.get_repository_factory() as factory:
                repo = factory.get_miner_log_repository()
                
                # Create log entry
                repo.create(
                    timestamp=datetime.utcnow(),
                    ip=ip,
                    hostname=validated_data.hostname,
                    temp=validated_data.temp,
                    hashRate=validated_data.hashRate,
                    power=validated_data.power,
                    voltage=validated_data.voltage,
                    frequency=validated_data.frequency,
                    coreVoltage=validated_data.coreVoltage,
                    fanrpm=validated_data.fanrpm,
                    sharesAccepted=validated_data.sharesAccepted,
                    sharesRejected=validated_data.sharesRejected,
                    uptime=validated_data.uptimeSeconds,
                    version=validated_data.version
                )
                
                return True
                
        except ValidationError as e:
            self.log_event(ip, "VALIDATION_ERROR", f"Invalid miner data: {e}", "ERROR")
            return False
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to log data: {e}", "ERROR")
            return False
    
    def log_event(self, ip: str, event_type: str, message: str, severity: str = "INFO") -> bool:
        """Log system event with validation"""
        try:
            # Validate event data
            validated_event = EventLogSchema(
                ip=ip,
                event_type=event_type,
                message=message,
                severity=severity
            )
            
            with self.get_repository_factory() as factory:
                repo = factory.get_protocol_event_repository()
                repo.log_event(
                    validated_event.ip,
                    validated_event.event_type,
                    validated_event.message,
                    validated_event.severity
                )
                return True
                
        except ValidationError as e:
            print(f"Event validation error: {e}")
            return False
        except Exception as e:
            print(f"Failed to log event: {e}")
            return False
    
    def save_benchmark_result(self, ip: str, frequency: int, core_voltage: int,
                            avg_hashrate: float, avg_temp: float, efficiency: float,
                            duration: int, **kwargs) -> bool:
        """Save benchmark result with validation"""
        try:
            # Prepare benchmark data
            benchmark_data = {
                'ip': ip,
                'frequency': frequency,
                'coreVoltage': core_voltage,
                'averageHashRate': avg_hashrate,
                'averageTemperature': avg_temp,
                'efficiencyJTH': efficiency,
                'duration': duration,
                **kwargs
            }
            
            # Validate benchmark data
            validated_result = BenchmarkResultSchema(**benchmark_data)
            
            with self.get_repository_factory() as factory:
                repo = factory.get_benchmark_result_repository()
                repo.save_result(
                    validated_result.ip,
                    validated_result.frequency,
                    validated_result.coreVoltage,
                    validated_result.averageHashRate,
                    validated_result.averageTemperature,
                    validated_result.efficiencyJTH,
                    validated_result.duration,
                    samples_count=validated_result.samples_count,
                    aborted=validated_result.aborted,
                    abort_reason=validated_result.abort_reason
                )
                return True
                
        except ValidationError as e:
            self.log_event(ip, "VALIDATION_ERROR", f"Invalid benchmark data: {e}", "ERROR")
            return False
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to save benchmark: {e}", "ERROR")
            return False
    
    def log_efficiency_marker(self, ip: str, efficiency: float, hashrate: float,
                            power: float, temperature: float, frequency: int,
                            core_voltage: int) -> bool:
        """Log efficiency marker with validation"""
        try:
            # Validate efficiency data
            validated_marker = EfficiencyMarkerSchema(
                ip=ip,
                efficiency=efficiency,
                hashRate=hashrate,
                power=power,
                temperature=temperature,
                frequency=frequency,
                coreVoltage=core_voltage
            )
            
            with self.get_repository_factory() as factory:
                repo = factory.get_efficiency_marker_repository()
                repo.log_efficiency(
                    validated_marker.ip,
                    validated_marker.efficiency,
                    validated_marker.hashRate,
                    validated_marker.power,
                    validated_marker.temperature,
                    validated_marker.frequency,
                    validated_marker.coreVoltage
                )
                return True
                
        except ValidationError as e:
            self.log_event(ip, "VALIDATION_ERROR", f"Invalid efficiency data: {e}", "ERROR")
            return False
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to log efficiency: {e}", "ERROR")
            return False
    
    def get_latest_status(self) -> List[Dict[str, Any]]:
        """Get latest status for all miners"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_miner_log_repository()
                miners = repo.get_latest_status_all()
                return [miner.to_dict() for miner in miners]
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get latest status: {e}", "ERROR")
            return []
    
    def get_latest_status_by_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get latest status for specific IP"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_miner_log_repository()
                miner = repo.get_latest_status_by_ip(ip)
                return miner.to_dict() if miner else None
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to get status: {e}", "ERROR")
            return None
    
    def get_event_log(self, limit: int = 100, ip_filter: str = None,
                     event_type_filter: str = None) -> List[Dict[str, Any]]:
        """Get recent events with optional filtering"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_protocol_event_repository()
                
                if ip_filter:
                    events = repo.get_events_by_ip(ip_filter, limit)
                elif event_type_filter:
                    events = repo.get_events_by_type(event_type_filter, limit)
                else:
                    events = repo.find_by(limit=limit, order_by='timestamp', desc_order=True)
                
                return [event.to_dict() for event in events]
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get events: {e}", "ERROR")
            return []
    
    def get_history_data(self, start: datetime, end: datetime,
                        ips: List[str] = None) -> Dict[str, Any]:
        """Get historical data for time range"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_miner_log_repository()
                history_data = repo.get_history_data(start, end, ips)
                
                # Convert to format expected by frontend
                result = {}
                for ip, records in history_data.items():
                    result[ip] = {
                        'traces': [record.to_dict() for record in records]
                    }
                
                return result
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get history: {e}", "ERROR")
            return {}
    
    def get_top_settings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing settings"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_tuning_status_repository()
                top_settings = repo.get_top_performers(limit)
                return [setting.to_dict() for setting in top_settings]
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get top settings: {e}", "ERROR")
            return []
    
    def get_best_efficiency_for_ip(self, ip: str) -> Optional[float]:
        """Get best efficiency for specific IP"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_tuning_status_repository()
                best_setting = repo.get_best_efficiency_by_ip(ip)
                return best_setting.avgEfficiency if best_setting else None
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to get best efficiency: {e}", "ERROR")
            return None
    
    def get_benchmark_results_for_ip(self, ip: str, limit: int = 10) -> List[Tuple]:
        """Get recent benchmark results for specific IP"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_benchmark_result_repository()
                results = repo.get_results_by_ip(ip, limit)
                
                # Convert to tuple format for backwards compatibility
                return [
                    (
                        result.frequency,
                        result.coreVoltage,
                        result.averageHashRate,
                        result.averageTemperature,
                        result.efficiencyJTH,
                        result.timestamp
                    )
                    for result in results
                ]
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to get benchmark results: {e}", "ERROR")
            return []
    
    def get_benchmark_results(self, limit: int = 50) -> List[Tuple]:
        """Get all benchmark results ordered by performance"""
        try:
            with self.get_repository_factory() as factory:
                repo = factory.get_benchmark_result_repository()
                results = repo.get_best_results(limit)
                
                # Convert to tuple format for backwards compatibility
                return [
                    (
                        result.ip,
                        result.frequency,
                        result.coreVoltage,
                        result.averageHashRate,
                        result.averageTemperature,
                        result.efficiencyJTH,
                        result.timestamp,
                        result.duration
                    )
                    for result in results
                ]
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get benchmark results: {e}", "ERROR")
            return []
    
    def get_efficiency_data_for_export(self, ip: str, start_str: str,
                                     end_str: str) -> Tuple[List[str], List[Tuple]]:
        """Get efficiency data for CSV export"""
        try:
            start_time = datetime.fromisoformat(start_str)
            end_time = datetime.fromisoformat(end_str)
            
            with self.get_repository_factory() as factory:
                repo = factory.get_efficiency_marker_repository()
                markers = repo.get_by_time_range(start_time, end_time)
                
                # Filter by IP
                ip_markers = [m for m in markers if m.ip == ip]
                
                # Create headers and data
                headers = ['timestamp', 'ip', 'efficiency', 'hashRate', 'power', 
                          'temperature', 'frequency', 'coreVoltage']
                
                rows = [
                    (
                        marker.timestamp,
                        marker.ip,
                        marker.efficiency,
                        marker.hashRate,
                        marker.power,
                        marker.temperature,
                        marker.frequency,
                        marker.coreVoltage
                    )
                    for marker in ip_markers
                ]
                
                return headers, rows
                
        except Exception as e:
            self.log_event(ip, "DATABASE_ERROR", f"Failed to get export data: {e}", "ERROR")
            return [], []
    
    def update_best_settings_from_benchmarks(self) -> bool:
        """Update best settings based on latest benchmark results"""
        try:
            with self.get_repository_factory() as factory:
                benchmark_repo = factory.get_benchmark_result_repository()
                tuning_repo = factory.get_tuning_status_repository()
                
                # Get distinct IPs from benchmarks
                all_results = benchmark_repo.get_best_efficiency_results(1000)
                ips = set(result.ip for result in all_results)
                
                # Update best settings for each IP
                for ip in ips:
                    best_result = benchmark_repo.get_results_by_ip(ip, 1)
                    if best_result:
                        result = best_result[0]
                        tuning_repo.update_best_settings(
                            ip,
                            result.frequency,
                            result.coreVoltage,
                            result.efficiencyJTH or 0
                        )
                
                return True
                
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to update best settings: {e}", "ERROR")
            return False
    
    def cleanup_old_data(self, days: int = 30) -> Dict[str, int]:
        """Cleanup old data from all tables"""
        cleanup_counts = {}
        
        try:
            with self.get_repository_factory() as factory:
                # Cleanup old logs
                log_repo = factory.get_miner_log_repository()
                cleanup_counts['logs'] = log_repo.cleanup_old_records(days)
                
                # Cleanup old events
                event_repo = factory.get_protocol_event_repository()
                cleanup_counts['events'] = event_repo.cleanup_old_records(days)
                
                # Cleanup old efficiency markers
                eff_repo = factory.get_efficiency_marker_repository()
                cleanup_counts['efficiency_markers'] = eff_repo.cleanup_old_records(days)
                
                self.log_event("SYSTEM", "CLEANUP_COMPLETED", 
                             f"Cleaned up {sum(cleanup_counts.values())} old records", "INFO")
                
        except Exception as e:
            self.log_event("SYSTEM", "CLEANUP_ERROR", f"Failed to cleanup: {e}", "ERROR")
        
        return cleanup_counts
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_repository_factory() as factory:
                stats = {}
                
                # Count records in each table
                stats['logs_count'] = factory.get_miner_log_repository().count()
                stats['events_count'] = factory.get_protocol_event_repository().count()
                stats['benchmarks_count'] = factory.get_benchmark_result_repository().count()
                stats['efficiency_markers_count'] = factory.get_efficiency_marker_repository().count()
                
                # Database connection info
                stats['connection_info'] = self.db_manager.get_connection_info()
                
                return stats
                
        except Exception as e:
            self.log_event("SYSTEM", "DATABASE_ERROR", f"Failed to get stats: {e}", "ERROR")
            return {}