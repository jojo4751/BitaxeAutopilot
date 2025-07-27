from sqlalchemy.orm import Session
from typing import Dict, Type, TypeVar

from .miner_repositories import (
    MinerLogRepository,
    TuningStatusRepository,
    ProtocolEventRepository,
    EfficiencyMarkerRepository,
    BenchmarkResultRepository,
    MinerConfigurationRepository,
    SystemConfigurationRepository
)

R = TypeVar('R')


class RepositoryFactory:
    """Factory for creating repository instances"""
    
    def __init__(self, session: Session):
        self.session = session
        self._repositories: Dict[str, object] = {}
    
    def get_miner_log_repository(self) -> MinerLogRepository:
        """Get MinerLog repository"""
        if 'miner_log' not in self._repositories:
            self._repositories['miner_log'] = MinerLogRepository(self.session)
        return self._repositories['miner_log']
    
    def get_tuning_status_repository(self) -> TuningStatusRepository:
        """Get TuningStatus repository"""
        if 'tuning_status' not in self._repositories:
            self._repositories['tuning_status'] = TuningStatusRepository(self.session)
        return self._repositories['tuning_status']
    
    def get_protocol_event_repository(self) -> ProtocolEventRepository:
        """Get ProtocolEvent repository"""
        if 'protocol_event' not in self._repositories:
            self._repositories['protocol_event'] = ProtocolEventRepository(self.session)
        return self._repositories['protocol_event']
    
    def get_efficiency_marker_repository(self) -> EfficiencyMarkerRepository:
        """Get EfficiencyMarker repository"""
        if 'efficiency_marker' not in self._repositories:
            self._repositories['efficiency_marker'] = EfficiencyMarkerRepository(self.session)
        return self._repositories['efficiency_marker']
    
    def get_benchmark_result_repository(self) -> BenchmarkResultRepository:
        """Get BenchmarkResult repository"""
        if 'benchmark_result' not in self._repositories:
            self._repositories['benchmark_result'] = BenchmarkResultRepository(self.session)
        return self._repositories['benchmark_result']
    
    def get_miner_configuration_repository(self) -> MinerConfigurationRepository:
        """Get MinerConfiguration repository"""
        if 'miner_configuration' not in self._repositories:
            self._repositories['miner_configuration'] = MinerConfigurationRepository(self.session)
        return self._repositories['miner_configuration']
    
    def get_system_configuration_repository(self) -> SystemConfigurationRepository:
        """Get SystemConfiguration repository"""
        if 'system_configuration' not in self._repositories:
            self._repositories['system_configuration'] = SystemConfigurationRepository(self.session)
        return self._repositories['system_configuration']
    
    def clear_cache(self):
        """Clear repository cache (useful for testing)"""
        self._repositories.clear()