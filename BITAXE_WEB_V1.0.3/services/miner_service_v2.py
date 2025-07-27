import requests
from typing import Dict, Any, Optional, List
from datetime import datetime

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import (
    MinerConnectionError, MinerTimeoutError, MinerAPIError, 
    MinerOfflineError, ValidationError as CustomValidationError,
    OverheatError, TemperatureError
)
from utils.retry_decorators import retry_miner_request, RetryConfig
from utils.error_handlers import operation_error_boundary
from models.validation_schemas import MinerDataSchema, MinerSettingsSchema

logger = get_logger("bitaxe.miner")


class MinerServiceV2:
    """Enhanced miner service with logging, exceptions, and retry logic"""
    
    def __init__(self, config_service, database_service):
        self.config_service = config_service
        self.database_service = database_service
        self.timeout = 5
        self.logger = logger
    
    @retry_miner_request("unknown", RetryConfig(max_attempts=3, base_delay=0.5))
    def fetch_miner_data(self, ip: str) -> Optional[Dict[str, Any]]:
        """Fetch current data from a miner with retry logic"""
        with operation_error_boundary(f"fetch_miner_data_{ip}", self.logger):
            try:
                self.logger.debug("Fetching miner data", miner_ip=ip)
                
                response = requests.get(
                    f"http://{ip}/api/system/info", 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Validate the response data
                try:
                    validated_data = MinerDataSchema(**data)
                    result = validated_data.dict(exclude_none=True)
                    
                    self.logger.debug("Miner data fetched successfully",
                                    miner_ip=ip,
                                    hashrate=result.get('hashRate', 0),
                                    temperature=result.get('temp', 0),
                                    power=result.get('power', 0))
                    
                    # Log to database
                    success = self.database_service.validate_and_log_miner_data(ip, result)
                    if not success:
                        self.logger.warning("Failed to log miner data to database",
                                          miner_ip=ip)
                    
                    return result
                    
                except Exception as validation_error:
                    self.logger.warning("Miner data validation failed",
                                      miner_ip=ip,
                                      error=str(validation_error),
                                      raw_data=data)
                    raise CustomValidationError(
                        f"Invalid miner data from {ip}",
                        field="response_data",
                        value=data,
                        constraints={"schema": "MinerDataSchema"}
                    )
                
            except requests.exceptions.Timeout:
                self.logger.error("Miner request timed out", miner_ip=ip, timeout_seconds=self.timeout)
                raise MinerTimeoutError(ip, self.timeout)
            
            except requests.exceptions.ConnectionError as e:
                self.logger.error("Failed to connect to miner", miner_ip=ip, error=str(e))
                raise MinerConnectionError(ip, cause=e)
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                self.logger.error("Miner API returned error",
                                miner_ip=ip,
                                status_code=status_code,
                                error=str(e))
                raise MinerAPIError(ip, status_code, str(e))
    
    def set_miner_settings(self, ip: str, frequency: int, core_voltage: int, 
                          autofanspeed: bool = True) -> bool:
        """Set miner frequency and voltage settings with validation"""
        with operation_error_boundary(f"set_miner_settings_{ip}", self.logger):
            # Validate settings
            try:
                settings = MinerSettingsSchema(
                    frequency=frequency,
                    coreVoltage=core_voltage,
                    autofanspeed=autofanspeed
                )
            except Exception as e:
                self.logger.error("Invalid miner settings",
                                miner_ip=ip,
                                frequency=frequency,
                                core_voltage=core_voltage,
                                error=str(e))
                raise CustomValidationError(
                    f"Invalid settings for miner {ip}",
                    field="settings",
                    value={"frequency": frequency, "coreVoltage": core_voltage},
                    constraints={"frequency_range": self.config_service.freq_list,
                               "voltage_range": self.config_service.volt_list}
                )
            
            try:
                payload = settings.dict()
                
                self.logger.info("Setting miner configuration",
                               miner_ip=ip,
                               frequency=frequency,
                               core_voltage=core_voltage,
                               autofanspeed=autofanspeed)
                
                response = requests.patch(
                    f"http://{ip}/api/system", 
                    json=payload, 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                message = f"Settings applied: {frequency} MHz @ {core_voltage} mV"
                self.database_service.log_event(ip, "SETTINGS_CHANGED", message, "INFO")
                
                self.logger.info("Miner settings applied successfully",
                               miner_ip=ip,
                               frequency=frequency,
                               core_voltage=core_voltage)
                
                return True
                
            except requests.exceptions.Timeout:
                self.logger.error("Timeout setting miner configuration",
                                miner_ip=ip,
                                timeout_seconds=self.timeout)
                raise MinerTimeoutError(ip, self.timeout)
            
            except requests.exceptions.ConnectionError as e:
                self.logger.error("Connection failed setting miner configuration",
                                miner_ip=ip,
                                error=str(e))
                raise MinerConnectionError(ip, cause=e)
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                self.logger.error("API error setting miner configuration",
                                miner_ip=ip,
                                status_code=status_code,
                                error=str(e))
                raise MinerAPIError(ip, status_code, str(e))
    
    def restart_miner(self, ip: str) -> bool:
        """Restart a miner with proper error handling"""
        with operation_error_boundary(f"restart_miner_{ip}", self.logger):
            try:
                self.logger.info("Restarting miner", miner_ip=ip)
                
                response = requests.post(
                    f"http://{ip}/api/system/restart", 
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                self.database_service.log_event(ip, "RESTART", "Miner restart initiated", "INFO")
                
                self.logger.info("Miner restart command sent successfully", miner_ip=ip)
                
                return True
                
            except requests.exceptions.Timeout:
                self.logger.error("Timeout restarting miner", miner_ip=ip)
                raise MinerTimeoutError(ip, self.timeout)
            
            except requests.exceptions.ConnectionError as e:
                self.logger.error("Connection failed restarting miner", miner_ip=ip, error=str(e))
                raise MinerConnectionError(ip, cause=e)
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                self.logger.error("API error restarting miner",
                                miner_ip=ip,
                                status_code=status_code,
                                error=str(e))
                raise MinerAPIError(ip, status_code, str(e))
    
    def set_fallback_settings(self, ip: str) -> bool:
        """Set miner to fallback settings"""
        with operation_error_boundary(f"set_fallback_{ip}", self.logger):
            fallback = self.config_service.fallback_settings
            
            self.logger.warning("Applying fallback settings",
                              miner_ip=ip,
                              fallback_frequency=fallback['frequency'],
                              fallback_voltage=fallback['coreVoltage'])
            
            return self.set_miner_settings(
                ip, 
                fallback['frequency'], 
                fallback['coreVoltage']
            )
    
    def set_profile_settings(self, ip: str, profile_name: str) -> bool:
        """Set miner to a specific profile (max/eco)"""
        with operation_error_boundary(f"set_profile_{ip}_{profile_name}", self.logger):
            profile = self.config_service.get_profile(profile_name)
            if not profile:
                self.logger.error("Profile not found",
                                miner_ip=ip,
                                profile_name=profile_name)
                raise CustomValidationError(
                    f"Profile '{profile_name}' not found",
                    field="profile_name",
                    value=profile_name
                )
            
            self.logger.info("Applying profile settings",
                           miner_ip=ip,
                           profile_name=profile_name,
                           frequency=profile['frequency'],
                           voltage=profile['coreVoltage'])
            
            return self.set_miner_settings(
                ip,
                profile['frequency'],
                profile['coreVoltage']
            )
    
    def get_all_miners_status(self) -> List[Dict[str, Any]]:
        """Get current status for all configured miners"""
        with operation_error_boundary("get_all_miners_status", self.logger):
            miners_data = []
            
            self.logger.info("Fetching status for all miners",
                           total_miners=len(self.config_service.ips))
            
            for ip in self.config_service.ips:
                try:
                    data = self.fetch_miner_data(ip)
                    if data:
                        # Add additional calculated fields
                        hashrate = data.get('hashRate', 0)
                        power = data.get('power', 0)
                        data['efficiency'] = hashrate / power if power > 0 else 0
                        data['ip'] = ip
                        data['color'] = self.config_service.get_miner_color(ip)
                        miners_data.append(data)
                        
                except Exception as e:
                    self.logger.error("Failed to fetch miner status",
                                    miner_ip=ip,
                                    error=str(e))
                    # Continue with other miners
                    continue
            
            self.logger.info("Completed status fetch for all miners",
                           successful_miners=len(miners_data),
                           total_miners=len(self.config_service.ips))
            
            return miners_data
    
    def check_miner_health(self, ip: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check miner health and return status indicators"""
        with operation_error_boundary(f"check_health_{ip}", self.logger):
            health_status = {
                'ip': ip,
                'is_healthy': True,
                'warnings': [],
                'errors': []
            }
            
            temp = data.get('temp', 0)
            hashrate = data.get('hashRate', 0)
            
            # Temperature checks
            if temp >= self.config_service.temp_overheat:
                health_status['is_healthy'] = False
                health_status['errors'].append(f"Overheating: {temp}°C")
                
                self.logger.error("Miner overheating detected",
                                miner_ip=ip,
                                temperature=temp,
                                overheat_threshold=self.config_service.temp_overheat)
                
                # Raise exception for critical overheating
                raise OverheatError(ip, temp, self.config_service.temp_overheat)
                
            elif temp >= self.config_service.temp_limit:
                health_status['warnings'].append(f"High temperature: {temp}°C")
                
                self.logger.warning("Miner high temperature",
                                  miner_ip=ip,
                                  temperature=temp,
                                  temp_limit=self.config_service.temp_limit)
            
            # Hashrate checks
            if hashrate <= 0:
                health_status['is_healthy'] = False
                health_status['errors'].append("No hashrate detected")
                
                self.logger.error("No hashrate detected", miner_ip=ip)
            
            return health_status
    
    def get_miner_efficiency(self, ip: str, hashrate: float, power: float) -> float:
        """Calculate and log miner efficiency"""
        with operation_error_boundary(f"calculate_efficiency_{ip}", self.logger):
            if power <= 0:
                self.logger.warning("Cannot calculate efficiency: power is zero",
                                  miner_ip=ip,
                                  hashrate=hashrate,
                                  power=power)
                return 0
            
            efficiency = hashrate / power
            
            self.logger.debug("Efficiency calculated",
                            miner_ip=ip,
                            hashrate=hashrate,
                            power=power,
                            efficiency=efficiency)
            
            # Log efficiency marker to database
            data = self.fetch_miner_data(ip)
            if data:
                self.database_service.log_efficiency_marker(
                    ip, efficiency, hashrate, power,
                    data.get('temp', 0),
                    data.get('frequency', 0),
                    data.get('coreVoltage', 0)
                )
            
            return efficiency
    
    def is_miner_online(self, ip: str) -> bool:
        """Check if miner is online and responding"""
        try:
            response = requests.get(f"http://{ip}/api/system/info", timeout=2)
            is_online = response.status_code == 200
            
            self.logger.debug("Miner online check",
                            miner_ip=ip,
                            is_online=is_online,
                            status_code=response.status_code)
            
            return is_online
            
        except Exception as e:
            self.logger.debug("Miner online check failed",
                            miner_ip=ip,
                            error=str(e))
            return False
    
    def get_miner_uptime(self, ip: str) -> Optional[int]:
        """Get miner uptime in seconds"""
        try:
            data = self.fetch_miner_data(ip)
            uptime = data.get('uptimeSeconds') if data else None
            
            self.logger.debug("Miner uptime retrieved",
                            miner_ip=ip,
                            uptime_seconds=uptime)
            
            return uptime
        except Exception as e:
            self.logger.warning("Failed to get miner uptime",
                              miner_ip=ip,
                              error=str(e))
            return None
    
    def validate_settings(self, frequency: int, core_voltage: int) -> bool:
        """Validate frequency and voltage settings against configuration"""
        freq_list = self.config_service.freq_list
        volt_list = self.config_service.volt_list
        
        is_valid = frequency in freq_list and core_voltage in volt_list
        
        self.logger.debug("Settings validation",
                        frequency=frequency,
                        core_voltage=core_voltage,
                        is_valid=is_valid,
                        valid_frequencies=freq_list,
                        valid_voltages=volt_list)
        
        return is_valid
    
    def get_miners_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all miners"""
        with operation_error_boundary("get_miners_summary", self.logger):
            miners_data = self.get_all_miners_status()
            
            total_miners = len(self.config_service.ips)
            online_miners = len(miners_data)
            total_hashrate = sum(m.get('hashRate', 0) for m in miners_data)
            total_power = sum(m.get('power', 0) for m in miners_data)
            avg_temp = sum(m.get('temp', 0) for m in miners_data) / len(miners_data) if miners_data else 0
            total_efficiency = total_hashrate / total_power if total_power > 0 else 0
            
            summary = {
                'total_miners': total_miners,
                'online_miners': online_miners,
                'offline_miners': total_miners - online_miners,
                'total_hashrate': total_hashrate,
                'total_power': total_power,
                'total_efficiency': total_efficiency,
                'average_temperature': avg_temp,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("Miners summary calculated",
                           **summary)
            
            return summary