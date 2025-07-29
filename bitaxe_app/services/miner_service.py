import requests
from typing import Dict, Any, Optional, List
from datetime import datetime


class MinerService:
    def __init__(self, config_service, database_service):
        self.config_service = config_service
        self.database_service = database_service
        self.timeout = 5

    def fetch_miner_data(self, ip: str) -> Optional[Dict[str, Any]]:
        """Fetch current data from a miner"""
        try:
            response = requests.get(f"http://{ip}/api/system/info", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.database_service.log_event(ip, "FETCH_ERROR", f"API Error: {e}")
            return None
        except Exception as e:
            self.database_service.log_event(ip, "FETCH_ERROR", f"Unexpected error: {e}")
            return None

    def set_miner_settings(self, ip: str, frequency: int, core_voltage: int, 
                          autofanspeed: bool = True) -> bool:
        """Set miner frequency and voltage settings"""
        try:
            payload = {
                "frequency": frequency,
                "coreVoltage": core_voltage,
                "autofanspeed": autofanspeed
            }
            
            response = requests.patch(f"http://{ip}/api/system", json=payload, timeout=self.timeout)
            response.raise_for_status()
            
            message = f"Settings set: {frequency} MHz @ {core_voltage} mV"
            self.database_service.log_event(ip, "SETTINGS_CHANGED", message)
            return True
            
        except requests.exceptions.RequestException as e:
            self.database_service.log_event(ip, "SETTINGS_ERROR", f"Failed to set settings: {e}")
            return False
        except Exception as e:
            self.database_service.log_event(ip, "SETTINGS_ERROR", f"Unexpected error: {e}")
            return False

    def restart_miner(self, ip: str) -> bool:
        """Restart a miner"""
        try:
            response = requests.post(f"http://{ip}/api/system/restart", timeout=self.timeout)
            response.raise_for_status()
            
            self.database_service.log_event(ip, "RESTART", "Miner restart initiated")
            return True
            
        except requests.exceptions.RequestException as e:
            self.database_service.log_event(ip, "RESTART_ERROR", f"Failed to restart: {e}")
            return False
        except Exception as e:
            self.database_service.log_event(ip, "RESTART_ERROR", f"Unexpected error: {e}")
            return False

    def set_fallback_settings(self, ip: str) -> bool:
        """Set miner to fallback settings"""
        fallback = self.config_service.fallback_settings
        return self.set_miner_settings(
            ip, 
            fallback['frequency'], 
            fallback['coreVoltage']
        )

    def set_profile_settings(self, ip: str, profile_name: str) -> bool:
        """Set miner to a specific profile (max/eco)"""
        profile = self.config_service.get_profile(profile_name)
        if not profile:
            self.database_service.log_event(ip, "PROFILE_ERROR", f"Profile not found: {profile_name}")
            return False
            
        return self.set_miner_settings(
            ip,
            profile['frequency'],
            profile['coreVoltage']
        )

    def get_all_miners_status(self) -> List[Dict[str, Any]]:
        """Get current status for all configured miners"""
        miners_data = []
        
        for ip in self.config_service.ips:
            data = self.fetch_miner_data(ip)
            if data:
                # Add additional calculated fields
                hashrate = data.get('hashRate', 0)
                power = data.get('power', 0)
                data['efficiency'] = hashrate / power if power > 0 else 0
                data['ip'] = ip
                data['color'] = self.config_service.get_miner_color(ip)
                miners_data.append(data)
                
                # Log the data to database
                self.database_service.log_miner_data(ip, data)
        
        return miners_data

    def check_miner_health(self, ip: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check miner health and return status indicators"""
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
        elif temp >= self.config_service.temp_limit:
            health_status['warnings'].append(f"High temperature: {temp}°C")
            
        # Hashrate checks
        if hashrate <= 0:
            health_status['is_healthy'] = False
            health_status['errors'].append("No hashrate detected")
        
        return health_status

    def get_miner_efficiency(self, ip: str, hashrate: float, power: float) -> float:
        """Calculate and log miner efficiency"""
        if power <= 0:
            return 0
            
        efficiency = hashrate / power
        
        # Log efficiency marker
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
            return response.status_code == 200
        except:
            return False

    def get_miner_uptime(self, ip: str) -> Optional[int]:
        """Get miner uptime in seconds"""
        data = self.fetch_miner_data(ip)
        return data.get('uptimeSeconds') if data else None

    def validate_settings(self, frequency: int, core_voltage: int) -> bool:
        """Validate frequency and voltage settings against configuration"""
        freq_list = self.config_service.freq_list
        volt_list = self.config_service.volt_list
        
        return frequency in freq_list and core_voltage in volt_list