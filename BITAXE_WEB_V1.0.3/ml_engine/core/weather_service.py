"""
Weather Service

External weather data integration for environmental adaptation and optimization.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector

logger = get_logger("bitaxe.ml.weather_service")


@dataclass
class WeatherData:
    """Weather data structure"""
    temperature: float  # °C
    humidity: float  # %
    pressure: float  # hPa
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    cloud_cover: float  # %
    visibility: float  # km
    uv_index: float
    timestamp: datetime
    location: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_openweather_response(cls, data: Dict[str, Any], location: str = "") -> 'WeatherData':
        """Create from OpenWeatherMap API response"""
        main = data.get('main', {})
        wind = data.get('wind', {})
        clouds = data.get('clouds', {})
        
        return cls(
            temperature=main.get('temp', 20.0),
            humidity=main.get('humidity', 50.0),
            pressure=main.get('pressure', 1013.25),
            wind_speed=wind.get('speed', 0.0),
            wind_direction=wind.get('deg', 0.0),
            cloud_cover=clouds.get('all', 0.0),
            visibility=data.get('visibility', 10000) / 1000,  # Convert m to km
            uv_index=data.get('uvi', 0.0),
            timestamp=datetime.now(),
            location=location
        )


@dataclass
class WeatherForecast:
    """Weather forecast data"""
    forecasts: List[WeatherData]
    forecast_hours: int
    created_at: datetime
    location: str = ""
    
    def get_forecast_at_time(self, target_time: datetime) -> Optional[WeatherData]:
        """Get forecast closest to target time"""
        if not self.forecasts:
            return None
        
        closest_forecast = min(
            self.forecasts,
            key=lambda f: abs((f.timestamp - target_time).total_seconds())
        )
        
        return closest_forecast
    
    def get_temperature_trend(self) -> Dict[str, float]:
        """Get temperature trend statistics"""
        if not self.forecasts:
            return {}
        
        temps = [f.temperature for f in self.forecasts]
        
        return {
            'min_temp': min(temps),
            'max_temp': max(temps),
            'avg_temp': np.mean(temps),
            'temp_range': max(temps) - min(temps),
            'temp_trend': np.polyfit(range(len(temps)), temps, 1)[0] if len(temps) > 1 else 0
        }


class WeatherAPIClient:
    """Base class for weather API clients"""
    
    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather data"""
        raise NotImplementedError
    
    async def get_forecast(self, location: str, hours: int = 24) -> Optional[WeatherForecast]:
        """Get weather forecast"""
        raise NotImplementedError


class OpenWeatherMapClient(WeatherAPIClient):
    """OpenWeatherMap API client"""
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        self.api_key = api_key
        self.config = config or {}
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.timeout = self.config.get('timeout', 10)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 1.0)
        self.last_request_time = 0
        
        logger.info("OpenWeatherMap client initialized")
    
    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather from OpenWeatherMap"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return WeatherData.from_openweather_response(data, location)
                    else:
                        logger.error(f"OpenWeatherMap API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error("Failed to get current weather", error=str(e))
            return None
    
    async def get_forecast(self, location: str, hours: int = 24) -> Optional[WeatherForecast]:
        """Get weather forecast from OpenWeatherMap"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/forecast"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(40, hours // 3)  # 3-hour intervals, max 40 entries
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        forecasts = []
                        
                        for item in data.get('list', []):
                            forecast_time = datetime.fromtimestamp(item['dt'])
                            weather_data = WeatherData.from_openweather_response(item, location)
                            weather_data.timestamp = forecast_time
                            forecasts.append(weather_data)
                        
                        return WeatherForecast(
                            forecasts=forecasts,
                            forecast_hours=hours,
                            created_at=datetime.now(),
                            location=location
                        )
                    else:
                        logger.error(f"OpenWeatherMap forecast API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error("Failed to get weather forecast", error=str(e))
            return None
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()


class LocalSensorClient(WeatherAPIClient):
    """Client for local weather sensors (DHT22, BMP280, etc.)"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sensor_config = self.config.get('sensors', {})
        self.mock_mode = self.config.get('mock_mode', True)
        
        logger.info("Local sensor client initialized", mock_mode=self.mock_mode)
    
    async def get_current_weather(self, location: str = "local") -> Optional[WeatherData]:
        """Get weather data from local sensors"""
        try:
            if self.mock_mode:
                # Mock sensor data for development
                return WeatherData(
                    temperature=20.0 + np.random.normal(0, 2),
                    humidity=50.0 + np.random.normal(0, 10),
                    pressure=1013.25 + np.random.normal(0, 5),
                    wind_speed=np.random.exponential(2),
                    wind_direction=np.random.uniform(0, 360),
                    cloud_cover=np.random.uniform(0, 100),
                    visibility=10.0,
                    uv_index=np.random.uniform(0, 11),
                    timestamp=datetime.now(),
                    location="local_sensors"
                )
            else:
                # Real sensor implementation would go here
                # e.g., reading from GPIO pins, I2C sensors, etc.
                logger.warning("Real sensor support not implemented")
                return None
        
        except Exception as e:
            logger.error("Failed to read local sensors", error=str(e))
            return None
    
    async def get_forecast(self, location: str = "local", hours: int = 24) -> Optional[WeatherForecast]:
        """Local sensors don't provide forecasts"""
        return None


class WeatherService:
    """
    Comprehensive weather service for environmental adaptation
    
    Features:
    - Multiple weather data sources with fallback
    - Weather-based optimization strategies
    - Forecasting and trend analysis
    - Local sensor integration
    - Caching and rate limiting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Initialize weather clients
        self.clients: List[WeatherAPIClient] = []
        self._initialize_clients()
        
        # Configuration
        self.location = self.config.get('location', 'New York,US')
        self.update_interval = self.config.get('update_interval', 300)  # 5 minutes
        self.forecast_hours = self.config.get('forecast_hours', 24)
        
        # Data storage
        self.current_weather: Optional[WeatherData] = None
        self.current_forecast: Optional[WeatherForecast] = None
        self.weather_history: List[WeatherData] = []
        self.max_history = self.config.get('max_history', 288)  # 24 hours at 5min intervals
        
        # Background tasks
        self.update_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Environmental adaptation parameters
        self.adaptation_config = self.config.get('adaptation', {})
        self.cooling_strategies = self._initialize_cooling_strategies()
        
        logger.info("Weather service initialized",
                   location=self.location,
                   clients=len(self.clients),
                   update_interval=self.update_interval)
    
    def _initialize_clients(self):
        """Initialize weather API clients"""
        # OpenWeatherMap client
        openweather_key = self.config.get('openweathermap_api_key')
        if openweather_key:
            self.clients.append(OpenWeatherMapClient(
                openweather_key, 
                self.config.get('openweathermap_config', {})
            ))
        
        # Local sensor client
        if self.config.get('enable_local_sensors', True):
            self.clients.append(LocalSensorClient(
                self.config.get('local_sensor_config', {})
            ))
        
        if not self.clients:
            logger.warning("No weather clients configured")
    
    def _initialize_cooling_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cooling strategies based on weather conditions"""
        return {
            'hot_dry': {
                'description': 'Hot and dry conditions',
                'conditions': {'min_temp': 30, 'max_humidity': 40},
                'strategies': {
                    'reduce_frequency': True,
                    'reduce_voltage': True,
                    'increase_fan_speed': True,
                    'frequency_reduction': 0.15,  # 15% reduction
                    'voltage_reduction': 0.05     # 0.05V reduction
                }
            },
            'hot_humid': {
                'description': 'Hot and humid conditions',
                'conditions': {'min_temp': 28, 'min_humidity': 70},
                'strategies': {
                    'reduce_frequency': True,
                    'reduce_voltage': True,
                    'increase_fan_speed': True,
                    'frequency_reduction': 0.20,  # 20% reduction
                    'voltage_reduction': 0.08     # 0.08V reduction
                }
            },
            'optimal': {
                'description': 'Optimal cooling conditions',
                'conditions': {'max_temp': 25, 'max_humidity': 60},
                'strategies': {
                    'reduce_frequency': False,
                    'reduce_voltage': False,
                    'increase_fan_speed': False,
                    'frequency_boost': 0.05,      # 5% boost possible
                    'voltage_boost': 0.02         # 0.02V boost possible
                }
            },
            'cold': {
                'description': 'Cold conditions',
                'conditions': {'max_temp': 10},
                'strategies': {
                    'reduce_frequency': False,
                    'reduce_voltage': False,
                    'increase_fan_speed': False,
                    'frequency_boost': 0.10,      # 10% boost possible
                    'voltage_boost': 0.05         # 0.05V boost possible
                }
            }
        }
    
    async def start(self):
        """Start the weather service"""
        if self.is_running:
            return
        
        logger.info("Starting weather service")
        self.is_running = True
        
        # Initial weather update
        await self.update_weather_data()
        
        # Start background update task
        self.update_task = asyncio.create_task(self._weather_update_worker())
        
        self.metrics_collector.increment_counter('weather_service_starts_total')
        logger.info("Weather service started")
    
    async def stop(self):
        """Stop the weather service"""
        if not self.is_running:
            return
        
        logger.info("Stopping weather service")
        self.is_running = False
        
        # Cancel background task
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Weather service stopped")
    
    async def update_weather_data(self) -> bool:
        """Update current weather and forecast data"""
        try:
            start_time = datetime.now()
            
            # Try each client until one succeeds
            current_weather = None
            forecast = None
            
            for client in self.clients:
                try:
                    # Get current weather
                    if current_weather is None:
                        current_weather = await client.get_current_weather(self.location)
                    
                    # Get forecast (if supported)
                    if forecast is None:
                        forecast = await client.get_forecast(self.location, self.forecast_hours)
                    
                    if current_weather and forecast:
                        break
                
                except Exception as e:
                    logger.debug(f"Weather client failed: {e}")
                    continue
            
            # Update stored data
            if current_weather:
                self.current_weather = current_weather
                self.weather_history.append(current_weather)
                
                # Maintain history size
                if len(self.weather_history) > self.max_history:
                    self.weather_history = self.weather_history[-self.max_history:]
            
            if forecast:
                self.current_forecast = forecast
            
            update_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics_collector.record_metric('weather_update_duration', update_time)
            self.metrics_collector.increment_counter('weather_updates_total')
            
            if current_weather:
                self.metrics_collector.record_metric('weather_temperature', current_weather.temperature)
                self.metrics_collector.record_metric('weather_humidity', current_weather.humidity)
                self.metrics_collector.record_metric('weather_pressure', current_weather.pressure)
            
            success = current_weather is not None
            if success:
                logger.debug("Weather data updated successfully",
                           temperature=current_weather.temperature,
                           humidity=current_weather.humidity)
            else:
                logger.warning("Failed to update weather data from all sources")
                self.metrics_collector.increment_counter('weather_update_failures_total')
            
            return success
            
        except Exception as e:
            logger.error("Weather update failed", error=str(e))
            self.metrics_collector.increment_counter('weather_update_errors_total')
            return False
    
    async def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather data"""
        return self.current_weather
    
    async def get_forecast(self, hours: int = None) -> Optional[WeatherForecast]:
        """Get weather forecast"""
        if hours and self.current_forecast:
            # Filter forecast to requested hours
            cutoff_time = datetime.now() + timedelta(hours=hours)
            filtered_forecasts = [
                f for f in self.current_forecast.forecasts
                if f.timestamp <= cutoff_time
            ]
            
            return WeatherForecast(
                forecasts=filtered_forecasts,
                forecast_hours=hours,
                created_at=self.current_forecast.created_at,
                location=self.current_forecast.location
            )
        
        return self.current_forecast
    
    async def get_cooling_strategy(self, current_temp: float = None) -> Dict[str, Any]:
        """
        Get optimal cooling strategy based on current weather
        
        Args:
            current_temp: Current miner temperature (optional)
            
        Returns:
            Cooling strategy recommendations
        """
        try:
            if not self.current_weather:
                logger.warning("No weather data available for cooling strategy")
                return {'strategy': 'default', 'adjustments': {}}
            
            weather = self.current_weather
            
            # Determine weather conditions
            strategy_name = 'optimal'  # default
            
            for name, strategy in self.cooling_strategies.items():
                conditions = strategy['conditions']
                matches = True
                
                # Check temperature conditions
                if 'min_temp' in conditions and weather.temperature < conditions['min_temp']:
                    matches = False
                if 'max_temp' in conditions and weather.temperature > conditions['max_temp']:
                    matches = False
                
                # Check humidity conditions
                if 'min_humidity' in conditions and weather.humidity < conditions['min_humidity']:
                    matches = False
                if 'max_humidity' in conditions and weather.humidity > conditions['max_humidity']:
                    matches = False
                
                if matches:
                    strategy_name = name
                    break
            
            selected_strategy = self.cooling_strategies[strategy_name]
            
            # Apply additional adjustments based on current miner temperature
            adjustments = selected_strategy['strategies'].copy()
            
            if current_temp:
                if current_temp > 80:  # Critical temperature
                    adjustments['frequency_reduction'] = adjustments.get('frequency_reduction', 0) + 0.1
                    adjustments['voltage_reduction'] = adjustments.get('voltage_reduction', 0) + 0.1
                elif current_temp < 50:  # Cool running
                    adjustments['frequency_boost'] = adjustments.get('frequency_boost', 0) + 0.05
            
            # Calculate heat index for additional context
            heat_index = self._calculate_heat_index(weather.temperature, weather.humidity)
            
            result = {
                'strategy': strategy_name,
                'description': selected_strategy['description'],
                'weather': {
                    'temperature': weather.temperature,
                    'humidity': weather.humidity,
                    'heat_index': heat_index
                },
                'adjustments': adjustments,
                'recommendation_strength': self._calculate_recommendation_strength(weather, current_temp)
            }
            
            # Record metrics
            self.metrics_collector.record_metric('cooling_strategy_heat_index', heat_index)
            self.metrics_collector.increment_counter('cooling_strategy_requests_total',
                                                   tags={'strategy': strategy_name})
            
            return result
            
        except Exception as e:
            logger.error("Failed to determine cooling strategy", error=str(e))
            return {'strategy': 'default', 'adjustments': {}}
    
    def _calculate_heat_index(self, temp_c: float, humidity: float) -> float:
        """Calculate heat index from temperature and humidity"""
        # Convert Celsius to Fahrenheit for heat index calculation
        temp_f = temp_c * 9/5 + 32
        
        if temp_f < 80:
            return temp_c  # No heat index adjustment needed
        
        # Simplified heat index calculation
        hi_f = (temp_f + humidity) / 2 + 10
        
        # Convert back to Celsius
        return (hi_f - 32) * 5/9
    
    def _calculate_recommendation_strength(self, weather: WeatherData, current_temp: Optional[float]) -> float:
        """Calculate how strongly to apply weather-based recommendations"""
        strength = 0.5  # baseline
        
        # Temperature factors
        if weather.temperature > 35:
            strength += 0.3
        elif weather.temperature > 30:
            strength += 0.2
        elif weather.temperature < 5:
            strength += 0.2
        
        # Humidity factors
        if weather.humidity > 80:
            strength += 0.2
        elif weather.humidity < 20:
            strength += 0.1
        
        # Current miner temperature factor
        if current_temp:
            if current_temp > 85:
                strength += 0.3
            elif current_temp > 75:
                strength += 0.1
        
        return min(1.0, strength)
    
    async def get_weather_trend(self, hours: int = 6) -> Dict[str, Any]:
        """Get weather trend analysis"""
        try:
            if not self.weather_history:
                return {}
            
            # Get recent history
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_weather = [
                w for w in self.weather_history
                if w.timestamp >= cutoff_time
            ]
            
            if len(recent_weather) < 2:
                return {}
            
            # Calculate trends
            temps = [w.temperature for w in recent_weather]
            humidities = [w.humidity for w in recent_weather]
            pressures = [w.pressure for w in recent_weather]
            
            temp_trend = np.polyfit(range(len(temps)), temps, 1)[0]
            humidity_trend = np.polyfit(range(len(humidities)), humidities, 1)[0]
            pressure_trend = np.polyfit(range(len(pressures)), pressures, 1)[0]
            
            return {
                'temperature_trend': temp_trend,
                'humidity_trend': humidity_trend,
                'pressure_trend': pressure_trend,
                'temperature_range': max(temps) - min(temps),
                'humidity_range': max(humidities) - min(humidities),
                'data_points': len(recent_weather),
                'analysis_hours': hours
            }
            
        except Exception as e:
            logger.error("Weather trend analysis failed", error=str(e))
            return {}
    
    async def _weather_update_worker(self):
        """Background worker for weather updates"""
        logger.info("Weather update worker started")
        
        while self.is_running:
            try:
                await self.update_weather_data()
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Weather update worker error", error=str(e))
                await asyncio.sleep(60)  # Wait before retry
        
        logger.info("Weather update worker stopped")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get weather service status"""
        return {
            'is_running': self.is_running,
            'location': self.location,
            'clients_configured': len(self.clients),
            'current_weather_available': self.current_weather is not None,
            'forecast_available': self.current_forecast is not None,
            'weather_history_points': len(self.weather_history),
            'last_update': self.current_weather.timestamp.isoformat() if self.current_weather else None,
            'next_update_in': self.update_interval
        }


async def create_weather_service(config: Dict[str, Any] = None) -> WeatherService:
    """Factory function to create weather service"""
    service = WeatherService(config)
    await service.start()
    return service