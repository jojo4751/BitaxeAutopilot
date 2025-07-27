"""
Feature Engineering Pipeline

Transform raw miner telemetry data into ML-ready features for training and inference.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector

logger = get_logger("bitaxe.ml.feature_engineering")


@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: np.ndarray
    feature_names: List[str]
    target: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    timestamp: datetime = None


class FeatureEngineer:
    """
    Advanced feature engineering for mining optimization ML models
    
    Features:
    - Time-series feature extraction
    - Statistical aggregations and rolling windows
    - Environmental feature integration
    - Technical indicators for mining performance
    - Feature scaling and normalization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Feature configuration
        self.window_sizes = self.config.get('window_sizes', [5, 15, 30, 60])  # minutes
        self.statistical_features = self.config.get('statistical_features', [
            'mean', 'std', 'min', 'max', 'median', 'skew', 'kurt'
        ])
        self.technical_indicators = self.config.get('technical_indicators', [
            'rsi', 'macd', 'bollinger_bands', 'moving_averages'
        ])
        
        # Feature scaling parameters
        self.feature_scalers = {}
        self.scaling_method = self.config.get('scaling_method', 'standard')
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_usage_stats = {}
        
        logger.info("Feature engineer initialized", 
                   window_sizes=self.window_sizes,
                   scaling_method=self.scaling_method)
    
    async def engineer_features(self, 
                              miner_data: List[Dict[str, Any]], 
                              weather_data: Optional[List[Dict[str, Any]]] = None,
                              target_column: Optional[str] = None) -> FeatureSet:
        """
        Engineer comprehensive feature set from raw data
        
        Args:
            miner_data: Raw miner telemetry data
            weather_data: Optional weather data
            target_column: Target variable for supervised learning
            
        Returns:
            FeatureSet with engineered features
        """
        try:
            start_time = datetime.now()
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(miner_data)
            
            if df.empty:
                logger.warning("Empty miner data provided for feature engineering")
                return FeatureSet(features=np.array([]), feature_names=[])
            
            # Ensure timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            else:
                df['timestamp'] = pd.date_range(start=datetime.now(), periods=len(df), freq='30S')
            
            # Engineer different feature categories
            features_dict = {}
            
            # 1. Basic telemetry features
            basic_features = await self._extract_basic_features(df)
            features_dict.update(basic_features)
            
            # 2. Time-series statistical features
            time_features = await self._extract_time_series_features(df)
            features_dict.update(time_features)
            
            # 3. Technical indicators
            technical_features = await self._extract_technical_indicators(df)
            features_dict.update(technical_features)
            
            # 4. Environmental features
            if weather_data:
                env_features = await self._extract_environmental_features(weather_data)
                features_dict.update(env_features)
            
            # 5. Derived performance metrics
            performance_features = await self._extract_performance_features(df)
            features_dict.update(performance_features)
            
            # 6. Temporal features
            temporal_features = await self._extract_temporal_features(df)
            features_dict.update(temporal_features)
            
            # Combine all features
            feature_names = list(features_dict.keys())
            features_matrix = np.column_stack([features_dict[name] for name in feature_names])
            
            # Handle target variable
            target = None
            if target_column and target_column in df.columns:
                target = df[target_column].values
            
            # Create feature set
            feature_set = FeatureSet(
                features=features_matrix,
                feature_names=feature_names,
                target=target,
                metadata={
                    'data_points': len(df),
                    'feature_count': len(feature_names),
                    'time_range': (df['timestamp'].min(), df['timestamp'].max()),
                    'engineering_time': (datetime.now() - start_time).total_seconds()
                },
                timestamp=datetime.now()
            )
            
            # Record metrics
            self.metrics_collector.record_metric('feature_engineering_duration', 
                                                (datetime.now() - start_time).total_seconds())
            self.metrics_collector.set_gauge('feature_count', len(feature_names))
            self.metrics_collector.increment_counter('feature_engineering_runs_total')
            
            logger.info("Feature engineering completed", 
                       features=len(feature_names),
                       data_points=len(df),
                       duration_ms=(datetime.now() - start_time).total_seconds() * 1000)
            
            return feature_set
            
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e))
            self.metrics_collector.increment_counter('feature_engineering_errors_total')
            raise
    
    async def _extract_basic_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract basic telemetry features"""
        features = {}
        
        # Core mining metrics
        numeric_columns = ['hashRate', 'temp', 'power', 'voltage', 'frequency']
        
        for col in numeric_columns:
            if col in df.columns:
                # Current value
                features[f'{col}_current'] = df[col].fillna(df[col].median()).values
                
                # Change rate (first derivative)
                features[f'{col}_change_rate'] = df[col].diff().fillna(0).values
                
                # Acceleration (second derivative)
                features[f'{col}_acceleration'] = df[col].diff().diff().fillna(0).values
        
        # Efficiency metrics
        if 'hashRate' in df.columns and 'power' in df.columns:
            efficiency = df['hashRate'] / df['power'].replace(0, np.nan)
            features['efficiency'] = efficiency.fillna(efficiency.median()).values
            features['efficiency_change_rate'] = efficiency.diff().fillna(0).values
        
        # Temperature-power ratio
        if 'temp' in df.columns and 'power' in df.columns:
            temp_power_ratio = df['temp'] / df['power'].replace(0, np.nan)
            features['temp_power_ratio'] = temp_power_ratio.fillna(temp_power_ratio.median()).values
        
        return features
    
    async def _extract_time_series_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract time-series statistical features"""
        features = {}
        
        numeric_columns = ['hashRate', 'temp', 'power', 'voltage', 'frequency']
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            series = df[col].fillna(df[col].median())
            
            # Rolling window statistics
            for window in self.window_sizes:
                if len(series) >= window:
                    rolling = series.rolling(window=window, min_periods=1)
                    
                    for stat in self.statistical_features:
                        if stat == 'mean':
                            values = rolling.mean().values
                        elif stat == 'std':
                            values = rolling.std().fillna(0).values
                        elif stat == 'min':
                            values = rolling.min().values
                        elif stat == 'max':
                            values = rolling.max().values
                        elif stat == 'median':
                            values = rolling.median().values
                        elif stat == 'skew':
                            values = rolling.skew().fillna(0).values
                        elif stat == 'kurt':
                            values = rolling.kurt().fillna(0).values
                        else:
                            continue
                        
                        features[f'{col}_{stat}_{window}m'] = values
        
        return features
    
    async def _extract_technical_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract technical indicators commonly used in trading"""
        features = {}
        
        # Focus on hashRate and efficiency as primary indicators
        if 'hashRate' in df.columns:
            hashrate = df['hashRate'].fillna(df['hashRate'].median())
            
            # RSI (Relative Strength Index)
            rsi = self._calculate_rsi(hashrate)
            features['hashrate_rsi'] = rsi
            
            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal = self._calculate_macd(hashrate)
            features['hashrate_macd'] = macd
            features['hashrate_macd_signal'] = macd_signal
            features['hashrate_macd_histogram'] = macd - macd_signal
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(hashrate)
            features['hashrate_bb_upper'] = bb_upper
            features['hashrate_bb_lower'] = bb_lower
            features['hashrate_bb_middle'] = bb_middle
            features['hashrate_bb_position'] = (hashrate - bb_lower) / (bb_upper - bb_lower)
            
            # Moving averages
            for period in [5, 10, 20]:
                if len(hashrate) >= period:
                    ma = hashrate.rolling(window=period).mean().fillna(hashrate.mean())
                    features[f'hashrate_ma_{period}'] = ma.values
                    features[f'hashrate_ma_{period}_ratio'] = (hashrate / ma).fillna(1.0).values
        
        return features
    
    async def _extract_environmental_features(self, weather_data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract environmental features from weather data"""
        features = {}
        
        if not weather_data:
            return features
        
        weather_df = pd.DataFrame(weather_data)
        
        # Current weather conditions
        weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        
        for feature in weather_features:
            if feature in weather_df.columns:
                values = weather_df[feature].fillna(weather_df[feature].median()).values
                # Repeat values to match miner data length if needed
                features[f'weather_{feature}'] = np.tile(values, 
                                                        (len(features.get('temp_current', [1])) // len(values)) + 1)[:len(features.get('temp_current', [1]))]
        
        # Weather comfort index (for cooling efficiency)
        if 'temperature' in weather_df.columns and 'humidity' in weather_df.columns:
            temp = weather_df['temperature'].fillna(weather_df['temperature'].median())
            humidity = weather_df['humidity'].fillna(weather_df['humidity'].median())
            
            # Heat index calculation
            heat_index = self._calculate_heat_index(temp, humidity)
            features['weather_heat_index'] = np.tile(heat_index.values, 
                                                   (len(features.get('temp_current', [1])) // len(heat_index)) + 1)[:len(features.get('temp_current', [1]))]
        
        return features
    
    async def _extract_performance_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract derived performance metrics"""
        features = {}
        
        # Stability metrics
        if 'hashRate' in df.columns:
            hashrate = df['hashRate'].fillna(df['hashRate'].median())
            
            # Coefficient of variation (stability indicator)
            features['hashrate_stability'] = self._rolling_coefficient_variation(hashrate)
            
            # Performance consistency score
            features['performance_consistency'] = self._calculate_consistency_score(hashrate)
        
        # Thermal efficiency
        if 'temp' in df.columns and 'hashRate' in df.columns:
            temp = df['temp'].fillna(df['temp'].median())
            hashrate = df['hashRate'].fillna(df['hashRate'].median())
            
            # Thermal efficiency (hashrate per degree)
            thermal_efficiency = hashrate / temp.replace(0, np.nan)
            features['thermal_efficiency'] = thermal_efficiency.fillna(thermal_efficiency.median()).values
        
        # Power efficiency trends
        if 'power' in df.columns and 'hashRate' in df.columns:
            power = df['power'].fillna(df['power'].median())
            hashrate = df['hashRate'].fillna(df['hashRate'].median())
            
            # Efficiency trend (improvement/degradation over time)
            efficiency = hashrate / power.replace(0, np.nan)
            efficiency_trend = efficiency.rolling(window=10).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            ).fillna(0)
            features['efficiency_trend'] = efficiency_trend.values
        
        return features
    
    async def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract time-based features"""
        features = {}
        
        if 'timestamp' not in df.columns:
            return features
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Cyclical time features
        features['hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        features['month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)
        
        # Time since start (for trend analysis)
        start_time = timestamps.min()
        features['time_since_start'] = (timestamps - start_time).dt.total_seconds().values
        
        # Business logic features
        features['is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(float).values
        features['is_business_hours'] = ((timestamps.dt.hour >= 9) & 
                                       (timestamps.dt.hour <= 17)).astype(float).values
        
        return features
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values  # Neutral RSI for missing values
    
    def _calculate_macd(self, series: pd.Series, 
                       fast_period: int = 12, slow_period: int = 26, 
                       signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator"""
        ema_fast = series.ewm(span=fast_period).mean()
        ema_slow = series.ewm(span=slow_period).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_period).mean()
        return macd.fillna(0).values, macd_signal.fillna(0).values
    
    def _calculate_bollinger_bands(self, series: pd.Series, 
                                  period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        rolling_mean = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        
        return (upper_band.fillna(series.mean()).values,
                lower_band.fillna(series.mean()).values,
                rolling_mean.fillna(series.mean()).values)
    
    def _calculate_heat_index(self, temp: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity"""
        # Simplified heat index calculation
        hi = temp + 0.5 * (humidity - 10)
        return hi
    
    def _rolling_coefficient_variation(self, series: pd.Series, window: int = 10) -> np.ndarray:
        """Calculate rolling coefficient of variation"""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        cv = rolling_std / rolling_mean.replace(0, np.nan)
        return cv.fillna(0).values
    
    def _calculate_consistency_score(self, series: pd.Series, window: int = 20) -> np.ndarray:
        """Calculate performance consistency score"""
        # Based on percentage of values within 1 standard deviation
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        within_std = ((series >= (rolling_mean - rolling_std)) & 
                     (series <= (rolling_mean + rolling_std))).rolling(window=window).mean()
        
        return within_std.fillna(0.5).values
    
    async def scale_features(self, feature_set: FeatureSet, 
                           fit_scalers: bool = False) -> FeatureSet:
        """Scale features using specified scaling method"""
        try:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if self.scaling_method == 'standard':
                scaler_class = StandardScaler
            elif self.scaling_method == 'minmax':
                scaler_class = MinMaxScaler
            elif self.scaling_method == 'robust':
                scaler_class = RobustScaler
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}")
                return feature_set
            
            scaled_features = feature_set.features.copy()
            
            for i, feature_name in enumerate(feature_set.feature_names):
                if feature_name not in self.feature_scalers:
                    if fit_scalers:
                        self.feature_scalers[feature_name] = scaler_class()
                        self.feature_scalers[feature_name].fit(scaled_features[:, i].reshape(-1, 1))
                    else:
                        # Skip scaling if no fitted scaler available
                        continue
                
                if feature_name in self.feature_scalers:
                    scaled_features[:, i] = self.feature_scalers[feature_name].transform(
                        scaled_features[:, i].reshape(-1, 1)
                    ).flatten()
            
            return FeatureSet(
                features=scaled_features,
                feature_names=feature_set.feature_names,
                target=feature_set.target,
                metadata={**feature_set.metadata, 'scaled': True},
                timestamp=feature_set.timestamp
            )
            
        except Exception as e:
            logger.error("Feature scaling failed", error=str(e))
            return feature_set
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance.copy()
    
    def update_feature_importance(self, importance_scores: Dict[str, float]):
        """Update feature importance from model training"""
        self.feature_importance.update(importance_scores)
        
        # Record metrics
        for feature, importance in importance_scores.items():
            self.metrics_collector.record_metric('feature_importance', importance, 
                                                tags={'feature': feature})


async def create_feature_engineer(config: Dict[str, Any] = None) -> FeatureEngineer:
    """Factory function to create feature engineer"""
    return FeatureEngineer(config)