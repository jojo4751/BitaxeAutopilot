"""
BitAxe Statistical Analysis Service
Advanced statistical analysis and data science for mining operations
"""

import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import json
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical analysis results."""
    metric: str
    value: float
    confidence: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    significance: str  # 'high', 'medium', 'low'
    details: Dict[str, Any]


@dataclass
class PerformanceProfile:
    """Miner performance profile with statistical characteristics."""
    ip: str
    period_start: datetime
    period_end: datetime
    
    # Basic statistics
    avg_hashrate: float
    std_hashrate: float
    min_hashrate: float
    max_hashrate: float
    
    avg_temp: float
    std_temp: float
    min_temp: float
    max_temp: float
    
    avg_power: float
    std_power: float
    
    avg_efficiency: float
    std_efficiency: float
    
    # Advanced metrics
    stability_score: float
    consistency_score: float
    performance_grade: str  # A, B, C, D, F
    
    # Anomalies and patterns
    anomaly_count: int
    thermal_events: int
    power_events: int
    
    # Operational metrics
    uptime_percentage: float
    data_completeness: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ip': self.ip,
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'hashrate_stats': {
                'avg': self.avg_hashrate,
                'std': self.std_hashrate,
                'min': self.min_hashrate,
                'max': self.max_hashrate
            },
            'temperature_stats': {
                'avg': self.avg_temp,
                'std': self.std_temp,
                'min': self.min_temp,
                'max': self.max_temp
            },
            'power_stats': {
                'avg': self.avg_power,
                'std': self.std_power
            },
            'efficiency_stats': {
                'avg': self.avg_efficiency,
                'std': self.std_efficiency
            },
            'performance': {
                'stability_score': self.stability_score,
                'consistency_score': self.consistency_score,
                'grade': self.performance_grade
            },
            'events': {
                'anomalies': self.anomaly_count,
                'thermal_events': self.thermal_events,
                'power_events': self.power_events
            },
            'operational': {
                'uptime_percentage': self.uptime_percentage,
                'data_completeness': self.data_completeness
            }
        }


class StatisticalAnalysisService:
    """Advanced statistical analysis service for mining data."""
    
    def __init__(self, database_manager, config_service):
        self.database_manager = database_manager
        self.config_service = config_service
        
        # Analysis parameters
        self.significance_threshold = 0.05
        self.anomaly_threshold = 2.5  # Standard deviations
        self.trend_confidence_threshold = 0.7
        
        logger.info("Statistical Analysis Service initialized")
    
    def analyze_performance_profile(
        self,
        ip: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[PerformanceProfile]:
        """Generate comprehensive performance profile for a miner.
        
        Args:
            ip: Miner IP address
            start_time: Analysis period start
            end_time: Analysis period end
            
        Returns:
            Performance profile with statistical analysis
        """
        try:
            # Get historical data
            historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
            
            if not historical_data or ip not in historical_data:
                logger.warning(f"No data found for miner {ip} in specified period")
                return None
            
            data_points = historical_data[ip]
            
            if len(data_points) < 10:  # Need minimum data for analysis
                logger.warning(f"Insufficient data points ({len(data_points)}) for analysis")
                return None
            
            # Extract metrics
            hashrates = [d.get('hashRate', 0) for d in data_points if d.get('hashRate') is not None]
            temps = [d.get('temp', 0) for d in data_points if d.get('temp') is not None]
            powers = [d.get('power', 0) for d in data_points if d.get('power') is not None]
            
            # Calculate efficiencies
            efficiencies = []
            for d in data_points:
                hr = d.get('hashRate', 0)
                pw = d.get('power', 0)
                if hr and pw and pw > 0:
                    efficiencies.append(hr / pw)
            
            # Basic statistics
            hashrate_stats = self._calculate_basic_stats(hashrates) if hashrates else {}
            temp_stats = self._calculate_basic_stats(temps) if temps else {}
            power_stats = self._calculate_basic_stats(powers) if powers else {}
            efficiency_stats = self._calculate_basic_stats(efficiencies) if efficiencies else {}
            
            # Advanced metrics
            stability_score = self._calculate_stability_score(hashrates, temps)
            consistency_score = self._calculate_consistency_score(hashrates, efficiencies)
            performance_grade = self._calculate_performance_grade(stability_score, consistency_score, efficiency_stats.get('mean', 0))
            
            # Anomaly detection
            anomaly_count = self._detect_anomalies(data_points)
            thermal_events = self._count_thermal_events(temps)
            power_events = self._count_power_events(powers)
            
            # Operational metrics
            uptime_percentage = self._calculate_uptime_percentage(start_time, end_time, len(data_points))
            data_completeness = self._calculate_data_completeness(data_points)
            
            return PerformanceProfile(
                ip=ip,
                period_start=start_time,
                period_end=end_time,
                
                avg_hashrate=hashrate_stats.get('mean', 0),
                std_hashrate=hashrate_stats.get('std', 0),
                min_hashrate=hashrate_stats.get('min', 0),
                max_hashrate=hashrate_stats.get('max', 0),
                
                avg_temp=temp_stats.get('mean', 0),
                std_temp=temp_stats.get('std', 0),
                min_temp=temp_stats.get('min', 0),
                max_temp=temp_stats.get('max', 0),
                
                avg_power=power_stats.get('mean', 0),
                std_power=power_stats.get('std', 0),
                
                avg_efficiency=efficiency_stats.get('mean', 0),
                std_efficiency=efficiency_stats.get('std', 0),
                
                stability_score=stability_score,
                consistency_score=consistency_score,
                performance_grade=performance_grade,
                
                anomaly_count=anomaly_count,
                thermal_events=thermal_events,
                power_events=power_events,
                
                uptime_percentage=uptime_percentage,
                data_completeness=data_completeness
            )
            
        except Exception as e:
            logger.error(f"Error analyzing performance profile for {ip}: {e}")
            return None
    
    def _calculate_basic_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        if not values:
            return {}
        
        try:
            clean_values = [v for v in values if v is not None and not math.isnan(v)]
            
            if not clean_values:
                return {}
            
            return {
                'mean': statistics.mean(clean_values),
                'median': statistics.median(clean_values),
                'std': statistics.stdev(clean_values) if len(clean_values) > 1 else 0,
                'min': min(clean_values),
                'max': max(clean_values),
                'q25': np.percentile(clean_values, 25) if clean_values else 0,
                'q75': np.percentile(clean_values, 75) if clean_values else 0,
                'count': len(clean_values)
            }
        except Exception as e:
            logger.error(f"Error calculating basic stats: {e}")
            return {}
    
    def _calculate_stability_score(self, hashrates: List[float], temps: List[float]) -> float:
        """Calculate stability score based on coefficient of variation."""
        try:
            scores = []
            
            # Hashrate stability (lower CV = higher stability)
            if hashrates and len(hashrates) > 1:
                hr_mean = statistics.mean(hashrates)
                hr_std = statistics.stdev(hashrates)
                hr_cv = hr_std / hr_mean if hr_mean > 0 else 1
                hr_stability = max(0, 100 - (hr_cv * 100))  # Invert CV to stability score
                scores.append(hr_stability)
            
            # Temperature stability
            if temps and len(temps) > 1:
                temp_mean = statistics.mean(temps)
                temp_std = statistics.stdev(temps)
                temp_cv = temp_std / temp_mean if temp_mean > 0 else 1
                temp_stability = max(0, 100 - (temp_cv * 50))  # Less weight on temp variation
                scores.append(temp_stability)
            
            return statistics.mean(scores) if scores else 0
            
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 0
    
    def _calculate_consistency_score(self, hashrates: List[float], efficiencies: List[float]) -> float:
        """Calculate consistency score based on performance predictability."""
        try:
            scores = []
            
            # Hashrate consistency over time
            if hashrates and len(hashrates) >= 10:
                # Calculate moving average deviation
                window_size = min(10, len(hashrates) // 3)
                moving_avgs = []
                
                for i in range(len(hashrates) - window_size + 1):
                    window = hashrates[i:i + window_size]
                    moving_avgs.append(statistics.mean(window))
                
                if len(moving_avgs) > 1:
                    ma_std = statistics.stdev(moving_avgs)
                    ma_mean = statistics.mean(moving_avgs)
                    ma_cv = ma_std / ma_mean if ma_mean > 0 else 1
                    consistency = max(0, 100 - (ma_cv * 80))
                    scores.append(consistency)
            
            # Efficiency consistency
            if efficiencies and len(efficiencies) > 1:
                eff_std = statistics.stdev(efficiencies)
                eff_mean = statistics.mean(efficiencies)
                eff_cv = eff_std / eff_mean if eff_mean > 0 else 1
                eff_consistency = max(0, 100 - (eff_cv * 60))
                scores.append(eff_consistency)
            
            return statistics.mean(scores) if scores else 0
            
        except Exception as e:
            logger.error(f"Error calculating consistency score: {e}")
            return 0
    
    def _calculate_performance_grade(self, stability: float, consistency: float, avg_efficiency: float) -> str:
        """Calculate overall performance grade."""
        try:
            # Normalize efficiency to 0-100 scale (assume 15 GH/W as excellent)
            efficiency_score = min(100, (avg_efficiency / 15) * 100) if avg_efficiency > 0 else 0
            
            # Weighted overall score
            overall_score = (
                stability * 0.3 +
                consistency * 0.3 +
                efficiency_score * 0.4
            )
            
            if overall_score >= 90:
                return 'A'
            elif overall_score >= 80:
                return 'B'
            elif overall_score >= 70:
                return 'C'
            elif overall_score >= 60:
                return 'D'
            else:
                return 'F'
                
        except Exception as e:
            logger.error(f"Error calculating performance grade: {e}")
            return 'F'
    
    def _detect_anomalies(self, data_points: List[Dict[str, Any]]) -> int:
        """Detect anomalies using statistical methods."""
        try:
            anomaly_count = 0
            
            # Extract key metrics
            hashrates = [d.get('hashRate', 0) for d in data_points if d.get('hashRate')]
            temps = [d.get('temp', 0) for d in data_points if d.get('temp')]
            powers = [d.get('power', 0) for d in data_points if d.get('power')]
            
            # Check each metric for outliers
            for values, metric_name in [(hashrates, 'hashrate'), (temps, 'temperature'), (powers, 'power')]:
                if len(values) > 10:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values)
                    
                    # Count values beyond threshold standard deviations
                    for val in values:
                        z_score = abs(val - mean_val) / std_val if std_val > 0 else 0
                        if z_score > self.anomaly_threshold:
                            anomaly_count += 1
            
            return anomaly_count
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return 0
    
    def _count_thermal_events(self, temps: List[float]) -> int:
        """Count thermal events (overheating)."""
        try:
            if not temps:
                return 0
            
            temp_limit = self.config_service.temp_limit
            return sum(1 for temp in temps if temp >= temp_limit)
            
        except Exception as e:
            logger.error(f"Error counting thermal events: {e}")
            return 0
    
    def _count_power_events(self, powers: List[float]) -> int:
        """Count power anomalies."""
        try:
            if not powers or len(powers) < 10:
                return 0
            
            mean_power = statistics.mean(powers)
            std_power = statistics.stdev(powers)
            
            # Count significant power deviations
            threshold = 2.0  # Standard deviations
            events = 0
            
            for power in powers:
                if std_power > 0:
                    z_score = abs(power - mean_power) / std_power
                    if z_score > threshold:
                        events += 1
            
            return events
            
        except Exception as e:
            logger.error(f"Error counting power events: {e}")
            return 0
    
    def _calculate_uptime_percentage(self, start_time: datetime, end_time: datetime, data_points: int) -> float:
        """Calculate uptime percentage based on expected data points."""
        try:
            total_seconds = (end_time - start_time).total_seconds()
            expected_points = total_seconds / 30  # Assuming 30-second intervals
            
            if expected_points > 0:
                return min(100, (data_points / expected_points) * 100)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating uptime percentage: {e}")
            return 0
    
    def _calculate_data_completeness(self, data_points: List[Dict[str, Any]]) -> float:
        """Calculate data completeness score."""
        try:
            if not data_points:
                return 0
            
            required_fields = ['hashRate', 'temp', 'power', 'frequency', 'coreVoltage']
            complete_records = 0
            
            for point in data_points:
                if all(point.get(field) is not None for field in required_fields):
                    complete_records += 1
            
            return (complete_records / len(data_points)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating data completeness: {e}")
            return 0
    
    def compare_miners(
        self,
        ips: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Compare performance across multiple miners.
        
        Args:
            ips: List of miner IP addresses
            start_time: Comparison period start
            end_time: Comparison period end
            
        Returns:
            Comparative analysis results
        """
        try:
            profiles = {}
            
            # Generate profiles for each miner
            for ip in ips:
                profile = self.analyze_performance_profile(ip, start_time, end_time)
                if profile:
                    profiles[ip] = profile
            
            if not profiles:
                return {'error': 'No valid profiles generated'}
            
            # Comparative statistics
            comparison = {
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'miners': len(profiles),
                'profiles': {ip: profile.to_dict() for ip, profile in profiles.items()},
                'fleet_stats': self._calculate_fleet_statistics(profiles),
                'rankings': self._rank_miners(profiles),
                'recommendations': self._generate_recommendations(profiles)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing miners: {e}")
            return {'error': str(e)}
    
    def _calculate_fleet_statistics(self, profiles: Dict[str, PerformanceProfile]) -> Dict[str, Any]:
        """Calculate fleet-wide statistics."""
        try:
            if not profiles:
                return {}
            
            # Aggregate metrics
            hashrates = [p.avg_hashrate for p in profiles.values()]
            temps = [p.avg_temp for p in profiles.values()]
            efficiencies = [p.avg_efficiency for p in profiles.values()]
            stability_scores = [p.stability_score for p in profiles.values()]
            consistency_scores = [p.consistency_score for p in profiles.values()]
            
            return {
                'total_hashrate': sum(hashrates),
                'avg_hashrate': statistics.mean(hashrates),
                'hashrate_std': statistics.stdev(hashrates) if len(hashrates) > 1 else 0,
                
                'avg_temperature': statistics.mean(temps),
                'temp_range': max(temps) - min(temps) if temps else 0,
                
                'avg_efficiency': statistics.mean(efficiencies),
                'efficiency_std': statistics.stdev(efficiencies) if len(efficiencies) > 1 else 0,
                
                'avg_stability': statistics.mean(stability_scores),
                'avg_consistency': statistics.mean(consistency_scores),
                
                'grade_distribution': self._calculate_grade_distribution(profiles)
            }
            
        except Exception as e:
            logger.error(f"Error calculating fleet statistics: {e}")
            return {}
    
    def _calculate_grade_distribution(self, profiles: Dict[str, PerformanceProfile]) -> Dict[str, int]:
        """Calculate distribution of performance grades."""
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        
        for profile in profiles.values():
            grade = profile.performance_grade
            if grade in grades:
                grades[grade] += 1
        
        return grades
    
    def _rank_miners(self, profiles: Dict[str, PerformanceProfile]) -> Dict[str, List[Dict[str, Any]]]:
        """Rank miners by various performance metrics."""
        try:
            rankings = {
                'by_efficiency': [],
                'by_stability': [],
                'by_consistency': [],
                'by_overall_grade': []
            }
            
            # Efficiency ranking
            efficiency_sorted = sorted(
                profiles.items(),
                key=lambda x: x[1].avg_efficiency,
                reverse=True
            )
            
            for rank, (ip, profile) in enumerate(efficiency_sorted, 1):
                rankings['by_efficiency'].append({
                    'rank': rank,
                    'ip': ip,
                    'value': profile.avg_efficiency,
                    'grade': profile.performance_grade
                })
            
            # Stability ranking
            stability_sorted = sorted(
                profiles.items(),
                key=lambda x: x[1].stability_score,
                reverse=True
            )
            
            for rank, (ip, profile) in enumerate(stability_sorted, 1):
                rankings['by_stability'].append({
                    'rank': rank,
                    'ip': ip,
                    'value': profile.stability_score,
                    'grade': profile.performance_grade
                })
            
            # Consistency ranking
            consistency_sorted = sorted(
                profiles.items(),
                key=lambda x: x[1].consistency_score,
                reverse=True
            )
            
            for rank, (ip, profile) in enumerate(consistency_sorted, 1):
                rankings['by_consistency'].append({
                    'rank': rank,
                    'ip': ip,
                    'value': profile.consistency_score,
                    'grade': profile.performance_grade
                })
            
            # Overall grade ranking
            grade_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
            overall_sorted = sorted(
                profiles.items(),
                key=lambda x: (
                    grade_order.get(x[1].performance_grade, 0),
                    x[1].avg_efficiency,
                    x[1].stability_score
                ),
                reverse=True
            )
            
            for rank, (ip, profile) in enumerate(overall_sorted, 1):
                rankings['by_overall_grade'].append({
                    'rank': rank,
                    'ip': ip,
                    'grade': profile.performance_grade,
                    'efficiency': profile.avg_efficiency,
                    'stability': profile.stability_score
                })
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error ranking miners: {e}")
            return {}
    
    def _generate_recommendations(self, profiles: Dict[str, PerformanceProfile]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        try:
            for ip, profile in profiles.items():
                # Temperature recommendations
                if profile.avg_temp >= self.config_service.temp_limit * 0.9:
                    recommendations.append({
                        'type': 'temperature',
                        'priority': 'high',
                        'miner': ip,
                        'issue': f'High operating temperature ({profile.avg_temp:.1f}°C)',
                        'recommendation': 'Check cooling, reduce frequency, or improve ventilation'
                    })
                
                # Efficiency recommendations
                fleet_avg_efficiency = statistics.mean([p.avg_efficiency for p in profiles.values()])
                if profile.avg_efficiency < fleet_avg_efficiency * 0.85:
                    recommendations.append({
                        'type': 'efficiency',
                        'priority': 'medium',
                        'miner': ip,
                        'issue': f'Below-average efficiency ({profile.avg_efficiency:.3f} GH/W)',
                        'recommendation': 'Consider frequency/voltage optimization or hardware maintenance'
                    })
                
                # Stability recommendations
                if profile.stability_score < 70:
                    recommendations.append({
                        'type': 'stability',
                        'priority': 'medium',
                        'miner': ip,
                        'issue': f'Low stability score ({profile.stability_score:.1f})',
                        'recommendation': 'Check power supply stability and network connectivity'
                    })
                
                # Anomaly recommendations
                if profile.anomaly_count > len(profiles) * 5:  # More than 5x average
                    recommendations.append({
                        'type': 'anomaly',
                        'priority': 'high',
                        'miner': ip,
                        'issue': f'High anomaly count ({profile.anomaly_count})',
                        'recommendation': 'Investigate for hardware issues or environmental factors'
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def analyze_trends(
        self,
        ip: str,
        metric: str,
        start_time: datetime,
        end_time: datetime
    ) -> StatisticalResult:
        """Analyze trends in a specific metric over time.
        
        Args:
            ip: Miner IP address
            metric: Metric to analyze ('hashRate', 'temp', 'power', 'efficiency')
            start_time: Analysis period start
            end_time: Analysis period end
            
        Returns:
            Statistical analysis result with trend information
        """
        try:
            # Get historical data
            historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
            
            if not historical_data or ip not in historical_data:
                return StatisticalResult(
                    metric=metric,
                    value=0,
                    confidence=0,
                    trend='unknown',
                    significance='low',
                    details={'error': 'No data available'}
                )
            
            data_points = historical_data[ip]
            
            if len(data_points) < 10:
                return StatisticalResult(
                    metric=metric,
                    value=0,
                    confidence=0,
                    trend='insufficient_data',
                    significance='low',
                    details={'data_points': len(data_points)}
                )
            
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for point in data_points:
                if metric == 'efficiency':
                    hr = point.get('hashRate', 0)
                    pw = point.get('power', 0)
                    if hr and pw and pw > 0:
                        values.append(hr / pw)
                        timestamps.append(datetime.fromisoformat(point.get('timestamp', '')))
                else:
                    val = point.get(metric)
                    if val is not None:
                        values.append(val)
                        timestamps.append(datetime.fromisoformat(point.get('timestamp', '')))
            
            if len(values) < 10:
                return StatisticalResult(
                    metric=metric,
                    value=0,
                    confidence=0,
                    trend='insufficient_data',
                    significance='low',
                    details={'valid_points': len(values)}
                )
            
            # Calculate trend using linear regression
            x_values = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # Simple linear regression
            n = len(values)
            sum_x = sum(x_values)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_values, values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Slope calculation
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # R-squared for confidence
            y_mean = statistics.mean(values)
            ss_tot = sum((y - y_mean) ** 2 for y in values)
            
            # Predicted values
            intercept = (sum_y - slope * sum_x) / n
            y_pred = [slope * x + intercept for x in x_values]
            ss_res = sum((y - y_p) ** 2 for y, y_p in zip(values, y_pred))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Determine trend direction and significance
            if abs(slope) < 1e-10:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Significance based on R-squared and slope magnitude
            if r_squared > 0.7 and abs(slope) > statistics.stdev(values) / (24 * 3600):  # Daily change > 1 std
                significance = 'high'
            elif r_squared > 0.5:
                significance = 'medium'
            else:
                significance = 'low'
            
            return StatisticalResult(
                metric=metric,
                value=statistics.mean(values),
                confidence=r_squared,
                trend=trend,
                significance=significance,
                details={
                    'slope': slope,
                    'r_squared': r_squared,
                    'data_points': len(values),
                    'period_days': (end_time - start_time).days,
                    'value_range': [min(values), max(values)],
                    'std_dev': statistics.stdev(values)
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trends for {ip}/{metric}: {e}")
            return StatisticalResult(
                metric=metric,
                value=0,
                confidence=0,
                trend='error',
                significance='low',
                details={'error': str(e)}
            )