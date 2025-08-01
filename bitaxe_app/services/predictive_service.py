"""
BitAxe Predictive Analytics Service
Machine learning and forecasting for mining operations
"""

import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Container for prediction results."""
    metric: str
    current_value: float
    predicted_value: float
    confidence: float
    time_horizon: timedelta
    trend: str
    risk_level: str  # 'low', 'medium', 'high'
    recommendations: List[str]


@dataclass
class ForecastResult:
    """Container for multi-point forecast results."""
    metric: str
    timestamps: List[datetime]
    values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    assumptions: List[str]


class PredictiveService:
    """Predictive analytics and forecasting service."""
    
    def __init__(self, database_manager, statistical_service, config_service):
        self.database_manager = database_manager
        self.statistical_service = statistical_service
        self.config_service = config_service
        
        # Prediction parameters
        self.min_data_points = 50
        self.default_forecast_horizon = timedelta(hours=24)
        self.confidence_threshold = 0.6
        
        logger.info("Predictive Service initialized")
    
    def predict_next_failure(self, ip: str) -> Dict[str, Any]:
        """Predict when a miner might experience issues.
        
        Args:
            ip: Miner IP address
            
        Returns:
            Failure prediction analysis
        """
        try:
            # Get extended historical data for better predictions
            end_time = datetime.now()
            start_time = end_time - timedelta(days=14)  # 2 weeks of data
            
            historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
            
            if not historical_data or ip not in historical_data:
                return {'error': 'No historical data available'}
            
            data_points = historical_data[ip]
            
            if len(data_points) < self.min_data_points:
                return {'error': f'Insufficient data points ({len(data_points)} < {self.min_data_points})'}
            
            # Analyze degradation patterns
            degradation_analysis = self._analyze_degradation_patterns(data_points)
            
            # Health trend analysis
            health_trends = self._analyze_health_trends(data_points)
            
            # Predict maintenance needs
            maintenance_prediction = self._predict_maintenance_needs(data_points, degradation_analysis)
            
            # Risk assessment
            risk_assessment = self._assess_failure_risk(data_points, degradation_analysis, health_trends)
            
            return {
                'ip': ip,
                'analysis_period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'data_points': len(data_points)
                },
                'degradation_analysis': degradation_analysis,
                'health_trends': health_trends,
                'maintenance_prediction': maintenance_prediction,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_failure_prevention_recommendations(
                    degradation_analysis, health_trends, maintenance_prediction, risk_assessment
                )
            }
            
        except Exception as e:
            logger.error(f"Error predicting failure for {ip}: {e}")
            return {'error': str(e)}
    
    def _analyze_degradation_patterns(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance degradation patterns."""
        try:
            # Extract time series data
            timestamps = []
            hashrates = []
            temps = []
            efficiencies = []
            
            for point in data_points:
                if point.get('timestamp'):
                    timestamps.append(datetime.fromisoformat(point['timestamp']))
                    hashrates.append(point.get('hashRate', 0))
                    temps.append(point.get('temp', 0))
                    
                    # Calculate efficiency
                    hr = point.get('hashRate', 0)
                    pw = point.get('power', 0)
                    if hr and pw and pw > 0:
                        efficiencies.append(hr / pw)
                    else:
                        efficiencies.append(0)
            
            if len(timestamps) < 10:
                return {'error': 'Insufficient timestamped data'}
            
            # Calculate degradation rates
            degradation = {}
            
            # Hashrate degradation
            if hashrates:
                hr_trend = self._calculate_linear_trend(timestamps, hashrates)
                degradation['hashrate'] = {
                    'trend_per_day': hr_trend['slope'] * 86400,  # Convert per second to per day
                    'confidence': hr_trend['r_squared'],
                    'current_value': hashrates[-1],
                    'predicted_7d': hashrates[-1] + (hr_trend['slope'] * 86400 * 7),
                    'degradation_rate_percent_per_day': (hr_trend['slope'] * 86400 / statistics.mean(hashrates)) * 100 if statistics.mean(hashrates) > 0 else 0
                }
            
            # Temperature trend
            if temps:
                temp_trend = self._calculate_linear_trend(timestamps, temps)
                degradation['temperature'] = {
                    'trend_per_day': temp_trend['slope'] * 86400,
                    'confidence': temp_trend['r_squared'],
                    'current_value': temps[-1],
                    'predicted_7d': temps[-1] + (temp_trend['slope'] * 86400 * 7)
                }
            
            # Efficiency degradation
            if efficiencies:
                eff_trend = self._calculate_linear_trend(timestamps, efficiencies)
                degradation['efficiency'] = {
                    'trend_per_day': eff_trend['slope'] * 86400,
                    'confidence': eff_trend['r_squared'],
                    'current_value': efficiencies[-1],
                    'predicted_7d': efficiencies[-1] + (eff_trend['slope'] * 86400 * 7),
                    'degradation_rate_percent_per_day': (eff_trend['slope'] * 86400 / statistics.mean(efficiencies)) * 100 if statistics.mean(efficiencies) > 0 else 0
                }
            
            return degradation
            
        except Exception as e:
            logger.error(f"Error analyzing degradation patterns: {e}")
            return {'error': str(e)}
    
    def _calculate_linear_trend(self, timestamps: List[datetime], values: List[float]) -> Dict[str, float]:
        """Calculate linear trend using least squares regression."""
        try:
            if len(timestamps) != len(values) or len(timestamps) < 2:
                return {'slope': 0, 'intercept': 0, 'r_squared': 0}
            
            # Convert timestamps to seconds from first timestamp
            x_values = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            y_values = values
            
            # Remove None values
            valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if y is not None and not math.isnan(y)]
            
            if len(valid_pairs) < 2:
                return {'slope': 0, 'intercept': 0, 'r_squared': 0}
            
            x_values, y_values = zip(*valid_pairs)
            
            # Linear regression calculations
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope and intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return {'slope': 0, 'intercept': statistics.mean(y_values), 'r_squared': 0}
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_mean = statistics.mean(y_values)
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            if ss_tot == 0:
                r_squared = 1 if slope == 0 else 0
            else:
                y_pred = [slope * x + intercept for x in x_values]
                ss_res = sum((y - y_p) ** 2 for y, y_p in zip(y_values, y_pred))
                r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': max(0, r_squared)  # Ensure non-negative
            }
            
        except Exception as e:
            logger.error(f"Error calculating linear trend: {e}")
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}
    
    def _analyze_health_trends(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall health trends."""
        try:
            # Split data into time windows for trend analysis
            window_size = max(10, len(data_points) // 5)  # 5 windows
            windows = []
            
            for i in range(0, len(data_points), window_size):
                window = data_points[i:i + window_size]
                if len(window) >= 5:  # Minimum window size
                    windows.append(window)
            
            if len(windows) < 2:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Calculate health metrics for each window
            window_health = []
            
            for window in windows:
                hashrates = [d.get('hashRate', 0) for d in window if d.get('hashRate')]
                temps = [d.get('temp', 0) for d in window if d.get('temp')]
                powers = [d.get('power', 0) for d in window if d.get('power')]
                
                # Calculate window health score
                health_score = 0
                score_components = 0
                
                # Hashrate stability component
                if hashrates and len(hashrates) > 1:
                    hr_cv = statistics.stdev(hashrates) / statistics.mean(hashrates)
                    hr_health = max(0, 100 - (hr_cv * 100))
                    health_score += hr_health
                    score_components += 1
                
                # Temperature component
                if temps:
                    avg_temp = statistics.mean(temps)
                    temp_limit = self.config_service.temp_limit
                    temp_health = max(0, 100 - max(0, (avg_temp - temp_limit) * 5))
                    health_score += temp_health
                    score_components += 1
                
                # Efficiency component
                efficiencies = []
                for d in window:
                    hr = d.get('hashRate', 0)
                    pw = d.get('power', 0)
                    if hr and pw and pw > 0:
                        efficiencies.append(hr / pw)
                
                if efficiencies:
                    avg_eff = statistics.mean(efficiencies)
                    # Assume 12 GH/W as baseline good efficiency
                    eff_health = min(100, (avg_eff / 12) * 100)
                    health_score += eff_health
                    score_components += 1
                
                if score_components > 0:
                    window_health.append(health_score / score_components)
                else:
                    window_health.append(0)
            
            # Analyze health trend
            if len(window_health) >= 2:
                window_indices = list(range(len(window_health)))
                health_trend = self._calculate_linear_trend(
                    [datetime.now() + timedelta(hours=i) for i in window_indices],
                    window_health
                )
                
                return {
                    'current_health_score': window_health[-1],
                    'health_trend_per_window': health_trend['slope'],
                    'trend_confidence': health_trend['r_squared'],
                    'windows_analyzed': len(windows),
                    'health_trajectory': window_health,
                    'predicted_health_next_window': window_health[-1] + health_trend['slope']
                }
            else:
                return {'error': 'Insufficient windows for trend analysis'}
                
        except Exception as e:
            logger.error(f"Error analyzing health trends: {e}")
            return {'error': str(e)}
    
    def _predict_maintenance_needs(self, data_points: List[Dict[str, Any]], degradation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict when maintenance might be needed."""
        try:
            maintenance_predictions = {}
            
            # Hashrate-based maintenance prediction
            if 'hashrate' in degradation_analysis:
                hr_data = degradation_analysis['hashrate']
                current_hr = hr_data['current_value']
                degradation_rate = hr_data.get('degradation_rate_percent_per_day', 0)
                
                if degradation_rate < -0.5:  # Losing more than 0.5% per day
                    # Predict when hashrate drops to 80% of current
                    days_to_80_percent = abs(20 / degradation_rate) if degradation_rate != 0 else float('inf')
                    
                    maintenance_predictions['hashrate_degradation'] = {
                        'days_until_maintenance': days_to_80_percent,
                        'trigger_threshold': current_hr * 0.8,
                        'confidence': hr_data['confidence'],
                        'urgency': 'high' if days_to_80_percent < 30 else 'medium' if days_to_80_percent < 90 else 'low'
                    }
            
            # Temperature-based maintenance prediction
            if 'temperature' in degradation_analysis:
                temp_data = degradation_analysis['temperature']
                current_temp = temp_data['current_value']
                temp_trend = temp_data['trend_per_day']
                
                if temp_trend > 0.1:  # Temperature rising by more than 0.1°C per day
                    temp_limit = self.config_service.temp_limit
                    days_to_limit = (temp_limit - current_temp) / temp_trend if temp_trend > 0 else float('inf')
                    
                    maintenance_predictions['temperature_rise'] = {
                        'days_until_maintenance': days_to_limit,
                        'trigger_threshold': temp_limit,
                        'confidence': temp_data['confidence'],
                        'urgency': 'high' if days_to_limit < 7 else 'medium' if days_to_limit < 30 else 'low'
                    }
            
            # Efficiency-based maintenance prediction
            if 'efficiency' in degradation_analysis:
                eff_data = degradation_analysis['efficiency']
                current_eff = eff_data['current_value']
                eff_degradation_rate = eff_data.get('degradation_rate_percent_per_day', 0)
                
                if eff_degradation_rate < -0.3:  # Losing more than 0.3% efficiency per day
                    # Predict when efficiency drops to 70% of current
                    days_to_70_percent = abs(30 / eff_degradation_rate) if eff_degradation_rate != 0 else float('inf')
                    
                    maintenance_predictions['efficiency_degradation'] = {
                        'days_until_maintenance': days_to_70_percent,
                        'trigger_threshold': current_eff * 0.7,
                        'confidence': eff_data['confidence'],
                        'urgency': 'high' if days_to_70_percent < 30 else 'medium' if days_to_70_percent < 90 else 'low'
                    }
            
            # Overall maintenance recommendation
            if maintenance_predictions:
                min_days = min(pred['days_until_maintenance'] for pred in maintenance_predictions.values() if pred['days_until_maintenance'] != float('inf'))
                
                if min_days == float('inf'):
                    overall_urgency = 'low'
                    recommended_action = 'routine_monitoring'
                elif min_days < 7:
                    overall_urgency = 'critical'
                    recommended_action = 'immediate_maintenance'
                elif min_days < 30:
                    overall_urgency = 'high'
                    recommended_action = 'schedule_maintenance'
                elif min_days < 90:
                    overall_urgency = 'medium'
                    recommended_action = 'plan_maintenance'
                else:
                    overall_urgency = 'low'
                    recommended_action = 'routine_monitoring'
                
                maintenance_predictions['overall'] = {
                    'urgency': overall_urgency,
                    'recommended_action': recommended_action,
                    'days_until_next_check': min(min_days, 30) if min_days != float('inf') else 30
                }
            
            return maintenance_predictions
            
        except Exception as e:
            logger.error(f"Error predicting maintenance needs: {e}")
            return {'error': str(e)}
    
    def _assess_failure_risk(self, data_points: List[Dict[str, Any]], degradation_analysis: Dict[str, Any], health_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall failure risk."""
        try:
            risk_factors = []
            risk_score = 0
            
            # Degradation risk factors
            if 'hashrate' in degradation_analysis:
                hr_degradation = degradation_analysis['hashrate'].get('degradation_rate_percent_per_day', 0)
                if hr_degradation < -1.0:  # More than 1% per day
                    risk_factors.append('Severe hashrate degradation')
                    risk_score += 30
                elif hr_degradation < -0.5:
                    risk_factors.append('Moderate hashrate degradation')
                    risk_score += 15
            
            if 'efficiency' in degradation_analysis:
                eff_degradation = degradation_analysis['efficiency'].get('degradation_rate_percent_per_day', 0)
                if eff_degradation < -0.5:
                    risk_factors.append('Efficiency declining rapidly')
                    risk_score += 25
            
            if 'temperature' in degradation_analysis:
                temp_trend = degradation_analysis['temperature'].get('trend_per_day', 0)
                current_temp = degradation_analysis['temperature'].get('current_value', 0)
                temp_limit = self.config_service.temp_limit
                
                if current_temp >= temp_limit:
                    risk_factors.append('Operating above temperature limit')
                    risk_score += 40
                elif current_temp >= temp_limit * 0.95:
                    risk_factors.append('Operating near temperature limit')
                    risk_score += 20
                
                if temp_trend > 0.2:  # Rising more than 0.2°C per day
                    risk_factors.append('Temperature trending upward')
                    risk_score += 15
            
            # Health trend risk factors
            if 'current_health_score' in health_trends:
                health_score = health_trends['current_health_score']
                health_trend = health_trends.get('health_trend_per_window', 0)
                
                if health_score < 50:
                    risk_factors.append('Poor overall health score')
                    risk_score += 30
                elif health_score < 70:
                    risk_factors.append('Below average health score')
                    risk_score += 15
                
                if health_trend < -5:  # Health declining by more than 5 points per window
                    risk_factors.append('Health trending downward')
                    risk_score += 20
            
            # Data quality risk factors
            recent_data = data_points[-20:] if len(data_points) >= 20 else data_points
            missing_data_count = sum(1 for d in recent_data if not all(d.get(field) for field in ['hashRate', 'temp', 'power']))
            
            if missing_data_count > len(recent_data) * 0.3:  # More than 30% missing data
                risk_factors.append('High data loss rate')
                risk_score += 25
            elif missing_data_count > len(recent_data) * 0.1:
                risk_factors.append('Moderate data loss')
                risk_score += 10
            
            # Determine overall risk level
            if risk_score >= 70:
                risk_level = 'critical'
            elif risk_score >= 40:
                risk_level = 'high'
            elif risk_score >= 20:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'overall_risk_score': risk_score,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'assessment_confidence': 0.8 if len(data_points) >= 100 else 0.6,
                'next_assessment_recommended': 'daily' if risk_level in ['critical', 'high'] else 'weekly'
            }
            
        except Exception as e:
            logger.error(f"Error assessing failure risk: {e}")
            return {'error': str(e)}
    
    def _generate_failure_prevention_recommendations(self, degradation_analysis: Dict[str, Any], 
                                                   health_trends: Dict[str, Any], 
                                                   maintenance_prediction: Dict[str, Any], 
                                                   risk_assessment: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations for failure prevention."""
        recommendations = []
        
        try:
            risk_level = risk_assessment.get('risk_level', 'low')
            
            # Critical risk recommendations
            if risk_level == 'critical':
                recommendations.append({
                    'priority': 'critical',
                    'category': 'immediate_action',
                    'action': 'Stop mining operations and perform immediate maintenance',
                    'reason': 'Critical risk factors detected'
                })
            
            # Temperature-based recommendations
            if 'temperature' in degradation_analysis:
                temp_data = degradation_analysis['temperature']
                if temp_data.get('trend_per_day', 0) > 0.1:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'cooling',
                        'action': 'Improve cooling system or reduce operating frequency',
                        'reason': f'Temperature rising at {temp_data["trend_per_day"]:.2f}°C per day'
                    })
            
            # Performance-based recommendations
            if 'hashrate' in degradation_analysis:
                hr_data = degradation_analysis['hashrate']
                degradation_rate = hr_data.get('degradation_rate_percent_per_day', 0)
                
                if degradation_rate < -0.5:
                    recommendations.append({
                        'priority': 'high',
                        'category': 'performance',
                        'action': 'Check hardware components and consider frequency/voltage optimization',
                        'reason': f'Hashrate declining at {abs(degradation_rate):.2f}% per day'
                    })
            
            # Maintenance scheduling recommendations
            if 'overall' in maintenance_prediction:
                maintenance_data = maintenance_prediction['overall']
                urgency = maintenance_data.get('urgency', 'low')
                
                if urgency in ['critical', 'high']:
                    recommendations.append({
                        'priority': urgency,
                        'category': 'maintenance',
                        'action': f'Schedule maintenance within {maintenance_data.get("days_until_next_check", 30)} days',
                        'reason': 'Predictive maintenance threshold approaching'
                    })
            
            # Monitoring recommendations
            recommendations.append({
                'priority': 'medium',
                'category': 'monitoring',
                'action': f'Monitor daily and reassess risk {risk_assessment.get("next_assessment_recommended", "weekly")}',
                'reason': f'Current risk level: {risk_level}'
            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def forecast_metric(self, ip: str, metric: str, hours_ahead: int = 24) -> ForecastResult:
        """Generate forecast for a specific metric.
        
        Args:
            ip: Miner IP address
            metric: Metric to forecast ('hashRate', 'temp', 'power', 'efficiency')
            hours_ahead: Hours to forecast into the future
            
        Returns:
            Forecast result with predictions and confidence intervals
        """
        try:
            # Get historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Use a week of data
            
            historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
            
            if not historical_data or ip not in historical_data:
                return ForecastResult(
                    metric=metric,
                    timestamps=[],
                    values=[],
                    confidence_intervals=[],
                    model_accuracy=0,
                    assumptions=['No historical data available']
                )
            
            data_points = historical_data[ip]
            
            if len(data_points) < 50:
                return ForecastResult(
                    metric=metric,
                    timestamps=[],
                    values=[],
                    confidence_intervals=[],
                    model_accuracy=0,
                    assumptions=['Insufficient historical data for reliable forecast']
                )
            
            # Extract time series
            timestamps = []
            values = []
            
            for point in data_points:
                if point.get('timestamp'):
                    ts = datetime.fromisoformat(point['timestamp'])
                    timestamps.append(ts)
                    
                    if metric == 'efficiency':
                        hr = point.get('hashRate', 0)
                        pw = point.get('power', 0)
                        values.append(hr / pw if pw > 0 else 0)
                    else:
                        values.append(point.get(metric, 0))
            
            if len(values) < 50:
                return ForecastResult(
                    metric=metric,
                    timestamps=[],
                    values=[],
                    confidence_intervals=[],
                    model_accuracy=0,
                    assumptions=['Insufficient valid data points']
                )
            
            # Simple trend-based forecast
            trend = self._calculate_linear_trend(timestamps, values)
            
            # Generate forecast timestamps
            last_timestamp = timestamps[-1]
            forecast_timestamps = []
            for h in range(1, hours_ahead + 1):
                forecast_timestamps.append(last_timestamp + timedelta(hours=h))
            
            # Calculate predictions
            forecast_values = []
            confidence_intervals = []
            
            # Calculate prediction error for confidence intervals
            predicted_historical = [trend['slope'] * (ts - timestamps[0]).total_seconds() + trend['intercept'] for ts in timestamps]
            errors = [abs(actual - pred) for actual, pred in zip(values, predicted_historical)]
            avg_error = statistics.mean(errors) if errors else 0
            
            for ts in forecast_timestamps:
                seconds_from_start = (ts - timestamps[0]).total_seconds()
                predicted_value = trend['slope'] * seconds_from_start + trend['intercept']
                
                # Confidence interval based on historical prediction error
                confidence_margin = avg_error * 1.96  # 95% confidence interval
                
                forecast_values.append(predicted_value)
                confidence_intervals.append((
                    predicted_value - confidence_margin,
                    predicted_value + confidence_margin
                ))
            
            # Model accuracy based on R-squared
            model_accuracy = trend['r_squared']
            
            # Assumptions
            assumptions = [
                'Linear trend continuation',
                'No external interventions',
                'Stable operating conditions',
                f'Based on {len(data_points)} historical data points'
            ]
            
            if model_accuracy < 0.3:
                assumptions.append('Low model confidence due to high variability')
            
            return ForecastResult(
                metric=metric,
                timestamps=forecast_timestamps,
                values=forecast_values,
                confidence_intervals=confidence_intervals,
                model_accuracy=model_accuracy,
                assumptions=assumptions
            )
            
        except Exception as e:
            logger.error(f"Error forecasting {metric} for {ip}: {e}")
            return ForecastResult(
                metric=metric,
                timestamps=[],
                values=[],
                confidence_intervals=[],
                model_accuracy=0,
                assumptions=[f'Forecast error: {str(e)}']
            )