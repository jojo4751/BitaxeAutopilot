"""
BitAxe Analytics Service
Advanced data processing and analytics for mining operations
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import requests
import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Comprehensive analytics service for mining data analysis."""
    
    def __init__(self, config_service, database_manager, miner_service):
        self.config_service = config_service
        self.database_manager = database_manager
        self.miner_service = miner_service
        
        # Default profitability parameters
        self.default_electricity_cost = 0.12  # USD per kWh
        self.default_pool_fee = 1.0  # 1%
        
    def calculate_enhanced_metrics(self, miner_data: Dict[str, Any], ip: str) -> Dict[str, Any]:
        """Calculate enhanced metrics from raw miner data.
        
        Args:
            miner_data: Raw miner telemetry data
            ip: Miner IP address
            
        Returns:
            Dictionary with enhanced metrics
        """
        enhanced = miner_data.copy()
        
        try:
            # Calculate efficiency (GH/W)
            hashrate = miner_data.get('hashRate', 0)
            power = miner_data.get('power', 0)
            
            if power > 0:
                enhanced['efficiency_ghw'] = hashrate / power
            else:
                enhanced['efficiency_ghw'] = 0
            
            # Calculate temperature health score (0-100)
            temp = miner_data.get('temp', 0)
            temp_limit = self.config_service.temp_limit
            temp_overheat = self.config_service.temp_overheat
            
            if temp <= temp_limit:
                enhanced['temp_health_score'] = 100
            elif temp <= temp_overheat:
                # Linear decrease from 100 to 50 between temp_limit and temp_overheat
                enhanced['temp_health_score'] = 100 - (50 * (temp - temp_limit) / (temp_overheat - temp_limit))
            else:
                # Critical temperature - score drops to 0-50 range
                enhanced['temp_health_score'] = max(0, 50 - (temp - temp_overheat))
            
            # Calculate share efficiency
            shares_accepted = miner_data.get('sharesAccepted', 0)
            shares_rejected = miner_data.get('sharesRejected', 0)
            total_shares = shares_accepted + shares_rejected
            
            if total_shares > 0:
                enhanced['share_efficiency'] = (shares_accepted / total_shares) * 100
            else:
                enhanced['share_efficiency'] = 0
            
            # Calculate uptime score
            uptime = miner_data.get('uptime', 0)
            if uptime > 0:
                # Convert uptime to hours and calculate score
                uptime_hours = uptime / 3600
                enhanced['uptime_score'] = min(100, (uptime_hours / 24) * 100)
            else:
                enhanced['uptime_score'] = 0
            
            # Calculate WiFi health score
            wifi_rssi = miner_data.get('wifiRSSI', -100)
            if wifi_rssi >= -50:
                enhanced['wifi_health_score'] = 100
            elif wifi_rssi >= -70:
                enhanced['wifi_health_score'] = 75
            elif wifi_rssi >= -85:
                enhanced['wifi_health_score'] = 50
            else:
                enhanced['wifi_health_score'] = 25
            
            # Calculate overall performance score
            enhanced['performance_score'] = self._calculate_performance_score(enhanced)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics for {ip}: {e}")
            
        return enhanced
    
    def _calculate_performance_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100).
        
        Args:
            data: Enhanced miner data
            
        Returns:
            Performance score
        """
        try:
            # Weight factors for different metrics
            weights = {
                'efficiency': 0.3,
                'temperature': 0.25,
                'shares': 0.2,
                'uptime': 0.15,
                'wifi': 0.1
            }
            
            # Normalize efficiency score (assume 15 GH/W as excellent)
            efficiency_score = min(100, (data.get('efficiency_ghw', 0) / 15) * 100)
            
            # Get other scores
            temp_score = data.get('temp_health_score', 0)
            share_score = data.get('share_efficiency', 0)
            uptime_score = data.get('uptime_score', 0)
            wifi_score = data.get('wifi_health_score', 0)
            
            # Calculate weighted average
            performance_score = (
                efficiency_score * weights['efficiency'] +
                temp_score * weights['temperature'] +
                share_score * weights['shares'] +
                uptime_score * weights['uptime'] +
                wifi_score * weights['wifi']
            )
            
            return round(performance_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def calculate_profitability_metrics(self, miner_data: Dict[str, Any], ip: str) -> Dict[str, Any]:
        """Calculate profitability metrics.
        
        Args:
            miner_data: Miner telemetry data
            ip: Miner IP address
            
        Returns:
            Dictionary with profitability metrics
        """
        try:
            hashrate_th = miner_data.get('hashRate', 0) / 1000  # Convert GH to TH
            power_kw = miner_data.get('power', 0) / 1000  # Convert W to kW
            
            # Get current Bitcoin metrics (mock data for now)
            btc_metrics = self._get_bitcoin_metrics()
            
            # Calculate daily earnings
            network_hashrate_eh = btc_metrics['network_hashrate_eh']
            block_reward = btc_metrics['block_reward']
            blocks_per_day = 144  # Approximately 144 blocks per day
            
            if network_hashrate_eh > 0:
                daily_btc_earnings = (hashrate_th / (network_hashrate_eh * 1000000)) * block_reward * blocks_per_day
            else:
                daily_btc_earnings = 0
            
            # Calculate daily costs
            daily_power_cost = power_kw * 24 * self.default_electricity_cost
            
            # Calculate profitability
            btc_price = btc_metrics['btc_price_usd']
            daily_revenue = daily_btc_earnings * btc_price
            pool_fee = daily_revenue * (self.default_pool_fee / 100)
            daily_profit = daily_revenue - pool_fee - daily_power_cost
            
            profit_margin = (daily_profit / daily_revenue * 100) if daily_revenue > 0 else 0
            
            return {
                'hashrate': hashrate_th,
                'power_consumption': power_kw,
                'electricity_cost_kwh': self.default_electricity_cost,
                'btc_price_usd': btc_price,
                'network_difficulty': btc_metrics.get('difficulty', 0),
                'pool_fee_percentage': self.default_pool_fee,
                'estimated_daily_btc': daily_btc_earnings,
                'estimated_daily_usd': daily_revenue,
                'daily_power_cost_usd': daily_power_cost,
                'estimated_daily_profit_usd': daily_profit,
                'profit_margin_percentage': profit_margin
            }
            
        except Exception as e:
            logger.error(f"Error calculating profitability for {ip}: {e}")
            return {}
    
    def _get_bitcoin_metrics(self) -> Dict[str, Any]:
        """Get current Bitcoin network metrics (mock implementation).
        
        Returns:
            Dictionary with Bitcoin metrics
        """
        # Mock data - in production, this would fetch real data from APIs
        return {
            'btc_price_usd': 45000.0,
            'network_hashrate_eh': 400.0,  # Exahashes per second
            'difficulty': 50000000000000,
            'block_reward': 6.25
        }
    
    def calculate_health_score(self, miner_data: Dict[str, Any], ip: str) -> Dict[str, Any]:
        """Calculate comprehensive miner health score.
        
        Args:
            miner_data: Enhanced miner data
            ip: Miner IP address
            
        Returns:
            Dictionary with health assessment
        """
        try:
            # Get individual health scores
            temp_health = miner_data.get('temp_health_score', 0)
            efficiency_health = min(100, (miner_data.get('efficiency_ghw', 0) / 12) * 100)
            stability_health = miner_data.get('share_efficiency', 0)
            connectivity_health = miner_data.get('wifi_health_score', 0)
            
            # Calculate overall health score
            health_weights = {
                'temperature': 0.3,
                'efficiency': 0.25,
                'stability': 0.25,
                'connectivity': 0.2
            }
            
            overall_health = (
                temp_health * health_weights['temperature'] +
                efficiency_health * health_weights['efficiency'] +
                stability_health * health_weights['stability'] +
                connectivity_health * health_weights['connectivity']
            )
            
            # Determine maintenance status
            maintenance_due = (
                temp_health < 50 or
                efficiency_health < 40 or
                stability_health < 80 or
                connectivity_health < 30
            )
            
            return {
                'health_score': round(overall_health, 2),
                'temp_health': temp_health,
                'efficiency_health': round(efficiency_health, 2),
                'stability_health': stability_health,
                'connectivity_health': connectivity_health,
                'maintenance_due': maintenance_due,
                'last_restart': None,  # Would be tracked separately
                'consecutive_errors': 0,  # Would be tracked separately
                'notes': self._generate_health_notes(temp_health, efficiency_health, stability_health, connectivity_health)
            }
            
        except Exception as e:
            logger.error(f"Error calculating health score for {ip}: {e}")
            return {}
    
    def _generate_health_notes(self, temp_health: float, efficiency_health: float, 
                             stability_health: float, connectivity_health: float) -> str:
        """Generate health assessment notes.
        
        Returns:
            String with health notes
        """
        notes = []
        
        if temp_health < 50:
            notes.append("High temperature - check cooling")
        elif temp_health < 75:
            notes.append("Elevated temperature - monitor closely")
            
        if efficiency_health < 40:
            notes.append("Low efficiency - consider frequency/voltage adjustment")
        elif efficiency_health < 70:
            notes.append("Below optimal efficiency")
            
        if stability_health < 80:
            notes.append("High reject rate - check pool connection")
        elif stability_health < 95:
            notes.append("Some rejected shares detected")
            
        if connectivity_health < 30:
            notes.append("Poor WiFi signal - check antenna/position")
        elif connectivity_health < 60:
            notes.append("Weak WiFi signal")
        
        return "; ".join(notes) if notes else "All systems operating normally"
    
    def detect_anomalies(self, current_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in miner performance data.
        
        Args:
            current_data: Current miner data
            historical_data: List of historical data points
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(historical_data) < 10:
            return anomalies  # Need sufficient historical data
        
        try:
            # Check for significant deviations in key metrics
            metrics_to_check = ['hashRate', 'temp', 'power', 'efficiency_ghw']
            
            for metric in metrics_to_check:
                if metric in current_data:
                    current_value = current_data[metric]
                    historical_values = [d.get(metric, 0) for d in historical_data if d.get(metric) is not None]
                    
                    if len(historical_values) >= 5:
                        mean_val = statistics.mean(historical_values)
                        stdev_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                        
                        # Detect significant deviations (2+ standard deviations)
                        if stdev_val > 0:
                            z_score = abs(current_value - mean_val) / stdev_val
                            
                            if z_score > 2:
                                anomalies.append({
                                    'metric': metric,
                                    'current_value': current_value,
                                    'expected_value': mean_val,
                                    'deviation': z_score,
                                    'severity': 'high' if z_score > 3 else 'medium',
                                    'description': f"{metric} deviates significantly from normal range"
                                })
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def generate_performance_summary(self, period_type: str, start_time: datetime, 
                                   end_time: datetime, ip: str) -> Dict[str, Any]:
        """Generate performance summary for a specific period.
        
        Args:
            period_type: Type of period ('hourly', 'daily', 'weekly')
            start_time: Period start time
            end_time: Period end time
            ip: Miner IP address
            
        Returns:
            Performance summary dictionary
        """
        try:
            # Get historical data for the period
            historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
            
            if not historical_data or ip not in historical_data:
                return {}
            
            data_points = historical_data[ip]
            
            if not data_points:
                return {}
            
            # Calculate summary statistics
            hashrates = [d.get('hashRate', 0) for d in data_points if d.get('hashRate')]
            temps = [d.get('temp', 0) for d in data_points if d.get('temp')]
            powers = [d.get('power', 0) for d in data_points if d.get('power')]
            
            summary = {
                'period_type': period_type,
                'period_start': start_time,
                'period_end': end_time,
                'ip': ip,
                'data_points': len(data_points)
            }
            
            if hashrates:
                summary.update({
                    'avg_hashrate': statistics.mean(hashrates),
                    'min_hashrate': min(hashrates),
                    'max_hashrate': max(hashrates)
                })
            
            if temps:
                summary.update({
                    'avg_temp': statistics.mean(temps),
                    'min_temp': min(temps),
                    'max_temp': max(temps)
                })
            
            if powers:
                summary['avg_power'] = statistics.mean(powers)
            
            # Calculate efficiency
            if hashrates and powers:
                efficiencies = [h / p for h, p in zip(hashrates, powers) if p > 0]
                if efficiencies:
                    summary['avg_efficiency'] = statistics.mean(efficiencies)
            
            # Calculate uptime percentage
            expected_data_points = int((end_time - start_time).total_seconds() / 30)  # Every 30 seconds
            if expected_data_points > 0:
                summary['uptime_percentage'] = (len(data_points) / expected_data_points) * 100
            
            # Calculate shares
            shares_accepted = sum(d.get('sharesAccepted', 0) for d in data_points)
            shares_rejected = sum(d.get('sharesRejected', 0) for d in data_points)
            
            summary.update({
                'total_shares_accepted': shares_accepted,
                'total_shares_rejected': shares_rejected
            })
            
            # Calculate performance score
            temp_score = max(0, 100 - (summary.get('avg_temp', 0) - 50) * 2) if 'avg_temp' in summary else 0
            efficiency_score = min(100, (summary.get('avg_efficiency', 0) / 15) * 100) if 'avg_efficiency' in summary else 0
            uptime_score = summary.get('uptime_percentage', 0)
            
            summary['performance_score'] = (temp_score * 0.3 + efficiency_score * 0.4 + uptime_score * 0.3)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary for {ip}: {e}")
            return {}
    
    def check_alert_conditions(self, miner_data: Dict[str, Any], ip: str) -> List[Dict[str, Any]]:
        """Check for alert conditions in miner data.
        
        Args:
            miner_data: Enhanced miner data
            ip: Miner IP address
            
        Returns:
            List of alerts to be logged
        """
        alerts = []
        
        try:
            # Temperature alerts
            temp = miner_data.get('temp', 0)
            if temp >= self.config_service.temp_overheat:
                alerts.append({
                    'alert_type': 'temperature',
                    'severity': 'critical',
                    'threshold_value': self.config_service.temp_overheat,
                    'actual_value': temp,
                    'message': f"Critical temperature: {temp}°C (limit: {self.config_service.temp_overheat}°C)"
                })
            elif temp >= self.config_service.temp_limit:
                alerts.append({
                    'alert_type': 'temperature',
                    'severity': 'warning',
                    'threshold_value': self.config_service.temp_limit,
                    'actual_value': temp,
                    'message': f"High temperature: {temp}°C (limit: {self.config_service.temp_limit}°C)"
                })
            
            # Efficiency alerts
            efficiency = miner_data.get('efficiency_ghw', 0)
            if efficiency < 5 and efficiency > 0:  # Below 5 GH/W is concerning
                alerts.append({
                    'alert_type': 'efficiency',
                    'severity': 'warning',
                    'threshold_value': 5.0,
                    'actual_value': efficiency,
                    'message': f"Low efficiency: {efficiency:.2f} GH/W"
                })
            
            # Hashrate alerts
            hashrate = miner_data.get('hashRate', 0)
            if hashrate < 100 and hashrate > 0:  # Below 100 GH/s is concerning
                alerts.append({
                    'alert_type': 'hashrate',
                    'severity': 'warning',
                    'threshold_value': 100.0,
                    'actual_value': hashrate,
                    'message': f"Low hashrate: {hashrate:.2f} GH/s"
                })
            elif hashrate == 0:
                alerts.append({
                    'alert_type': 'hashrate',
                    'severity': 'critical',
                    'threshold_value': 0.0,
                    'actual_value': hashrate,
                    'message': "No hashrate detected - miner may be offline"
                })
            
            # Share efficiency alerts
            share_efficiency = miner_data.get('share_efficiency', 0)
            if share_efficiency < 95 and share_efficiency > 0:
                alerts.append({
                    'alert_type': 'shares',
                    'severity': 'warning',
                    'threshold_value': 95.0,
                    'actual_value': share_efficiency,
                    'message': f"High reject rate: {100 - share_efficiency:.1f}% rejected"
                })
            
        except Exception as e:
            logger.error(f"Error checking alert conditions for {ip}: {e}")
        
        return alerts
    
    def process_miner_data(self, ip: str, raw_data: Dict[str, Any]) -> None:
        """Process and store enhanced miner data with analytics.
        
        Args:
            ip: Miner IP address
            raw_data: Raw miner telemetry data
        """
        try:
            with self.database_manager.transaction() as conn:
                # Calculate enhanced metrics
                enhanced_data = self.calculate_enhanced_metrics(raw_data, ip)
                
                # Log standard miner data
                self.database_manager.log_miner_data(conn, ip, enhanced_data)
                
                # Calculate and log profitability metrics
                profitability_data = self.calculate_profitability_metrics(enhanced_data, ip)
                if profitability_data:
                    self.database_manager.log_profitability_data(conn, ip, profitability_data)
                
                # Calculate and log health score
                health_data = self.calculate_health_score(enhanced_data, ip)
                if health_data:
                    self.database_manager.log_miner_health(conn, ip, health_data)
                
                # Check for alerts
                alerts = self.check_alert_conditions(enhanced_data, ip)
                for alert in alerts:
                    self.database_manager.log_performance_alert(
                        conn, ip, alert['alert_type'], alert['severity'],
                        alert['threshold_value'], alert['actual_value'], alert['message']
                    )
                
                logger.debug(f"Processed enhanced data for {ip}")
                
        except Exception as e:
            logger.error(f"Error processing miner data for {ip}: {e}")