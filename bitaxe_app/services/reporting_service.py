"""
BitAxe Automated Reporting Service
Generate comprehensive analytics reports for mining operations
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section of a report."""
    title: str
    content: Union[str, Dict[str, Any], List[Any]]
    section_type: str  # 'text', 'table', 'chart', 'metrics'
    priority: str = 'medium'  # 'high', 'medium', 'low'


@dataclass
class Report:
    """Complete report structure."""
    title: str
    report_type: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    executive_summary: str
    sections: List[ReportSection]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'title': self.title,
            'report_type': self.report_type,
            'generated_at': self.generated_at.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'executive_summary': self.executive_summary,
            'sections': [asdict(section) for section in self.sections],
            'metadata': self.metadata
        }


class ReportingService:
    """Automated reporting service for mining analytics."""
    
    def __init__(self, database_manager, statistical_service, predictive_service, config_service):
        self.database_manager = database_manager
        self.statistical_service = statistical_service
        self.predictive_service = predictive_service
        self.config_service = config_service
        
        # Report configuration
        self.report_directory = Path("reports")
        self.report_directory.mkdir(exist_ok=True)
        
        logger.info("Reporting Service initialized")
    
    def generate_daily_report(self, target_date: Optional[datetime] = None) -> Report:
        """Generate comprehensive daily operations report.
        
        Args:
            target_date: Date for report (defaults to yesterday)
            
        Returns:
            Complete daily report
        """
        try:
            if target_date is None:
                target_date = datetime.now() - timedelta(days=1)
            
            # Define report period (full day)
            period_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
            
            logger.info(f"Generating daily report for {period_start.strftime('%Y-%m-%d')}")
            
            # Collect data for all miners
            fleet_data = self._collect_fleet_data(period_start, period_end)
            
            # Generate report sections
            sections = []
            
            # Fleet Overview Section
            sections.append(self._generate_fleet_overview_section(fleet_data, period_start, period_end))
            
            # Performance Analysis Section
            sections.append(self._generate_performance_analysis_section(fleet_data))
            
            # Individual Miner Analysis
            sections.extend(self._generate_individual_miner_sections(fleet_data, period_start, period_end))
            
            # Alerts and Issues Section
            sections.append(self._generate_alerts_section(period_start, period_end))
            
            # Recommendations Section
            sections.append(self._generate_recommendations_section(fleet_data))
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(fleet_data, sections)
            
            # Create report
            report = Report(
                title=f"Daily Mining Operations Report - {period_start.strftime('%Y-%m-%d')}",
                report_type="daily",
                generated_at=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                executive_summary=executive_summary,
                sections=sections,
                metadata={
                    'total_miners': len(fleet_data.get('miners', {})),
                    'data_quality_score': fleet_data.get('data_quality_score', 0),
                    'report_version': '2.0'
                }
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            raise
    
    def generate_weekly_report(self, target_week: Optional[datetime] = None) -> Report:
        """Generate comprehensive weekly analysis report."""
        try:
            if target_week is None:
                target_week = datetime.now() - timedelta(days=7)
            
            # Define report period (full week ending on target_week)
            period_end = target_week.replace(hour=23, minute=59, second=59, microsecond=999999)
            period_start = period_end - timedelta(days=6)
            period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            logger.info(f"Generating weekly report for {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            
            # Collect extended data
            fleet_data = self._collect_fleet_data(period_start, period_end)
            
            # Generate sections with trend analysis
            sections = []
            
            # Weekly Summary Section
            sections.append(self._generate_weekly_summary_section(fleet_data, period_start, period_end))
            
            # Trend Analysis Section
            sections.append(self._generate_trend_analysis_section(fleet_data, period_start, period_end))
            
            # Comparative Performance Section
            sections.append(self._generate_comparative_performance_section(fleet_data))
            
            # Efficiency Analysis Section
            sections.append(self._generate_efficiency_analysis_section(fleet_data))
            
            # Predictive Analysis Section
            sections.append(self._generate_predictive_analysis_section(fleet_data))
            
            # Weekly Recommendations Section
            sections.append(self._generate_weekly_recommendations_section(fleet_data))
            
            # Generate executive summary
            executive_summary = self._generate_weekly_executive_summary(fleet_data, sections)
            
            report = Report(
                title=f"Weekly Mining Analytics Report - Week of {period_start.strftime('%Y-%m-%d')}",
                report_type="weekly",
                generated_at=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                executive_summary=executive_summary,
                sections=sections,
                metadata={
                    'total_miners': len(fleet_data.get('miners', {})),
                    'analysis_depth': 'comprehensive',
                    'trend_analysis': True,
                    'predictive_analysis': True,
                    'report_version': '2.0'
                }
            )
            
            self._save_report(report)
            return report
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            raise
    
    def generate_performance_report(self, ip: str, days: int = 7) -> Report:
        """Generate detailed performance report for a specific miner."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            logger.info(f"Generating performance report for {ip} ({days} days)")
            
            # Get detailed performance profile
            profile = self.statistical_service.analyze_performance_profile(ip, start_time, end_time)
            
            if not profile:
                raise ValueError(f"Unable to generate performance profile for {ip}")
            
            # Get predictive analysis
            failure_prediction = self.predictive_service.predict_next_failure(ip)
            
            # Generate sections
            sections = []
            
            # Performance Overview
            sections.append(ReportSection(
                title="Performance Overview",
                content={
                    'overall_grade': profile.performance_grade,
                    'stability_score': profile.stability_score,
                    'consistency_score': profile.consistency_score,
                    'uptime_percentage': profile.uptime_percentage,
                    'data_completeness': profile.data_completeness
                },
                section_type='metrics',
                priority='high'
            ))
            
            # Detailed Statistics
            sections.append(ReportSection(
                title="Detailed Performance Statistics",
                content=profile.to_dict(),
                section_type='table',
                priority='medium'
            ))
            
            # Predictive Analysis
            if 'error' not in failure_prediction:
                sections.append(ReportSection(
                    title="Predictive Analysis",
                    content=failure_prediction,
                    section_type='text',
                    priority='high'
                ))
            
            # Generate executive summary
            executive_summary = self._generate_miner_executive_summary(profile, failure_prediction)
            
            report = Report(
                title=f"Miner Performance Report - {ip}",
                report_type="performance",
                generated_at=datetime.now(),
                period_start=start_time,
                period_end=end_time,
                executive_summary=executive_summary,
                sections=sections,
                metadata={
                    'miner_ip': ip,
                    'analysis_period_days': days,
                    'performance_grade': profile.performance_grade,
                    'report_version': '2.0'
                }
            )
            
            self._save_report(report)
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report for {ip}: {e}")
            raise
    
    def _collect_fleet_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Collect comprehensive fleet data for reporting."""
        try:
            fleet_data = {
                'miners': {},
                'fleet_metrics': {},
                'period_info': {
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': (end_time - start_time).total_seconds() / 3600
                }
            }
            
            # Get data for each configured miner
            for ip in self.config_service.ips:
                try:
                    # Get historical data
                    historical_data = self.database_manager.get_history_data(start_time, end_time, ip)
                    
                    if historical_data and ip in historical_data:
                        data_points = historical_data[ip]
                        
                        # Generate performance profile
                        profile = self.statistical_service.analyze_performance_profile(ip, start_time, end_time)
                        
                        fleet_data['miners'][ip] = {
                            'data_points': data_points,
                            'profile': profile.to_dict() if profile else None,
                            'data_count': len(data_points)
                        }
                    else:
                        fleet_data['miners'][ip] = {
                            'data_points': [],
                            'profile': None,
                            'data_count': 0
                        }
                        
                except Exception as e:
                    logger.warning(f"Error collecting data for miner {ip}: {e}")
                    fleet_data['miners'][ip] = {
                        'data_points': [],
                        'profile': None,
                        'data_count': 0,
                        'error': str(e)
                    }
            
            # Calculate fleet metrics
            fleet_data['fleet_metrics'] = self._calculate_fleet_metrics(fleet_data['miners'])
            
            return fleet_data
            
        except Exception as e:
            logger.error(f"Error collecting fleet data: {e}")
            return {'miners': {}, 'fleet_metrics': {}, 'error': str(e)}
    
    def _calculate_fleet_metrics(self, miners_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fleet-wide metrics."""
        try:
            metrics = {
                'total_miners': len(miners_data),
                'active_miners': 0,
                'total_hashrate': 0,
                'total_power': 0,
                'avg_efficiency': 0,
                'avg_temperature': 0,
                'avg_uptime': 0,
                'total_data_points': 0,
                'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
            }
            
            valid_profiles = []
            all_temps = []
            all_efficiencies = []
            all_uptimes = []
            
            for ip, miner_data in miners_data.items():
                metrics['total_data_points'] += miner_data.get('data_count', 0)
                
                profile_data = miner_data.get('profile')
                if profile_data:
                    valid_profiles.append(profile_data)
                    
                    # Accumulate metrics
                    metrics['total_hashrate'] += profile_data.get('avg_hashrate', 0)
                    metrics['total_power'] += profile_data.get('avg_power', 0)
                    
                    all_temps.append(profile_data.get('avg_temp', 0))
                    all_efficiencies.append(profile_data.get('avg_efficiency', 0))
                    all_uptimes.append(profile_data.get('uptime_percentage', 0))
                    
                    # Count grades
                    grade = profile_data.get('performance', {}).get('grade', 'F')
                    if grade in metrics['grade_distribution']:
                        metrics['grade_distribution'][grade] += 1
                    
                    if profile_data.get('avg_hashrate', 0) > 0:
                        metrics['active_miners'] += 1
            
            # Calculate averages
            if all_temps:
                metrics['avg_temperature'] = statistics.mean(all_temps)
            
            if all_efficiencies:
                metrics['avg_efficiency'] = statistics.mean(all_efficiencies)
            
            if all_uptimes:
                metrics['avg_uptime'] = statistics.mean(all_uptimes)
            
            # Calculate fleet efficiency
            if metrics['total_power'] > 0:
                metrics['fleet_efficiency'] = metrics['total_hashrate'] / metrics['total_power']
            else:
                metrics['fleet_efficiency'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating fleet metrics: {e}")
            return {}
    
    def _generate_fleet_overview_section(self, fleet_data: Dict[str, Any], start_time: datetime, end_time: datetime) -> ReportSection:
        """Generate fleet overview section."""
        fleet_metrics = fleet_data.get('fleet_metrics', {})
        
        content = {
            'reporting_period': f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}",
            'fleet_status': {
                'total_miners': fleet_metrics.get('total_miners', 0),
                'active_miners': fleet_metrics.get('active_miners', 0),
                'offline_miners': fleet_metrics.get('total_miners', 0) - fleet_metrics.get('active_miners', 0)
            },
            'performance_summary': {
                'total_hashrate_ths': fleet_metrics.get('total_hashrate', 0) / 1000,
                'total_power_kw': fleet_metrics.get('total_power', 0) / 1000,
                'fleet_efficiency_thw': fleet_metrics.get('fleet_efficiency', 0),
                'avg_temperature_c': fleet_metrics.get('avg_temperature', 0),
                'avg_uptime_percent': fleet_metrics.get('avg_uptime', 0)
            },
            'grade_distribution': fleet_metrics.get('grade_distribution', {}),
            'data_quality': {
                'total_data_points': fleet_metrics.get('total_data_points', 0),
                'expected_data_points': len(fleet_data.get('miners', {})) * (end_time - start_time).total_seconds() / 30,
                'collection_rate_percent': 0
            }
        }
        
        # Calculate collection rate
        expected = content['data_quality']['expected_data_points']
        actual = content['data_quality']['total_data_points']
        if expected > 0:
            content['data_quality']['collection_rate_percent'] = (actual / expected) * 100
        
        return ReportSection(
            title="Fleet Overview",
            content=content,
            section_type='metrics',
            priority='high'
        )
    
    def _generate_performance_analysis_section(self, fleet_data: Dict[str, Any]) -> ReportSection:
        """Generate performance analysis section."""
        miners_data = fleet_data.get('miners', {})
        
        performance_data = []
        
        for ip, miner_data in miners_data.items():
            profile = miner_data.get('profile')
            if profile:
                performance_data.append({
                    'ip': ip,
                    'grade': profile.get('performance', {}).get('grade', 'F'),
                    'stability_score': profile.get('performance', {}).get('stability_score', 0),
                    'consistency_score': profile.get('performance', {}).get('consistency_score', 0),
                    'avg_hashrate': profile.get('hashrate_stats', {}).get('avg', 0),
                    'avg_efficiency': profile.get('efficiency_stats', {}).get('avg', 0),
                    'avg_temperature': profile.get('temperature_stats', {}).get('avg', 0),
                    'uptime_percentage': profile.get('operational', {}).get('uptime_percentage', 0)
                })
        
        # Sort by performance grade and efficiency
        grade_order = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
        performance_data.sort(key=lambda x: (grade_order.get(x['grade'], 0), x['avg_efficiency']), reverse=True)
        
        return ReportSection(
            title="Performance Analysis",
            content={
                'top_performers': performance_data[:5],
                'all_miners': performance_data,
                'performance_insights': self._generate_performance_insights(performance_data)
            },
            section_type='table',
            priority='high'
        )
    
    def _generate_performance_insights(self, performance_data: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from performance data."""
        insights = []
        
        if not performance_data:
            return ["No performance data available for analysis"]
        
        try:
            # Grade distribution insight
            grades = [p['grade'] for p in performance_data]
            grade_counts = {grade: grades.count(grade) for grade in set(grades)}
            
            if grade_counts.get('A', 0) > len(performance_data) * 0.5:
                insights.append("Excellent fleet performance with majority of miners achieving Grade A")
            elif grade_counts.get('F', 0) > len(performance_data) * 0.3:
                insights.append("Performance concerns detected with high number of failing miners")
            
            # Efficiency insights
            efficiencies = [p['avg_efficiency'] for p in performance_data if p['avg_efficiency'] > 0]
            if efficiencies:
                avg_eff = statistics.mean(efficiencies)
                if avg_eff > 12:
                    insights.append(f"Fleet efficiency is excellent at {avg_eff:.2f} GH/W")
                elif avg_eff < 8:
                    insights.append(f"Fleet efficiency needs improvement at {avg_eff:.2f} GH/W")
            
            # Temperature insights
            temps = [p['avg_temperature'] for p in performance_data if p['avg_temperature'] > 0]
            if temps:
                avg_temp = statistics.mean(temps)
                max_temp = max(temps)
                
                if max_temp >= self.config_service.temp_limit:
                    insights.append(f"Temperature concerns detected with max temperature at {max_temp:.1f}°C")
                elif avg_temp > self.config_service.temp_limit * 0.9:
                    insights.append(f"Fleet running warm with average temperature at {avg_temp:.1f}°C")
            
            # Uptime insights
            uptimes = [p['uptime_percentage'] for p in performance_data if p['uptime_percentage'] > 0]
            if uptimes:
                avg_uptime = statistics.mean(uptimes)
                if avg_uptime < 95:
                    insights.append(f"Uptime concerns with fleet average at {avg_uptime:.1f}%")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            return ["Error generating performance insights"]
    
    def _generate_individual_miner_sections(self, fleet_data: Dict[str, Any], start_time: datetime, end_time: datetime) -> List[ReportSection]:
        """Generate sections for individual miners with issues."""
        sections = []
        miners_data = fleet_data.get('miners', {})
        
        # Focus on miners with issues or top performers
        issue_miners = []
        top_performers = []
        
        for ip, miner_data in miners_data.items():
            profile = miner_data.get('profile')
            if profile:
                grade = profile.get('performance', {}).get('grade', 'F')
                stability = profile.get('performance', {}).get('stability_score', 0)
                
                if grade in ['D', 'F'] or stability < 60:
                    issue_miners.append((ip, profile))
                elif grade == 'A' and stability > 90:
                    top_performers.append((ip, profile))
        
        # Add sections for issue miners
        for ip, profile in issue_miners[:3]:  # Limit to top 3 issues
            sections.append(ReportSection(
                title=f"Miner Analysis: {ip} (Issues Detected)",
                content={
                    'miner_ip': ip,
                    'performance_grade': profile.get('performance', {}).get('grade', 'F'),
                    'key_issues': self._identify_miner_issues(profile),
                    'detailed_stats': profile,
                    'recommendations': self._generate_miner_recommendations(profile)
                },
                section_type='text',
                priority='high'
            ))
        
        # Add section for top performer if any
        if top_performers:
            ip, profile = top_performers[0]  # Best performer
            sections.append(ReportSection(
                title=f"Top Performer: {ip}",
                content={
                    'miner_ip': ip,
                    'performance_grade': profile.get('performance', {}).get('grade', 'A'),
                    'success_factors': self._identify_success_factors(profile),
                    'detailed_stats': profile
                },
                section_type='text',
                priority='medium'
            ))
        
        return sections
    
    def _identify_miner_issues(self, profile: Dict[str, Any]) -> List[str]:
        """Identify specific issues with a miner."""
        issues = []
        
        try:
            # Temperature issues
            avg_temp = profile.get('temperature_stats', {}).get('avg', 0)
            if avg_temp >= self.config_service.temp_limit:
                issues.append(f"Operating above temperature limit at {avg_temp:.1f}°C")
            
            # Efficiency issues
            avg_efficiency = profile.get('efficiency_stats', {}).get('avg', 0)
            if avg_efficiency < 8:  # Below 8 GH/W is poor
                issues.append(f"Low efficiency at {avg_efficiency:.2f} GH/W")
            
            # Stability issues
            stability_score = profile.get('performance', {}).get('stability_score', 0)
            if stability_score < 70:
                issues.append(f"Poor stability with score of {stability_score:.1f}")
            
            # Uptime issues
            uptime = profile.get('operational', {}).get('uptime_percentage', 0)
            if uptime < 90:
                issues.append(f"Low uptime at {uptime:.1f}%")
            
            # Anomaly issues
            anomaly_count = profile.get('events', {}).get('anomalies', 0)
            if anomaly_count > 10:
                issues.append(f"High anomaly count: {anomaly_count}")
            
            return issues if issues else ["No specific issues identified"]
            
        except Exception as e:
            logger.error(f"Error identifying miner issues: {e}")
            return ["Error analyzing miner issues"]
    
    def _identify_success_factors(self, profile: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to good performance."""
        factors = []
        
        try:
            # Temperature management
            avg_temp = profile.get('temperature_stats', {}).get('avg', 0)
            if avg_temp < self.config_service.temp_limit * 0.8:
                factors.append(f"Excellent temperature management at {avg_temp:.1f}°C")
            
            # High efficiency
            avg_efficiency = profile.get('efficiency_stats', {}).get('avg', 0)
            if avg_efficiency > 12:
                factors.append(f"Outstanding efficiency at {avg_efficiency:.2f} GH/W")
            
            # High stability
            stability_score = profile.get('performance', {}).get('stability_score', 0)
            if stability_score > 90:
                factors.append(f"Excellent stability with score of {stability_score:.1f}")
            
            # High uptime
            uptime = profile.get('operational', {}).get('uptime_percentage', 0)
            if uptime > 98:
                factors.append(f"Outstanding uptime at {uptime:.1f}%")
            
            return factors if factors else ["Consistent overall performance"]
            
        except Exception as e:
            logger.error(f"Error identifying success factors: {e}")
            return ["Error analyzing success factors"]
    
    def _generate_miner_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for a miner."""
        recommendations = []
        
        try:
            # Temperature recommendations
            avg_temp = profile.get('temperature_stats', {}).get('avg', 0)
            if avg_temp >= self.config_service.temp_limit:
                recommendations.append("Immediate: Improve cooling or reduce operating frequency")
            elif avg_temp >= self.config_service.temp_limit * 0.9:
                recommendations.append("Check ventilation and consider frequency adjustment")
            
            # Efficiency recommendations
            avg_efficiency = profile.get('efficiency_stats', {}).get('avg', 0)
            if avg_efficiency < 10:
                recommendations.append("Optimize frequency/voltage settings for better efficiency")
            
            # Stability recommendations
            stability_score = profile.get('performance', {}).get('stability_score', 0)
            if stability_score < 70:
                recommendations.append("Check power supply stability and network connectivity")
            
            # Uptime recommendations
            uptime = profile.get('operational', {}).get('uptime_percentage', 0)
            if uptime < 95:
                recommendations.append("Investigate connectivity issues and power stability")
            
            return recommendations if recommendations else ["Continue current operation"]
            
        except Exception as e:
            logger.error(f"Error generating miner recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _generate_alerts_section(self, start_time: datetime, end_time: datetime) -> ReportSection:
        """Generate alerts and issues section."""
        try:
            # Get alerts from the period
            alerts = self.database_manager.get_event_log(limit=100)
            
            # Filter alerts for the period
            period_alerts = []
            for alert in alerts:
                alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
                if start_time <= alert_time <= end_time:
                    period_alerts.append(alert)
            
            # Categorize alerts
            critical_alerts = [a for a in period_alerts if a.get('severity') == 'critical']
            warning_alerts = [a for a in period_alerts if a.get('severity') == 'warning']
            
            content = {
                'summary': {
                    'total_alerts': len(period_alerts),
                    'critical_alerts': len(critical_alerts),
                    'warning_alerts': len(warning_alerts)
                },
                'critical_alerts': critical_alerts[:10],  # Top 10 critical
                'warning_alerts': warning_alerts[:10],   # Top 10 warnings
                'alert_trends': self._analyze_alert_trends(period_alerts)
            }
            
            return ReportSection(
                title="Alerts and Issues",
                content=content,
                section_type='table',
                priority='high' if critical_alerts else 'medium'
            )
            
        except Exception as e:
            logger.error(f"Error generating alerts section: {e}")
            return ReportSection(
                title="Alerts and Issues",
                content={'error': str(e)},
                section_type='text',
                priority='medium'
            )
    
    def _analyze_alert_trends(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in alerts."""
        try:
            if not alerts:
                return {'trend': 'No alerts to analyze'}
            
            # Group by type
            alert_types = {}
            for alert in alerts:
                alert_type = alert.get('event_type', 'unknown')
                if alert_type not in alert_types:
                    alert_types[alert_type] = 0
                alert_types[alert_type] += 1
            
            # Most common alert type
            most_common = max(alert_types.items(), key=lambda x: x[1])
            
            return {
                'most_common_type': most_common[0],
                'most_common_count': most_common[1],
                'alert_type_distribution': alert_types,
                'total_unique_types': len(alert_types)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing alert trends: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations_section(self, fleet_data: Dict[str, Any]) -> ReportSection:
        """Generate fleet-wide recommendations."""
        fleet_metrics = fleet_data.get('fleet_metrics', {})
        miners_data = fleet_data.get('miners', {})
        
        recommendations = []
        
        try:
            # Fleet efficiency recommendations
            fleet_efficiency = fleet_metrics.get('fleet_efficiency', 0)
            if fleet_efficiency < 10:
                recommendations.append({
                    'priority': 'high',
                    'category': 'efficiency',
                    'recommendation': 'Fleet efficiency is below optimal. Review frequency/voltage settings.',
                    'impact': 'Significant power cost reduction potential'
                })
            
            # Temperature management recommendations
            avg_temperature = fleet_metrics.get('avg_temperature', 0)
            if avg_temperature > self.config_service.temp_limit * 0.9:
                recommendations.append({
                    'priority': 'high',
                    'category': 'cooling',
                    'recommendation': 'Fleet running warm. Improve ventilation or reduce operating temperatures.',
                    'impact': 'Prevent thermal throttling and extend hardware life'
                })
            
            # Uptime recommendations
            avg_uptime = fleet_metrics.get('avg_uptime', 0)
            if avg_uptime < 95:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'reliability',
                    'recommendation': 'Fleet uptime below 95%. Investigate connectivity and power stability.',
                    'impact': 'Increase mining revenue through better availability'
                })
            
            # Performance grade recommendations
            grade_dist = fleet_metrics.get('grade_distribution', {})
            failing_miners = grade_dist.get('F', 0) + grade_dist.get('D', 0)
            total_miners = fleet_metrics.get('total_miners', 1)
            
            if failing_miners / total_miners > 0.2:  # More than 20% failing
                recommendations.append({
                    'priority': 'high',
                    'category': 'maintenance',
                    'recommendation': f'{failing_miners} miners need attention. Schedule maintenance review.',
                    'impact': 'Restore fleet performance and prevent failures'
                })
            
            return ReportSection(
                title="Fleet Recommendations",
                content={
                    'priority_actions': [r for r in recommendations if r['priority'] == 'high'],
                    'improvement_opportunities': [r for r in recommendations if r['priority'] == 'medium'],
                    'all_recommendations': recommendations
                },
                section_type='text',
                priority='high'
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ReportSection(
                title="Fleet Recommendations",
                content={'error': str(e)},
                section_type='text',
                priority='medium'
            )
    
    def _generate_executive_summary(self, fleet_data: Dict[str, Any], sections: List[ReportSection]) -> str:
        """Generate executive summary for daily report."""
        try:
            fleet_metrics = fleet_data.get('fleet_metrics', {})
            
            # Key metrics
            total_miners = fleet_metrics.get('total_miners', 0)
            active_miners = fleet_metrics.get('active_miners', 0)
            total_hashrate = fleet_metrics.get('total_hashrate', 0) / 1000  # Convert to TH/s
            fleet_efficiency = fleet_metrics.get('fleet_efficiency', 0)
            avg_uptime = fleet_metrics.get('avg_uptime', 0)
            
            # Grade distribution
            grade_dist = fleet_metrics.get('grade_distribution', {})
            grade_a_count = grade_dist.get('A', 0)
            
            # Build summary
            summary_parts = []
            
            # Fleet status
            if active_miners == total_miners:
                summary_parts.append(f"All {total_miners} miners are operational")
            else:
                offline = total_miners - active_miners
                summary_parts.append(f"{active_miners} of {total_miners} miners active ({offline} offline)")
            
            # Performance summary
            summary_parts.append(f"Fleet producing {total_hashrate:.1f} TH/s at {fleet_efficiency:.2f} TH/W efficiency")
            
            # Quality assessment
            if grade_a_count > total_miners * 0.7:
                summary_parts.append("Excellent fleet performance with majority achieving Grade A")
            elif grade_a_count > total_miners * 0.4:
                summary_parts.append("Good fleet performance with room for optimization")
            else:
                summary_parts.append("Fleet performance needs attention with multiple miners underperforming")
            
            # Uptime comment
            if avg_uptime > 98:
                summary_parts.append(f"Outstanding uptime at {avg_uptime:.1f}%")
            elif avg_uptime > 95:
                summary_parts.append(f"Good uptime at {avg_uptime:.1f}%")
            else:
                summary_parts.append(f"Uptime concerns at {avg_uptime:.1f}%")
            
            # Priority issues
            high_priority_sections = [s for s in sections if s.priority == 'high']
            if len(high_priority_sections) > 2:
                summary_parts.append("Multiple high-priority issues require immediate attention")
            elif high_priority_sections:
                summary_parts.append("Some issues require attention")
            else:
                summary_parts.append("No critical issues detected")
            
            return ". ".join(summary_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Error generating executive summary."
    
    def _save_report(self, report: Report) -> str:
        """Save report to file system."""
        try:
            # Create filename
            timestamp = report.generated_at.strftime('%Y%m%d_%H%M%S')
            filename = f"{report.report_type}_report_{timestamp}.json"
            filepath = self.report_directory / filename
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return ""
    
    def get_available_reports(self) -> List[Dict[str, Any]]:
        """Get list of available reports."""
        try:
            reports = []
            
            for report_file in self.report_directory.glob("*.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    reports.append({
                        'filename': report_file.name,
                        'title': report_data.get('title', 'Unknown'),
                        'type': report_data.get('report_type', 'unknown'),
                        'generated_at': report_data.get('generated_at', ''),
                        'period_start': report_data.get('period_start', ''),
                        'period_end': report_data.get('period_end', ''),
                        'file_size': report_file.stat().st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Error reading report file {report_file}: {e}")
            
            # Sort by generation time (newest first)
            reports.sort(key=lambda x: x['generated_at'], reverse=True)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error getting available reports: {e}")
            return []
    
    # Additional methods for weekly reports...
    def _generate_weekly_summary_section(self, fleet_data, start_time, end_time):
        """Generate weekly summary section - placeholder."""
        return ReportSection(
            title="Weekly Summary",
            content="Weekly summary implementation needed",
            section_type='text',
            priority='high'
        )
    
    def _generate_trend_analysis_section(self, fleet_data, start_time, end_time):
        """Generate trend analysis section - placeholder."""
        return ReportSection(
            title="Trend Analysis",
            content="Trend analysis implementation needed", 
            section_type='text',
            priority='medium'
        )
    
    def _generate_comparative_performance_section(self, fleet_data):
        """Generate comparative performance section - placeholder."""
        return ReportSection(
            title="Comparative Performance",
            content="Comparative analysis implementation needed",
            section_type='table',
            priority='medium'
        )
    
    def _generate_efficiency_analysis_section(self, fleet_data):
        """Generate efficiency analysis section - placeholder."""
        return ReportSection(
            title="Efficiency Analysis", 
            content="Efficiency analysis implementation needed",
            section_type='metrics',
            priority='medium'
        )
    
    def _generate_predictive_analysis_section(self, fleet_data):
        """Generate predictive analysis section - placeholder."""
        return ReportSection(
            title="Predictive Analysis",
            content="Predictive analysis implementation needed",
            section_type='text',
            priority='low'
        )
    
    def _generate_weekly_recommendations_section(self, fleet_data):
        """Generate weekly recommendations section - placeholder.""" 
        return ReportSection(
            title="Weekly Recommendations",
            content="Weekly recommendations implementation needed",
            section_type='text',
            priority='medium'
        )
    
    def _generate_weekly_executive_summary(self, fleet_data, sections):
        """Generate weekly executive summary - placeholder."""
        return "Weekly executive summary implementation needed."
    
    def _generate_miner_executive_summary(self, profile, failure_prediction):
        """Generate miner-specific executive summary - placeholder."""
        return "Miner executive summary implementation needed."