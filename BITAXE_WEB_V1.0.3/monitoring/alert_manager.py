"""
Advanced Alert Management System

Comprehensive alerting system with rule-based alerts, escalation policies,
notification channels, and smart alert correlation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
import requests
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from collections import defaultdict, deque
import hashlib

from bitaxe_logging.structured_logger import get_logger

logger = get_logger("bitaxe.alert_manager")


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.FIRING
    timestamp: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'message': self.message,
            'severity': self.severity.value,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'tags': self.tags,
            'metadata': self.metadata,
            'source': self.source,
            'details': self.details
        }


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    message: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    tags: Dict[str, str] = field(default_factory=dict)
    escalation_delay: int = 1800  # 30 minutes before escalation
    auto_resolve: bool = True
    resolve_condition: Optional[str] = None
    
    # Thresholds for common patterns
    threshold_value: Optional[float] = None
    threshold_operator: str = ">"  # >, <, >=, <=, ==, !=
    duration_seconds: int = 60  # How long condition must be true
    
    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert rule condition"""
        try:
            # Create safe evaluation context
            context = {
                'metrics': metrics,
                'threshold': self.threshold_value,
                **metrics  # Allow direct access to metric values
            }
            
            # Evaluate the condition
            if self.condition:
                return eval(self.condition, {"__builtins__": {}}, context)
            elif self.threshold_value is not None:
                # Simple threshold evaluation
                metric_value = metrics.get(self.name.lower().replace(' ', '_'), 0)
                if self.threshold_operator == '>':
                    return metric_value > self.threshold_value
                elif self.threshold_operator == '<':
                    return metric_value < self.threshold_value
                elif self.threshold_operator == '>=':
                    return metric_value >= self.threshold_value
                elif self.threshold_operator == '<=':
                    return metric_value <= self.threshold_value
                elif self.threshold_operator == '==':
                    return metric_value == self.threshold_value
                elif self.threshold_operator == '!=':
                    return metric_value != self.threshold_value
            
            return False
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}", error=str(e))
            return False


class NotificationChannel:
    """Base notification channel"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert"""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send email notification"""
        if not self.enabled:
            return False
        
        try:
            smtp_host = self.config.get('smtp_host', 'localhost')
            smtp_port = self.config.get('smtp_port', 587)
            username = self.config.get('username')
            password = self.config.get('password')
            from_email = self.config.get('from_email')
            to_emails = self.config.get('to_emails', [])
            
            if not to_emails:
                logger.warning("No email recipients configured")
                return False
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            
            # Email body
            body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Tags: {json.dumps(alert.tags, indent=2)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification for alert {alert.id}", error=str(e))
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send Slack notification"""
        if not self.enabled:
            return False
        
        try:
            webhook_url = self.config.get('webhook_url')
            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False
            
            # Color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.HIGH: "#FF8C00",
                AlertSeverity.MEDIUM: "#FFD700",
                AlertSeverity.LOW: "#32CD32",
                AlertSeverity.INFO: "#1E90FF"
            }
            
            # Create Slack message
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"{alert.name}",
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "BitAxe Alert Manager",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification for alert {alert.id}", error=str(e))
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel"""
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send webhook notification"""
        if not self.enabled:
            return False
        
        try:
            url = self.config.get('url')
            if not url:
                logger.warning("Webhook URL not configured")
                return False
            
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            method = self.config.get('method', 'POST').upper()
            
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'source': 'bitaxe-alert-manager'
            }
            
            if method == 'POST':
                response = requests.post(url, json=payload, headers=headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, json=payload, headers=headers, timeout=10)
            else:
                logger.warning(f"Unsupported webhook method: {method}")
                return False
            
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification for alert {alert.id}", error=str(e))
            return False


class AlertCorrelator:
    """Correlate related alerts to reduce noise"""
    
    def __init__(self, correlation_window: int = 300):
        self.correlation_window = correlation_window  # seconds
        self.alert_groups: Dict[str, List[Alert]] = defaultdict(list)
        self.correlation_rules: List[Dict[str, Any]] = []
    
    def add_correlation_rule(self, rule: Dict[str, Any]):
        """Add a correlation rule"""
        self.correlation_rules.append(rule)
    
    def correlate_alert(self, alert: Alert) -> Optional[str]:
        """Correlate an alert with existing groups"""
        current_time = datetime.now()
        
        # Find matching correlation group
        for rule in self.correlation_rules:
            if self._matches_rule(alert, rule):
                group_key = self._generate_group_key(alert, rule)
                
                # Clean up old alerts from group
                self.alert_groups[group_key] = [
                    a for a in self.alert_groups[group_key]
                    if (current_time - a.timestamp).total_seconds() < self.correlation_window
                ]
                
                # Add to group
                self.alert_groups[group_key].append(alert)
                
                # Check if this should be suppressed
                if len(self.alert_groups[group_key]) > 1:
                    return group_key
        
        return None
    
    def _matches_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches correlation rule"""
        # Match by name pattern
        if 'name_pattern' in rule:
            if rule['name_pattern'] not in alert.name:
                return False
        
        # Match by tags
        if 'tags' in rule:
            for tag_key, tag_value in rule['tags'].items():
                if alert.tags.get(tag_key) != tag_value:
                    return False
        
        # Match by source
        if 'source' in rule:
            if alert.source != rule['source']:
                return False
        
        return True
    
    def _generate_group_key(self, alert: Alert, rule: Dict[str, Any]) -> str:
        """Generate group key for correlation"""
        key_parts = []
        
        if 'group_by' in rule:
            for field in rule['group_by']:
                if field == 'name':
                    key_parts.append(alert.name)
                elif field == 'source':
                    key_parts.append(alert.source)
                elif field.startswith('tag:'):
                    tag_name = field[4:]
                    key_parts.append(alert.tags.get(tag_name, 'unknown'))
        else:
            key_parts.append(alert.name)
        
        return ':'.join(key_parts)


class AlertManager:
    """
    Comprehensive Alert Management System
    
    Features:
    - Rule-based alerting with flexible conditions
    - Multiple notification channels (email, Slack, webhooks)
    - Alert correlation and deduplication
    - Escalation policies
    - Silence and acknowledgment management
    - Historical alert tracking
    - Dashboard integration
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Alert rules
        self.alert_rules: Dict[str, AlertRule] = {}
        self.rule_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Notification channels
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self._setup_notification_channels()
        
        # Alert correlation
        self.correlator = AlertCorrelator()
        self._setup_correlation_rules()
        
        # Silencing
        self.silenced_rules: Dict[str, datetime] = {}
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Metrics callback
        self.metrics_callback: Optional[Callable[[], Dict[str, Any]]] = None
        
        # Rate limiting
        self.notification_rate_limit: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info("Alert Manager initialized")
    
    def _setup_notification_channels(self):
        """Setup notification channels from config"""
        channels_config = self.config.get('notification_channels', {})
        
        for channel_name, channel_config in channels_config.items():
            channel_type = channel_config.get('type', '').lower()
            
            if channel_type == 'email':
                self.notification_channels[channel_name] = EmailNotificationChannel(
                    channel_name, channel_config
                )
            elif channel_type == 'slack':
                self.notification_channels[channel_name] = SlackNotificationChannel(
                    channel_name, channel_config
                )
            elif channel_type == 'webhook':
                self.notification_channels[channel_name] = WebhookNotificationChannel(
                    channel_name, channel_config
                )
            else:
                logger.warning(f"Unknown notification channel type: {channel_type}")
    
    def _setup_correlation_rules(self):
        """Setup alert correlation rules"""
        correlation_rules = self.config.get('correlation_rules', [])
        for rule in correlation_rules:
            self.correlator.add_correlation_rule(rule)
    
    def set_metrics_callback(self, callback: Callable[[], Dict[str, Any]]):
        """Set callback function to get current metrics"""
        self.metrics_callback = callback
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def create_manual_alert(self, name: str, message: str, severity: AlertSeverity,
                           source: str = "manual", tags: Dict[str, str] = None,
                           details: Dict[str, Any] = None) -> Alert:
        """Create a manual alert"""
        alert_id = self._generate_alert_id(name, tags or {})
        
        alert = Alert(
            id=alert_id,
            name=name,
            message=message,
            severity=severity,
            source=source,
            tags=tags or {},
            details=details or {}
        )
        
        self._process_new_alert(alert)
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            # Move to resolved alerts
            self.resolved_alerts[alert_id] = alert
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def silence_rule(self, rule_name: str, duration_minutes: int = 60):
        """Silence an alert rule for a specified duration"""
        silence_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.silenced_rules[rule_name] = silence_until
        
        logger.info(f"Alert rule silenced: {rule_name} until {silence_until}")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.active_alerts)
        severity_counts = defaultdict(int)
        
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            'total_active_alerts': total_alerts,
            'severity_breakdown': dict(severity_counts),
            'total_resolved_alerts': len(self.resolved_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'silenced_rules': len(self.silenced_rules)
        }
    
    async def start(self):
        """Start the alert manager"""
        if self.is_running:
            return
        
        logger.info("Starting Alert Manager")
        self.is_running = True
        
        # Start background tasks
        self.evaluation_task = asyncio.create_task(self._evaluation_worker())
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("Alert Manager started")
    
    async def stop(self):
        """Stop the alert manager"""
        if not self.is_running:
            return
        
        logger.info("Stopping Alert Manager")
        self.is_running = False
        
        # Cancel background tasks
        if self.evaluation_task:
            self.evaluation_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to finish
        tasks = [t for t in [self.evaluation_task, self.cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Alert Manager stopped")
    
    def _generate_alert_id(self, name: str, tags: Dict[str, str]) -> str:
        """Generate unique alert ID"""
        tag_str = json.dumps(sorted(tags.items()), sort_keys=True)
        hash_input = f"{name}:{tag_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _process_new_alert(self, alert: Alert):
        """Process a new alert"""
        # Check correlation
        correlation_group = self.correlator.correlate_alert(alert)
        if correlation_group:
            logger.debug(f"Alert {alert.id} correlated to group {correlation_group}")
            # Could suppress notification or modify alert based on correlation
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        logger.info(f"New alert created: {alert.name} ({alert.severity.value})")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        # Rate limiting check
        current_time = time.time()
        alert_key = f"{alert.name}:{alert.severity.value}"
        
        # Check if we're within rate limit
        recent_notifications = self.notification_rate_limit[alert_key]
        recent_notifications.append(current_time)
        
        # Count notifications in last hour
        hour_ago = current_time - 3600
        recent_count = len([t for t in recent_notifications if t > hour_ago])
        
        # Rate limit: max 10 notifications per hour for same alert type
        if recent_count > 10:
            logger.warning(f"Rate limit exceeded for alert {alert.name}")
            return
        
        # Send to all configured channels
        for channel_name, channel in self.notification_channels.items():
            try:
                success = await channel.send_notification(alert)
                if success:
                    logger.debug(f"Notification sent via {channel_name} for alert {alert.id}")
                else:
                    logger.warning(f"Failed to send notification via {channel_name} for alert {alert.id}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name} for alert {alert.id}", error=str(e))
    
    async def _evaluation_worker(self):
        """Background worker to evaluate alert rules"""
        logger.debug("Alert evaluation worker started")
        
        while self.is_running:
            try:
                await self._evaluate_alert_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert evaluation worker error", error=str(e))
                await asyncio.sleep(60)
        
        logger.debug("Alert evaluation worker stopped")
    
    async def _evaluate_alert_rules(self):
        """Evaluate all alert rules"""
        if not self.metrics_callback:
            return
        
        try:
            # Get current metrics
            current_metrics = self.metrics_callback()
            current_time = datetime.now()
            
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule is silenced
                if rule_name in self.silenced_rules:
                    if current_time < self.silenced_rules[rule_name]:
                        continue
                    else:
                        # Remove expired silence
                        del self.silenced_rules[rule_name]
                
                # Evaluate rule
                try:
                    is_firing = rule.evaluate(current_metrics)
                    await self._handle_rule_evaluation(rule, is_firing, current_metrics)
                    
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_name}", error=str(e))
                    
        except Exception as e:
            logger.error("Error in rule evaluation", error=str(e))
    
    async def _handle_rule_evaluation(self, rule: AlertRule, is_firing: bool, metrics: Dict[str, Any]):
        """Handle the result of rule evaluation"""
        rule_state = self.rule_states[rule.name]
        current_time = datetime.now()
        
        # Track firing duration
        if is_firing:
            if 'first_fire_time' not in rule_state:
                rule_state['first_fire_time'] = current_time
            
            firing_duration = (current_time - rule_state['first_fire_time']).total_seconds()
            
            # Check if we should fire the alert (duration threshold met)
            if firing_duration >= rule.duration_seconds:
                # Check cooldown
                last_fired = rule_state.get('last_fired')
                if last_fired:
                    time_since_last = (current_time - last_fired).total_seconds()
                    if time_since_last < rule.cooldown_seconds:
                        return  # Still in cooldown
                
                # Create alert
                alert_id = self._generate_alert_id(rule.name, rule.tags)
                
                # Check if alert already exists
                if alert_id not in self.active_alerts:
                    alert = Alert(
                        id=alert_id,
                        name=rule.name,
                        message=rule.message,
                        severity=rule.severity,
                        source="alert_rule",
                        tags=rule.tags,
                        details={
                            'rule_name': rule.name,
                            'firing_duration': firing_duration,
                            'threshold_value': rule.threshold_value,
                            'current_metrics': metrics
                        }
                    )
                    
                    self._process_new_alert(alert)
                    rule_state['last_fired'] = current_time
        
        else:
            # Rule is not firing
            rule_state.pop('first_fire_time', None)
            
            # Auto-resolve if enabled
            if rule.auto_resolve:
                alert_id = self._generate_alert_id(rule.name, rule.tags)
                if alert_id in self.active_alerts:
                    # Check resolve condition if specified
                    should_resolve = True
                    if rule.resolve_condition:
                        try:
                            context = {'metrics': metrics, **metrics}
                            should_resolve = eval(rule.resolve_condition, {"__builtins__": {}}, context)
                        except Exception as e:
                            logger.error(f"Error evaluating resolve condition for {rule.name}", error=str(e))
                    
                    if should_resolve:
                        self.resolve_alert(alert_id)
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        logger.debug("Alert cleanup worker started")
        
        while self.is_running:
            try:
                # Clean up old resolved alerts
                cutoff_time = datetime.now() - timedelta(days=7)
                
                old_alerts = [
                    alert_id for alert_id, alert in self.resolved_alerts.items()
                    if alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in old_alerts:
                    del self.resolved_alerts[alert_id]
                
                if old_alerts:
                    logger.debug(f"Cleaned up {len(old_alerts)} old resolved alerts")
                
                # Clean up expired silences
                current_time = datetime.now()
                expired_silences = [
                    rule_name for rule_name, until_time in self.silenced_rules.items()
                    if current_time >= until_time
                ]
                
                for rule_name in expired_silences:
                    del self.silenced_rules[rule_name]
                
                if expired_silences:
                    logger.debug(f"Cleaned up {len(expired_silences)} expired silences")
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert cleanup worker error", error=str(e))
                await asyncio.sleep(300)
        
        logger.debug("Alert cleanup worker stopped")