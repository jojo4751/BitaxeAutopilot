"""
Predictive Analytics Service

High-level service for coordinating all predictive analytics models.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from ml_engine.models.predictive_models import (
    TemperaturePredictor, 
    EfficiencyForecaster, 
    FailureDetector, 
    PerformanceOptimizer,
    create_predictive_models
)
from ml_engine.data.feature_engineering import FeatureEngineer

logger = get_logger("bitaxe.ml.predictive_analytics")


@dataclass
class PredictiveInsights:
    """Aggregated predictive insights"""
    miner_ip: str
    timestamp: datetime
    temperature_forecast: Optional[float] = None
    temperature_confidence: Optional[float] = None
    efficiency_forecast: Optional[float] = None
    efficiency_confidence: Optional[float] = None
    anomaly_score: Optional[float] = None
    failure_risk: str = "low"  # low, medium, high
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class PredictiveAnalytics:
    """
    Comprehensive predictive analytics service
    
    Coordinates temperature prediction, efficiency forecasting, failure detection,
    and performance optimization to provide actionable insights.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Predictive models
        self.temperature_predictor: Optional[TemperaturePredictor] = None
        self.efficiency_forecaster: Optional[EfficiencyForecaster] = None
        self.failure_detector: Optional[FailureDetector] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        
        # Feature engineering
        self.feature_engineer: Optional[FeatureEngineer] = None
        
        # Configuration
        self.prediction_interval = self.config.get('prediction_interval', 300)  # 5 minutes
        self.enable_temperature_prediction = self.config.get('enable_temperature_prediction', True)
        self.enable_efficiency_forecasting = self.config.get('enable_efficiency_forecasting', True)
        self.enable_failure_detection = self.config.get('enable_failure_detection', True)
        self.enable_performance_optimization = self.config.get('enable_performance_optimization', True)
        
        # Thresholds for insights
        self.insight_thresholds = self.config.get('insight_thresholds', {
            'high_temperature_threshold': 80.0,
            'low_efficiency_threshold': 80.0,
            'high_anomaly_threshold': 0.7,
            'critical_anomaly_threshold': 0.9
        })
        
        # Background tasks
        self.prediction_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Historical data for training
        self.historical_data: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Predictive analytics service initialized")
    
    async def start(self):
        """Start predictive analytics service"""
        if self.is_running:
            return
        
        logger.info("Starting predictive analytics service")
        self.is_running = True
        
        try:
            # Initialize models
            await self._initialize_models()
            
            # Start background prediction task
            self.prediction_task = asyncio.create_task(self._prediction_loop())
            
            self.metrics_collector.increment_counter('predictive_analytics_starts_total')
            logger.info("Predictive analytics service started")
            
        except Exception as e:
            logger.error("Failed to start predictive analytics service", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop predictive analytics service"""
        if not self.is_running:
            return
        
        logger.info("Stopping predictive analytics service")
        self.is_running = False
        
        # Cancel prediction task
        if self.prediction_task:
            self.prediction_task.cancel()
            try:
                await self.prediction_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Predictive analytics service stopped")
    
    async def _initialize_models(self):
        """Initialize all predictive models"""
        logger.info("Initializing predictive models")
        
        # Create models
        models = await create_predictive_models(self.config.get('models', {}))
        
        if self.enable_temperature_prediction:
            self.temperature_predictor = models.get('temperature_predictor')
        
        if self.enable_efficiency_forecasting:
            self.efficiency_forecaster = models.get('efficiency_forecaster')
        
        if self.enable_failure_detection:
            self.failure_detector = models.get('failure_detector')
        
        if self.enable_performance_optimization:
            self.performance_optimizer = models.get('performance_optimizer')
        
        # Initialize feature engineer
        feature_config = self.config.get('feature_engineering', {})
        from ml_engine.data.feature_engineering import create_feature_engineer
        self.feature_engineer = await create_feature_engineer(feature_config)
        
        logger.info("Predictive models initialized")
    
    async def generate_insights(self, 
                              miner_ip: str,
                              recent_data: List[Dict[str, Any]],
                              weather_data: Optional[Dict[str, Any]] = None) -> PredictiveInsights:
        """
        Generate comprehensive predictive insights for a miner
        
        Args:
            miner_ip: Miner IP address
            recent_data: Recent telemetry data
            weather_data: Current weather conditions
            
        Returns:
            Predictive insights
        """
        try:
            start_time = datetime.now()
            
            insights = PredictiveInsights(
                miner_ip=miner_ip,
                timestamp=start_time
            )
            
            # Temperature prediction
            if self.enable_temperature_prediction and self.temperature_predictor:
                temp_result = await self.temperature_predictor.predict(recent_data, weather_data)
                if temp_result:
                    insights.temperature_forecast = temp_result.predictions[0]
                    insights.temperature_confidence = temp_result.confidence[0] if temp_result.confidence is not None else 0.5
            
            # Efficiency forecasting
            if self.enable_efficiency_forecasting and self.efficiency_forecaster:
                eff_result = await self.efficiency_forecaster.predict(recent_data)
                if eff_result:
                    insights.efficiency_forecast = eff_result.predictions[0]
                    insights.efficiency_confidence = eff_result.confidence[0] if eff_result.confidence is not None else 0.5
            
            # Failure detection
            if self.enable_failure_detection and self.failure_detector:
                anomaly_result = await self.failure_detector.detect_anomalies(recent_data)
                if anomaly_result:
                    insights.anomaly_score = float(np.mean(anomaly_result.predictions))
            
            # Generate recommendations
            insights.recommendations = await self._generate_recommendations(insights, recent_data)
            
            # Determine failure risk
            insights.failure_risk = self._assess_failure_risk(insights)
            
            # Record metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_timer('predictive_insights_generation_time', processing_time)
            self.metrics_collector.increment_counter('predictive_insights_generated_total')
            
            if insights.temperature_forecast:
                self.metrics_collector.record_metric('predicted_temperature', insights.temperature_forecast,
                                                   tags={'miner_ip': miner_ip})
            
            if insights.efficiency_forecast:
                self.metrics_collector.record_metric('predicted_efficiency', insights.efficiency_forecast,
                                                   tags={'miner_ip': miner_ip})
            
            if insights.anomaly_score:
                self.metrics_collector.record_metric('anomaly_score', insights.anomaly_score,
                                                   tags={'miner_ip': miner_ip})
            
            logger.debug(f"Generated predictive insights for {miner_ip}",
                        temp_forecast=insights.temperature_forecast,
                        eff_forecast=insights.efficiency_forecast,
                        anomaly_score=insights.anomaly_score,
                        failure_risk=insights.failure_risk)
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights for {miner_ip}", error=str(e))
            return PredictiveInsights(
                miner_ip=miner_ip,
                timestamp=datetime.now(),
                recommendations=["Prediction failed - check manually"]
            )
    
    async def _generate_recommendations(self, 
                                      insights: PredictiveInsights,
                                      recent_data: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        try:
            current_data = recent_data[-1] if recent_data else {}
            current_temp = current_data.get('temp', 0)
            current_efficiency = current_data.get('hashRate', 0) / current_data.get('power', 1)
            
            # Temperature-based recommendations
            if insights.temperature_forecast:
                if insights.temperature_forecast > self.insight_thresholds['high_temperature_threshold']:
                    recommendations.append(
                        f"High temperature predicted ({insights.temperature_forecast:.1f}°C) - "
                        "consider reducing frequency or improving cooling"
                    )
                elif insights.temperature_forecast > current_temp + 5:
                    recommendations.append(
                        f"Temperature rising trend detected - monitor cooling performance"
                    )
            
            # Efficiency-based recommendations
            if insights.efficiency_forecast:
                if insights.efficiency_forecast < self.insight_thresholds['low_efficiency_threshold']:
                    recommendations.append(
                        f"Low efficiency predicted ({insights.efficiency_forecast:.1f} GH/W) - "
                        "optimize frequency/voltage settings"
                    )
                elif insights.efficiency_forecast < current_efficiency * 0.9:
                    recommendations.append(
                        "Efficiency degradation predicted - check for hardware issues"
                    )
            
            # Anomaly-based recommendations
            if insights.anomaly_score:
                if insights.anomaly_score > self.insight_thresholds['critical_anomaly_threshold']:
                    recommendations.append(
                        "Critical anomaly detected - immediate investigation required"
                    )
                elif insights.anomaly_score > self.insight_thresholds['high_anomaly_threshold']:
                    recommendations.append(
                        "Anomaly detected - monitor performance closely"
                    )
            
            # Proactive maintenance recommendations
            if len(recent_data) >= 10:
                temp_trend = np.polyfit(range(len(recent_data)), 
                                      [d.get('temp', 0) for d in recent_data], 1)[0]
                if temp_trend > 0.1:  # Temperature increasing over time
                    recommendations.append(
                        "Gradual temperature increase detected - schedule cooling system maintenance"
                    )
            
            # Performance optimization recommendations
            if self.enable_performance_optimization and self.performance_optimizer:
                try:
                    # This would use the genetic algorithm optimizer
                    # For now, add a placeholder recommendation
                    recommendations.append(
                        "Performance optimization available - run optimization algorithm"
                    )
                except Exception as e:
                    logger.debug(f"Performance optimization recommendation failed: {e}")
            
            return recommendations
            
        except Exception as e:
            logger.error("Recommendation generation failed", error=str(e))
            return ["Failed to generate recommendations"]
    
    def _assess_failure_risk(self, insights: PredictiveInsights) -> str:
        """Assess overall failure risk level"""
        try:
            risk_score = 0.0
            
            # Temperature risk
            if insights.temperature_forecast:
                if insights.temperature_forecast > 90:
                    risk_score += 0.4
                elif insights.temperature_forecast > 80:
                    risk_score += 0.2
            
            # Efficiency risk
            if insights.efficiency_forecast:
                if insights.efficiency_forecast < 50:
                    risk_score += 0.3
                elif insights.efficiency_forecast < 80:
                    risk_score += 0.1
            
            # Anomaly risk
            if insights.anomaly_score:
                if insights.anomaly_score > 0.9:
                    risk_score += 0.5
                elif insights.anomaly_score > 0.7:
                    risk_score += 0.3
                elif insights.anomaly_score > 0.5:
                    risk_score += 0.1
            
            # Determine risk level
            if risk_score >= 0.7:
                return "high"
            elif risk_score >= 0.3:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error("Failure risk assessment failed", error=str(e))
            return "unknown"
    
    async def train_models(self, historical_data: List[Dict[str, Any]]) -> bool:
        """Train all predictive models with historical data"""
        try:
            logger.info("Training predictive models with historical data")
            training_success = True
            
            # Train temperature predictor
            if self.temperature_predictor:
                temp_success = await self.temperature_predictor.train(historical_data)
                if temp_success:
                    logger.info("Temperature predictor trained successfully")
                else:
                    logger.warning("Temperature predictor training failed")
                    training_success = False
            
            # Train efficiency forecaster
            if self.efficiency_forecaster:
                eff_success = await self.efficiency_forecaster.train(historical_data)
                if eff_success:
                    logger.info("Efficiency forecaster trained successfully")
                else:
                    logger.warning("Efficiency forecaster training failed")
                    training_success = False
            
            # Train failure detector
            if self.failure_detector:
                fail_success = await self.failure_detector.train(historical_data)
                if fail_success:
                    logger.info("Failure detector trained successfully")
                else:
                    logger.warning("Failure detector training failed")
                    training_success = False
            
            # Record training metrics
            if training_success:
                self.metrics_collector.increment_counter('predictive_models_trained_total')
                logger.info("All predictive models trained successfully")
            else:
                self.metrics_collector.increment_counter('predictive_model_training_failures_total')
                logger.warning("Some predictive models failed to train")
            
            return training_success
            
        except Exception as e:
            logger.error("Predictive model training failed", error=str(e))
            self.metrics_collector.increment_counter('predictive_model_training_errors_total')
            return False
    
    async def _prediction_loop(self):
        """Background prediction loop"""
        logger.info("Prediction loop started")
        
        while self.is_running:
            try:
                # This would typically run predictions for all active miners
                # For now, just update metrics
                
                self.metrics_collector.set_gauge('predictive_analytics_active', 1)
                
                await asyncio.sleep(self.prediction_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Prediction loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("Prediction loop stopped")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get predictive analytics service status"""
        return {
            'is_running': self.is_running,
            'models_enabled': {
                'temperature_prediction': self.enable_temperature_prediction,
                'efficiency_forecasting': self.enable_efficiency_forecasting,
                'failure_detection': self.enable_failure_detection,
                'performance_optimization': self.enable_performance_optimization
            },
            'models_initialized': {
                'temperature_predictor': self.temperature_predictor is not None,
                'efficiency_forecaster': self.efficiency_forecaster is not None,
                'failure_detector': self.failure_detector is not None,
                'performance_optimizer': self.performance_optimizer is not None
            },
            'prediction_interval': self.prediction_interval,
            'insight_thresholds': self.insight_thresholds
        }


async def create_predictive_analytics(config: Dict[str, Any] = None) -> PredictiveAnalytics:
    """Factory function to create predictive analytics service"""
    service = PredictiveAnalytics(config)
    await service.start()
    return service