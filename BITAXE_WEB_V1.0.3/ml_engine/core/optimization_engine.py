"""
ML Optimization Engine

Core optimization engine that integrates all ML components for autonomous mining optimization.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import json

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector
from ml_engine.core.ml_model_service import MLModelService, create_ml_model_service
from ml_engine.core.weather_service import WeatherService, create_weather_service
from ml_engine.models.rl_agent import FrequencyVoltageOptimizer, create_frequency_voltage_optimizer
from ml_engine.models.predictive_models import create_predictive_models
from ml_engine.data.feature_engineering import FeatureEngineer, create_feature_engineer

logger = get_logger("bitaxe.ml.optimization_engine")


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    miner_ip: str
    original_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    predicted_improvements: Dict[str, float]
    confidence_score: float
    optimization_strategy: str
    applied_adjustments: Dict[str, Any]
    timestamp: datetime
    
    def get_config_changes(self) -> Dict[str, Any]:
        """Get the changes made to configuration"""
        changes = {}
        for key, new_value in self.optimized_config.items():
            original_value = self.original_config.get(key)
            if original_value != new_value:
                changes[key] = {
                    'from': original_value,
                    'to': new_value,
                    'change': new_value - original_value if isinstance(new_value, (int, float)) else None
                }
        return changes


@dataclass
class MinerState:
    """Current state of a miner for optimization"""
    ip: str
    telemetry: Dict[str, Any]
    current_config: Dict[str, Any]
    health_status: str
    last_optimization: Optional[datetime] = None
    optimization_history: List[OptimizationResult] = None
    
    def __post_init__(self):
        if self.optimization_history is None:
            self.optimization_history = []


class OptimizationEngine:
    """
    Core ML optimization engine for autonomous mining fleet management
    
    Features:
    - Multi-objective optimization (efficiency, temperature, stability)
    - Reinforcement learning-based parameter tuning
    - Predictive analytics for proactive optimization
    - Weather-aware environmental adaptation
    - Safety-constrained optimization
    - A/B testing and performance validation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Core ML services
        self.ml_model_service: Optional[MLModelService] = None
        self.weather_service: Optional[WeatherService] = None
        self.rl_optimizer: Optional[FrequencyVoltageOptimizer] = None
        self.predictive_models: Dict[str, Any] = {}
        self.feature_engineer: Optional[FeatureEngineer] = None
        
        # Optimization configuration
        self.optimization_interval = self.config.get('optimization_interval', 300)  # 5 minutes
        self.min_optimization_gap = self.config.get('min_optimization_gap', 900)  # 15 minutes between optimizations
        self.enable_predictive_optimization = self.config.get('enable_predictive', True)
        self.enable_weather_adaptation = self.config.get('enable_weather', True)
        self.enable_rl_optimization = self.config.get('enable_rl', True)
        
        # Safety constraints
        self.safety_config = self.config.get('safety', {})
        self.max_temperature = self.safety_config.get('max_temperature', 85.0)
        self.min_efficiency = self.safety_config.get('min_efficiency', 50.0)
        self.max_power = self.safety_config.get('max_power', 200.0)
        
        # Optimization targets
        self.target_efficiency = self.config.get('target_efficiency', 120.0)  # GH/W
        self.target_temperature = self.config.get('target_temperature', 70.0)  # °C
        
        # State tracking
        self.miner_states: Dict[str, MinerState] = {}
        self.optimization_active = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'efficiency_improvements': [],
            'temperature_reductions': [],
            'safety_violations': 0
        }
        
        logger.info("Optimization engine initialized",
                   optimization_interval=self.optimization_interval,
                   safety_constraints=self.safety_config)
    
    async def start(self):
        """Start the optimization engine"""
        if self.optimization_active:
            return
        
        logger.info("Starting ML optimization engine")
        
        try:
            # Initialize ML services
            await self._initialize_ml_services()
            
            # Start optimization loop
            self.optimization_active = True
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            self.metrics_collector.increment_counter('optimization_engine_starts_total')
            logger.info("ML optimization engine started successfully")
            
        except Exception as e:
            logger.error("Failed to start optimization engine", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the optimization engine"""
        if not self.optimization_active:
            return
        
        logger.info("Stopping ML optimization engine")
        self.optimization_active = False
        
        # Cancel optimization task
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
                pass
        
        # Stop ML services
        if self.ml_model_service:
            await self.ml_model_service.stop()
        if self.weather_service:
            await self.weather_service.stop()
        if self.rl_optimizer:
            await self.rl_optimizer.stop_optimization()
        
        logger.info("ML optimization engine stopped")
    
    async def _initialize_ml_services(self):
        """Initialize all ML services"""
        logger.info("Initializing ML services")
        
        # ML Model Service
        model_config = self.config.get('model_service', {})
        self.ml_model_service = await create_ml_model_service(
            model_dir=model_config.get('model_dir', 'models'),
            config=model_config
        )
        
        # Weather Service
        weather_config = self.config.get('weather_service', {})
        self.weather_service = await create_weather_service(weather_config)
        
        # RL Optimizer
        if self.enable_rl_optimization:
            rl_config = self.config.get('rl_optimizer', {})
            self.rl_optimizer = await create_frequency_voltage_optimizer(rl_config)
            await self.rl_optimizer.start_optimization()
        
        # Predictive Models
        if self.enable_predictive_optimization:
            predictive_config = self.config.get('predictive_models', {})
            self.predictive_models = await create_predictive_models(predictive_config)
        
        # Feature Engineer
        feature_config = self.config.get('feature_engineering', {})
        self.feature_engineer = await create_feature_engineer(feature_config)
        
        logger.info("ML services initialized successfully")
    
    async def optimize_miner(self, 
                           miner_ip: str, 
                           telemetry_data: Dict[str, Any],
                           current_config: Dict[str, Any]) -> Optional[OptimizationResult]:
        """
        Optimize a single miner using all available ML techniques
        
        Args:
            miner_ip: Miner IP address
            telemetry_data: Current miner telemetry
            current_config: Current miner configuration
            
        Returns:
            Optimization result or None if no optimization needed
        """
        try:
            start_time = datetime.now()
            
            # Update miner state
            await self._update_miner_state(miner_ip, telemetry_data, current_config)
            
            # Safety checks
            if not await self._safety_check(miner_ip, telemetry_data):
                logger.warning(f"Safety check failed for miner {miner_ip}")
                return None
            
            # Check if optimization is needed
            if not await self._should_optimize(miner_ip):
                return None
            
            # Gather optimization context
            context = await self._gather_optimization_context(miner_ip, telemetry_data)
            
            # Generate optimization strategies
            strategies = await self._generate_optimization_strategies(context)
            
            # Select best strategy
            best_strategy = await self._select_optimization_strategy(strategies, context)
            
            if not best_strategy:
                logger.debug(f"No optimization strategy selected for miner {miner_ip}")
                return None
            
            # Apply optimization
            optimized_config = await self._apply_optimization_strategy(
                best_strategy, current_config, context
            )
            
            # Validate optimization
            if not await self._validate_optimization(current_config, optimized_config, context):
                logger.warning(f"Optimization validation failed for miner {miner_ip}")
                return None
            
            # Calculate predicted improvements
            predicted_improvements = await self._predict_improvements(
                current_config, optimized_config, context
            )
            
            # Create optimization result
            result = OptimizationResult(
                miner_ip=miner_ip,
                original_config=current_config,
                optimized_config=optimized_config,
                predicted_improvements=predicted_improvements,
                confidence_score=best_strategy.get('confidence', 0.5),
                optimization_strategy=best_strategy.get('name', 'unknown'),
                applied_adjustments=best_strategy.get('adjustments', {}),
                timestamp=datetime.now()
            )
            
            # Update optimization history
            await self._record_optimization_result(miner_ip, result)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Record metrics
            self.metrics_collector.record_timer('miner_optimization_duration', optimization_time)
            self.metrics_collector.increment_counter('miner_optimizations_total')
            
            logger.info(f"Miner optimization completed for {miner_ip}",
                       strategy=best_strategy.get('name'),
                       confidence=best_strategy.get('confidence'),
                       optimization_time_ms=optimization_time * 1000)
            
            return result
            
        except Exception as e:
            logger.error(f"Miner optimization failed for {miner_ip}", error=str(e))
            self.metrics_collector.increment_counter('miner_optimization_errors_total')
            return None
    
    async def _update_miner_state(self, 
                                 miner_ip: str, 
                                 telemetry_data: Dict[str, Any],
                                 current_config: Dict[str, Any]):
        """Update internal miner state tracking"""
        if miner_ip not in self.miner_states:
            self.miner_states[miner_ip] = MinerState(
                ip=miner_ip,
                telemetry=telemetry_data,
                current_config=current_config,
                health_status='unknown'
            )
        else:
            state = self.miner_states[miner_ip]
            state.telemetry = telemetry_data
            state.current_config = current_config
    
    async def _safety_check(self, miner_ip: str, telemetry_data: Dict[str, Any]) -> bool:
        """Perform comprehensive safety checks"""
        try:
            # Temperature safety
            temp = telemetry_data.get('temp', 0)
            if temp > self.max_temperature:
                logger.warning(f"Temperature too high for {miner_ip}: {temp}°C")
                self.optimization_stats['safety_violations'] += 1
                return False
            
            # Power safety
            power = telemetry_data.get('power', 0)
            if power > self.max_power:
                logger.warning(f"Power too high for {miner_ip}: {power}W")
                return False
            
            # Efficiency safety
            hashrate = telemetry_data.get('hashRate', 0)
            efficiency = hashrate / power if power > 0 else 0
            if efficiency < self.min_efficiency:
                logger.warning(f"Efficiency too low for {miner_ip}: {efficiency} GH/W")
                return False
            
            # Basic data validity
            if hashrate <= 0 or power <= 0:
                logger.warning(f"Invalid telemetry data for {miner_ip}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed for {miner_ip}", error=str(e))
            return False
    
    async def _should_optimize(self, miner_ip: str) -> bool:
        """Determine if miner needs optimization"""
        try:
            state = self.miner_states.get(miner_ip)
            if not state:
                return True  # New miner, should optimize
            
            # Check time since last optimization
            if state.last_optimization:
                time_since_last = datetime.now() - state.last_optimization
                if time_since_last.total_seconds() < self.min_optimization_gap:
                    return False
            
            # Check if current performance is suboptimal
            telemetry = state.telemetry
            temp = telemetry.get('temp', 0)
            efficiency = telemetry.get('hashRate', 0) / telemetry.get('power', 1)
            
            # Optimize if temperature is high or efficiency is low
            if temp > self.target_temperature or efficiency < self.target_efficiency:
                return True
            
            # Use predictive models to determine if optimization would be beneficial
            if self.enable_predictive_optimization:
                should_optimize = await self._predictive_optimization_check(miner_ip)
                if should_optimize:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Should optimize check failed for {miner_ip}", error=str(e))
            return False
    
    async def _gather_optimization_context(self, 
                                         miner_ip: str, 
                                         telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive context for optimization"""
        context = {
            'miner_ip': miner_ip,
            'telemetry': telemetry_data,
            'timestamp': datetime.now()
        }
        
        # Weather context
        if self.enable_weather_adaptation and self.weather_service:
            try:
                weather_data = await self.weather_service.get_current_weather()
                if weather_data:
                    context['weather'] = {
                        'temperature': weather_data.temperature,
                        'humidity': weather_data.humidity,
                        'pressure': weather_data.pressure
                    }
                
                # Get cooling strategy
                cooling_strategy = await self.weather_service.get_cooling_strategy(
                    telemetry_data.get('temp')
                )
                context['cooling_strategy'] = cooling_strategy
                
            except Exception as e:
                logger.debug(f"Failed to get weather context: {e}")
        
        # Historical context
        state = self.miner_states.get(miner_ip)
        if state and state.optimization_history:
            context['optimization_history'] = state.optimization_history[-5:]  # Last 5 optimizations
        
        # Predictive context
        if self.enable_predictive_optimization:
            try:
                # Temperature prediction
                if 'temperature_predictor' in self.predictive_models:
                    temp_prediction = await self.predictive_models['temperature_predictor'].predict([telemetry_data])
                    if temp_prediction:
                        context['temperature_prediction'] = temp_prediction.predictions[0]
                
                # Efficiency forecast
                if 'efficiency_forecaster' in self.predictive_models:
                    efficiency_forecast = await self.predictive_models['efficiency_forecaster'].predict([telemetry_data])
                    if efficiency_forecast:
                        context['efficiency_forecast'] = efficiency_forecast.predictions[0]
                
                # Anomaly detection
                if 'failure_detector' in self.predictive_models:
                    anomaly_result = await self.predictive_models['failure_detector'].detect_anomalies([telemetry_data])
                    if anomaly_result:
                        context['anomaly_score'] = anomaly_result.predictions[0]
                
            except Exception as e:
                logger.debug(f"Failed to get predictive context: {e}")
        
        return context
    
    async def _generate_optimization_strategies(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate possible optimization strategies"""
        strategies = []
        
        telemetry = context['telemetry']
        current_temp = telemetry.get('temp', 0)
        current_power = telemetry.get('power', 0)
        current_efficiency = telemetry.get('hashRate', 0) / current_power if current_power > 0 else 0
        
        # Strategy 1: Temperature-focused optimization
        if current_temp > self.target_temperature:
            strategies.append({
                'name': 'temperature_reduction',
                'priority': 'high',
                'confidence': 0.8,
                'adjustments': {
                    'frequency_reduction': min(0.1, (current_temp - self.target_temperature) / 100),
                    'voltage_reduction': min(0.05, (current_temp - self.target_temperature) / 200)
                },
                'expected_temp_reduction': min(10, current_temp - self.target_temperature)
            })
        
        # Strategy 2: Efficiency-focused optimization
        if current_efficiency < self.target_efficiency:
            strategies.append({
                'name': 'efficiency_improvement',
                'priority': 'medium',
                'confidence': 0.7,
                'adjustments': {
                    'frequency_optimization': 0.05,
                    'voltage_optimization': 0.02
                },
                'expected_efficiency_gain': self.target_efficiency - current_efficiency
            })
        
        # Strategy 3: Weather-adaptive optimization
        if 'cooling_strategy' in context:
            cooling = context['cooling_strategy']
            if cooling.get('strategy') != 'optimal':
                strategies.append({
                    'name': 'weather_adaptive',
                    'priority': 'medium',
                    'confidence': cooling.get('recommendation_strength', 0.5),
                    'adjustments': cooling.get('adjustments', {}),
                    'weather_strategy': cooling.get('strategy')
                })
        
        # Strategy 4: Reinforcement Learning optimization
        if self.enable_rl_optimization and self.rl_optimizer:
            try:
                rl_config = await self.rl_optimizer.optimize_miner(telemetry, context.get('weather'))
                if rl_config:
                    strategies.append({
                        'name': 'reinforcement_learning',
                        'priority': 'high',
                        'confidence': 0.9,
                        'adjustments': rl_config,
                        'source': 'rl_agent'
                    })
            except Exception as e:
                logger.debug(f"RL optimization failed: {e}")
        
        # Strategy 5: Predictive optimization
        if 'temperature_prediction' in context:
            predicted_temp = context['temperature_prediction']
            if predicted_temp > self.max_temperature * 0.9:  # 90% of max temp
                strategies.append({
                    'name': 'predictive_cooling',
                    'priority': 'high',
                    'confidence': 0.8,
                    'adjustments': {
                        'frequency_reduction': 0.15,
                        'voltage_reduction': 0.08
                    },
                    'predicted_temp': predicted_temp
                })
        
        return strategies
    
    async def _select_optimization_strategy(self, 
                                          strategies: List[Dict[str, Any]], 
                                          context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select the best optimization strategy"""
        if not strategies:
            return None
        
        # Score strategies based on priority, confidence, and context
        scored_strategies = []
        
        for strategy in strategies:
            score = 0.0
            
            # Priority scoring
            priority = strategy.get('priority', 'low')
            if priority == 'high':
                score += 3.0
            elif priority == 'medium':
                score += 2.0
            else:
                score += 1.0
            
            # Confidence scoring
            confidence = strategy.get('confidence', 0.5)
            score += confidence * 2.0
            
            # Context-specific scoring
            telemetry = context['telemetry']
            current_temp = telemetry.get('temp', 0)
            
            # Boost temperature strategies if temperature is critical
            if current_temp > self.max_temperature * 0.9 and 'temperature' in strategy['name']:
                score += 2.0
            
            # Boost RL strategies if they have good historical performance
            if strategy['name'] == 'reinforcement_learning':
                score += 1.0  # RL generally performs well
            
            scored_strategies.append((score, strategy))
        
        # Sort by score and return best strategy
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        best_strategy = scored_strategies[0][1]
        
        logger.debug(f"Selected optimization strategy: {best_strategy['name']} (score: {scored_strategies[0][0]:.2f})")
        
        return best_strategy
    
    async def _apply_optimization_strategy(self, 
                                         strategy: Dict[str, Any], 
                                         current_config: Dict[str, Any],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the selected optimization strategy"""
        optimized_config = current_config.copy()
        adjustments = strategy.get('adjustments', {})
        
        # Apply frequency adjustments
        if 'frequency_reduction' in adjustments:
            current_freq = current_config.get('frequency', 600)
            reduction = adjustments['frequency_reduction']
            new_freq = current_freq * (1 - reduction)
            optimized_config['frequency'] = max(400, min(800, new_freq))
        
        if 'frequency_optimization' in adjustments:
            current_freq = current_config.get('frequency', 600)
            optimization = adjustments['frequency_optimization']
            new_freq = current_freq * (1 + optimization)
            optimized_config['frequency'] = max(400, min(800, new_freq))
        
        # Apply voltage adjustments
        if 'voltage_reduction' in adjustments:
            current_voltage = current_config.get('voltage', 12.0)
            reduction = adjustments['voltage_reduction']
            new_voltage = current_voltage - reduction
            optimized_config['voltage'] = max(10.0, min(14.0, new_voltage))
        
        if 'voltage_optimization' in adjustments:
            current_voltage = current_config.get('voltage', 12.0)
            optimization = adjustments['voltage_optimization']
            new_voltage = current_voltage + optimization
            optimized_config['voltage'] = max(10.0, min(14.0, new_voltage))
        
        # Apply RL-specific adjustments
        if strategy.get('source') == 'rl_agent':
            # RL adjustments are already in the correct format
            for key, value in adjustments.items():
                if key in ['frequency', 'voltage']:
                    optimized_config[key] = max(
                        10.0 if key == 'voltage' else 400,
                        min(14.0 if key == 'voltage' else 800, value)
                    )
        
        return optimized_config
    
    async def _validate_optimization(self, 
                                   current_config: Dict[str, Any],
                                   optimized_config: Dict[str, Any],
                                   context: Dict[str, Any]) -> bool:
        """Validate that optimization is safe and beneficial"""
        try:
            # Check that changes are within safe bounds
            freq_change = abs(optimized_config.get('frequency', 600) - current_config.get('frequency', 600))
            voltage_change = abs(optimized_config.get('voltage', 12.0) - current_config.get('voltage', 12.0))
            
            # Limit maximum single-step changes
            if freq_change > 100:  # MHz
                logger.warning("Frequency change too large, rejecting optimization")
                return False
            
            if voltage_change > 0.5:  # V
                logger.warning("Voltage change too large, rejecting optimization")
                return False
            
            # Check absolute bounds
            freq = optimized_config.get('frequency', 600)
            voltage = optimized_config.get('voltage', 12.0)
            
            if not (400 <= freq <= 800):
                logger.warning(f"Frequency out of bounds: {freq}")
                return False
            
            if not (10.0 <= voltage <= 14.0):
                logger.warning(f"Voltage out of bounds: {voltage}")
                return False
            
            return True
            
        except Exception as e:
            logger.error("Optimization validation failed", error=str(e))
            return False
    
    async def _predict_improvements(self, 
                                  current_config: Dict[str, Any],
                                  optimized_config: Dict[str, Any],
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """Predict the improvements from optimization"""
        improvements = {}
        
        try:
            # Simple prediction based on configuration changes
            current_freq = current_config.get('frequency', 600)
            new_freq = optimized_config.get('frequency', 600)
            freq_change_ratio = new_freq / current_freq
            
            current_voltage = current_config.get('voltage', 12.0)
            new_voltage = optimized_config.get('voltage', 12.0)
            voltage_change = new_voltage - current_voltage
            
            # Estimate efficiency change (simplified model)
            efficiency_change = (freq_change_ratio - 1) * 50  # Rough approximation
            improvements['efficiency_change_percent'] = efficiency_change
            
            # Estimate temperature change
            temp_change = voltage_change * 10 + (freq_change_ratio - 1) * 20
            improvements['temperature_change_celsius'] = temp_change
            
            # Estimate power change
            power_change = voltage_change * 5 + (freq_change_ratio - 1) * 30
            improvements['power_change_watts'] = power_change
            
        except Exception as e:
            logger.error("Improvement prediction failed", error=str(e))
        
        return improvements
    
    async def _record_optimization_result(self, miner_ip: str, result: OptimizationResult):
        """Record optimization result in miner state"""
        state = self.miner_states.get(miner_ip)
        if state:
            state.optimization_history.append(result)
            state.last_optimization = result.timestamp
            
            # Keep only recent history
            if len(state.optimization_history) > 50:
                state.optimization_history = state.optimization_history[-50:]
        
        # Update global stats
        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['successful_optimizations'] += 1
        
        if 'efficiency_change_percent' in result.predicted_improvements:
            self.optimization_stats['efficiency_improvements'].append(
                result.predicted_improvements['efficiency_change_percent']
            )
        
        if 'temperature_change_celsius' in result.predicted_improvements:
            self.optimization_stats['temperature_reductions'].append(
                -result.predicted_improvements['temperature_change_celsius']  # Negative for reduction
            )
    
    async def _predictive_optimization_check(self, miner_ip: str) -> bool:
        """Use predictive models to determine if optimization would be beneficial"""
        try:
            # This would use trained models to predict if optimization is needed
            # For now, return a simple heuristic
            return True
        except Exception as e:
            logger.debug(f"Predictive optimization check failed: {e}")
            return False
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        logger.info("Optimization loop started")
        
        while self.optimization_active:
            try:
                # Get list of miners to optimize
                # This would typically come from the miner service
                # For now, we'll optimize any miners in our state tracking
                
                optimization_tasks = []
                for miner_ip, state in self.miner_states.items():
                    if await self._should_optimize(miner_ip):
                        task = asyncio.create_task(
                            self.optimize_miner(miner_ip, state.telemetry, state.current_config)
                        )
                        optimization_tasks.append(task)
                
                # Wait for all optimizations to complete
                if optimization_tasks:
                    results = await asyncio.gather(*optimization_tasks, return_exceptions=True)
                    successful_optimizations = sum(1 for r in results if isinstance(r, OptimizationResult))
                    
                    logger.info(f"Optimization cycle completed: {successful_optimizations}/{len(optimization_tasks)} successful")
                
                await asyncio.sleep(self.optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Optimization loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("Optimization loop stopped")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization engine status"""
        return {
            'active': self.optimization_active,
            'tracked_miners': len(self.miner_states),
            'optimization_interval': self.optimization_interval,
            'ml_services_status': {
                'model_service': self.ml_model_service is not None,
                'weather_service': self.weather_service is not None,
                'rl_optimizer': self.rl_optimizer is not None,
                'predictive_models': len(self.predictive_models),
                'feature_engineer': self.feature_engineer is not None
            },
            'optimization_stats': self.optimization_stats.copy(),
            'safety_config': self.safety_config,
            'feature_flags': {
                'predictive_optimization': self.enable_predictive_optimization,
                'weather_adaptation': self.enable_weather_adaptation,
                'rl_optimization': self.enable_rl_optimization
            }
        }


async def create_optimization_engine(config: Dict[str, Any] = None) -> OptimizationEngine:
    """Factory function to create optimization engine"""
    engine = OptimizationEngine(config)
    await engine.start()
    return engine