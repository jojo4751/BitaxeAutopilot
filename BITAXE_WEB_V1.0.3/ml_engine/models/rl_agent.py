"""
Reinforcement Learning Agent for Frequency/Voltage Optimization

PPO-based agent for autonomous BitAxe mining optimization with safety constraints
and multi-objective reward functions.
"""

import asyncio
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pickle
from pathlib import Path

from logging.structured_logger import get_logger
from monitoring.metrics_collector import get_metrics_collector

logger = get_logger("bitaxe.ml.rl_agent")


@dataclass
class RLState:
    """Reinforcement learning state representation"""
    temperature: float
    hashrate: float
    power: float
    efficiency: float
    voltage: float
    frequency: float
    weather_temp: float = 20.0
    weather_humidity: float = 50.0
    time_of_day: float = 0.5
    stability_score: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array"""
        return np.array([
            self.temperature,
            self.hashrate,
            self.power,
            self.efficiency,
            self.voltage,
            self.frequency,
            self.weather_temp,
            self.weather_humidity,
            self.time_of_day,
            self.stability_score
        ], dtype=np.float32)
    
    @classmethod
    def from_miner_data(cls, miner_data: Dict[str, Any], weather_data: Optional[Dict[str, Any]] = None) -> 'RLState':
        """Create state from miner telemetry data"""
        now = datetime.now()
        time_of_day = (now.hour * 60 + now.minute) / (24 * 60)  # Normalize to 0-1
        
        # Calculate efficiency
        hashrate = miner_data.get('hashRate', 0)
        power = miner_data.get('power', 1)
        efficiency = hashrate / power if power > 0 else 0
        
        # Weather data
        weather_temp = weather_data.get('temperature', 20.0) if weather_data else 20.0
        weather_humidity = weather_data.get('humidity', 50.0) if weather_data else 50.0
        
        return cls(
            temperature=miner_data.get('temp', 50.0),
            hashrate=hashrate,
            power=power,
            efficiency=efficiency,
            voltage=miner_data.get('voltage', 12.0),
            frequency=miner_data.get('frequency', 600),
            weather_temp=weather_temp,
            weather_humidity=weather_humidity,
            time_of_day=time_of_day,
            stability_score=miner_data.get('stability_score', 1.0)
        )


@dataclass
class RLAction:
    """Reinforcement learning action representation"""
    frequency_delta: float  # Frequency change in MHz
    voltage_delta: float    # Voltage change in V
    
    def to_array(self) -> np.ndarray:
        """Convert action to numpy array"""
        return np.array([self.frequency_delta, self.voltage_delta], dtype=np.float32)
    
    @classmethod
    def from_array(cls, action_array: np.ndarray) -> 'RLAction':
        """Create action from numpy array"""
        return cls(
            frequency_delta=float(action_array[0]),
            voltage_delta=float(action_array[1])
        )
    
    def apply_to_miner_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply action to current miner configuration"""
        new_config = current_config.copy()
        
        # Apply frequency change with safety limits
        current_freq = current_config.get('frequency', 600)
        new_freq = current_freq + self.frequency_delta
        new_freq = np.clip(new_freq, 400, 800)  # Safety limits
        
        # Apply voltage change with safety limits
        current_voltage = current_config.get('voltage', 12.0)
        new_voltage = current_voltage + self.voltage_delta
        new_voltage = np.clip(new_voltage, 10.0, 14.0)  # Safety limits
        
        new_config.update({
            'frequency': new_freq,
            'voltage': new_voltage
        })
        
        return new_config


class RewardFunction:
    """
    Multi-objective reward function for mining optimization
    
    Balances:
    - Mining efficiency (primary objective)
    - Temperature control (safety constraint)
    - Stability (longevity consideration)
    - Power consumption (economic factor)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Reward weights
        self.efficiency_weight = self.config.get('efficiency_weight', 1.0)
        self.temperature_weight = self.config.get('temperature_weight', 0.3)
        self.stability_weight = self.config.get('stability_weight', 0.2)
        self.power_weight = self.config.get('power_weight', 0.1)
        
        # Temperature thresholds
        self.temp_target = self.config.get('temp_target', 65.0)  # °C
        self.temp_max = self.config.get('temp_max', 85.0)  # °C
        self.temp_critical = self.config.get('temp_critical', 95.0)  # °C
        
        # Efficiency targets
        self.efficiency_baseline = self.config.get('efficiency_baseline', 100.0)  # GH/W
        
        logger.info("Reward function initialized",
                   weights={
                       'efficiency': self.efficiency_weight,
                       'temperature': self.temperature_weight,
                       'stability': self.stability_weight,
                       'power': self.power_weight
                   })
    
    def calculate_reward(self, 
                        prev_state: RLState, 
                        action: RLAction, 
                        new_state: RLState) -> float:
        """Calculate reward for state transition"""
        try:
            reward = 0.0
            
            # 1. Efficiency reward (primary objective)
            efficiency_reward = self._calculate_efficiency_reward(prev_state, new_state)
            reward += self.efficiency_weight * efficiency_reward
            
            # 2. Temperature penalty (safety constraint)
            temperature_penalty = self._calculate_temperature_penalty(new_state)
            reward -= self.temperature_weight * temperature_penalty
            
            # 3. Stability reward (longevity)
            stability_reward = self._calculate_stability_reward(prev_state, new_state)
            reward += self.stability_weight * stability_reward
            
            # 4. Power efficiency (economic factor)
            power_reward = self._calculate_power_reward(prev_state, new_state)
            reward += self.power_weight * power_reward
            
            # 5. Critical safety penalties
            safety_penalty = self._calculate_safety_penalty(new_state)
            reward -= safety_penalty
            
            # Normalize reward to reasonable range
            reward = np.clip(reward, -10.0, 10.0)
            
            return float(reward)
            
        except Exception as e:
            logger.error("Reward calculation failed", error=str(e))
            return -1.0  # Penalty for calculation errors
    
    def _calculate_efficiency_reward(self, prev_state: RLState, new_state: RLState) -> float:
        """Calculate efficiency improvement reward"""
        efficiency_improvement = new_state.efficiency - prev_state.efficiency
        
        # Normalize by baseline efficiency
        normalized_improvement = efficiency_improvement / self.efficiency_baseline
        
        # Additional bonus for achieving high efficiency
        efficiency_bonus = 0.0
        if new_state.efficiency > self.efficiency_baseline * 1.2:
            efficiency_bonus = 0.5
        
        return normalized_improvement + efficiency_bonus
    
    def _calculate_temperature_penalty(self, state: RLState) -> float:
        """Calculate temperature-based penalty"""
        temp = state.temperature
        
        if temp <= self.temp_target:
            return 0.0  # No penalty for optimal temperature
        elif temp <= self.temp_max:
            # Linear penalty up to max temperature
            return (temp - self.temp_target) / (self.temp_max - self.temp_target)
        elif temp <= self.temp_critical:
            # Exponential penalty between max and critical
            excess = (temp - self.temp_max) / (self.temp_critical - self.temp_max)
            return 1.0 + 2.0 * (excess ** 2)
        else:
            # Severe penalty for critical temperatures
            return 5.0
    
    def _calculate_stability_reward(self, prev_state: RLState, new_state: RLState) -> float:
        """Calculate stability reward"""
        # Reward maintaining stable hashrate
        hashrate_stability = 1.0 - abs(new_state.hashrate - prev_state.hashrate) / prev_state.hashrate
        hashrate_stability = max(0.0, hashrate_stability)
        
        # Reward maintaining stable temperature
        temp_stability = 1.0 - abs(new_state.temperature - prev_state.temperature) / 10.0
        temp_stability = max(0.0, temp_stability)
        
        return (hashrate_stability + temp_stability) / 2.0
    
    def _calculate_power_reward(self, prev_state: RLState, new_state: RLState) -> float:
        """Calculate power efficiency reward"""
        # Reward for reducing power while maintaining/improving hashrate
        power_change = new_state.power - prev_state.power
        hashrate_change = new_state.hashrate - prev_state.hashrate
        
        if power_change <= 0 and hashrate_change >= 0:
            # Best case: less power, same or better hashrate
            return 1.0
        elif power_change > 0 and hashrate_change > 0:
            # Increased power but better hashrate - check if efficiency improved
            efficiency_change = new_state.efficiency - prev_state.efficiency
            return efficiency_change / prev_state.efficiency if prev_state.efficiency > 0 else 0.0
        else:
            # Power increased without hashrate benefit
            return -0.5
    
    def _calculate_safety_penalty(self, state: RLState) -> float:
        """Calculate severe safety violation penalties"""
        penalty = 0.0
        
        # Critical temperature violation
        if state.temperature > self.temp_critical:
            penalty += 10.0
        
        # Impossible/dangerous values
        if state.voltage < 8.0 or state.voltage > 16.0:
            penalty += 5.0
        
        if state.frequency < 200 or state.frequency > 1000:
            penalty += 5.0
        
        if state.power <= 0 or state.hashrate <= 0:
            penalty += 3.0
        
        return penalty


class PPOAgent:
    """
    Proximal Policy Optimization Agent for mining optimization
    
    Features:
    - Continuous action space for frequency/voltage control
    - Safety-constrained exploration
    - Multi-objective optimization
    - Experience replay and training
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # RL hyperparameters
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.value_loss_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        
        # Network architecture
        self.state_dim = 10  # RLState dimensions
        self.action_dim = 2  # frequency_delta, voltage_delta
        self.hidden_dim = self.config.get('hidden_dim', 64)
        
        # Safety constraints
        self.action_bounds = {
            'frequency_delta': (-50, 50),  # MHz
            'voltage_delta': (-0.5, 0.5)  # V
        }
        
        # Experience buffer
        self.experience_buffer = []
        self.buffer_size = self.config.get('buffer_size', 10000)
        
        # Training settings
        self.batch_size = self.config.get('batch_size', 64)
        self.training_epochs = self.config.get('training_epochs', 10)
        self.training_interval = self.config.get('training_interval', 100)  # steps
        
        # Model state
        self.is_trained = False
        self.total_steps = 0
        self.episode_rewards = []
        
        # Reward function
        self.reward_function = RewardFunction(self.config.get('reward_config', {}))
        
        # Initialize neural network (simplified version)
        self.policy_network = None
        self.value_network = None
        self._initialize_networks()
        
        logger.info("PPO Agent initialized",
                   state_dim=self.state_dim,
                   action_dim=self.action_dim,
                   learning_rate=self.learning_rate)
    
    def _initialize_networks(self):
        """Initialize policy and value networks"""
        try:
            # This is a simplified implementation
            # In production, you would use TensorFlow/PyTorch
            
            # For demonstration, we'll use a simple policy
            self.policy_params = {
                'mean': np.zeros(self.action_dim),
                'std': np.ones(self.action_dim) * 0.1
            }
            
            self.value_params = {
                'weights': np.random.normal(0, 0.1, (self.state_dim, 1)),
                'bias': 0.0
            }
            
            logger.info("Neural networks initialized")
            
        except Exception as e:
            logger.error("Failed to initialize networks", error=str(e))
    
    async def select_action(self, state: RLState, exploration: bool = True) -> RLAction:
        """
        Select action using current policy
        
        Args:
            state: Current environment state
            exploration: Whether to add exploration noise
            
        Returns:
            Selected action
        """
        try:
            state_array = state.to_array()
            
            # Simple policy: adjust based on temperature and efficiency
            frequency_delta = 0.0
            voltage_delta = 0.0
            
            # Temperature-based adjustments
            if state.temperature > 75.0:
                # Too hot - reduce frequency/voltage
                frequency_delta = -10.0 - (state.temperature - 75.0) * 0.5
                voltage_delta = -0.1 - (state.temperature - 75.0) * 0.01
            elif state.temperature < 60.0:
                # Cool enough - can increase performance
                if state.efficiency < 120.0:  # Below target efficiency
                    frequency_delta = 5.0
                    voltage_delta = 0.05
            
            # Efficiency-based adjustments
            if state.efficiency < 80.0:
                # Very low efficiency - emergency adjustment
                frequency_delta = min(frequency_delta + 10.0, 20.0)
                voltage_delta = min(voltage_delta + 0.1, 0.2)
            
            # Add exploration noise if enabled
            if exploration:
                frequency_delta += np.random.normal(0, 5.0)
                voltage_delta += np.random.normal(0, 0.05)
            
            # Apply safety constraints
            frequency_delta = np.clip(frequency_delta, 
                                    self.action_bounds['frequency_delta'][0],
                                    self.action_bounds['frequency_delta'][1])
            voltage_delta = np.clip(voltage_delta,
                                  self.action_bounds['voltage_delta'][0],
                                  self.action_bounds['voltage_delta'][1])
            
            action = RLAction(frequency_delta, voltage_delta)
            
            # Record metrics
            self.metrics_collector.record_metric('rl_action_frequency_delta', frequency_delta)
            self.metrics_collector.record_metric('rl_action_voltage_delta', voltage_delta)
            
            return action
            
        except Exception as e:
            logger.error("Action selection failed", error=str(e))
            # Return safe default action
            return RLAction(0.0, 0.0)
    
    async def store_experience(self, 
                             state: RLState, 
                             action: RLAction, 
                             reward: float, 
                             next_state: RLState, 
                             done: bool):
        """Store experience in replay buffer"""
        try:
            experience = {
                'state': state.to_array(),
                'action': action.to_array(),
                'reward': reward,
                'next_state': next_state.to_array(),
                'done': done,
                'timestamp': datetime.now()
            }
            
            self.experience_buffer.append(experience)
            
            # Maintain buffer size
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer = self.experience_buffer[-self.buffer_size:]
            
            self.total_steps += 1
            
            # Record metrics
            self.metrics_collector.record_metric('rl_reward', reward)
            self.metrics_collector.set_gauge('rl_experience_buffer_size', len(self.experience_buffer))
            
            # Check if training is needed
            if (self.total_steps % self.training_interval == 0 and 
                len(self.experience_buffer) >= self.batch_size):
                await self.train()
            
        except Exception as e:
            logger.error("Failed to store experience", error=str(e))
    
    async def train(self):
        """Train the PPO agent on collected experiences"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return
            
            start_time = datetime.now()
            
            # Sample batch from experience buffer
            batch_indices = np.random.choice(len(self.experience_buffer), 
                                           self.batch_size, replace=False)
            batch = [self.experience_buffer[i] for i in batch_indices]
            
            # Extract batch data
            states = np.array([exp['state'] for exp in batch])
            actions = np.array([exp['action'] for exp in batch])
            rewards = np.array([exp['reward'] for exp in batch])
            next_states = np.array([exp['next_state'] for exp in batch])
            dones = np.array([exp['done'] for exp in batch])
            
            # Calculate advantages (simplified)
            values = self._estimate_values(states)
            next_values = self._estimate_values(next_states)
            
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
            
            # Update policy and value networks (simplified)
            policy_loss = self._update_policy(states, actions, advantages)
            value_loss = self._update_value_function(states, rewards, next_values, dones)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Record training metrics
            self.metrics_collector.record_metric('rl_training_duration', training_time)
            self.metrics_collector.record_metric('rl_policy_loss', policy_loss)
            self.metrics_collector.record_metric('rl_value_loss', value_loss)
            self.metrics_collector.increment_counter('rl_training_steps_total')
            
            self.is_trained = True
            
            logger.info("PPO training completed",
                       batch_size=self.batch_size,
                       policy_loss=policy_loss,
                       value_loss=value_loss,
                       training_time_ms=training_time * 1000)
            
        except Exception as e:
            logger.error("PPO training failed", error=str(e))
            self.metrics_collector.increment_counter('rl_training_errors_total')
    
    def _estimate_values(self, states: np.ndarray) -> np.ndarray:
        """Estimate state values using value network"""
        # Simplified value estimation
        values = np.dot(states, self.value_params['weights']).flatten() + self.value_params['bias']
        return values
    
    def _update_policy(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray) -> float:
        """Update policy network"""
        # Simplified policy update
        # In practice, this would use gradient ascent on policy objective
        
        # Calculate policy gradient approximation
        policy_gradient = np.mean(advantages)
        
        # Update policy parameters (very simplified)
        learning_rate = self.learning_rate * 0.1
        self.policy_params['mean'] += learning_rate * policy_gradient * 0.01
        
        return abs(policy_gradient)
    
    def _update_value_function(self, states: np.ndarray, rewards: np.ndarray, 
                              next_values: np.ndarray, dones: np.ndarray) -> float:
        """Update value function"""
        # Calculate target values
        targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Calculate value loss (MSE)
        current_values = self._estimate_values(states)
        value_loss = np.mean((targets - current_values) ** 2)
        
        # Update value function parameters (simplified)
        value_gradient = np.mean((targets - current_values).reshape(-1, 1) * states, axis=0)
        learning_rate = self.learning_rate * 0.1
        
        self.value_params['weights'] += learning_rate * value_gradient.reshape(-1, 1)
        self.value_params['bias'] += learning_rate * np.mean(targets - current_values)
        
        return value_loss
    
    async def save_model(self, filepath: str):
        """Save trained model"""
        try:
            model_data = {
                'policy_params': self.policy_params,
                'value_params': self.value_params,
                'config': self.config,
                'is_trained': self.is_trained,
                'total_steps': self.total_steps,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
    
    async def load_model(self, filepath: str):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.policy_params = model_data['policy_params']
            self.value_params = model_data['value_params']
            self.is_trained = model_data['is_trained']
            self.total_steps = model_data['total_steps']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'is_trained': self.is_trained,
            'total_steps': self.total_steps,
            'buffer_size': len(self.experience_buffer),
            'episode_count': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }


class FrequencyVoltageOptimizer:
    """
    High-level RL-based optimizer for BitAxe frequency and voltage
    
    Combines PPO agent with safety systems and performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = get_metrics_collector()
        
        # Initialize PPO agent
        self.agent = PPOAgent(self.config.get('agent_config', {}))
        
        # Safety system
        self.safety_enabled = self.config.get('safety_enabled', True)
        self.max_temp_threshold = self.config.get('max_temp_threshold', 90.0)
        self.min_efficiency_threshold = self.config.get('min_efficiency_threshold', 50.0)
        
        # Optimization settings
        self.optimization_interval = self.config.get('optimization_interval', 60)  # seconds
        self.exploration_rate = self.config.get('exploration_rate', 0.1)
        self.exploration_decay = self.config.get('exploration_decay', 0.995)
        
        # State tracking
        self.current_state: Optional[RLState] = None
        self.last_action: Optional[RLAction] = None
        self.optimization_active = False
        
        logger.info("Frequency/Voltage Optimizer initialized")
    
    async def start_optimization(self):
        """Start autonomous optimization"""
        self.optimization_active = True
        logger.info("RL optimization started")
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
    
    async def stop_optimization(self):
        """Stop autonomous optimization"""
        self.optimization_active = False
        logger.info("RL optimization stopped")
    
    async def optimize_miner(self, 
                           miner_data: Dict[str, Any],
                           weather_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Optimize single miner configuration
        
        Args:
            miner_data: Current miner telemetry
            weather_data: Current weather conditions
            
        Returns:
            Optimized configuration or None if no changes needed
        """
        try:
            # Create current state
            new_state = RLState.from_miner_data(miner_data, weather_data)
            
            # Safety checks
            if self.safety_enabled and not self._safety_check(new_state):
                logger.warning("Safety check failed, skipping optimization")
                return None
            
            # Calculate reward if we have previous state and action
            if self.current_state and self.last_action:
                reward = self.agent.reward_function.calculate_reward(
                    self.current_state, self.last_action, new_state
                )
                
                await self.agent.store_experience(
                    self.current_state, self.last_action, reward, new_state, False
                )
            
            # Select new action
            action = await self.agent.select_action(new_state, 
                                                  exploration=np.random.random() < self.exploration_rate)
            
            # Apply action to get new configuration
            current_config = {
                'frequency': miner_data.get('frequency', 600),
                'voltage': miner_data.get('voltage', 12.0)
            }
            
            new_config = action.apply_to_miner_config(current_config)
            
            # Update state tracking
            self.current_state = new_state
            self.last_action = action
            
            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(self.exploration_rate, 0.05)
            
            # Record metrics
            self.metrics_collector.increment_counter('rl_optimizations_total')
            self.metrics_collector.record_metric('rl_exploration_rate', self.exploration_rate)
            
            logger.debug("Miner optimization completed",
                        frequency_change=action.frequency_delta,
                        voltage_change=action.voltage_delta,
                        new_frequency=new_config['frequency'],
                        new_voltage=new_config['voltage'])
            
            return new_config
            
        except Exception as e:
            logger.error("Miner optimization failed", error=str(e))
            self.metrics_collector.increment_counter('rl_optimization_errors_total')
            return None
    
    def _safety_check(self, state: RLState) -> bool:
        """Perform safety checks on current state"""
        try:
            # Temperature safety
            if state.temperature > self.max_temp_threshold:
                logger.warning(f"Temperature too high: {state.temperature}°C")
                return False
            
            # Efficiency safety
            if state.efficiency < self.min_efficiency_threshold:
                logger.warning(f"Efficiency too low: {state.efficiency} GH/W")
                return False
            
            # Power safety
            if state.power <= 0 or state.power > 200:  # Reasonable power bounds
                logger.warning(f"Power out of bounds: {state.power}W")
                return False
            
            # Voltage safety
            if state.voltage < 8.0 or state.voltage > 16.0:
                logger.warning(f"Voltage out of bounds: {state.voltage}V")
                return False
            
            return True
            
        except Exception as e:
            logger.error("Safety check failed", error=str(e))
            return False
    
    async def _optimization_loop(self):
        """Background optimization loop"""
        logger.info("RL optimization loop started")
        
        while self.optimization_active:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                # Record optimization status
                self.metrics_collector.set_gauge('rl_optimization_active', 1 if self.optimization_active else 0)
                
            except Exception as e:
                logger.error("Optimization loop error", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("RL optimization loop stopped")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'active': self.optimization_active,
            'exploration_rate': self.exploration_rate,
            'agent_stats': self.agent.get_training_stats(),
            'current_state': asdict(self.current_state) if self.current_state else None,
            'last_action': asdict(self.last_action) if self.last_action else None,
            'safety_enabled': self.safety_enabled
        }
    
    async def save_agent(self, filepath: str):
        """Save trained agent"""
        await self.agent.save_model(filepath)
    
    async def load_agent(self, filepath: str):
        """Load trained agent"""
        await self.agent.load_model(filepath)


async def create_frequency_voltage_optimizer(config: Dict[str, Any] = None) -> FrequencyVoltageOptimizer:
    """Factory function to create RL optimizer"""
    optimizer = FrequencyVoltageOptimizer(config)
    return optimizer