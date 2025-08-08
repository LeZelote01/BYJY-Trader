"""
PPO Agent for Reinforcement Learning Trading

Implements Proximal Policy Optimization (PPO) agent for autonomous trading.
PPO is a policy gradient method that maintains a balance between sample efficiency 
and performance stability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from .base_agent import BaseRLAgent
from core.logger import get_logger

logger = get_logger(__name__)

class PPONetwork(nn.Module):
    """
    PPO Actor-Critic Network
    
    Combines both policy (actor) and value (critic) networks
    with shared feature extraction layers.
    """
    
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128]):
        """
        Initialize PPO Network
        
        Args:
            observation_dim: Input observation dimension
            action_dim: Output action dimension  
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy network)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value network) 
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            observations: Batch of observations
            
        Returns:
            Tuple of (action_distribution, state_values)
        """
        # Shared feature extraction
        features = self.shared_layers(observations)
        
        # Actor: get action distribution
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd.clamp(-5, 2))  # Clamp for stability
        action_dist = Normal(action_mean, action_std)
        
        # Critic: get state value
        state_values = self.critic(features)
        
        return action_dist, state_values
    
    def get_action(self, observations: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions from observations
        
        Args:
            observations: Input observations
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (actions, log_probabilities)
        """
        action_dist, _ = self.forward(observations)
        
        if deterministic:
            actions = action_dist.mean
        else:
            actions = action_dist.sample()
        
        # Clamp actions to valid range
        actions = torch.tanh(actions)  # Ensures actions in [-1, 1]
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        return actions, log_probs
    
    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations
        
        Args:
            observations: Input observations  
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_probabilities, state_values, entropy)
        """
        action_dist, state_values = self.forward(observations)
        
        # Convert tanh actions back to original space for log prob calculation
        pre_tanh_actions = torch.atanh(torch.clamp(actions, -0.9999, 0.9999))
        
        log_probs = action_dist.log_prob(pre_tanh_actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_probs, state_values, entropy


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization Agent
    
    Features:
    - Clipped policy gradient updates
    - Value function learning
    - Entropy regularization
    - Generalized Advantage Estimation (GAE)
    """
    
    def __init__(self,
                 observation_dim: int,
                 action_dim: int = 2,  # [position_change, confidence]
                 learning_rate: float = 3e-4,
                 hidden_dims: List[int] = [256, 256, 128],
                 clip_ratio: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 gae_lambda: float = 0.95,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 epochs_per_update: int = 10,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize PPO Agent
        
        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            hidden_dims: Network hidden dimensions
            clip_ratio: PPO clipping parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            gae_lambda: GAE lambda parameter
            gamma: Discount factor
            batch_size: Training batch size
            epochs_per_update: Training epochs per update
            max_grad_norm: Max gradient norm for clipping
            device: Computing device
        """
        super().__init__("PPO", observation_dim, action_dim, learning_rate, device)
        
        # PPO hyperparameters
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.max_grad_norm = max_grad_norm
        
        # Initialize network
        self.network = PPONetwork(observation_dim, action_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer(gamma=gamma, gae_lambda=gae_lambda)
        
        # Training metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.clip_fractions = []
        
        logger.info(f"PPO Agent initialized with obs_dim={observation_dim}, action_dim={action_dim}")
        logger.info(f"PPO hyperparameters: clip_ratio={clip_ratio}, value_coef={value_coef}, entropy_coef={entropy_coef}")
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using current policy
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            if training:
                actions, log_probs = self.network.get_action(obs_tensor, deterministic=False)
                
                # Store for buffer if needed
                self._last_obs = observation
                self._last_action = actions.squeeze(0).cpu().numpy()
                self._last_log_prob = log_probs.squeeze(0).cpu().numpy()
                
                # Get value estimate
                _, value = self.network.forward(obs_tensor)
                self._last_value = value.squeeze(0).cpu().numpy()
                
            else:
                # Deterministic action for evaluation
                actions, _ = self.network.get_action(obs_tensor, deterministic=True)
                self._last_action = actions.squeeze(0).cpu().numpy()
        
        return self._last_action
    
    def store_transition(self, observation: np.ndarray, action: np.ndarray, 
                        reward: float, next_observation: np.ndarray, done: bool):
        """Store transition in buffer"""
        if hasattr(self, '_last_obs'):
            self.buffer.store(
                obs=self._last_obs,
                act=action,
                rew=reward,
                val=self._last_value if hasattr(self, '_last_value') else 0.0,
                logp=self._last_log_prob if hasattr(self, '_last_log_prob') else 0.0
            )
            
            if done:
                self.buffer.finish_path(last_val=0.0)
            
            self.step_count += 1
    
    def update(self, experiences: Dict = None) -> Dict:
        """
        Update agent with PPO algorithm
        
        Args:
            experiences: Not used, PPO uses internal buffer
            
        Returns:
            Training metrics
        """
        if not self.buffer.ready_for_update():
            return {}
        
        # Get training data from buffer
        data = self.buffer.get()
        
        # Convert to tensors
        observations = torch.FloatTensor(data['obs']).to(self.device)
        actions = torch.FloatTensor(data['act']).to(self.device)
        advantages = torch.FloatTensor(data['adv']).to(self.device)
        returns = torch.FloatTensor(data['ret']).to(self.device)
        old_log_probs = torch.FloatTensor(data['logp']).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # Multiple epochs of updates
        for epoch in range(self.epochs_per_update):
            # Create random batches
            indices = torch.randperm(len(observations))
            
            for start_idx in range(0, len(observations), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(observations))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_acts = actions[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_ret = returns[batch_indices]
                batch_old_logp = old_log_probs[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.network.evaluate_actions(batch_obs, batch_acts)
                
                # Policy loss with PPO clipping
                ratio = torch.exp(log_probs - batch_old_logp)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                policy_loss1 = -batch_adv * ratio
                policy_loss2 = -batch_adv * clipped_ratio
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_ret)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Clip fraction (measure of how much clipping occurs)
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                    clip_fractions.append(clip_fraction)
        
        # Store training metrics
        metrics = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions),
            'total_loss': np.mean(policy_losses) + self.value_coef * np.mean(value_losses) + self.entropy_coef * np.mean(entropy_losses)
        }
        
        # Update stored metrics
        self.policy_losses.extend(policy_losses)
        self.value_losses.extend(value_losses)
        self.entropy_losses.extend(entropy_losses)
        self.clip_fractions.extend(clip_fractions)
        
        # Clear buffer
        self.buffer.clear()
        
        logger.debug(f"PPO update completed: policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, clip_fraction={metrics['clip_fraction']:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save PPO model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'observation_dim': self.observation_dim,
                'action_dim': self.action_dim,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'gae_lambda': self.gae_lambda,
                'gamma': self.gamma
            },
            'training_stats': self.get_stats()
        }, filepath)
        
        logger.info(f"PPO model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load PPO model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.clip_ratio = hyperparams.get('clip_ratio', self.clip_ratio)
            self.value_coef = hyperparams.get('value_coef', self.value_coef)
            self.entropy_coef = hyperparams.get('entropy_coef', self.entropy_coef)
            self.gae_lambda = hyperparams.get('gae_lambda', self.gae_lambda)
            self.gamma = hyperparams.get('gamma', self.gamma)
        
        logger.info(f"PPO model loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict:
        """Get detailed training metrics"""
        base_stats = self.get_stats()
        
        ppo_stats = {
            'recent_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0.0,
            'recent_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0.0,
            'recent_entropy_loss': np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0.0,
            'recent_clip_fraction': np.mean(self.clip_fractions[-100:]) if self.clip_fractions else 0.0,
        }
        
        return {**base_stats, **ppo_stats}


class PPOBuffer:
    """
    Buffer for storing experiences and computing GAE advantages
    """
    
    def __init__(self, size: int = 10000, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize PPO Buffer
        
        Args:
            size: Buffer size
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Buffers
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        
        # Episode tracking
        self.path_start_idx = 0
        
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
        """Store single transition"""
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.values.append(val)
        self.log_probs.append(logp)
    
    def finish_path(self, last_val: float = 0.0):
        """
        Finish current episode path and compute advantages/returns
        
        Args:
            last_val: Value estimate for final state
        """
        path_slice = slice(self.path_start_idx, len(self.rewards))
        rews = self.rewards[path_slice] + [last_val]
        vals = self.values[path_slice] + [last_val]
        
        # Compute GAE advantages
        deltas = [r + self.gamma * nv - v for r, v, nv in zip(rews[:-1], vals[:-1], vals[1:])]
        
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + self.gamma * self.gae_lambda * adv
            advantages.append(adv)
        advantages.reverse()
        
        # Compute returns
        returns = [adv + val for adv, val in zip(advantages, vals[:-1])]
        
        # Store advantages and returns
        self.advantages.extend(advantages)
        self.returns.extend(returns)
        
        self.path_start_idx = len(self.rewards)
    
    def get(self) -> Dict:
        """Get all stored data"""
        data = {
            'obs': np.array(self.observations),
            'act': np.array(self.actions),
            'ret': np.array(self.returns),
            'adv': np.array(self.advantages),
            'logp': np.array(self.log_probs)
        }
        return data
    
    def ready_for_update(self) -> bool:
        """Check if buffer has enough data for update"""
        return len(self.advantages) > 0
    
    def clear(self):
        """Clear all buffers"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.path_start_idx = 0