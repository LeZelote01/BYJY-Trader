"""
A3C Agent for Reinforcement Learning Trading

Implements Asynchronous Advantage Actor-Critic (A3C) agent for autonomous trading.
A3C uses multiple parallel workers to collect experiences asynchronously.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import threading
import time
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from .base_agent import BaseRLAgent
from core.logger import get_logger

logger = get_logger(__name__)

class A3CNetwork(nn.Module):
    """
    A3C Actor-Critic Network
    
    Shared network for both policy and value function with
    separate heads for actor and critic.
    """
    
    def __init__(self, 
                 observation_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [256, 128]):
        """
        Initialize A3C Network
        
        Args:
            observation_dim: Input observation dimension
            action_dim: Output action dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Shared layers
        layers = []
        prev_dim = observation_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass
        
        Args:
            observations: Input observations
            
        Returns:
            Tuple of (action_distribution, state_values)
        """
        features = self.shared_layers(observations)
        
        # Actor output
        action_mean = torch.tanh(self.actor_mean(features))  # Bound actions to [-1, 1]
        action_std = torch.exp(self.actor_logstd.clamp(-5, 2))
        action_dist = Normal(action_mean, action_std)
        
        # Critic output
        state_values = self.critic(features)
        
        return action_dist, state_values
    
    def get_action_and_value(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value for given observations
        
        Args:
            observations: Input observations
            
        Returns:
            Tuple of (actions, log_probabilities, state_values)
        """
        action_dist, values = self.forward(observations)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        return actions, log_probs, values


class A3CWorker:
    """
    A3C Worker for parallel experience collection
    """
    
    def __init__(self, 
                 worker_id: int,
                 global_network: A3CNetwork,
                 local_network: A3CNetwork,
                 optimizer: torch.optim.Optimizer,
                 env_creator,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 40.0,
                 update_freq: int = 20):
        """
        Initialize A3C Worker
        
        Args:
            worker_id: Worker identification number
            global_network: Shared global network
            local_network: Local worker network
            optimizer: Global optimizer
            env_creator: Function to create environment
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            update_freq: Frequency of global network updates
        """
        self.worker_id = worker_id
        self.global_network = global_network
        self.local_network = local_network
        self.optimizer = optimizer
        self.env_creator = env_creator
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_freq = update_freq
        
        # Worker statistics
        self.episodes_completed = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        
        # Create environment
        self.env = env_creator()
        
    def run(self, max_episodes: int = 1000):
        """
        Run worker training loop
        
        Args:
            max_episodes: Maximum number of episodes to run
        """
        logger.info(f"Worker {self.worker_id} starting training")
        
        for episode in range(max_episodes):
            self._run_episode()
            
            if episode % 100 == 0:
                mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
                logger.info(f"Worker {self.worker_id}: Episode {episode}, Mean Reward: {mean_reward:.2f}")
        
        logger.info(f"Worker {self.worker_id} completed training")
    
    def _run_episode(self):
        """Run single episode"""
        # Sync local network with global
        self.local_network.load_state_dict(self.global_network.state_dict())
        
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        observation = self.env.reset()
        episode_reward = 0.0
        steps = 0
        
        while steps < self.update_freq:
            # Get action from local network
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.local_network.get_action_and_value(obs_tensor)
                action_np = action.squeeze(0).numpy()
                log_prob_np = log_prob.squeeze(0).numpy()
                value_np = value.squeeze(0).numpy()
            
            # Take environment step
            next_observation, reward, done, info = self.env.step(action_np)
            
            # Store experience
            observations.append(observation)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value_np)
            log_probs.append(log_prob_np)
            
            observation = next_observation
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        # Calculate returns and advantages
        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                _, _, next_value_tensor = self.local_network.get_action_and_value(obs_tensor)
                next_value = next_value_tensor.squeeze(0).numpy()
        
        returns = self._compute_returns(rewards, values, next_value)
        advantages = np.array(returns) - np.array(values)
        
        # Update global network
        self._update_global_network(observations, actions, returns, advantages, log_probs)
        
        # Update statistics
        if done:
            self.episodes_completed += 1
            self.total_reward += episode_reward
            self.episode_rewards.append(episode_reward)
    
    def _compute_returns(self, rewards: List[float], values: List[float], next_value: float) -> List[float]:
        """
        Compute discounted returns
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate for next state
            
        Returns:
            List of discounted returns
        """
        returns = []
        R = next_value
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def _update_global_network(self, 
                              observations: List[np.ndarray], 
                              actions: List[np.ndarray],
                              returns: List[float],
                              advantages: List[float],
                              log_probs: List[float]):
        """
        Update global network using collected experiences
        """
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations))
        act_tensor = torch.FloatTensor(np.array(actions))
        ret_tensor = torch.FloatTensor(returns)
        adv_tensor = torch.FloatTensor(advantages)
        old_log_probs_tensor = torch.FloatTensor(log_probs)
        
        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        # Forward pass through local network
        action_dist, values = self.local_network.forward(obs_tensor)
        
        # Calculate losses
        # Policy loss
        new_log_probs = action_dist.log_prob(act_tensor).sum(dim=-1)
        policy_loss = -(adv_tensor * new_log_probs).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), ret_tensor)
        
        # Entropy loss (for exploration)
        entropy = action_dist.entropy().sum(dim=-1).mean()
        entropy_loss = -entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), self.max_grad_norm)
        
        # Copy gradients to global network
        for local_param, global_param in zip(self.local_network.parameters(), self.global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad
        
        self.optimizer.step()


class A3CAgent(BaseRLAgent):
    """
    Asynchronous Advantage Actor-Critic Agent
    
    Features:
    - Multiple parallel workers
    - Asynchronous learning
    - Shared global network
    - Entropy regularization
    """
    
    def __init__(self,
                 observation_dim: int,
                 action_dim: int = 2,
                 learning_rate: float = 3e-4,
                 hidden_dims: List[int] = [256, 128],
                 num_workers: int = 4,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 40.0,
                 update_freq: int = 20,
                 device: str = 'cpu'):
        """
        Initialize A3C Agent
        
        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate
            hidden_dims: Network hidden dimensions
            num_workers: Number of parallel workers
            gamma: Discount factor
            entropy_coef: Entropy coefficient
            value_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            update_freq: Worker update frequency
            device: Computing device
        """
        super().__init__("A3C", observation_dim, action_dim, learning_rate, device)
        
        # A3C hyperparameters
        self.num_workers = num_workers
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_freq = update_freq
        
        # Initialize global network
        self.global_network = A3CNetwork(observation_dim, action_dim, hidden_dims).to(device)
        self.global_network.share_memory()  # For multiprocessing
        
        # Global optimizer
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=learning_rate)
        
        # Workers
        self.workers = []
        self.worker_threads = []
        
        # Training flag
        self.training = False
        
        logger.info(f"A3C Agent initialized with {num_workers} workers")
    
    def create_workers(self, env_creator):
        """
        Create worker threads
        
        Args:
            env_creator: Function to create training environments
        """
        self.workers = []
        
        for worker_id in range(self.num_workers):
            # Create local network for worker
            local_network = A3CNetwork(self.observation_dim, self.action_dim).to(self.device)
            
            # Create worker
            worker = A3CWorker(
                worker_id=worker_id,
                global_network=self.global_network,
                local_network=local_network,
                optimizer=self.optimizer,
                env_creator=env_creator,
                gamma=self.gamma,
                entropy_coef=self.entropy_coef,
                value_coef=self.value_coef,
                max_grad_norm=self.max_grad_norm,
                update_freq=self.update_freq
            )
            
            self.workers.append(worker)
        
        logger.info(f"Created {len(self.workers)} A3C workers")
    
    def train_parallel(self, max_episodes_per_worker: int = 1000):
        """
        Train A3C agent with parallel workers
        
        Args:
            max_episodes_per_worker: Maximum episodes per worker
        """
        if not self.workers:
            raise ValueError("No workers created. Call create_workers() first.")
        
        self.training = True
        self.worker_threads = []
        
        logger.info(f"Starting A3C training with {self.num_workers} workers")
        
        # Start worker threads
        for worker in self.workers:
            thread = threading.Thread(target=worker.run, args=(max_episodes_per_worker,))
            thread.start()
            self.worker_threads.append(thread)
        
        # Wait for all workers to complete
        for thread in self.worker_threads:
            thread.join()
        
        self.training = False
        
        # Collect training statistics
        total_episodes = sum(worker.episodes_completed for worker in self.workers)
        total_reward = sum(worker.total_reward for worker in self.workers)
        
        logger.info(f"A3C training completed: {total_episodes} episodes, total reward: {total_reward:.2f}")
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using global network
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_dist, _ = self.global_network.forward(obs_tensor)
            
            if training:
                action = action_dist.sample()
            else:
                action = action_dist.mean
            
            return action.squeeze(0).cpu().numpy()
    
    def update(self, experiences: Dict = None) -> Dict:
        """
        A3C updates happen asynchronously in workers
        
        Returns:
            Training metrics from workers
        """
        if not self.workers:
            return {}
        
        # Collect worker statistics
        total_episodes = sum(worker.episodes_completed for worker in self.workers)
        total_reward = sum(worker.total_reward for worker in self.workers)
        
        all_rewards = []
        for worker in self.workers:
            all_rewards.extend(worker.episode_rewards)
        
        if all_rewards:
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            max_reward = np.max(all_rewards)
            min_reward = np.min(all_rewards)
        else:
            mean_reward = std_reward = max_reward = min_reward = 0.0
        
        return {
            'total_episodes': total_episodes,
            'total_reward': total_reward,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'num_workers': len(self.workers)
        }
    
    def save_model(self, filepath: str):
        """Save A3C global network"""
        torch.save({
            'global_network_state_dict': self.global_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'observation_dim': self.observation_dim,
                'action_dim': self.action_dim,
                'num_workers': self.num_workers,
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            },
            'training_stats': self.get_stats()
        }, filepath)
        
        logger.info(f"A3C model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load A3C global network"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.global_network.load_state_dict(checkpoint['global_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters if available
        if 'hyperparameters' in checkpoint:
            hyperparams = checkpoint['hyperparameters']
            self.num_workers = hyperparams.get('num_workers', self.num_workers)
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.entropy_coef = hyperparams.get('entropy_coef', self.entropy_coef)
            self.value_coef = hyperparams.get('value_coef', self.value_coef)
        
        logger.info(f"A3C model loaded from {filepath}")
    
    def get_worker_stats(self) -> Dict:
        """Get detailed worker statistics"""
        if not self.workers:
            return {}
        
        worker_stats = {}
        for i, worker in enumerate(self.workers):
            worker_stats[f'worker_{i}'] = {
                'episodes_completed': worker.episodes_completed,
                'total_reward': worker.total_reward,
                'mean_reward': np.mean(worker.episode_rewards) if worker.episode_rewards else 0.0,
                'recent_rewards': worker.episode_rewards[-10:] if worker.episode_rewards else []
            }
        
        return worker_stats
    
    def stop_training(self):
        """Stop training (for early termination)"""
        self.training = False
        logger.info("A3C training stop requested")