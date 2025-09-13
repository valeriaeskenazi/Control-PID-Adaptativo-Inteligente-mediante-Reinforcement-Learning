"""
Proximal Policy Optimization (PPO) Agent for PID controller tuning.

This implementation uses all the modular components we've built:
- StochasticActor for policy network
- CriticNetwork for value estimation  
- EpisodeBuffer for experience collection
- StochasticPolicy for action selection
- All integrated with the AbstractPolicyGradientAgent interface
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

from ..base_agent import AbstractPolicyGradientAgent, PIDAgentConfig
from ..networks.actor_critic import StochasticActor, CriticNetwork
from ..buffer import EpisodeBuffer, PolicyExperience
from ..policy import StochasticPolicy


class PPOAgent(AbstractPolicyGradientAgent):
    """
    Proximal Policy Optimization agent for PID parameter tuning.
    
    PPO is particularly well-suited for PID control because:
    1. Handles continuous action spaces naturally
    2. Stable training with policy clipping
    3. Sample efficient with GAE
    4. Good exploration with stochastic policies
    """
    
    def __init__(self, config: PIDAgentConfig):
        super().__init__(
            state_dim=config.state_dim if hasattr(config, 'state_dim') else 6,
            action_dim=config.action_dim if hasattr(config, 'action_dim') else 3,
            device=config.device,
            seed=config.seed
        )
        
        self.config = config
        
        # PPO-specific hyperparameters
        self.clip_ratio = getattr(config, 'clip_ratio', 0.2)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 10)
        self.mini_batch_size = getattr(config, 'mini_batch_size', 64)
        self.value_loss_coef = getattr(config, 'value_loss_coef', 0.5)
        self.entropy_coef = getattr(config, 'entropy_coef', 0.01)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.gamma = config.gamma
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        
        # Create networks using our modular components
        # Disable batch norm for single sample inference
        self.actor = StochasticActor(
            input_dim=self.state_dim,
            hidden_dims=config.hidden_dims,
            kp_range=config.kp_range,
            ki_range=config.ki_range,
            kd_range=config.kd_range,
            dropout_rate=0.0  # Disable dropout for stability
        ).to(self.device)
        
        self.critic = CriticNetwork(
            input_dim=self.state_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=0.0  # Disable dropout for stability
        ).to(self.device)
        
        # Create policy wrapper
        action_bounds = (
            torch.tensor([config.kp_range[0], config.ki_range[0], config.kd_range[0]]),
            torch.tensor([config.kp_range[1], config.ki_range[1], config.kd_range[1]])
        )
        self.policy = StochasticPolicy(
            network=self.actor,
            action_bounds=action_bounds,
            device=self.device
        )
        
        # Experience buffer
        buffer_capacity = getattr(config, 'buffer_capacity', 10000)
        self.buffer = EpisodeBuffer(capacity=buffer_capacity, device=self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate,
            eps=getattr(config, 'adam_eps', 1e-5)
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate,
            eps=getattr(config, 'adam_eps', 1e-5)
        )
        
        # Training metrics
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"PPOAgent_{id(self)}")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using stochastic policy.
        
        Args:
            state: Current state [PV, setpoint, error, error_prev, error_integral, error_derivative]
            training: Whether in training mode (affects exploration)
        
        Returns:
            action: PID parameters [Kp, Ki, Kd]
        """
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            if training:
                # Sample from policy distribution
                action, log_prob = self.policy.get_action_and_log_prob(state_tensor)
                
                # Also get value estimate for experience storage
                value = self.critic(state_tensor)
                
                # Store for potential experience collection
                self._last_log_prob = log_prob.item()
                self._last_value = value.item()
            else:
                # Use mean action for evaluation
                action = self.policy.select_action(state_tensor, training=False)
                self._last_log_prob = 0.0
                self._last_value = 0.0
        
        return self.postprocess_action(action)
    
    def store_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Store experience in episode buffer.
        
        Args:
            state: Current state
            action: Action taken  
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = PolicyExperience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value
        )
        
        self.buffer.add(experience)
        
        if done:
            self.buffer.finish_episode()
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Update PPO agent using collected experiences.
        
        Args:
            batch_data: Optional batch data (PPO uses its own buffer)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Get all experiences from buffer
        batch = self.buffer.sample()
        
        if batch['states'].shape[0] == 0:
            return {}
        
        # Compute returns and advantages using GAE
        returns_and_advantages = self.buffer.compute_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        if returns_and_advantages['returns'].shape[0] == 0:
            return {}
        
        returns = returns_and_advantages['returns']
        advantages = returns_and_advantages['advantages']
        
        # PPO update for multiple epochs
        total_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0
        }
        
        dataset_size = batch['states'].shape[0]
        num_batches = 0
        
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                mini_batch_indices = indices[start_idx:end_idx]
                
                if len(mini_batch_indices) < 4:  # Skip very small batches
                    continue
                
                # Get mini-batch data
                mini_batch = {
                    'states': batch['states'][mini_batch_indices],
                    'actions': batch['actions'][mini_batch_indices],
                    'old_log_probs': batch['log_probs'][mini_batch_indices],
                    'returns': returns[mini_batch_indices],
                    'advantages': advantages[mini_batch_indices],
                    'old_values': batch['values'][mini_batch_indices]
                }
                
                # Update networks
                metrics = self._update_networks(mini_batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    total_metrics[key] += value
                num_batches += 1
                
                # Early stopping if KL divergence gets too large
                if metrics['approx_kl'] > 0.01:  # KL threshold
                    break
            
            # Early stopping if KL divergence gets too large
            if num_batches > 0 and total_metrics['approx_kl'] / num_batches > 0.01:
                break
        
        # Average metrics over all mini-batches
        if num_batches > 0:
            for key in total_metrics:
                total_metrics[key] /= num_batches
        
        # Clear buffer for next round of collection
        self.buffer.clear()
        
        # Update training step
        self.training_step += 1
        
        # Log metrics
        for key, value in total_metrics.items():
            self.training_metrics[key].append(value)
        
        return total_metrics
    
    def _update_networks(self, mini_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update actor and critic networks on mini-batch."""
        states = mini_batch['states']
        actions = mini_batch['actions']
        old_log_probs = mini_batch['old_log_probs']
        returns = mini_batch['returns']
        advantages = mini_batch['advantages']
        old_values = mini_batch['old_values']
        
        # Get current policy outputs
        mean, log_std = self.actor(states)
        std = torch.exp(log_std)
        
        # Create distribution and compute log probabilities
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1).mean()
        
        # Get current values
        values = self.critic(states)
        
        # Compute policy loss
        policy_loss = self.compute_policy_loss(
            new_log_probs, old_log_probs, advantages
        )
        
        # Compute value loss  
        value_loss = self.compute_value_loss(values, returns, old_values)
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Update actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Compute additional metrics
        with torch.no_grad():
            # Approximate KL divergence
            approx_kl = (old_log_probs - new_log_probs).mean()
            
            # Clipping fraction
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.abs(ratio - 1.0) > self.clip_ratio
            clip_fraction = clipped.float().mean()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item()
        }
    
    def compute_policy_loss(
        self, 
        new_log_probs: torch.Tensor, 
        old_log_probs: torch.Tensor, 
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PPO policy loss with clipping.
        
        Args:
            new_log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
        
        Returns:
            policy_loss: PPO clipped policy loss
        """
        # Compute importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # Take minimum (conservative policy update)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_value_loss(
        self, 
        values: torch.Tensor, 
        returns: torch.Tensor,
        old_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            values: Current value estimates
            returns: Target returns
            old_values: Previous value estimates
        
        Returns:
            value_loss: Value function loss
        """
        # Option 1: Simple MSE loss
        value_loss_simple = F.mse_loss(values.squeeze(), returns)
        
        # Option 2: Clipped value loss (similar to policy clipping)
        value_clipped = old_values + torch.clamp(
            values.squeeze() - old_values, 
            -self.clip_ratio, 
            self.clip_ratio
        )
        value_loss_clipped = F.mse_loss(value_clipped, returns)
        
        # Use maximum of clipped and unclipped (conservative update)
        value_loss = torch.max(value_loss_simple, value_loss_clipped)
        
        return value_loss
    
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config,
            'training_metrics': self.training_metrics
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.training_metrics = checkpoint.get('training_metrics', {
            'policy_loss': [], 'value_loss': [], 'entropy': [], 
            'approx_kl': [], 'clip_fraction': []
        })
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        base_info = super().get_training_info()
        
        ppo_info = {
            'clip_ratio': self.clip_ratio,
            'ppo_epochs': self.ppo_epochs,
            'mini_batch_size': self.mini_batch_size,
            'buffer_size': len(self.buffer),
            'recent_policy_loss': self.training_metrics['policy_loss'][-10:] if self.training_metrics['policy_loss'] else [],
            'recent_value_loss': self.training_metrics['value_loss'][-10:] if self.training_metrics['value_loss'] else [],
            'recent_entropy': self.training_metrics['entropy'][-10:] if self.training_metrics['entropy'] else []
        }
        
        return {**base_info, **ppo_info}
    
    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
    
    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()


def create_ppo_config(**kwargs) -> PIDAgentConfig:
    """
    Create PIDAgentConfig with PPO-specific default values.
    
    Args:
        **kwargs: Override default values
    
    Returns:
        config: PIDAgentConfig with PPO defaults
    """
    ppo_defaults = {
        # PPO-specific hyperparameters
        'clip_ratio': 0.2,
        'ppo_epochs': 10,
        'mini_batch_size': 64,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'buffer_capacity': 10000,
        'adam_eps': 1e-5,
        
        # General RL hyperparameters
        'learning_rate': 3e-4,
        'batch_size': 256,  # Not used directly in PPO, but kept for compatibility
        'gamma': 0.99,
        
        # Network architecture
        'hidden_dims': [128, 128, 64],
        'dropout_rate': 0.1,
        
        # PID parameter ranges
        'kp_range': (0.1, 10.0),
        'ki_range': (0.01, 5.0),
        'kd_range': (0.001, 2.0),
        
        # System
        'device': 'cpu',
        'seed': None
    }
    
    # Override with provided kwargs
    final_config = {**ppo_defaults, **kwargs}
    
    return PIDAgentConfig(**final_config)


def test_ppo_agent():
    """Test PPO agent creation and basic functionality."""
    print("Testing PPO Agent...")
    
    # Create config
    config = create_ppo_config(
        learning_rate=1e-3,
        hidden_dims=[64, 64],
        device='cpu'
    )
    
    # Create agent
    agent = PPOAgent(config)
    
    # Test action selection
    sample_state = np.array([50.0, 50.0, 0.0, 0.0, 0.0, 0.0])  # Perfect state
    action = agent.select_action(sample_state, training=True)
    
    print(f"Agent created successfully")
    print(f"Sample action: {action}")
    print(f"Action shape: {action.shape}")
    print(f"PID ranges: Kp{config.kp_range}, Ki{config.ki_range}, Kd{config.kd_range}")
    
    # Test experience storage
    agent.store_experience(
        state=sample_state,
        action=action,
        reward=1.0,
        next_state=sample_state,
        done=False
    )
    
    # Finish episode
    agent.store_experience(
        state=sample_state,
        action=action,
        reward=1.0,
        next_state=sample_state,
        done=True
    )
    
    print(f"Buffer size after episode: {len(agent.buffer)}")
    
    # Test update (with minimal data)
    if len(agent.buffer) > 0:
        metrics = agent.update()
        print(f"Update metrics: {list(metrics.keys())}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        agent.save(f.name)
        print(f"Agent saved to {f.name}")
        
        # Create new agent and load
        new_agent = PPOAgent(config)
        new_agent.load(f.name)
        print("Agent loaded successfully")
    
    print("PPO Agent test completed!")


if __name__ == "__main__":
    test_ppo_agent()