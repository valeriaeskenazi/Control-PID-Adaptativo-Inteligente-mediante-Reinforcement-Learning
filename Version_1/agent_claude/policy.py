"""
Policy definitions and utilities for PID reinforcement learning.

This module provides different policy implementations and utilities
for action selection, exploration strategies, and policy evaluation
specific to PID parameter tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from enum import Enum

from .networks.actor_critic import StochasticActor, ActorNetwork


class ExplorationStrategy(Enum):
    """Enumeration of exploration strategies."""
    EPSILON_GREEDY = "epsilon_greedy"
    GAUSSIAN_NOISE = "gaussian_noise"
    PARAMETER_NOISE = "parameter_noise"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    BETA_DISTRIBUTION = "beta_distribution"


class AbstractPolicy(ABC):
    """
    Abstract base class for all policies.
    
    Defines the interface for action selection and policy updates.
    """
    
    def __init__(self, action_dim: int = 3, device: str = 'cpu'):
        self.action_dim = action_dim
        self.device = torch.device(device)
    
    @abstractmethod
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Select action given state."""
        pass
    
    @abstractmethod
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action given state."""
        pass
    
    def preprocess_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess state for policy input."""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        state = state.to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        return state


class DeterministicPolicy(AbstractPolicy):
    """
    Deterministic policy for DDPG-style algorithms.
    
    Uses neural network to directly output PID parameters with
    optional noise for exploration.
    """
    
    def __init__(
        self,
        network: ActorNetwork,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.GAUSSIAN_NOISE,
        noise_scale: float = 0.1,
        noise_clip: float = 0.5,
        device: str = 'cpu'
    ):
        super().__init__(action_dim=3, device=device)
        self.network = network.to(self.device)
        self.exploration_strategy = exploration_strategy
        self.noise_scale = noise_scale
        self.noise_clip = noise_clip
        
        # Initialize noise processes
        if exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK:
            self.ou_noise = OrnsteinUhlenbeckNoise(
                size=3, mu=0.0, theta=0.15, sigma=0.2
            )
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Select deterministic action with optional exploration noise."""
        state = self.preprocess_state(state)
        
        with torch.no_grad():
            action = self.network(state)
        
        if training and self.exploration_strategy != ExplorationStrategy.EPSILON_GREEDY:
            action = self._add_exploration_noise(action)
        
        return action.squeeze(0) if action.shape[0] == 1 else action
    
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        For deterministic policies, log probability is not well-defined.
        Return zeros for compatibility.
        """
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        return torch.zeros(batch_size, 1).to(self.device)
    
    def _add_exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
        """Add exploration noise to deterministic actions."""
        if self.exploration_strategy == ExplorationStrategy.GAUSSIAN_NOISE:
            noise = torch.randn_like(action) * self.noise_scale
            if self.noise_clip > 0:
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            return action + noise
        
        elif self.exploration_strategy == ExplorationStrategy.ORNSTEIN_UHLENBECK:
            noise = torch.FloatTensor(self.ou_noise()).to(self.device)
            if len(action.shape) > 1:
                noise = noise.expand_as(action)
            return action + noise
        
        else:
            return action


class StochasticPolicy(AbstractPolicy):
    """
    Stochastic policy for PPO-style algorithms.
    
    Uses neural network to output distribution parameters and
    samples actions from the distribution.
    """
    
    def __init__(
        self,
        network: StochasticActor,
        action_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        device: str = 'cpu'
    ):
        super().__init__(action_dim=3, device=device)
        self.network = network.to(self.device)
        
        # Action bounds for clipping
        if action_bounds is not None:
            self.action_min, self.action_max = action_bounds
            self.action_min = self.action_min.to(self.device)
            self.action_max = self.action_max.to(self.device)
        else:
            self.action_min = None
            self.action_max = None
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Sample action from policy distribution."""
        state = self.preprocess_state(state)
        
        if training:
            # Sample from distribution
            action, log_prob = self.network.get_action_and_log_prob(state)
        else:
            # Use mean action for evaluation
            mean, _ = self.network(state)
            action = mean
        
        # Clip actions if bounds are specified
        if self.action_min is not None and self.action_max is not None:
            action = torch.clamp(action, self.action_min, self.action_max)
        
        return action.squeeze(0) if action.shape[0] == 1 else action
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and its log probability."""
        state = self.preprocess_state(state)
        action, log_prob = self.network.get_action_and_log_prob(state)
        
        # Clip actions if bounds are specified
        if self.action_min is not None and self.action_max is not None:
            action = torch.clamp(action, self.action_min, self.action_max)
        
        return action, log_prob
    
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of given action."""
        state = self.preprocess_state(state)
        
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        return self.network.get_log_prob(state, action)


class EpsilonGreedyPolicy(AbstractPolicy):
    """
    Epsilon-greedy policy for discrete action spaces.
    
    Note: For continuous PID parameters, this would require
    discretization or action selection from a predefined set.
    """
    
    def __init__(
        self,
        network: nn.Module,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        action_candidates: Optional[torch.Tensor] = None,
        device: str = 'cpu'
    ):
        super().__init__(action_dim=3, device=device)
        self.network = network.to(self.device)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Predefined PID parameter candidates for discrete action space
        if action_candidates is None:
            self.action_candidates = self._generate_pid_candidates()
        else:
            self.action_candidates = action_candidates.to(self.device)
    
    def _generate_pid_candidates(self) -> torch.Tensor:
        """Generate predefined PID parameter combinations."""
        kp_values = torch.linspace(0.1, 10.0, 10)
        ki_values = torch.linspace(0.01, 5.0, 10)
        kd_values = torch.linspace(0.001, 2.0, 10)
        
        candidates = []
        for kp in kp_values:
            for ki in ki_values:
                for kd in kd_values:
                    candidates.append([kp.item(), ki.item(), kd.item()])
        
        return torch.FloatTensor(candidates).to(self.device)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Select action using epsilon-greedy strategy."""
        if training and np.random.random() < self.epsilon:
            # Random action
            idx = np.random.randint(0, len(self.action_candidates))
            action = self.action_candidates[idx]
        else:
            # Greedy action based on Q-values
            state = self.preprocess_state(state)
            with torch.no_grad():
                q_values = self.network(state)  # Assuming network outputs Q-values
                best_idx = q_values.argmax().item()
                action = self.action_candidates[best_idx]
        
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action
    
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        For epsilon-greedy, compute log probability based on policy.
        This is approximate since the policy is not fully differentiable.
        """
        # Find closest action candidate
        distances = torch.norm(self.action_candidates - action.unsqueeze(0), dim=1)
        closest_idx = distances.argmin()
        
        # Compute probability: (1 - epsilon) for greedy + epsilon/|A| for random
        prob_random = self.epsilon / len(self.action_candidates)
        
        state = self.preprocess_state(state)
        with torch.no_grad():
            q_values = self.network(state)
            is_greedy = (q_values.argmax() == closest_idx).float()
        
        prob = prob_random + (1 - self.epsilon) * is_greedy
        log_prob = torch.log(prob + 1e-8)  # Add small epsilon to avoid log(0)
        
        return log_prob.unsqueeze(0)


class BetaPolicy(AbstractPolicy):
    """
    Beta distribution policy for bounded continuous actions.
    
    Particularly suitable for PID parameters since they are naturally
    bounded and the Beta distribution can model various shapes.
    """
    
    def __init__(
        self,
        network: nn.Module,
        action_bounds: Tuple[torch.Tensor, torch.Tensor],
        device: str = 'cpu'
    ):
        super().__init__(action_dim=3, device=device)
        self.network = network.to(self.device)
        self.action_min, self.action_max = action_bounds
        self.action_min = self.action_min.to(self.device)
        self.action_max = self.action_max.to(self.device)
        
        # Network should output alpha and beta parameters for each action dimension
        # So output_dim should be 6 (3 alphas + 3 betas)
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Sample action from Beta distribution."""
        state = self.preprocess_state(state)
        
        # Get distribution parameters
        params = self.network(state)  # [batch_size, 6]
        alphas = F.softplus(params[:, :3]) + 1.0  # Ensure > 1
        betas = F.softplus(params[:, 3:]) + 1.0   # Ensure > 1
        
        if training:
            # Sample from Beta distribution
            beta_dist = torch.distributions.Beta(alphas, betas)
            normalized_action = beta_dist.sample()
        else:
            # Use mode of Beta distribution: (alpha - 1) / (alpha + beta - 2)
            normalized_action = (alphas - 1) / (alphas + betas - 2)
            normalized_action = torch.clamp(normalized_action, 0.01, 0.99)
        
        # Scale to action bounds
        action = self.action_min + normalized_action * (self.action_max - self.action_min)
        
        return action.squeeze(0) if action.shape[0] == 1 else action
    
    def get_action_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action under Beta policy."""
        state = self.preprocess_state(state)
        
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Get distribution parameters
        params = self.network(state)
        alphas = F.softplus(params[:, :3]) + 1.0
        betas = F.softplus(params[:, 3:]) + 1.0
        
        # Normalize action to [0, 1]
        normalized_action = (action - self.action_min) / (self.action_max - self.action_min)
        normalized_action = torch.clamp(normalized_action, 1e-6, 1 - 1e-6)
        
        # Compute log probability
        beta_dist = torch.distributions.Beta(alphas, betas)
        log_prob = beta_dist.log_prob(normalized_action).sum(dim=1, keepdim=True)
        
        return log_prob


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck noise process for exploration.
    
    Provides temporally correlated noise that's often more effective
    than uncorrelated Gaussian noise for continuous control.
    """
    
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
    
    def reset(self):
        """Reset noise to mean."""
        self.state = np.ones(self.size) * self.mu
    
    def __call__(self) -> np.ndarray:
        """Generate next noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class PIDPolicyEvaluator:
    """
    Utility class for evaluating PID policies across different processes.
    
    Useful for transfer learning analysis and policy performance assessment.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.evaluation_results = {}
    
    def evaluate_policy(
        self,
        policy: AbstractPolicy,
        env,
        num_episodes: int = 10,
        process_difficulty: str = None
    ) -> Dict[str, float]:
        """
        Evaluate policy performance on given environment.
        
        Args:
            policy: Policy to evaluate
            env: Environment instance
            num_episodes: Number of episodes to run
            process_difficulty: Specific process difficulty to test
        
        Returns:
            Dictionary of evaluation metrics
        """
        policy.network.eval()
        
        episode_rewards = []
        episode_lengths = []
        final_errors = []
        settling_times = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if process_difficulty:
                # Set specific process difficulty if supported
                env.process_difficulty = process_difficulty
            
            total_reward = 0
            step_count = 0
            settled = False
            settling_time = None
            
            while True:
                with torch.no_grad():
                    action = policy.select_action(torch.FloatTensor(state), training=False)
                
                next_state, reward, done, truncated, info = env.step(action.cpu().numpy())
                
                total_reward += reward
                step_count += 1
                
                # Check if system has settled (error within dead band)
                if not settled and abs(info.get('error', float('inf'))) <= env.dead_band:
                    settled = True
                    settling_time = step_count
                
                if done or truncated:
                    break
                
                state = next_state
            
            episode_rewards.append(total_reward)
            episode_lengths.append(step_count)
            final_errors.append(abs(info.get('error', 0)))
            
            if settling_time is not None:
                settling_times.append(settling_time)
        
        # Compute evaluation metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_final_error': np.mean(final_errors),
            'success_rate': np.mean([err <= env.dead_band for err in final_errors]),
            'mean_settling_time': np.mean(settling_times) if settling_times else float('inf'),
            'settling_success_rate': len(settling_times) / num_episodes
        }
        
        # Store results
        key = f"{process_difficulty or 'default'}_{num_episodes}ep"
        self.evaluation_results[key] = metrics
        
        policy.network.train()
        return metrics
    
    def compare_policies(
        self,
        policies: Dict[str, AbstractPolicy],
        env,
        num_episodes: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple policies on the same environment."""
        comparison_results = {}
        
        for name, policy in policies.items():
            results = self.evaluate_policy(policy, env, num_episodes)
            comparison_results[name] = results
        
        return comparison_results
    
    def get_transfer_learning_metrics(
        self,
        source_results: Dict[str, float],
        target_results: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute transfer learning specific metrics."""
        transfer_metrics = {
            'reward_transfer_ratio': target_results['mean_reward'] / max(source_results['mean_reward'], 1e-6),
            'settling_time_improvement': (source_results.get('mean_settling_time', float('inf')) - 
                                        target_results.get('mean_settling_time', float('inf'))),
            'success_rate_improvement': target_results['success_rate'] - source_results['success_rate']
        }
        
        return transfer_metrics


def create_policy(
    policy_type: str,
    network: nn.Module,
    device: str = 'cpu',
    **kwargs
) -> AbstractPolicy:
    """
    Factory function to create policies.
    
    Args:
        policy_type: Type of policy ('deterministic', 'stochastic', 'epsilon_greedy', 'beta')
        network: Neural network for the policy
        device: Device to run on
        **kwargs: Additional arguments for specific policies
    
    Returns:
        Initialized policy
    """
    policy_type = policy_type.lower()
    
    if policy_type == 'deterministic':
        return DeterministicPolicy(network, device=device, **kwargs)
    elif policy_type == 'stochastic':
        return StochasticPolicy(network, device=device, **kwargs)
    elif policy_type == 'epsilon_greedy':
        return EpsilonGreedyPolicy(network, device=device, **kwargs)
    elif policy_type == 'beta':
        return BetaPolicy(network, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def test_policies():
    """Test different policy implementations."""
    print("Testing policies...")
    
    # Create dummy networks
    from .networks.actor_critic import ActorNetwork, StochasticActor
    
    # Test deterministic policy
    actor_net = ActorNetwork(input_dim=6)
    det_policy = DeterministicPolicy(actor_net)
    
    sample_state = torch.randn(6)
    action = det_policy.select_action(sample_state)
    print(f"Deterministic policy action shape: {action.shape}")
    print(f"Sample action: {action.detach().numpy()}")
    
    # Test stochastic policy
    stoch_net = StochasticActor(input_dim=6)
    stoch_policy = StochasticPolicy(stoch_net)
    
    action = stoch_policy.select_action(sample_state)
    action_train, log_prob = stoch_policy.get_action_and_log_prob(sample_state)
    print(f"\nStochastic policy action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    print("\nAll policy tests passed!")


if __name__ == "__main__":
    test_policies()