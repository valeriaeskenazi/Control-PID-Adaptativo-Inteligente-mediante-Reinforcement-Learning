"""
Actor-Critic network architectures for policy gradient methods.

Specialized networks for PPO, A2C, A3C and other policy gradient algorithms
that use actor-critic architecture for PID parameter tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

from .base_networks import FeatureExtractor, PIDOutputLayer, ValueHead


class ActorNetwork(nn.Module):
    """
    Actor network for policy gradient methods.
    
    Outputs PID parameters given process state.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0),
        kd_range: Tuple[float, float] = (0.001, 2.0),
        dropout_rate: float = 0.1
    ):
        """
        Initialize Actor Network.
        
        Args:
            input_dim: State dimension (6 for PID environment)
            hidden_dims: Hidden layer dimensions
            kp_range, ki_range, kd_range: PID parameter ranges
            dropout_rate: Dropout probability
        """
        super(ActorNetwork, self).__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False  # Disable batch norm to avoid single batch issues
        )
        
        self.pid_output = PIDOutputLayer(
            input_dim=self.feature_extractor.output_dim,
            kp_range=kp_range,
            ki_range=ki_range,
            kd_range=kd_range
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get PID parameters.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            pid_params: PID parameters [batch_size, 3]
        """
        features = self.feature_extractor(state)
        pid_params = self.pid_output(features)
        return pid_params


class CriticNetwork(nn.Module):
    """
    Critic network for value function estimation.
    
    Estimates state values for policy gradient methods.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        dropout_rate: float = 0.1
    ):
        """
        Initialize Critic Network.
        
        Args:
            input_dim: State dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(CriticNetwork, self).__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False  # Disable batch norm to avoid single batch issues
        )
        
        self.value_head = ValueHead(
            input_dim=self.feature_extractor.output_dim
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to estimate state value.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            value: State value estimate [batch_size, 1]
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value


class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic network for parameter efficiency.
    
    Uses shared feature extractor with separate actor and critic heads.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        shared_dims: List[int] = [128, 128],
        actor_dims: List[int] = [64],
        critic_dims: List[int] = [64],
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0),
        kd_range: Tuple[float, float] = (0.001, 2.0),
        dropout_rate: float = 0.1
    ):
        """
        Initialize shared Actor-Critic network.
        
        Args:
            input_dim: State dimension
            shared_dims: Shared feature extractor dimensions
            actor_dims: Actor-specific layer dimensions
            critic_dims: Critic-specific layer dimensions
            kp_range, ki_range, kd_range: PID parameter ranges
            dropout_rate: Dropout probability
        """
        super(SharedActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared_features = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=shared_dims,
            dropout_rate=dropout_rate
        )
        
        # Actor-specific layers
        if actor_dims:
            self.actor_features = FeatureExtractor(
                input_dim=self.shared_features.output_dim,
                hidden_dims=actor_dims,
                dropout_rate=dropout_rate
            )
            actor_input_dim = self.actor_features.output_dim
        else:
            self.actor_features = nn.Identity()
            actor_input_dim = self.shared_features.output_dim
        
        # Critic-specific layers
        if critic_dims:
            self.critic_features = FeatureExtractor(
                input_dim=self.shared_features.output_dim,
                hidden_dims=critic_dims,
                dropout_rate=dropout_rate
            )
            critic_input_dim = self.critic_features.output_dim
        else:
            self.critic_features = nn.Identity()
            critic_input_dim = self.shared_features.output_dim
        
        # Output heads
        self.pid_output = PIDOutputLayer(
            input_dim=actor_input_dim,
            kp_range=kp_range,
            ki_range=ki_range,
            kd_range=kd_range
        )
        
        self.value_head = ValueHead(input_dim=critic_input_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through shared network.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            pid_params: PID parameters [batch_size, 3]
            value: State value [batch_size, 1]
        """
        # Shared feature extraction
        shared_features = self.shared_features(state)
        
        # Actor forward pass
        actor_features = self.actor_features(shared_features)
        pid_params = self.pid_output(actor_features)
        
        # Critic forward pass
        critic_features = self.critic_features(shared_features)
        value = self.value_head(critic_features)
        
        return pid_params, value
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action from actor only."""
        shared_features = self.shared_features(state)
        actor_features = self.actor_features(shared_features)
        return self.pid_output(actor_features)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value from critic only."""
        shared_features = self.shared_features(state)
        critic_features = self.critic_features(shared_features)
        return self.value_head(critic_features)


class StochasticActor(nn.Module):
    """
    Stochastic actor network that outputs action distribution parameters.
    
    Useful for PPO and other stochastic policy gradient methods.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0),
        kd_range: Tuple[float, float] = (0.001, 2.0),
        dropout_rate: float = 0.1,
        min_log_std: float = -10.0,
        max_log_std: float = 2.0
    ):
        """
        Initialize stochastic actor.
        
        Args:
            input_dim: State dimension
            hidden_dims: Hidden layer dimensions
            kp_range, ki_range, kd_range: PID parameter ranges
            dropout_rate: Dropout probability
            min_log_std: Minimum log standard deviation
            max_log_std: Maximum log standard deviation
        """
        super(StochasticActor, self).__init__()
        
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False  # Disable batch norm to avoid single batch issues
        )
        
        # Mean (deterministic PID parameters)
        self.pid_mean = PIDOutputLayer(
            input_dim=self.feature_extractor.output_dim,
            kp_range=kp_range,
            ki_range=ki_range,
            kd_range=kd_range
        )
        
        # Log standard deviation (for exploration)
        self.log_std_head = nn.Linear(self.feature_extractor.output_dim, 3)
        
        # Initialize log_std with small values
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            mean: Mean PID parameters [batch_size, 3]
            log_std: Log standard deviation [batch_size, 3]
        """
        features = self.feature_extractor(state)
        
        mean = self.pid_mean(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            action: Sampled PID parameters [batch_size, 3]
            log_prob: Log probability [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Sample from normal distribution
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()  # Reparameterized sampling
        
        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given action.
        
        Args:
            state: Process state [batch_size, 6]
            action: PID parameters [batch_size, 3]
        
        Returns:
            log_prob: Log probability [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        
        return log_prob


def test_actor_critic_networks():
    """Test actor-critic network architectures."""
    batch_size = 16
    state_dim = 6
    
    # Sample input
    sample_state = torch.randn(batch_size, state_dim)
    
    print("Testing Actor-Critic Networks...")
    
    # Test separate actor and critic
    actor = ActorNetwork()
    critic = CriticNetwork()
    
    pid_params = actor(sample_state)
    values = critic(sample_state)
    
    print(f"Actor output shape: {pid_params.shape}")
    print(f"Critic output shape: {values.shape}")
    print(f"Sample PID params: {pid_params[0].detach().numpy()}")
    
    # Test shared network
    shared_ac = SharedActorCritic()
    pid_params_shared, values_shared = shared_ac(sample_state)
    
    print(f"Shared AC PID params shape: {pid_params_shared.shape}")
    print(f"Shared AC values shape: {values_shared.shape}")
    
    # Test stochastic actor
    stochastic_actor = StochasticActor()
    mean, log_std = stochastic_actor(sample_state)
    action, log_prob = stochastic_actor.get_action_and_log_prob(sample_state)
    
    print(f"Stochastic actor mean shape: {mean.shape}")
    print(f"Stochastic actor log_std shape: {log_std.shape}")
    print(f"Sampled action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    
    # Parameter counts
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    shared_params = sum(p.numel() for p in shared_ac.parameters())
    stochastic_params = sum(p.numel() for p in stochastic_actor.parameters())
    
    print(f"\nParameter counts:")
    print(f"Actor: {actor_params:,}")
    print(f"Critic: {critic_params:,}")
    print(f"Shared AC: {shared_params:,}")
    print(f"Stochastic Actor: {stochastic_params:,}")
    print(f"Efficiency (shared vs separate): {shared_params / (actor_params + critic_params):.2%}")


if __name__ == "__main__":
    test_actor_critic_networks()