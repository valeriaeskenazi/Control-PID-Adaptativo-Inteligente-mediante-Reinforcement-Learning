"""
Q-network architectures for value-based reinforcement learning methods.

This module provides Q-networks for DQN, DDQN, Dueling DQN and other
value-based algorithms adapted for PID parameter tuning.

For continuous PID parameters, we use discretization strategies or
function approximation approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import itertools

from .base_networks import FeatureExtractor, build_mlp, DuelingHead


class DiscretePIDQNetwork(nn.Module):
    """
    Q-Network for discrete PID parameter selection.
    
    Discretizes the continuous PID parameter space into a finite
    set of combinations for standard DQN training.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        kp_values: List[float] = None,
        ki_values: List[float] = None,
        kd_values: List[float] = None,
        dropout_rate: float = 0.0
    ):
        """
        Initialize discrete PID Q-network.
        
        Args:
            input_dim: State dimension (6 for PID environment)
            hidden_dims: Hidden layer dimensions
            kp_values: Discrete Kp values to choose from
            ki_values: Discrete Ki values to choose from  
            kd_values: Discrete Kd values to choose from
            dropout_rate: Dropout probability
        """
        super(DiscretePIDQNetwork, self).__init__()
        
        # Default discrete values if not provided
        if kp_values is None:
            kp_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        if ki_values is None:
            ki_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        if kd_values is None:
            kd_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        
        self.kp_values = kp_values
        self.ki_values = ki_values
        self.kd_values = kd_values
        
        # Generate all combinations of PID parameters
        self.action_combinations = list(itertools.product(kp_values, ki_values, kd_values))
        self.num_actions = len(self.action_combinations)
        
        print(f"Discrete PID Q-Network: {self.num_actions} action combinations")
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False
        )
        
        # Q-value head
        self.q_head = nn.Linear(self.feature_extractor.output_dim, self.num_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values for all discrete actions.
        
        Args:
            state: Process state [batch_size, 6]
        
        Returns:
            q_values: Q-values for all actions [batch_size, num_actions]
        """
        features = self.feature_extractor(state)
        q_values = self.q_head(features)
        return q_values
    
    def get_action_from_index(self, action_index: int) -> np.ndarray:
        """Convert action index to PID parameters."""
        if action_index >= self.num_actions:
            action_index = self.num_actions - 1
        return np.array(self.action_combinations[action_index], dtype=np.float32)
    
    def get_index_from_action(self, action: np.ndarray) -> int:
        """Find closest action index for given PID parameters."""
        action_tuple = tuple(action)
        
        # Find closest combination
        min_distance = float('inf')
        best_index = 0
        
        for i, combo in enumerate(self.action_combinations):
            distance = sum((a - c) ** 2 for a, c in zip(action_tuple, combo))
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        return best_index
    
    def get_all_actions(self) -> torch.Tensor:
        """Get all possible actions as tensor."""
        return torch.FloatTensor(self.action_combinations)


class DuelingPIDQNetwork(nn.Module):
    """
    Dueling Q-Network for PID parameter selection.
    
    Separates state value V(s) and advantage A(s,a) for better
    learning in environments where actions don't always matter.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        kp_values: List[float] = None,
        ki_values: List[float] = None,
        kd_values: List[float] = None,
        dropout_rate: float = 0.0
    ):
        super(DuelingPIDQNetwork, self).__init__()
        
        # Same discretization as regular Q-network
        if kp_values is None:
            kp_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        if ki_values is None:
            ki_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        if kd_values is None:
            kd_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        
        self.kp_values = kp_values
        self.ki_values = ki_values
        self.kd_values = kd_values
        self.action_combinations = list(itertools.product(kp_values, ki_values, kd_values))
        self.num_actions = len(self.action_combinations)
        
        # Shared feature extraction
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=False
        )
        
        # Dueling head
        self.dueling_head = DuelingHead(
            input_dim=self.feature_extractor.output_dim,
            action_dim=self.num_actions
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling network."""
        features = self.feature_extractor(state)
        q_values = self.dueling_head(features)
        return q_values
    
    def get_action_from_index(self, action_index: int) -> np.ndarray:
        """Convert action index to PID parameters."""
        if action_index >= self.num_actions:
            action_index = self.num_actions - 1
        return np.array(self.action_combinations[action_index], dtype=np.float32)
    
    def get_index_from_action(self, action: np.ndarray) -> int:
        """Find closest action index for given PID parameters."""
        action_tuple = tuple(action)
        
        min_distance = float('inf')
        best_index = 0
        
        for i, combo in enumerate(self.action_combinations):
            distance = sum((a - c) ** 2 for a, c in zip(action_tuple, combo))
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        return best_index


class ContinuousQNetwork(nn.Module):
    """
    Q-Network for continuous PID parameters using function approximation.
    
    Takes both state and action as input and outputs Q(s,a).
    More suitable for continuous control but requires different training.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        hidden_dims: List[int] = [128, 128, 64],
        dropout_rate: float = 0.0
    ):
        super(ContinuousQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State processing branch
        self.state_processor = FeatureExtractor(
            input_dim=state_dim,
            hidden_dims=hidden_dims[:-1],  # All but last layer
            dropout_rate=dropout_rate,
            use_batch_norm=False
        )
        
        # Combined state-action processing
        combined_input_dim = self.state_processor.output_dim + action_dim
        
        self.q_network = build_mlp(
            input_dim=combined_input_dim,
            output_dim=1,
            hidden_dims=[hidden_dims[-1]],  # Last layer
            dropout_rate=dropout_rate,
            use_batch_norm=False
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for continuous Q-network.
        
        Args:
            state: Process state [batch_size, state_dim]
            action: PID parameters [batch_size, action_dim]
        
        Returns:
            q_value: Q-value for state-action pair [batch_size, 1]
        """
        # Process state
        state_features = self.state_processor(state)
        
        # Concatenate state features and action
        combined = torch.cat([state_features, action], dim=1)
        
        # Compute Q-value
        q_value = self.q_network(combined)
        
        return q_value


class PIDActionDiscretizer:
    """
    Utility class for handling PID parameter discretization.
    
    Provides different discretization strategies and conversion utilities.
    """
    
    def __init__(
        self,
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0),
        kd_range: Tuple[float, float] = (0.001, 2.0),
        discretization_levels: Tuple[int, int, int] = (6, 6, 6),
        discretization_type: str = 'linear'
    ):
        """
        Initialize action discretizer.
        
        Args:
            kp_range, ki_range, kd_range: Parameter ranges
            discretization_levels: Number of discrete values per parameter
            discretization_type: 'linear' or 'log' spacing
        """
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        self.discretization_levels = discretization_levels
        self.discretization_type = discretization_type
        
        # Generate discrete values
        self.kp_values = self._generate_values(kp_range, discretization_levels[0])
        self.ki_values = self._generate_values(ki_range, discretization_levels[1])
        self.kd_values = self._generate_values(kd_range, discretization_levels[2])
        
        # Generate all combinations
        self.action_combinations = list(itertools.product(
            self.kp_values, self.ki_values, self.kd_values
        ))
        self.num_actions = len(self.action_combinations)
    
    def _generate_values(self, value_range: Tuple[float, float], num_levels: int) -> List[float]:
        """Generate discrete values within range."""
        min_val, max_val = value_range
        
        if self.discretization_type == 'linear':
            values = np.linspace(min_val, max_val, num_levels)
        elif self.discretization_type == 'log':
            # Use log spacing (better for PID parameters with wide ranges)
            log_min = np.log10(max(min_val, 1e-6))  # Avoid log(0)
            log_max = np.log10(max_val)
            values = np.logspace(log_min, log_max, num_levels)
        else:
            raise ValueError(f"Unknown discretization type: {self.discretization_type}")
        
        return values.tolist()
    
    def get_action_from_index(self, action_index: int) -> np.ndarray:
        """Convert action index to PID parameters."""
        if action_index >= self.num_actions:
            action_index = self.num_actions - 1
        return np.array(self.action_combinations[action_index], dtype=np.float32)
    
    def get_index_from_action(self, action: np.ndarray) -> int:
        """Find closest action index for given PID parameters."""
        min_distance = float('inf')
        best_index = 0
        
        for i, combo in enumerate(self.action_combinations):
            distance = sum((a - c) ** 2 for a, c in zip(action, combo))
            if distance < min_distance:
                min_distance = distance
                best_index = i
        
        return best_index
    
    def get_all_actions(self) -> np.ndarray:
        """Get all possible actions as numpy array."""
        return np.array(self.action_combinations, dtype=np.float32)
    
    def get_discretization_info(self) -> Dict[str, Any]:
        """Get information about the discretization."""
        return {
            'num_actions': self.num_actions,
            'kp_values': self.kp_values,
            'ki_values': self.ki_values,
            'kd_values': self.kd_values,
            'discretization_levels': self.discretization_levels,
            'discretization_type': self.discretization_type
        }


def create_q_network(
    network_type: str,
    input_dim: int = 6,
    hidden_dims: List[int] = [128, 128, 64],
    **kwargs
) -> nn.Module:
    """
    Factory function for creating Q-networks.
    
    Args:
        network_type: Type of Q-network ('discrete', 'dueling', 'continuous')
        input_dim: Input state dimension
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional arguments for specific network types
    
    Returns:
        q_network: Initialized Q-network
    """
    network_type = network_type.lower()
    
    if network_type == 'discrete':
        return DiscretePIDQNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    elif network_type == 'dueling':
        return DuelingPIDQNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    elif network_type == 'continuous':
        return ContinuousQNetwork(
            state_dim=input_dim,
            hidden_dims=hidden_dims,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown Q-network type: {network_type}")


def test_q_networks():
    """Test Q-network implementations."""
    print("Testing Q-Networks...")
    
    batch_size = 32
    state_dim = 6
    action_dim = 3
    
    # Test discrete Q-network
    discrete_q = create_q_network('discrete', input_dim=state_dim, hidden_dims=[64, 64])
    sample_state = torch.randn(batch_size, state_dim)
    
    q_values = discrete_q(sample_state)
    print(f"Discrete Q-network output shape: {q_values.shape}")
    print(f"Number of discrete actions: {discrete_q.num_actions}")
    
    # Test action conversion
    action_index = 0
    pid_params = discrete_q.get_action_from_index(action_index)
    recovered_index = discrete_q.get_index_from_action(pid_params)
    print(f"Action conversion test: {action_index} -> {pid_params} -> {recovered_index}")
    
    # Test dueling Q-network
    dueling_q = create_q_network('dueling', input_dim=state_dim, hidden_dims=[64, 64])
    dueling_q_values = dueling_q(sample_state)
    print(f"Dueling Q-network output shape: {dueling_q_values.shape}")
    
    # Test continuous Q-network
    continuous_q = create_q_network('continuous', input_dim=state_dim, hidden_dims=[64, 64])
    sample_actions = torch.randn(batch_size, action_dim)
    continuous_q_values = continuous_q(sample_state, sample_actions)
    print(f"Continuous Q-network output shape: {continuous_q_values.shape}")
    
    # Test action discretizer
    discretizer = PIDActionDiscretizer(
        discretization_levels=(4, 4, 4),
        discretization_type='log'
    )
    
    info = discretizer.get_discretization_info()
    print(f"Discretizer info: {info['num_actions']} actions")
    print(f"Kp values: {info['kp_values']}")
    
    print("All Q-network tests passed!")


if __name__ == "__main__":
    test_q_networks()