"""
Base neural network components for PID reinforcement learning.

This module provides reusable building blocks that can be composed
into different network architectures for various RL algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class PIDOutputLayer(nn.Module):
    """
    Output layer specifically designed for PID parameters.
    
    Ensures positive outputs within specified ranges for Kp, Ki, Kd.
    """
    
    def __init__(
        self,
        input_dim: int,
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0),
        kd_range: Tuple[float, float] = (0.001, 2.0)
    ):
        """
        Initialize PID output layer.
        
        Args:
            input_dim: Input feature dimension
            kp_range: (min, max) range for proportional gain
            ki_range: (min, max) range for integral gain
            kd_range: (min, max) range for derivative gain
        """
        super(PIDOutputLayer, self).__init__()
        
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        
        # Separate heads for each PID parameter
        self.kp_head = nn.Linear(input_dim, 1)
        self.ki_head = nn.Linear(input_dim, 1)
        self.kd_head = nn.Linear(input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small positive bias."""
        for head in [self.kp_head, self.ki_head, self.kd_head]:
            nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
            if head.bias is not None:
                nn.init.constant_(head.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate PID parameters.
        
        Args:
            x: Input features [batch_size, input_dim]
        
        Returns:
            pid_params: PID parameters [batch_size, 3] [Kp, Ki, Kd]
        """
        kp_raw = self.kp_head(x)
        ki_raw = self.ki_head(x)
        kd_raw = self.kd_head(x)
        
        # Apply activation and scaling
        kp = self._scale_output(F.softplus(kp_raw), self.kp_range)
        ki = self._scale_output(F.softplus(ki_raw), self.ki_range)
        kd = self._scale_output(F.softplus(kd_raw), self.kd_range)
        
        return torch.cat([kp, ki, kd], dim=1)
    
    def _scale_output(self, x: torch.Tensor, output_range: Tuple[float, float]) -> torch.Tensor:
        """Scale output to specified range."""
        min_val, max_val = output_range
        # Use tanh for bounded output, then scale to range
        scaled = torch.tanh(x) * (max_val - min_val) / 2 + (max_val + min_val) / 2
        return scaled


class FeatureExtractor(nn.Module):
    """
    Generic feature extraction network for process states.
    
    Can be used as backbone for different RL architectures.
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: List[int] = [128, 128, 64],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize feature extractor.
        
        Args:
            input_dim: Input dimension (6 for PID environment)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(FeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self.activation)
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        return self.layers(x)


class ValueHead(nn.Module):
    """Value function head for critic networks."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super(ValueHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class QValueHead(nn.Module):
    """Q-value head for value-based methods."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 32):
        super(QValueHead, self).__init__()
        
        # For continuous control, we typically concatenate state and action
        self.q_layers = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value for state-action pair.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
        
        Returns:
            q_value: Q-value [batch_size, 1]
        """
        x = torch.cat([state, action], dim=1)
        return self.q_layers(x)


class NoiseLayer(nn.Module):
    """
    Noise layer for exploration in continuous control.
    
    Useful for DDPG, TD3, and other deterministic policy methods.
    """
    
    def __init__(self, action_dim: int, noise_scale: float = 0.1):
        super(NoiseLayer, self).__init__()
        self.action_dim = action_dim
        self.noise_scale = noise_scale
    
    def forward(self, actions: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Add noise to actions during training."""
        if not training:
            return actions
        
        noise = torch.randn_like(actions) * self.noise_scale
        return actions + noise


class DuelingHead(nn.Module):
    """
    Dueling network head that separates state value and advantage.
    
    Useful for DQN variants. For continuous control, this would typically
    be used with action discretization.
    """
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 32):
        super(DuelingHead, self).__init__()
        
        # State value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values using dueling architecture.
        
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        """
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = 'relu',
    output_activation: str = 'linear',
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False
) -> nn.Module:
    """
    Build a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Hidden layer activation
        output_activation: Output layer activation
        dropout_rate: Dropout rate
        use_batch_norm: Whether to use batch normalization
    
    Returns:
        mlp: Sequential MLP module
    """
    # Choose activation functions
    activation_map = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'elu': nn.ELU(),
        'sigmoid': nn.Sigmoid(),
        'linear': nn.Identity()
    }
    
    act_fn = activation_map.get(activation, nn.ReLU())
    output_act_fn = activation_map.get(output_activation, nn.Identity())
    
    layers = []
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(act_fn)
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        prev_dim = hidden_dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    
    if output_activation != 'linear':
        layers.append(output_act_fn)
    
    mlp = nn.Sequential(*layers)
    
    # Initialize weights
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    return mlp


if __name__ == "__main__":
    # Test components
    batch_size = 16
    state_dim = 6
    action_dim = 3
    
    print("Testing base network components...")
    
    # Test feature extractor
    feature_extractor = FeatureExtractor(input_dim=state_dim, hidden_dims=[64, 32])
    sample_state = torch.randn(batch_size, state_dim)
    features = feature_extractor(sample_state)
    print(f"Feature extractor output shape: {features.shape}")
    
    # Test PID output layer
    pid_output = PIDOutputLayer(input_dim=features.shape[1])
    pid_params = pid_output(features)
    print(f"PID output shape: {pid_params.shape}")
    print(f"Sample PID params: {pid_params[0].detach().numpy()}")
    
    # Test value head
    value_head = ValueHead(input_dim=features.shape[1])
    values = value_head(features)
    print(f"Value head output shape: {values.shape}")
    
    # Test Q-value head
    q_head = QValueHead(input_dim=state_dim, action_dim=action_dim)
    sample_action = torch.randn(batch_size, action_dim)
    q_values = q_head(sample_state, sample_action)
    print(f"Q-value head output shape: {q_values.shape}")
    
    print("\nAll components working correctly!")