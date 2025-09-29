from typing import Dict, Any, Tuple, Optional
from abstract_agent import AbstractPIDAgent
import torch
import numpy as np


class PIDAgentConfig:
    """
    Configuration class for PID RL agents.
    
    Contains all hyperparameters and settings that can be shared
    across different agent implementations.
    """
    
    def __init__(
        self,
        # Network architecture
        hidden_dims: list = [128, 128, 64],
        dropout_rate: float = 0.1,
        
        # PID parameter ranges
        kp_range: Tuple[float, float] = (0.1, 10.0),
        ki_range: Tuple[float, float] = (0.01, 5.0), 
        kd_range: Tuple[float, float] = (0.001, 2.0),
        
        # Training parameters
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        
        # Device and reproducibility
        device: str = 'cpu',
        seed: Optional[int] = None,
        
        # Logging and saving
        log_interval: int = 100,
        save_interval: int = 1000,
        
        **kwargs
    ):
        """Initialize configuration with default values."""
        # Network
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # PID ranges
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        
        # Training
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        
        # System
        self.device = device
        self.seed = seed
        
        # Logging
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Store any additional algorithm-specific parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PIDAgentConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_agent(agent_type: str, config: PIDAgentConfig) -> AbstractPIDAgent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_type: Type of agent ('ppo', 'dqn', 'sac', etc.)
        config: Agent configuration
    
    Returns:
        agent: Initialized agent of specified type
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'ppo':
        from .algorithms.ppo_agent import PPOAgent
        return PPOAgent(config)
    elif agent_type == 'dqn':
        from .algorithms.dqn_agent import DQNAgent
        return DQNAgent(config)
    elif agent_type == 'sac':
        # from .algorithms.sac_agent import SACAgent
        # return SACAgent(config)
        raise NotImplementedError("SAC agent not yet implemented")
    elif agent_type == 'ddpg':
        # from .algorithms.ddpg_agent import DDPGAgent
        # return DDPGAgent(config)
        raise NotImplementedError("DDPG agent not yet implemented")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: ['ppo', 'dqn']")


if __name__ == "__main__":
    # Test configuration
    config = PIDAgentConfig(
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        batch_size=32,
        custom_param=42  # Example of algorithm-specific parameter
    )
    
    print("PID Agent Configuration:")
    print(f"Hidden dims: {config.hidden_dims}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Custom param: {config.custom_param}")
    print(f"PID ranges: Kp{config.kp_range}, Ki{config.ki_range}, Kd{config.kd_range}")