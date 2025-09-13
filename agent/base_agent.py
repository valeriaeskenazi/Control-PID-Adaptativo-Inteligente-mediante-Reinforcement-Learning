"""
Abstract base classes for PID controller reinforcement learning agents.

This module defines the interface that all RL agents must implement,
allowing for easy experimentation with different algorithms while
maintaining consistent behavior for PID parameter tuning.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np


class AbstractPIDAgent(ABC):
    """
    Abstract base class for PID tuning reinforcement learning agents.
    
    All RL agents (PPO, DQN, SAC, etc.) must inherit from this class
    and implement the required methods.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space (6 for PID environment)
            action_dim: Dimension of action space (3 for Kp, Ki, Kd)
            device: PyTorch device ('cpu' or 'cuda')
            seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        if seed is not None:
            self.set_seed(seed)
        
        self.training_step = 0
        self.episode_count = 0
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action (PID parameters) given current state.
        
        Args:
            state: Current state [PV, setpoint, error, error_prev, error_integral, error_derivative]
            training: Whether in training mode (affects exploration)
        
        Returns:
            action: PID parameters [Kp, Ki, Kd]
        """
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent parameters using a batch of experience.
        
        Args:
            batch_data: Dictionary containing experience batch
                       Must include: states, actions, rewards, next_states, dones
        
        Returns:
            metrics: Dictionary of training metrics (losses, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        pass
    
    def set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess state for neural network input.
        
        Args:
            state: Raw state from environment
        
        Returns:
            processed_state: Normalized tensor ready for network
        """
        # Convert to tensor and add batch dimension if needed
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        # Add batch dimension if single state
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        return state_tensor
    
    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Postprocess action from neural network output.
        
        Args:
            action: Raw network output
        
        Returns:
            processed_action: PID parameters ready for environment
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Remove batch dimension if single action
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        
        return action
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training information."""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'device': str(self.device)
        }


class AbstractPolicyGradientAgent(AbstractPIDAgent):
    """
    Abstract base class for policy gradient agents (PPO, A2C, REINFORCE, etc.).
    
    Extends AbstractPIDAgent with policy gradient specific methods.
    """
    
    @abstractmethod
    def compute_policy_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute policy gradient loss."""
        pass
    
    @abstractmethod
    def compute_value_loss(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        pass


class AbstractValueBasedAgent(AbstractPIDAgent):
    """
    Abstract base class for value-based agents (DQN, DDQN, etc.).
    
    Note: For continuous control, these agents typically use discretization
    or other techniques to handle continuous action spaces.
    """
    
    @abstractmethod
    def compute_q_loss(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor) -> torch.Tensor:
        """Compute Q-learning loss."""
        pass
    
    @abstractmethod
    def get_epsilon(self) -> float:
        """Get current exploration epsilon for epsilon-greedy."""
        pass


class AbstractActorCriticAgent(AbstractPIDAgent):
    """
    Abstract base class for actor-critic agents (DDPG, TD3, SAC, etc.).
    
    Extends AbstractPIDAgent with actor-critic specific methods.
    """
    
    @abstractmethod
    def compute_actor_loss(self, states: torch.Tensor) -> torch.Tensor:
        """Compute actor network loss."""
        pass
    
    @abstractmethod
    def compute_critic_loss(self, states: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        """Compute critic network loss.""" 
        pass


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
        from .algorithms.sac_agent import SACAgent
        return SACAgent(config)
    elif agent_type == 'ddpg':
        from .algorithms.ddpg_agent import DDPGAgent
        return DDPGAgent(config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


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