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
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 3,
        device: str = 'cpu',
        seed: Optional[int] = None,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Initialize value-based agent with epsilon-greedy parameters.
        
        Args:
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate  
            epsilon_decay: Decay factor per step
        """
        super().__init__(state_dim, action_dim, device, seed)
        
        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def update_epsilon(self) -> None:
        """
        Update epsilon using exponential decay.
        
        Common implementation for all value-based agents.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """
        Get current exploration epsilon for epsilon-greedy policy.
        
        Returns:
            epsilon: Current exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
            
        Note:
            - Decreases over time during training via update_epsilon()
            - Used for epsilon-greedy action selection
            - Typical range: starts at 1.0, decays to 0.01-0.1
        """
        return self.epsilon
    
    def reset_epsilon(self, epsilon_start: float = 1.0) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = epsilon_start
    
    @abstractmethod
    def compute_q_loss(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor) -> torch.Tensor:
        """Compute Q-learning loss."""
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