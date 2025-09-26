import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import deque, namedtuple
import random
from abc import ABC, abstractmethod


# Experience tuple for different algorithms
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])

# Extended experience for policy gradient methods
PolicyExperience = namedtuple('PolicyExperience', [
    'state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'
])

# PID-specific experience with process metadata
PIDExperience = namedtuple('PIDExperience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'process_difficulty', 'setpoint', 'pv', 'error'
])


class AbstractReplayBuffer(ABC):
    """
    Abstract base class for all replay buffers.
    
    Defines the interface that all buffers must implement.
    """
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        """
        Initialize buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 0
    
    @abstractmethod
    def add(self, experience: Union[Experience, PolicyExperience, PIDExperience]) -> None:
        """Add experience to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        pass
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size


class SimpleReplayBuffer(AbstractReplayBuffer):
    """
    Simple replay buffer for DQN-style algorithms.
    
    Stores experiences in a circular buffer and samples uniformly.
    """
    
    def __init__(self, capacity: int = 100000, device: str = 'cpu'):
        super().__init__(capacity, device)
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch of experiences."""
        if batch_size > self.size:
            batch_size = self.size
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.size = 0

class PriorityReplayBuffer(AbstractReplayBuffer):
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on TD-error priorities, useful for
    improving sample efficiency in value-based methods.
    """
    
    def __init__(
        self, 
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        device: str = 'cpu'
    ):
        super().__init__(capacity, device)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_beta = 1.0
        
        # Use sum tree for efficient priority sampling
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Experience, priority: float = None) -> None:
        """Add experience with priority."""
        if priority is None:
            priority = self.max_priority
        
        self.buffer.append(experience)
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
        self.size = len(self.buffer)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch based on priorities."""
        if batch_size > self.size:
            batch_size = self.size
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Sample experiences
        batch = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Update beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights_tensor,
            'indices': indices
        }
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority
            self.max_priority = max(self.max_priority, priority + 1e-6)
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.size = 0
        self.max_priority = 1.0        