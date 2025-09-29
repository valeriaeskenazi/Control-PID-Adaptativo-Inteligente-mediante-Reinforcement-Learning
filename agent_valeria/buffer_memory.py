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

#Priority Replay Buffer Implementation, adapted from original paper
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tamaño del árbol binario completo
        self.data = np.zeros(capacity, dtype=object)  # Aquí guardamos las experiencias
        self.write = 0  # Puntero circular para escribir
        self.n_entries = 0  # Cuántas entradas tenemos

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # Índice del padre
        self.tree[parent] += change  # Sumar el cambio
        if parent != 0:  # Si no es la raíz, seguir propagando
            self._propagate(parent, change)    

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1 # Índice en las hojas del árbol
        self.data[self.write] = data # Guardar la experiencia
        self.update(idx, p) # Actualizar prioridad y propagar hacia arriba
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1     

    def update(self, idx, p):
        change = p - self.tree[idx] # Cuánto cambió la prioridad
        self.tree[idx] = p # Actualizar la hoja
        self._propagate(idx, change) # Propagar el cambio hacia la raíz

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
    

class PriorityReplayBuffer(AbstractReplayBuffer):
    def __init__(
        self, 
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        total_training_steps: int = 1000000,
        device: str = 'cpu'
    ):
        super().__init__(capacity, device)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / total_training_steps
        
        self.tree = SumTree(capacity)
        self.epsilon = 1e-6

    def add(self, experience: Experience, td_error: float = None) -> None:
        if td_error is None:
            priority = self.tree.max_priority if self.tree.n_entries > 0 else 1.0
        else:
            priority = abs(td_error) + self.epsilon
        
        self.tree.add(priority, experience)
        self.size = self.tree.n_entries

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            batch.append(experience)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        weights /= weights.max()
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'weights': weights_tensor,
            'indices': indices
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.tree.update(idx, priority)        