"""
Experience replay buffers for PID reinforcement learning agents.

This module provides different types of replay buffers suitable for
various RL algorithms and the specific needs of PID control learning.
"""

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


class EpisodeBuffer(AbstractReplayBuffer):
    """
    Episode-based buffer for policy gradient methods.
    
    Stores complete episodes and can compute returns and advantages.
    Suitable for PPO, A2C, and other on-policy methods.
    """
    
    def __init__(self, capacity: int = 10000, device: str = 'cpu'):
        super().__init__(capacity, device)
        self.episodes = []
        self.current_episode = []
    
    def add(self, experience: PolicyExperience) -> None:
        """Add experience to current episode."""
        self.current_episode.append(experience)
    
    def finish_episode(self) -> None:
        """Mark current episode as complete and start new one."""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            
            # Remove old episodes if over capacity
            while len(self.episodes) > self.capacity:
                self.episodes.pop(0)
            
            self.size = sum(len(episode) for episode in self.episodes)
    
    def sample(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """Return all experiences from all complete episodes."""
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode)
        
        if not all_experiences:
            return self._empty_batch()
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in all_experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in all_experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in all_experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in all_experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in all_experiences]).to(self.device)
        log_probs = torch.FloatTensor([e.log_prob for e in all_experiences]).to(self.device)
        values = torch.FloatTensor([e.value for e in all_experiences]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'log_probs': log_probs,
            'values': values
        }
    
    def compute_returns_and_advantages(
        self, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """
        Compute returns and GAE advantages for policy gradient methods.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Dictionary with returns and advantages
        """
        all_returns = []
        all_advantages = []
        
        for episode in self.episodes:
            if not episode:
                continue
            
            episode_rewards = [e.reward for e in episode]
            episode_values = [e.value for e in episode]
            episode_dones = [e.done for e in episode]
            
            # Compute returns
            returns = self._compute_returns(episode_rewards, gamma)
            
            # Compute GAE advantages
            advantages = self._compute_gae_advantages(
                episode_rewards, episode_values, episode_dones, gamma, gae_lambda
            )
            
            all_returns.extend(returns)
            all_advantages.extend(advantages)
        
        if not all_returns:
            return {'returns': torch.empty(0), 'advantages': torch.empty(0)}
        
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        advantages_tensor = torch.FloatTensor(all_advantages).to(self.device)
        
        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return {
            'returns': returns_tensor,
            'advantages': advantages_tensor
        }
    
    def _compute_returns(self, rewards: List[float], gamma: float) -> List[float]:
        """Compute discounted returns."""
        returns = []
        R = 0
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.append(R)
        return list(reversed(returns))
    
    def _compute_gae_advantages(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool],
        gamma: float, 
        gae_lambda: float
    ) -> List[float]:
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.append(gae)
        
        return list(reversed(advantages))
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        """Return empty batch tensors."""
        return {
            'states': torch.empty(0, 6).to(self.device),
            'actions': torch.empty(0, 3).to(self.device),
            'rewards': torch.empty(0).to(self.device),
            'next_states': torch.empty(0, 6).to(self.device),
            'dones': torch.empty(0, dtype=torch.bool).to(self.device),
            'log_probs': torch.empty(0).to(self.device),
            'values': torch.empty(0).to(self.device)
        }
    
    def clear(self) -> None:
        """Clear all episodes."""
        self.episodes.clear()
        self.current_episode.clear()
        self.size = 0


class PIDSpecificBuffer(AbstractReplayBuffer):
    """
    PID-specific replay buffer that stores process metadata.
    
    Useful for transfer learning and analyzing performance across
    different process types and difficulties.
    """
    
    def __init__(self, capacity: int = 100000, device: str = 'cpu'):
        super().__init__(capacity, device)
        self.buffer = deque(maxlen=capacity)
        self.process_stats = {
            'EASY': {'count': 0, 'avg_reward': 0.0},
            'MEDIUM': {'count': 0, 'avg_reward': 0.0},
            'DIFFICULT': {'count': 0, 'avg_reward': 0.0},
            'UNKNOWN': {'count': 0, 'avg_reward': 0.0}
        }
    
    def add(self, experience: PIDExperience) -> None:
        """Add PID experience with process metadata."""
        self.buffer.append(experience)
        self.size = len(self.buffer)
        
        # Update process statistics
        difficulty = experience.process_difficulty
        if difficulty in self.process_stats:
            stats = self.process_stats[difficulty]
            old_count = stats['count']
            old_avg = stats['avg_reward']
            
            stats['count'] += 1
            stats['avg_reward'] = (old_avg * old_count + experience.reward) / stats['count']
    
    def sample(self, batch_size: int, process_difficulty: str = None) -> Dict[str, torch.Tensor]:
        """
        Sample batch, optionally filtered by process difficulty.
        
        Args:
            batch_size: Number of experiences to sample
            process_difficulty: If specified, only sample from this difficulty
        
        Returns:
            Dictionary of tensors
        """
        if process_difficulty:
            # Filter by process difficulty
            filtered_buffer = [exp for exp in self.buffer 
                             if exp.process_difficulty == process_difficulty]
            if not filtered_buffer:
                return self._empty_batch()
            
            sample_pool = filtered_buffer
        else:
            sample_pool = list(self.buffer)
        
        if batch_size > len(sample_pool):
            batch_size = len(sample_pool)
        
        batch = random.sample(sample_pool, batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # PID-specific metadata
        difficulties = [e.process_difficulty for e in batch]
        setpoints = torch.FloatTensor([e.setpoint for e in batch]).to(self.device)
        pvs = torch.FloatTensor([e.pv for e in batch]).to(self.device)
        errors = torch.FloatTensor([e.error for e in batch]).to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'difficulties': difficulties,
            'setpoints': setpoints,
            'pvs': pvs,
            'errors': errors
        }
    
    def sample_balanced(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample balanced batch across process difficulties."""
        difficulties = ['EASY', 'MEDIUM', 'DIFFICULT', 'UNKNOWN']
        samples_per_difficulty = batch_size // len(difficulties)
        
        all_samples = []
        for difficulty in difficulties:
            difficulty_samples = [exp for exp in self.buffer 
                                if exp.process_difficulty == difficulty]
            if difficulty_samples:
                n_samples = min(samples_per_difficulty, len(difficulty_samples))
                samples = random.sample(difficulty_samples, n_samples)
                all_samples.extend(samples)
        
        # Fill remaining slots randomly
        remaining = batch_size - len(all_samples)
        if remaining > 0:
            other_samples = random.sample(list(self.buffer), 
                                        min(remaining, len(self.buffer)))
            all_samples.extend(other_samples)
        
        # Convert to same format as regular sample
        if not all_samples:
            return self._empty_batch()
        
        batch = all_samples[:batch_size]  # Ensure exact batch size
        
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
            'dones': dones,
            'difficulties': [e.process_difficulty for e in batch],
            'setpoints': torch.FloatTensor([e.setpoint for e in batch]).to(self.device),
            'pvs': torch.FloatTensor([e.pv for e in batch]).to(self.device),
            'errors': torch.FloatTensor([e.error for e in batch]).to(self.device)
        }
    
    def get_process_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about process performance."""
        return self.process_stats.copy()
    
    def _empty_batch(self) -> Dict[str, torch.Tensor]:
        """Return empty batch tensors."""
        return {
            'states': torch.empty(0, 6).to(self.device),
            'actions': torch.empty(0, 3).to(self.device),
            'rewards': torch.empty(0).to(self.device),
            'next_states': torch.empty(0, 6).to(self.device),
            'dones': torch.empty(0, dtype=torch.bool).to(self.device),
            'difficulties': [],
            'setpoints': torch.empty(0).to(self.device),
            'pvs': torch.empty(0).to(self.device),
            'errors': torch.empty(0).to(self.device)
        }
    
    def clear(self) -> None:
        """Clear buffer and reset statistics."""
        self.buffer.clear()
        self.size = 0
        for difficulty in self.process_stats:
            self.process_stats[difficulty] = {'count': 0, 'avg_reward': 0.0}


def create_buffer(
    buffer_type: str, 
    capacity: int = 100000, 
    device: str = 'cpu',
    **kwargs
) -> AbstractReplayBuffer:
    """
    Factory function to create replay buffers.
    
    Args:
        buffer_type: Type of buffer ('simple', 'priority', 'episode', 'pid')
        capacity: Buffer capacity
        device: Device to store tensors on
        **kwargs: Additional arguments for specific buffer types
    
    Returns:
        Initialized replay buffer
    """
    buffer_type = buffer_type.lower()
    
    if buffer_type == 'simple':
        return SimpleReplayBuffer(capacity, device)
    elif buffer_type == 'priority':
        return PriorityReplayBuffer(capacity, device=device, **kwargs)
    elif buffer_type == 'episode':
        return EpisodeBuffer(capacity, device)
    elif buffer_type == 'pid':
        return PIDSpecificBuffer(capacity, device)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


def test_buffers():
    """Test different buffer implementations."""
    print("Testing replay buffers...")
    
    # Test simple buffer
    simple_buffer = SimpleReplayBuffer(capacity=1000)
    
    for i in range(100):
        exp = Experience(
            state=np.random.randn(6),
            action=np.random.randn(3),
            reward=np.random.randn(),
            next_state=np.random.randn(6),
            done=np.random.random() < 0.1
        )
        simple_buffer.add(exp)
    
    batch = simple_buffer.sample(32)
    print(f"Simple buffer: {len(simple_buffer)} experiences, batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # Test episode buffer
    episode_buffer = EpisodeBuffer(capacity=100)
    
    for episode in range(5):
        for step in range(10):
            exp = PolicyExperience(
                state=np.random.randn(6),
                action=np.random.randn(3),
                reward=np.random.randn(),
                next_state=np.random.randn(6),
                done=(step == 9),
                log_prob=np.random.randn(),
                value=np.random.randn()
            )
            episode_buffer.add(exp)
        episode_buffer.finish_episode()
    
    batch = episode_buffer.sample()
    returns_and_advantages = episode_buffer.compute_returns_and_advantages()
    print(f"\nEpisode buffer: {len(episode_buffer)} experiences")
    print(f"Returns shape: {returns_and_advantages['returns'].shape}")
    print(f"Advantages shape: {returns_and_advantages['advantages'].shape}")
    
    # Test PID buffer
    pid_buffer = PIDSpecificBuffer(capacity=1000)
    
    difficulties = ['EASY', 'MEDIUM', 'DIFFICULT']
    for i in range(150):
        exp = PIDExperience(
            state=np.random.randn(6),
            action=np.random.randn(3),
            reward=np.random.randn(),
            next_state=np.random.randn(6),
            done=np.random.random() < 0.1,
            process_difficulty=np.random.choice(difficulties),
            setpoint=50.0 + np.random.randn() * 10,
            pv=50.0 + np.random.randn() * 15,
            error=np.random.randn() * 5
        )
        pid_buffer.add(exp)
    
    balanced_batch = pid_buffer.sample_balanced(32)
    stats = pid_buffer.get_process_stats()
    print(f"\nPID buffer: {len(pid_buffer)} experiences")
    print("Process statistics:")
    for difficulty, stat in stats.items():
        print(f"  {difficulty}: {stat['count']} samples, avg reward: {stat['avg_reward']:.3f}")
    
    print("\nAll buffer tests passed!")


if __name__ == "__main__":
    test_buffers()