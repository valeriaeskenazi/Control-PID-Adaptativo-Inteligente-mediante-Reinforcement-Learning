"""
Deep Q-Network (DQN) Agent for PID controller tuning.

This implementation adapts DQN to continuous PID parameters through discretization
and uses all our modular components:
- DiscretePIDQNetwork or DuelingPIDQNetwork for Q-value estimation
- SimpleReplayBuffer or PriorityReplayBuffer for experience replay  
- EpsilonGreedyPolicy for exploration
- Target network for stable training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import random
from copy import deepcopy

from ..base_agent import AbstractValueBasedAgent, PIDAgentConfig
from ..networks.q_networks import DiscretePIDQNetwork, DuelingPIDQNetwork, PIDActionDiscretizer, create_q_network
from ..buffer import SimpleReplayBuffer, PriorityReplayBuffer, Experience, create_buffer
from ..policy import EpsilonGreedyPolicy


class DQNAgent(AbstractValueBasedAgent):
    """
    Deep Q-Network agent for PID parameter tuning.
    
    DQN characteristics for PID control:
    1. Discretizes continuous PID parameter space
    2. Uses experience replay for sample efficiency
    3. Target network for stable Q-learning
    4. Epsilon-greedy exploration
    5. Double DQN optional for reduced overestimation
    """
    
    def __init__(self, config: PIDAgentConfig):
        super().__init__(
            state_dim=getattr(config, 'state_dim', 6),
            action_dim=getattr(config, 'action_dim', 3),
            device=config.device,
            seed=config.seed
        )
        
        self.config = config
        
        # DQN-specific hyperparameters
        self.epsilon = getattr(config, 'epsilon', 1.0)
        self.epsilon_min = getattr(config, 'epsilon_min', 0.01)
        self.epsilon_decay = getattr(config, 'epsilon_decay', 0.995)
        self.target_update_freq = getattr(config, 'target_update_freq', 1000)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.double_dqn = getattr(config, 'double_dqn', True)
        self.dueling = getattr(config, 'dueling', False)
        
        # PID discretization parameters
        self.discretization_levels = getattr(config, 'discretization_levels', (6, 6, 6))
        self.discretization_type = getattr(config, 'discretization_type', 'log')
        
        # Create action discretizer
        self.action_discretizer = PIDActionDiscretizer(
            kp_range=config.kp_range,
            ki_range=config.ki_range,
            kd_range=config.kd_range,
            discretization_levels=self.discretization_levels,
            discretization_type=self.discretization_type
        )
        
        self.num_actions = self.action_discretizer.num_actions
        print(f"DQN Agent: {self.num_actions} discrete PID combinations")
        
        # Create Q-networks
        network_type = 'dueling' if self.dueling else 'discrete'
        
        self.q_network = create_q_network(
            network_type=network_type,
            input_dim=self.state_dim,
            hidden_dims=config.hidden_dims,
            kp_values=self.action_discretizer.kp_values,
            ki_values=self.action_discretizer.ki_values,
            kd_values=self.action_discretizer.kd_values,
            dropout_rate=0.0  # Disable dropout for stability
        ).to(self.device)
        
        # Create target network (frozen copy)
        self.target_network = create_q_network(
            network_type=network_type,
            input_dim=self.state_dim,
            hidden_dims=config.hidden_dims,
            kp_values=self.action_discretizer.kp_values,
            ki_values=self.action_discretizer.ki_values,
            kd_values=self.action_discretizer.kd_values,
            dropout_rate=0.0
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Create experience replay buffer
        buffer_type = getattr(config, 'buffer_type', 'simple')
        buffer_capacity = getattr(config, 'buffer_capacity', 100000)
        
        if buffer_type == 'priority':
            self.replay_buffer = PriorityReplayBuffer(
                capacity=buffer_capacity,
                alpha=getattr(config, 'priority_alpha', 0.6),
                beta=getattr(config, 'priority_beta', 0.4),
                device=self.device
            )
            self.use_priority_buffer = True
        else:
            self.replay_buffer = SimpleReplayBuffer(
                capacity=buffer_capacity,
                device=self.device
            )
            self.use_priority_buffer = False
        
        # Create epsilon-greedy policy
        # Note: We'll handle epsilon-greedy manually since we need discrete actions
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate,
            eps=getattr(config, 'adam_eps', 1e-5)
        )
        
        # Training metrics
        self.training_metrics = {
            'q_loss': [],
            'epsilon': [],
            'target_updates': 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"DQNAgent_{id(self)}")
        
        print(f"DQN Agent initialized:")
        print(f"  Discretization: {self.discretization_levels} ({self.discretization_type})")
        print(f"  Actions: {self.num_actions}")
        print(f"  Double DQN: {self.double_dqn}")
        print(f"  Dueling: {self.dueling}")
        print(f"  Buffer: {buffer_type} ({buffer_capacity})")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state [PV, setpoint, error, error_prev, error_integral, error_derivative]
            training: Whether in training mode (affects exploration)
        
        Returns:
            action: PID parameters [Kp, Ki, Kd]
        """
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            # Random action
            action_index = random.randint(0, self.num_actions - 1)
        else:
            # Greedy action based on Q-values
            state_tensor = self.preprocess_state(state)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax().item()
        
        # Convert action index to PID parameters
        pid_params = self.action_discretizer.get_action_from_index(action_index)
        
        # Store for experience collection
        self._last_action_index = action_index
        
        return pid_params
    
    def store_experience(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken (PID parameters)
            reward: Reward received
            next_state: Next state  
            done: Whether episode ended
        """
        experience = Experience(
            state=state,
            action=action,  # Store continuous action for compatibility
            reward=reward,
            next_state=next_state,
            done=done
        )
        
        if self.use_priority_buffer:
            # For priority buffer, we need to compute initial priority
            # Use TD error as priority (computed during training)
            self.replay_buffer.add(experience, priority=None)  # Will use max_priority
        else:
            self.replay_buffer.add(experience)
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Update DQN using experience replay.
        
        Args:
            batch_data: Optional batch data (DQN uses its own buffer)
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert continuous actions back to indices for Q-learning
        action_indices = []
        for action in batch['actions']:
            action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
            action_index = self.action_discretizer.get_index_from_action(action_np)
            action_indices.append(action_index)
        
        action_indices = torch.LongTensor(action_indices).to(self.device)
        
        # Compute Q-learning loss
        if self.use_priority_buffer:
            q_loss, td_errors = self.compute_q_loss_with_priorities(
                batch['states'], action_indices, batch['rewards'], 
                batch['next_states'], batch['dones'], batch['weights']
            )
            
            # Update priorities in buffer
            priorities = td_errors.detach().cpu().numpy()
            self.replay_buffer.update_priorities(batch['indices'], priorities)
        else:
            q_loss = self.compute_q_loss(
                batch['states'], action_indices, batch['rewards'],
                batch['next_states'], batch['dones']
            )
        
        # Update Q-network
        self.optimizer.zero_grad()
        q_loss.backward()
        
        # Gradient clipping
        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_grad_norm)
        
        self.optimizer.step()
        
        # Update target network periodically
        if (self.training_step + 1) % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.training_metrics['target_updates'] += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update training step
        self.training_step += 1
        
        # Store metrics
        metrics = {
            'q_loss': q_loss.item(),
            'epsilon': self.epsilon
        }
        
        self.training_metrics['q_loss'].append(metrics['q_loss'])
        self.training_metrics['epsilon'].append(metrics['epsilon'])
        
        return metrics
    
    def compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-learning loss.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of action indices [batch_size]
            rewards: Batch of rewards [batch_size]
            next_states: Batch of next states [batch_size, state_dim] 
            dones: Batch of done flags [batch_size]
        
        Returns:
            q_loss: Q-learning loss
        """
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values for computing targets
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_q_values_main = self.q_network(next_states)
                next_actions = next_q_values_main.argmax(dim=1)
                
                next_q_values_target = self.target_network(next_states)
                next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.max(dim=1)[0]
            
            # Compute target Q-values
            target_q = rewards + (self.gamma * next_q * (~dones))
        
        # Compute loss (Huber loss is more stable than MSE for Q-learning)
        q_loss = F.smooth_l1_loss(current_q, target_q)
        
        return q_loss
    
    def compute_q_loss_with_priorities(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-loss with importance sampling weights for priority replay."""
        batch_size = states.shape[0]
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values
        with torch.no_grad():
            if self.double_dqn:
                next_q_values_main = self.q_network(next_states)
                next_actions = next_q_values_main.argmax(dim=1)
                
                next_q_values_target = self.target_network(next_states)
                next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states)
                next_q = next_q_values.max(dim=1)[0]
            
            target_q = rewards + (self.gamma * next_q * (~dones))
        
        # Compute TD errors for priority updates
        td_errors = torch.abs(current_q - target_q)
        
        # Weighted loss (importance sampling)
        elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (weights * elementwise_loss).mean()
        
        return weighted_loss, td_errors
    
    def get_epsilon(self) -> float:
        """Get current exploration epsilon."""
        return self.epsilon
    
    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'config': self.config,
            'training_metrics': self.training_metrics,
            'action_discretizer': self.action_discretizer
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.training_metrics = checkpoint.get('training_metrics', {
            'q_loss': [], 'epsilon': [], 'target_updates': 0
        })
        
        if 'action_discretizer' in checkpoint:
            self.action_discretizer = checkpoint['action_discretizer']
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        base_info = super().get_training_info()
        
        dqn_info = {
            'epsilon': self.epsilon,
            'num_actions': self.num_actions,
            'buffer_size': len(self.replay_buffer),
            'target_updates': self.training_metrics['target_updates'],
            'double_dqn': self.double_dqn,
            'dueling': self.dueling,
            'discretization_info': self.action_discretizer.get_discretization_info(),
            'recent_q_loss': self.training_metrics['q_loss'][-10:] if self.training_metrics['q_loss'] else []
        }
        
        return {**base_info, **dqn_info}
    
    def set_eval_mode(self) -> None:
        """Set networks to evaluation mode."""
        self.q_network.eval()
    
    def set_train_mode(self) -> None:
        """Set networks to training mode."""
        self.q_network.train()


def create_dqn_config(**kwargs) -> PIDAgentConfig:
    """
    Create PIDAgentConfig with DQN-specific default values.
    
    Args:
        **kwargs: Override default values
    
    Returns:
        config: PIDAgentConfig with DQN defaults
    """
    dqn_defaults = {
        # DQN-specific hyperparameters
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000,
        'double_dqn': True,
        'dueling': False,
        
        # Discretization parameters
        'discretization_levels': (6, 6, 6),
        'discretization_type': 'log',
        
        # Buffer parameters
        'buffer_type': 'simple',  # or 'priority'
        'buffer_capacity': 100000,
        'priority_alpha': 0.6,
        'priority_beta': 0.4,
        
        # General RL hyperparameters
        'learning_rate': 1e-3,
        'batch_size': 64,
        'gamma': 0.99,
        'max_grad_norm': 1.0,
        'adam_eps': 1e-5,
        
        # Network architecture
        'hidden_dims': [128, 128, 64],
        
        # PID parameter ranges
        'kp_range': (0.1, 10.0),
        'ki_range': (0.01, 5.0),
        'kd_range': (0.001, 2.0),
        
        # System
        'device': 'cpu',
        'seed': None
    }
    
    # Override with provided kwargs
    final_config = {**dqn_defaults, **kwargs}
    
    return PIDAgentConfig(**final_config)


def test_dqn_agent():
    """Test DQN agent creation and basic functionality."""
    print("Testing DQN Agent...")
    
    # Create config
    config = create_dqn_config(
        learning_rate=1e-3,
        hidden_dims=[64, 64],
        discretization_levels=(4, 4, 4),  # Smaller for testing
        device='cpu',
        epsilon=0.5
    )
    
    # Create agent
    agent = DQNAgent(config)
    
    # Test action selection
    sample_state = np.array([50.0, 50.0, 0.0, 0.0, 0.0, 0.0])  # Perfect state
    
    # Test with exploration
    action_train = agent.select_action(sample_state, training=True)
    print(f"Training action: {action_train}")
    
    # Test without exploration
    agent.epsilon = 0.0  # Force greedy
    action_eval = agent.select_action(sample_state, training=False)
    print(f"Evaluation action: {action_eval}")
    
    print(f"Agent created successfully")
    print(f"Number of discrete actions: {agent.num_actions}")
    print(f"Current epsilon: {agent.epsilon}")
    
    # Test experience storage
    agent.store_experience(
        state=sample_state,
        action=action_train,
        reward=1.0,
        next_state=sample_state,
        done=False
    )
    
    print(f"Buffer size after storage: {len(agent.replay_buffer)}")
    
    # Add more experiences to test update
    for _ in range(100):
        random_state = np.random.randn(6)
        random_action = agent.select_action(random_state, training=True)
        agent.store_experience(
            state=random_state,
            action=random_action,
            reward=np.random.randn(),
            next_state=np.random.randn(6),
            done=np.random.random() < 0.1
        )
    
    print(f"Buffer size after random experiences: {len(agent.replay_buffer)}")
    
    # Test update
    if agent.replay_buffer.is_ready(agent.batch_size):
        metrics = agent.update()
        print(f"Update metrics: {list(metrics.keys())}")
        if metrics:
            print(f"Q-loss: {metrics['q_loss']:.4f}")
            print(f"Epsilon: {metrics['epsilon']:.4f}")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        agent.save(f.name)
        print(f"Agent saved to {f.name}")
        
        # Create new agent and load
        new_agent = DQNAgent(config)
        new_agent.load(f.name)
        print("Agent loaded successfully")
    
    print("DQN Agent test completed!")


if __name__ == "__main__":
    test_dqn_agent()