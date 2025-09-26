"""
Reinforcement Learning algorithms for PID controller tuning.

This package contains specific algorithm implementations:
- PPOAgent: Proximal Policy Optimization
- DQNAgent: Deep Q-Network (planned)
- SACAgent: Soft Actor-Critic (planned)
- DDPGAgent: Deep Deterministic Policy Gradient (planned)

All agents inherit from the abstract base classes and use
the modular network and utility components.
"""

from .ppo_agent import PPOAgent, create_ppo_config
from .dqn_agent import DQNAgent, create_dqn_config

__all__ = [
    'PPOAgent',
    'create_ppo_config',
    'DQNAgent', 
    'create_dqn_config'
]