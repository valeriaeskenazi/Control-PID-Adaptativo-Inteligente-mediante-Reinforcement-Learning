"""
Neural network components for PID reinforcement learning.

This package provides modular network building blocks that can be
composed for different RL algorithms:

- base_networks: Core components (feature extractors, output layers, heads)
- actor_critic: Networks for policy gradient methods (PPO, A2C, A3C)
- q_networks: Networks for value-based methods (DQN, DDQN, Dueling DQN)
- policy_networks: Networks for actor-critic methods (DDPG, TD3, SAC)
"""

# Base components
from .base_networks import (
    FeatureExtractor,
    PIDOutputLayer,
    ValueHead,
    QValueHead,
    NoiseLayer,
    DuelingHead,
    build_mlp
)

# Actor-Critic networks
from .actor_critic import (
    ActorNetwork,
    CriticNetwork,
    SharedActorCritic,
    StochasticActor
)

# Q-networks
from .q_networks import (
    DiscretePIDQNetwork,
    DuelingPIDQNetwork,
    ContinuousQNetwork,
    PIDActionDiscretizer,
    create_q_network
)

__all__ = [
    # Base components
    'FeatureExtractor',
    'PIDOutputLayer', 
    'ValueHead',
    'QValueHead',
    'NoiseLayer',
    'DuelingHead',
    'build_mlp',
    
    # Actor-Critic
    'ActorNetwork',
    'CriticNetwork',
    'SharedActorCritic',
    'StochasticActor',
    
    # Q-Networks
    'DiscretePIDQNetwork',
    'DuelingPIDQNetwork', 
    'ContinuousQNetwork',
    'PIDActionDiscretizer',
    'create_q_network'
]