"""
Agent module for PID controller tuning with reinforcement learning.

This module contains all agent-related components organized in a modular way:
- base_agent: Abstract base classes for different RL paradigms
- networks: Neural network building blocks and architectures
- algorithms: Specific RL algorithm implementations (PPO, DQN, SAC, etc.)

Design Philosophy:
- AbstractPIDAgent: Base interface all agents must implement
- Modular networks: Reusable components for different algorithms
- Algorithm-specific agents: PPOAgent, DQNAgent, SACAgent, etc.
"""

# Base agent classes
from .base_agent import (
    AbstractPIDAgent,
    AbstractPolicyGradientAgent,
    AbstractValueBasedAgent,
    AbstractActorCriticAgent,
    PIDAgentConfig,
    create_agent
)

# Network components
from .networks import (
    FeatureExtractor,
    PIDOutputLayer,
    ActorNetwork,
    CriticNetwork,
    SharedActorCritic,
    StochasticActor
)

__all__ = [
    # Base classes
    'AbstractPIDAgent',
    'AbstractPolicyGradientAgent',
    'AbstractValueBasedAgent', 
    'AbstractActorCriticAgent',
    'PIDAgentConfig',
    'create_agent',
    
    # Network components
    'FeatureExtractor',
    'PIDOutputLayer',
    'ActorNetwork',
    'CriticNetwork',
    'SharedActorCritic',
    'StochasticActor'
]