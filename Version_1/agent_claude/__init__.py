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

# Experience replay and learning components
from .buffer import (
    SimpleReplayBuffer,
    PriorityReplayBuffer,
    EpisodeBuffer,
    PIDSpecificBuffer,
    create_buffer,
    Experience,
    PolicyExperience,
    PIDExperience
)

# Policy components
from .policy import (
    DeterministicPolicy,
    StochasticPolicy,
    EpsilonGreedyPolicy,
    BetaPolicy,
    PIDPolicyEvaluator,
    ExplorationStrategy,
    create_policy
)

# Transfer learning components
from .transfer_learning import (
    TransferLearningManager,
    ProcessCharacteristics,
    TransferLearningMetrics,
    TransferMethod,
    ProcessSimilarityAnalyzer,
    create_transfer_learner
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
    'StochasticActor',
    
    # Buffer components
    'SimpleReplayBuffer',
    'PriorityReplayBuffer',
    'EpisodeBuffer', 
    'PIDSpecificBuffer',
    'create_buffer',
    'Experience',
    'PolicyExperience',
    'PIDExperience',
    
    # Policy components
    'DeterministicPolicy',
    'StochasticPolicy',
    'EpsilonGreedyPolicy',
    'BetaPolicy',
    'PIDPolicyEvaluator',
    'ExplorationStrategy',
    'create_policy',
    
    # Transfer learning components
    'TransferLearningManager',
    'ProcessCharacteristics',
    'TransferLearningMetrics',
    'TransferMethod',
    'ProcessSimilarityAnalyzer',
    'create_transfer_learner'
]