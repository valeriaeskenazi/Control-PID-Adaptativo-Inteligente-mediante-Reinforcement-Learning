"""
Transfer learning utilities for PID reinforcement learning across different processes.

This module provides tools for transferring knowledge learned on one type of
industrial process to another, which is the core of the universal PID agent concept.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from copy import deepcopy
import os
from dataclasses import dataclass
from enum import Enum

from .base_agent import AbstractPIDAgent, PIDAgentConfig
from .buffer import PIDSpecificBuffer, AbstractReplayBuffer


class TransferMethod(Enum):
    """Transfer learning methods."""
    FINE_TUNING = "fine_tuning"
    FEATURE_EXTRACTION = "feature_extraction" 
    PROGRESSIVE_NETWORKS = "progressive_networks"
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    MULTI_TASK_LEARNING = "multi_task"


@dataclass
class ProcessCharacteristics:
    """Characteristics of an industrial process for transfer learning."""
    difficulty: str  # EASY, MEDIUM, DIFFICULT, UNKNOWN
    time_constant: float  # Process response time
    dead_time: float  # Process dead time
    nonlinearity: float  # Degree of nonlinearity (0-1)
    noise_level: float  # Measurement noise level
    disturbance_level: float  # External disturbance level
    operating_range: Tuple[float, float]  # (min, max) operating values
    typical_setpoint: float  # Typical setpoint value
    process_type: str  # 'level', 'temperature', 'pressure', 'flow', etc.


@dataclass
class TransferLearningMetrics:
    """Metrics for evaluating transfer learning performance."""
    source_performance: Dict[str, float]
    target_performance_scratch: Dict[str, float]  # Training from scratch
    target_performance_transfer: Dict[str, float]  # With transfer learning
    
    # Transfer-specific metrics
    convergence_speedup: float  # How much faster convergence was
    sample_efficiency: float  # Reduction in samples needed
    final_performance_ratio: float  # Transfer perf / scratch perf
    negative_transfer: bool  # Whether transfer hurt performance


class AbstractTransferLearner(ABC):
    """Abstract base class for transfer learning methods."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.source_agents: Dict[str, AbstractPIDAgent] = {}
        self.transfer_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def transfer_to_target(
        self,
        source_agent: AbstractPIDAgent,
        target_config: PIDAgentConfig,
        source_characteristics: ProcessCharacteristics,
        target_characteristics: ProcessCharacteristics
    ) -> AbstractPIDAgent:
        """Transfer knowledge from source to target domain."""
        pass
    
    @abstractmethod
    def evaluate_transfer(
        self,
        source_agent: AbstractPIDAgent,
        target_agent: AbstractPIDAgent,
        target_env,
        num_episodes: int = 100
    ) -> TransferLearningMetrics:
        """Evaluate transfer learning effectiveness."""
        pass


class FineTuningTransfer(AbstractTransferLearner):
    """
    Fine-tuning transfer learning.
    
    Copies source network weights and fine-tunes on target domain.
    Most common and straightforward transfer method.
    """
    
    def __init__(
        self,
        freeze_layers: List[str] = None,
        learning_rate_reduction: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__(device)
        self.freeze_layers = freeze_layers or []
        self.learning_rate_reduction = learning_rate_reduction
    
    def transfer_to_target(
        self,
        source_agent: AbstractPIDAgent,
        target_config: PIDAgentConfig,
        source_characteristics: ProcessCharacteristics,
        target_characteristics: ProcessCharacteristics
    ) -> AbstractPIDAgent:
        """Transfer via fine-tuning."""
        # Create target agent with same architecture
        from .base_agent import create_agent
        target_agent = create_agent(type(source_agent).__name__.lower().replace('agent', ''), target_config)
        
        # Copy source weights
        self._copy_network_weights(source_agent, target_agent)
        
        # Freeze specified layers
        self._freeze_layers(target_agent)
        
        # Reduce learning rate for fine-tuning
        self._adjust_learning_rates(target_agent)
        
        # Log transfer
        self.transfer_history.append({
            'method': 'fine_tuning',
            'source_difficulty': source_characteristics.difficulty,
            'target_difficulty': target_characteristics.difficulty,
            'source_type': source_characteristics.process_type,
            'target_type': target_characteristics.process_type,
            'frozen_layers': self.freeze_layers
        })
        
        return target_agent
    
    def _copy_network_weights(self, source_agent: AbstractPIDAgent, target_agent: AbstractPIDAgent):
        """Copy network weights from source to target."""
        # This is a simplified version - in practice, you'd need to handle
        # different agent types and network architectures more carefully
        
        source_state_dict = source_agent.actor.state_dict() if hasattr(source_agent, 'actor') else None
        target_state_dict = target_agent.actor.state_dict() if hasattr(target_agent, 'actor') else None
        
        if source_state_dict and target_state_dict:
            # Copy compatible layers
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            target_agent.actor.load_state_dict(target_state_dict)
        
        # Also copy critic if available
        if hasattr(source_agent, 'critic') and hasattr(target_agent, 'critic'):
            source_critic_dict = source_agent.critic.state_dict()
            target_critic_dict = target_agent.critic.state_dict()
            
            for name, param in source_critic_dict.items():
                if name in target_critic_dict and param.shape == target_critic_dict[name].shape:
                    target_critic_dict[name].copy_(param)
            
            target_agent.critic.load_state_dict(target_critic_dict)
    
    def _freeze_layers(self, agent: AbstractPIDAgent):
        """Freeze specified layers."""
        if hasattr(agent, 'actor'):
            for name, param in agent.actor.named_parameters():
                if any(layer in name for layer in self.freeze_layers):
                    param.requires_grad = False
    
    def _adjust_learning_rates(self, agent: AbstractPIDAgent):
        """Reduce learning rates for fine-tuning."""
        if hasattr(agent, 'actor_optimizer'):
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] *= self.learning_rate_reduction
        
        if hasattr(agent, 'critic_optimizer'):
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] *= self.learning_rate_reduction
    
    def evaluate_transfer(
        self,
        source_agent: AbstractPIDAgent,
        target_agent: AbstractPIDAgent,
        target_env,
        num_episodes: int = 100
    ) -> TransferLearningMetrics:
        """Evaluate fine-tuning transfer performance."""
        # This would need to be implemented based on specific evaluation protocol
        # For now, return placeholder metrics
        return TransferLearningMetrics(
            source_performance={'mean_reward': 0.0},
            target_performance_scratch={'mean_reward': 0.0},
            target_performance_transfer={'mean_reward': 0.0},
            convergence_speedup=1.0,
            sample_efficiency=1.0,
            final_performance_ratio=1.0,
            negative_transfer=False
        )


class ProgressiveNetworksTransfer(AbstractTransferLearner):
    """
    Progressive Networks for transfer learning.
    
    Adds new columns for each new task while keeping previous
    knowledge frozen. Good for avoiding catastrophic forgetting.
    """
    
    def __init__(self, device: str = 'cpu'):
        super().__init__(device)
        self.task_columns: List[nn.Module] = []
        self.lateral_connections: List[nn.Module] = []
    
    def transfer_to_target(
        self,
        source_agent: AbstractPIDAgent,
        target_config: PIDAgentConfig,
        source_characteristics: ProcessCharacteristics,
        target_characteristics: ProcessCharacteristics
    ) -> AbstractPIDAgent:
        """Transfer via progressive networks."""
        # Freeze source network
        self._freeze_source_network(source_agent)
        
        # Create new column for target task
        target_column = self._create_target_column(target_config)
        
        # Add lateral connections from source to target
        lateral_connections = self._create_lateral_connections(source_agent, target_column)
        
        # Create modified target agent
        target_agent = self._create_progressive_agent(
            source_agent, target_column, lateral_connections, target_config
        )
        
        return target_agent
    
    def _freeze_source_network(self, source_agent: AbstractPIDAgent):
        """Freeze all parameters in source network."""
        if hasattr(source_agent, 'actor'):
            for param in source_agent.actor.parameters():
                param.requires_grad = False
        if hasattr(source_agent, 'critic'):
            for param in source_agent.critic.parameters():
                param.requires_grad = False
    
    def _create_target_column(self, config: PIDAgentConfig) -> nn.Module:
        """Create new network column for target task."""
        # This would create a new network architecture
        # For simplicity, returning placeholder
        return nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def _create_lateral_connections(self, source_agent: AbstractPIDAgent, target_column: nn.Module) -> nn.ModuleList:
        """Create lateral connections from source to target."""
        # Simplified lateral connections
        return nn.ModuleList([
            nn.Linear(128, 128),  # Connection from source layer 1 to target layer 1
            nn.Linear(64, 64)     # Connection from source layer 2 to target layer 2
        ])
    
    def _create_progressive_agent(
        self,
        source_agent: AbstractPIDAgent,
        target_column: nn.Module,
        lateral_connections: nn.ModuleList,
        config: PIDAgentConfig
    ) -> AbstractPIDAgent:
        """Create agent with progressive network architecture."""
        # This is a complex implementation that would require
        # creating a new agent class with progressive network support
        # For now, return a modified version of the source agent
        return source_agent
    
    def evaluate_transfer(
        self,
        source_agent: AbstractPIDAgent,
        target_agent: AbstractPIDAgent,
        target_env,
        num_episodes: int = 100
    ) -> TransferLearningMetrics:
        """Evaluate progressive networks transfer."""
        return TransferLearningMetrics(
            source_performance={'mean_reward': 0.0},
            target_performance_scratch={'mean_reward': 0.0},
            target_performance_transfer={'mean_reward': 0.0},
            convergence_speedup=1.0,
            sample_efficiency=1.0,
            final_performance_ratio=1.0,
            negative_transfer=False
        )


class ElasticWeightConsolidationTransfer(AbstractTransferLearner):
    """
    Elastic Weight Consolidation (EWC) for transfer learning.
    
    Adds penalty terms to prevent important weights from changing
    too much when learning new tasks, reducing catastrophic forgetting.
    """
    
    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        fisher_estimation_samples: int = 1000,
        device: str = 'cpu'
    ):
        super().__init__(device)
        self.ewc_lambda = ewc_lambda
        self.fisher_estimation_samples = fisher_estimation_samples
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def transfer_to_target(
        self,
        source_agent: AbstractPIDAgent,
        target_config: PIDAgentConfig,
        source_characteristics: ProcessCharacteristics,
        target_characteristics: ProcessCharacteristics
    ) -> AbstractPIDAgent:
        """Transfer using EWC."""
        # Estimate Fisher information matrix for source task
        self._estimate_fisher_information(source_agent)
        
        # Store optimal parameters for source task
        self._store_optimal_parameters(source_agent)
        
        # Create target agent and copy source weights
        from .base_agent import create_agent
        target_agent = create_agent(type(source_agent).__name__.lower().replace('agent', ''), target_config)
        
        # Copy weights
        self._copy_network_weights(source_agent, target_agent)
        
        # Modify target agent to include EWC penalty
        self._add_ewc_penalty(target_agent)
        
        return target_agent
    
    def _estimate_fisher_information(self, source_agent: AbstractPIDAgent):
        """Estimate Fisher Information Matrix."""
        # This is a simplified version of Fisher information estimation
        # In practice, you'd need access to the source task data
        
        if hasattr(source_agent, 'actor'):
            for name, param in source_agent.actor.named_parameters():
                if param.requires_grad:
                    # Simplified Fisher estimation using gradient variance
                    self.fisher_information[name] = param.grad.pow(2).clone() if param.grad is not None else torch.zeros_like(param)
    
    def _store_optimal_parameters(self, source_agent: AbstractPIDAgent):
        """Store optimal parameters from source task."""
        if hasattr(source_agent, 'actor'):
            for name, param in source_agent.actor.named_parameters():
                if param.requires_grad:
                    self.optimal_params[name] = param.data.clone()
    
    def _copy_network_weights(self, source_agent: AbstractPIDAgent, target_agent: AbstractPIDAgent):
        """Copy network weights (same as fine-tuning)."""
        if hasattr(source_agent, 'actor') and hasattr(target_agent, 'actor'):
            source_state_dict = source_agent.actor.state_dict()
            target_state_dict = target_agent.actor.state_dict()
            
            for name, param in source_state_dict.items():
                if name in target_state_dict and param.shape == target_state_dict[name].shape:
                    target_state_dict[name].copy_(param)
            
            target_agent.actor.load_state_dict(target_state_dict)
    
    def _add_ewc_penalty(self, target_agent: AbstractPIDAgent):
        """Add EWC penalty to target agent's loss computation."""
        # This would require modifying the agent's update method
        # to include EWC penalty terms. For now, just store the reference.
        target_agent._ewc_fisher = self.fisher_information
        target_agent._ewc_optimal = self.optimal_params
        target_agent._ewc_lambda = self.ewc_lambda
    
    def compute_ewc_penalty(self, agent: AbstractPIDAgent) -> torch.Tensor:
        """Compute EWC penalty term."""
        penalty = 0
        
        if hasattr(agent, 'actor') and hasattr(agent, '_ewc_fisher'):
            for name, param in agent.actor.named_parameters():
                if name in agent._ewc_fisher and name in agent._ewc_optimal:
                    penalty += (agent._ewc_fisher[name] * (param - agent._ewc_optimal[name]).pow(2)).sum()
        
        return agent._ewc_lambda * penalty
    
    def evaluate_transfer(
        self,
        source_agent: AbstractPIDAgent,
        target_agent: AbstractPIDAgent,
        target_env,
        num_episodes: int = 100
    ) -> TransferLearningMetrics:
        """Evaluate EWC transfer."""
        return TransferLearningMetrics(
            source_performance={'mean_reward': 0.0},
            target_performance_scratch={'mean_reward': 0.0},
            target_performance_transfer={'mean_reward': 0.0},
            convergence_speedup=1.0,
            sample_efficiency=1.0,
            final_performance_ratio=1.0,
            negative_transfer=False
        )


class MultiTaskLearning(AbstractTransferLearner):
    """
    Multi-task learning for simultaneous training on multiple processes.
    
    Learns shared representations across different process types
    while maintaining task-specific heads.
    """
    
    def __init__(
        self,
        shared_layers: List[str] = None,
        task_specific_layers: List[str] = None,
        device: str = 'cpu'
    ):
        super().__init__(device)
        self.shared_layers = shared_layers or ['feature_extractor']
        self.task_specific_layers = task_specific_layers or ['output_layer']
        self.task_heads: Dict[str, nn.Module] = {}
        self.shared_backbone: Optional[nn.Module] = None
    
    def create_multitask_agent(
        self,
        task_configs: Dict[str, PIDAgentConfig],
        task_characteristics: Dict[str, ProcessCharacteristics]
    ) -> AbstractPIDAgent:
        """Create multi-task agent that can handle multiple process types."""
        # Create shared backbone
        self.shared_backbone = self._create_shared_backbone(task_configs)
        
        # Create task-specific heads
        for task_name, config in task_configs.items():
            self.task_heads[task_name] = self._create_task_head(config, task_characteristics[task_name])
        
        # Create multi-task agent
        multitask_agent = self._create_multitask_agent_class(
            self.shared_backbone, self.task_heads, task_configs
        )
        
        return multitask_agent
    
    def _create_shared_backbone(self, task_configs: Dict[str, PIDAgentConfig]) -> nn.Module:
        """Create shared feature extraction backbone."""
        # Use configuration from first task as template
        first_config = next(iter(task_configs.values()))
        
        from .networks.base_networks import FeatureExtractor
        backbone = FeatureExtractor(
            input_dim=6,
            hidden_dims=first_config.hidden_dims,
            dropout_rate=first_config.dropout_rate
        )
        
        return backbone
    
    def _create_task_head(self, config: PIDAgentConfig, characteristics: ProcessCharacteristics) -> nn.Module:
        """Create task-specific output head."""
        from .networks.base_networks import PIDOutputLayer
        
        head = PIDOutputLayer(
            input_dim=config.hidden_dims[-1],
            kp_range=config.kp_range,
            ki_range=config.ki_range,
            kd_range=config.kd_range
        )
        
        return head
    
    def _create_multitask_agent_class(
        self,
        backbone: nn.Module,
        task_heads: Dict[str, nn.Module],
        task_configs: Dict[str, PIDAgentConfig]
    ) -> AbstractPIDAgent:
        """Create multi-task agent implementation."""
        # This would require creating a new agent class
        # For now, return a placeholder
        from .base_agent import create_agent
        first_config = next(iter(task_configs.values()))
        return create_agent('ppo', first_config)  # Placeholder
    
    def transfer_to_target(
        self,
        source_agent: AbstractPIDAgent,
        target_config: PIDAgentConfig,
        source_characteristics: ProcessCharacteristics,
        target_characteristics: ProcessCharacteristics
    ) -> AbstractPIDAgent:
        """Transfer from multi-task to single task."""
        # Extract relevant components from multi-task agent
        target_task_head = self._extract_task_head(source_agent, target_characteristics.process_type)
        shared_features = self._extract_shared_features(source_agent)
        
        # Create specialized single-task agent
        from .base_agent import create_agent
        target_agent = create_agent(type(source_agent).__name__.lower().replace('agent', ''), target_config)
        
        # Transfer shared features and task-specific head
        self._transfer_multitask_components(target_agent, shared_features, target_task_head)
        
        return target_agent
    
    def _extract_task_head(self, agent: AbstractPIDAgent, task_type: str) -> nn.Module:
        """Extract task-specific head from multi-task agent."""
        return self.task_heads.get(task_type, list(self.task_heads.values())[0])
    
    def _extract_shared_features(self, agent: AbstractPIDAgent) -> nn.Module:
        """Extract shared feature extractor."""
        return self.shared_backbone
    
    def _transfer_multitask_components(
        self,
        target_agent: AbstractPIDAgent,
        shared_features: nn.Module,
        task_head: nn.Module
    ):
        """Transfer components from multi-task to single-task agent."""
        # This would involve copying weights to appropriate parts of target agent
        pass
    
    def evaluate_transfer(
        self,
        source_agent: AbstractPIDAgent,
        target_agent: AbstractPIDAgent,
        target_env,
        num_episodes: int = 100
    ) -> TransferLearningMetrics:
        """Evaluate multi-task transfer."""
        return TransferLearningMetrics(
            source_performance={'mean_reward': 0.0},
            target_performance_scratch={'mean_reward': 0.0},
            target_performance_transfer={'mean_reward': 0.0},
            convergence_speedup=1.0,
            sample_efficiency=1.0,
            final_performance_ratio=1.0,
            negative_transfer=False
        )


class ProcessSimilarityAnalyzer:
    """
    Analyzer for determining similarity between different processes.
    
    Helps decide when transfer learning is likely to be beneficial
    and which source processes to use for transfer.
    """
    
    def __init__(self):
        self.similarity_metrics: Dict[Tuple[str, str], float] = {}
        self.process_embeddings: Dict[str, np.ndarray] = {}
    
    def compute_process_similarity(
        self,
        process1: ProcessCharacteristics,
        process2: ProcessCharacteristics
    ) -> float:
        """
        Compute similarity between two processes.
        
        Returns similarity score between 0 and 1.
        """
        # Feature vector for process characteristics
        features1 = self._extract_features(process1)
        features2 = self._extract_features(process2)
        
        # Compute cosine similarity
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        
        return max(0, similarity)  # Ensure non-negative
    
    def _extract_features(self, process: ProcessCharacteristics) -> np.ndarray:
        """Extract numerical features from process characteristics."""
        # Normalize features to [0, 1] range
        difficulty_map = {'EASY': 0.0, 'MEDIUM': 0.5, 'DIFFICULT': 1.0, 'UNKNOWN': 0.25}
        
        features = np.array([
            difficulty_map.get(process.difficulty, 0.25),
            np.clip(process.time_constant / 100.0, 0, 1),  # Normalize assuming max 100s
            np.clip(process.dead_time / 10.0, 0, 1),       # Normalize assuming max 10s
            process.nonlinearity,
            np.clip(process.noise_level, 0, 1),
            np.clip(process.disturbance_level, 0, 1),
            np.clip((process.operating_range[1] - process.operating_range[0]) / 1000.0, 0, 1),
            np.clip(process.typical_setpoint / 1000.0, 0, 1)
        ])
        
        return features
    
    def recommend_source_processes(
        self,
        target_process: ProcessCharacteristics,
        available_processes: List[ProcessCharacteristics],
        top_k: int = 3
    ) -> List[Tuple[ProcessCharacteristics, float]]:
        """
        Recommend most similar source processes for transfer learning.
        
        Returns list of (process, similarity_score) tuples.
        """
        similarities = []
        
        for source_process in available_processes:
            similarity = self.compute_process_similarity(target_process, source_process)
            similarities.append((source_process, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def should_use_transfer(
        self,
        target_process: ProcessCharacteristics,
        source_process: ProcessCharacteristics,
        similarity_threshold: float = 0.6
    ) -> bool:
        """Determine if transfer learning is recommended."""
        similarity = self.compute_process_similarity(target_process, source_process)
        return similarity >= similarity_threshold


class TransferLearningManager:
    """
    High-level manager for transfer learning operations.
    
    Orchestrates the entire transfer learning pipeline including
    similarity analysis, method selection, and evaluation.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.similarity_analyzer = ProcessSimilarityAnalyzer()
        self.transfer_methods = {
            TransferMethod.FINE_TUNING: FineTuningTransfer(device=device),
            TransferMethod.PROGRESSIVE_NETWORKS: ProgressiveNetworksTransfer(device=device),
            TransferMethod.ELASTIC_WEIGHT_CONSOLIDATION: ElasticWeightConsolidationTransfer(device=device),
            TransferMethod.MULTI_TASK_LEARNING: MultiTaskLearning(device=device)
        }
        
        self.agent_repository: Dict[str, AbstractPIDAgent] = {}
        self.process_repository: Dict[str, ProcessCharacteristics] = {}
        self.transfer_results: List[TransferLearningMetrics] = {}
    
    def register_agent(self, agent_id: str, agent: AbstractPIDAgent, process: ProcessCharacteristics):
        """Register trained agent with process characteristics."""
        self.agent_repository[agent_id] = agent
        self.process_repository[agent_id] = process
    
    def transfer_to_new_process(
        self,
        target_config: PIDAgentConfig,
        target_process: ProcessCharacteristics,
        method: TransferMethod = TransferMethod.FINE_TUNING,
        source_agent_id: Optional[str] = None
    ) -> Tuple[AbstractPIDAgent, Dict[str, Any]]:
        """
        Transfer learning to new process.
        
        Returns:
            target_agent: Agent for new process
            transfer_info: Information about transfer process
        """
        # Find best source agent if not specified
        if source_agent_id is None:
            source_agent_id = self._find_best_source_agent(target_process)
        
        if source_agent_id not in self.agent_repository:
            raise ValueError(f"Source agent {source_agent_id} not found")
        
        source_agent = self.agent_repository[source_agent_id]
        source_process = self.process_repository[source_agent_id]
        
        # Perform transfer
        transfer_method = self.transfer_methods[method]
        target_agent = transfer_method.transfer_to_target(
            source_agent, target_config, source_process, target_process
        )
        
        # Compile transfer information
        transfer_info = {
            'source_agent_id': source_agent_id,
            'source_process': source_process,
            'target_process': target_process,
            'method': method,
            'similarity_score': self.similarity_analyzer.compute_process_similarity(
                source_process, target_process
            )
        }
        
        return target_agent, transfer_info
    
    def _find_best_source_agent(self, target_process: ProcessCharacteristics) -> str:
        """Find most suitable source agent for transfer."""
        if not self.process_repository:
            raise ValueError("No source agents available")
        
        available_processes = list(self.process_repository.values())
        recommendations = self.similarity_analyzer.recommend_source_processes(
            target_process, available_processes, top_k=1
        )
        
        if not recommendations:
            # Return first available agent as fallback
            return list(self.agent_repository.keys())[0]
        
        # Find agent ID for most similar process
        best_process = recommendations[0][0]
        for agent_id, process in self.process_repository.items():
            if process == best_process:
                return agent_id
        
        return list(self.agent_repository.keys())[0]
    
    def evaluate_all_transfer_methods(
        self,
        target_config: PIDAgentConfig,
        target_process: ProcessCharacteristics,
        target_env,
        source_agent_id: Optional[str] = None,
        num_episodes: int = 100
    ) -> Dict[TransferMethod, TransferLearningMetrics]:
        """Evaluate all transfer methods on target process."""
        results = {}
        
        for method in TransferMethod:
            try:
                target_agent, transfer_info = self.transfer_to_new_process(
                    target_config, target_process, method, source_agent_id
                )
                
                # Evaluate transfer
                source_agent = self.agent_repository[transfer_info['source_agent_id']]
                metrics = self.transfer_methods[method].evaluate_transfer(
                    source_agent, target_agent, target_env, num_episodes
                )
                
                results[method] = metrics
                
            except Exception as e:
                print(f"Failed to evaluate {method}: {e}")
                continue
        
        return results
    
    def get_transfer_recommendations(
        self,
        target_process: ProcessCharacteristics
    ) -> Dict[str, Any]:
        """Get recommendations for transfer learning setup."""
        # Find similar processes
        available_processes = list(self.process_repository.values())
        recommendations = self.similarity_analyzer.recommend_source_processes(
            target_process, available_processes, top_k=3
        )
        
        # Recommend transfer method based on process similarity
        if not recommendations:
            recommended_method = TransferMethod.FINE_TUNING
        elif recommendations[0][1] > 0.8:
            recommended_method = TransferMethod.FINE_TUNING
        elif recommendations[0][1] > 0.6:
            recommended_method = TransferMethod.ELASTIC_WEIGHT_CONSOLIDATION
        else:
            recommended_method = TransferMethod.MULTI_TASK_LEARNING
        
        return {
            'recommended_sources': recommendations,
            'recommended_method': recommended_method,
            'should_use_transfer': len(recommendations) > 0 and recommendations[0][1] > 0.5,
            'expected_benefit': 'High' if recommendations and recommendations[0][1] > 0.7 else 'Medium'
        }


def create_transfer_learner(
    method: TransferMethod,
    device: str = 'cpu',
    **kwargs
) -> AbstractTransferLearner:
    """Factory function to create transfer learning methods."""
    if method == TransferMethod.FINE_TUNING:
        return FineTuningTransfer(device=device, **kwargs)
    elif method == TransferMethod.PROGRESSIVE_NETWORKS:
        return ProgressiveNetworksTransfer(device=device, **kwargs)
    elif method == TransferMethod.ELASTIC_WEIGHT_CONSOLIDATION:
        return ElasticWeightConsolidationTransfer(device=device, **kwargs)
    elif method == TransferMethod.MULTI_TASK_LEARNING:
        return MultiTaskLearning(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown transfer method: {method}")


def test_transfer_learning():
    """Test transfer learning components."""
    print("Testing transfer learning components...")
    
    # Test process similarity
    analyzer = ProcessSimilarityAnalyzer()
    
    process1 = ProcessCharacteristics(
        difficulty='EASY',
        time_constant=5.0,
        dead_time=1.0,
        nonlinearity=0.1,
        noise_level=0.05,
        disturbance_level=0.1,
        operating_range=(0, 100),
        typical_setpoint=50.0,
        process_type='temperature'
    )
    
    process2 = ProcessCharacteristics(
        difficulty='EASY',
        time_constant=6.0,
        dead_time=1.2,
        nonlinearity=0.15,
        noise_level=0.08,
        disturbance_level=0.12,
        operating_range=(0, 120),
        typical_setpoint=60.0,
        process_type='temperature'
    )
    
    similarity = analyzer.compute_process_similarity(process1, process2)
    print(f"Process similarity: {similarity:.3f}")
    
    # Test transfer learning manager
    manager = TransferLearningManager()
    
    recommendations = manager.get_transfer_recommendations(process1)
    print(f"Transfer recommendations: {recommendations}")
    
    print("Transfer learning tests completed!")


if __name__ == "__main__":
    test_transfer_learning()