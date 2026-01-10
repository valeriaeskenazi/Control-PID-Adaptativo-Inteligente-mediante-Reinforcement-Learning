from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import torch
import numpy as np


class AbstractPIDAgent(ABC):
    """
    Todos los agentes RL (PPO, DQN, SAC, etc.) deben heredar de esta clase.
    """
    
    def __init__(
        self,
        state_dim: int = 6, # 6 dimensiones de estado [PV, setpoint, error, error_prev, error_integral, error_derivative]
        action_dim: int = 7,  # 7 acciones discretas para DeltaPIDActionSpace, 1 tupla del tipo (kp,ki,kd) para control directo
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        if seed is not None:
            self.set_seed(seed)
        
        self.training_step = 0
        self.episode_count = 0
    
    @abstractmethod
    def select_action(
        self,                       
        state: np.ndarray,          
        training: bool = True
    ) -> Union[int, np.ndarray]:
        """
        Args:
            state: Estado actual [PV, setpoint, error, error_prev, error_integral, error_derivative]
            training: Indica si el agente está en modo entrenamiento (True) o evaluación (False)
        
        Devuelve:
            action: 
                - int: índice de acción discreta (0-6) para agentes value-based en modo 'pid_tuning'
                - np.ndarray: acción continua para agentes policy-based en modo 'direct'
        """
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualiza los parámetros del agente usando un lote de experiencias.
        
        Args:
            batch_data: Diccionario con datos del lote.
                       Debe incluir: states, actions, rewards, next_states, dones
        
        Devuelve:
            metrics: Diccionario con métricas de entrenamiento (pérdidas, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Guardar el estado del agente en un archivo."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Cargar el estado del agente desde un archivo."""
        pass
    
    def set_seed(self, seed: int) -> None:
        """Para reproducibilidad."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """ Procesa el estado antes de pasarlo a la red neuronal."""
        # Convertir a tensor y mover a dispositivo
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.to(self.device)
        
        # Agregar dimensión de batch si es necesario
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        return state_tensor # Shape: (1, state_dim)
    
    def postprocess_action(self, action: torch.Tensor) -> Union[int, np.ndarray]:
        """Procesa la salida de la red neuronal para obtener la acción final. """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Sacar dimensión de batch si es necesario
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        
        # Si es acción discreta (un solo valor entero)
        if action.shape == () or (len(action.shape) == 1 and action.shape[0] == 1):
            return int(action)
        
        # Si es acción continua
        return action
    
    def get_training_info(self) -> Dict[str, Any]:
        """Devolver la información de entrenamiento actual del agente."""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'device': str(self.device),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }


class AbstractPolicyGradientAgent(AbstractPIDAgent):
    """
    Abstract base class for policy gradient agents (PPO, A2C, REINFORCE, etc.).
    
    Extends AbstractPIDAgent with policy gradient specific methods.
    Estos agentes típicamente usan acciones continuas.
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
    
    Estos agentes usan acciones discretas (índices 0-6 para DeltaPIDActionSpace).
    Implementan epsilon-greedy para exploración.
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 7,  # 7 acciones discretas
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
        """Update epsilon using exponential decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        """Get current exploration epsilon."""
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
    Estos agentes típicamente usan acciones continuas.
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