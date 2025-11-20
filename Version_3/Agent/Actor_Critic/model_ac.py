"""
Redes neuronales para Actor-Critic con acciones continuas.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Red Actor para acciones continuas.
    
    Entrada: Estado del proceso (6 dims para PID, variable para orquestador)
    Salida: Acciones continuas en rango [-1, 1] (con tanh)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        """
        Inicializar red Actor.
        
        Args:
            state_dim: Dimensión del estado
            action_dim: Dimensión de la acción (continua)
            hidden_dims: Dimensiones de capas ocultas
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Construir capas dinámicamente
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Capa de salida con tanh para limitar a [-1, 1]
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Inicializar pesos usando Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red Actor.
        
        Args:
            state: Estado del proceso [batch_size, state_dim]
        
        Returns:
            actions: Acciones continuas [batch_size, action_dim] en [-1, 1]
        """
        # Asegurar dimensión de batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        actions = self.network(state)
        
        return actions


class CriticNetwork(nn.Module):
    """
    Red Critic (estado-valor).
    
    Entrada: Estado del proceso
    Salida: Valor estimado V(s)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        """
        Inicializar red Critic.
        
        Args:
            state_dim: Dimensión del estado
            hidden_dims: Dimensiones de capas ocultas
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        # Construir capas dinámicamente
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Capa de salida (valor escalar)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Inicializar pesos usando Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de la red Critic.
        
        Args:
            state: Estado del proceso [batch_size, state_dim]
        
        Returns:
            value: Valor estimado V(s) [batch_size, 1]
        """
        # Asegurar dimensión de batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        value = self.network(state)
        
        return value
