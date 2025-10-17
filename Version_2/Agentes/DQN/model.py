"""
Red neuronal para DQN - PID Control con acciones incrementales.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DQN_Network(nn.Module):
    """
    Red neuronal para DQN con acciones discretas incrementales.
    
    Entrada: Estado del proceso PID (6 dimensiones)
    Salida: Q-values para cada acción discreta (7 acciones)
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        n_actions: int = 7,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        """
        Inicializar red neuronal.
        
        Args:
            state_dim: Dimensión del estado (6: PV, SP, error, error_prev, error_int, error_der)
            n_actions: Número de acciones discretas (7 para DeltaPIDActionSpace)
            hidden_dims: Tupla con dimensiones de capas ocultas
        """
        super(DQN_Network, self).__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims
        
        # Construir capas dinámicamente
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(input_dim, n_actions))
        
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
        Forward pass de la red.
        
        Args:
            state: Estado del proceso [batch_size, state_dim] o [state_dim]
        
        Returns:
            q_values: Q-values para cada acción [batch_size, n_actions]
        """
        # Asegurar que tenga dimensión de batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        q_values = self.network(state)
        
        return q_values