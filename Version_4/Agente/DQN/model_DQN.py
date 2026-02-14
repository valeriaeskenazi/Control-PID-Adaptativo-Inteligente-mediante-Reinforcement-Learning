import torch
import torch.nn as nn
from typing import Tuple


class DQN_Network(nn.Module):    
    def __init__(
        self,
        state_dim: int ,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
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
        # Inicializar pesos usando Kaiming initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Forward pass
        q_values = self.network(state)
        
        return q_values