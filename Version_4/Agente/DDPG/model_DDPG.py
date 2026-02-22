import torch
import torch.nn as nn
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Red Actor para DDPG - produce deltas continuos.
    
    Para ORCH: produce [delta_Tc, delta_F] en [-1, 1] (translate los escala)
    Para CTRL: produce [delta_Kp, delta_Ki, delta_Kd] x n_vars en [-1, 1]
    
    Salida siempre en [-1, 1] via tanh, translate se encarga de escalar al rango real.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,          # n_vars * params_por_var (ej: orch=2, ctrl=2*3=6)
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Tanh: salida en [-1, 1], translate la escala al rango real de delta
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
        # Última capa con pesos pequeños para acciones iniciales cerca de 0
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -0.003, 0.003)
        nn.init.constant_(last_linear.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.network(state)  # (batch, action_dim) en [-1, 1]


class CriticNetwork(nn.Module):
    """
    Red Critic para DDPG - estima Q(s, a).
    
    A diferencia del AC viejo (que estimaba V(s)), 
    este Critic recibe estado + acción y devuelve Q-value escalar.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        # Critic recibe [state, action] concatenados
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))  # Q-value escalar

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
        last_linear = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(last_linear.weight, -0.003, 0.003)
        nn.init.constant_(last_linear.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        x = torch.cat([state, action], dim=1)  # (batch, state_dim + action_dim)
        return self.network(x)  # (batch, 1)
