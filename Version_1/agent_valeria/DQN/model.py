"""
Red neuronal simple para DQN - PID Control
Solo la arquitectura de la red neuronal
"""
import torch
import torch.nn as nn

class DQN_Network(nn.Module):
    """
    Red neuronal simple para DQN
    
    Entrada: Estado del proceso PID (6 dimensiones)
    Salida: Q-values para cada acción discreta
    """
    
    def __init__(self, state_dim=6, n_actions=64, hidden_size=128):
        """
        Inicializar la red neuronal
        
        Args:
            state_dim: Dimensiones del estado (6 para PID: PV, SP, error, error_prev, error_int, error_der)
            n_actions: Número de acciones discretas (combinaciones de Kp, Ki, Kd)
            hidden_size: Tamaño de las capas ocultas
        """
        super(DQN_Network, self).__init__()
        
        # Arquitectura de la red - 
        self.fc1 = nn.Linear(state_dim, hidden_size)      # 6 → 128
        self.fc2 = nn.Linear(hidden_size, hidden_size)    # 128 → 128  
        self.fc3 = nn.Linear(hidden_size, hidden_size//2) # 128 → 64
        self.fc4 = nn.Linear(hidden_size//2, n_actions)   # 64 → n_actions
        
        # Activación
        self.relu = nn.ReLU()
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicializar pesos de manera inteligente"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.01)
    
    def forward(self, state):
        """
        Paso forward de la red
        
        Args:
            state: Estado del proceso [batch_size, 6] o [6]
            
        Returns:
            q_values: Q-values para cada acción [batch_size, n_actions]
        """
        # Asegurar que sea tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        
        # Si es un solo estado, agregar dimensión batch
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Paso a través de la red
        x = self.relu(self.fc1(state))    # 6 → 128 + ReLU
        x = self.relu(self.fc2(x))        # 128 → 128 + ReLU
        x = self.relu(self.fc3(x))        # 128 → 64 + ReLU
        q_values = self.fc4(x)            # 64 → n_actions (sin activación)
        
        return q_values