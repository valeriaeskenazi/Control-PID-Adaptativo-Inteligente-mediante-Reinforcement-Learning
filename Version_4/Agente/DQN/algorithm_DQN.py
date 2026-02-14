import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional

from .model_DQN import DQN_Network
from ..memory import AbstractReplayBuffer, SimpleReplayBuffer
from ..abstract_agent import AbstractValueBasedAgent


class DQNAgent(AbstractValueBasedAgent):
    def __init__(
        self,
        state_dim: int,          
        action_dim: int,
        agent_type: str,
        hidden_dims: tuple = (128, 128, 64),
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer: Optional[AbstractReplayBuffer] = None,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):

        # Llamar al constructor padre
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_type=agent_type,
            device=device,
            seed=seed,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay
        )
        
        # Parámetros específicos de DQN
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_dims = hidden_dims
        
        # Redes neuronales
        self.q_network = DQN_Network(
            state_dim=state_dim,
            n_actions=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_network = DQN_Network(
            state_dim=state_dim,
            n_actions=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Replay buffer
        if replay_buffer is not None:
            self.memory = replay_buffer  
        else:
            # Default: SimpleReplayBuffer
            self.memory = SimpleReplayBuffer(capacity=memory_size, device=device)

        # Copiar pesos a red objetivo
        self.update_target_network()
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = SimpleReplayBuffer(capacity=memory_size, device=device)
        
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> int:
       
        # Usar epsilon solo en entrenamiento
        current_epsilon = self.get_epsilon() if training else 0.0
        
        # Preprocesar estado
        state_tensor = self.preprocess_state(state)

        # Calcular cuántas variables hay (state_dim / 5 porque cada var tiene 5 obs)
        n_vars = self.state_dim // 5

        actions = []
        for i in range(n_vars):
            # Extraer estado de esta variable
            var_state = state[i*5:(i+1)*5]
            state_tensor = self.preprocess_state(var_state)
            
            # Epsilon-greedy
            if training and np.random.random() < self.get_epsilon():
                action_idx = np.random.randint(0, self.action_dim)
            else:
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action_idx = q_values.argmax(dim=1).item()
            
            actions.append(action_idx)
        
        return np.array(actions, dtype=np.int64)
    
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        # Verificar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return {}
        
        # Muestrear batch del buffer
        batch = self.memory.sample(self.batch_size)
        
        states = batch['states']
        actions = batch['actions'].long()
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Calcular Q-values actuales (CON gradientes)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Calcular Q-values objetivo (SIN gradientes)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # PARA PRIORITY BUFFER: calcular TD errors
        if 'weights' in batch:
            weights = batch['weights']
            
            # Loss ponderado por importancia
            td_errors_tensor = current_q - target_q  # CON gradientes para backprop
            loss = (weights * (td_errors_tensor ** 2)).mean()
            
            # Actualizar prioridades (necesita detach para numpy)
            if hasattr(self.memory, 'update_priorities'):
                td_errors_np = td_errors_tensor.abs().detach().cpu().numpy()
                self.memory.update_priorities(batch['indices'], td_errors_np)
        
        # PARA SIMPLE BUFFER
        else:
            loss = self.compute_q_loss(states, actions, rewards, next_states, dones)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Actualizar epsilon
        self.update_epsilon()
        
        # Actualizar red objetivo periódicamente
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Incrementar contador
        self.training_step += 1
        
        # Devolver métricas
        return {
            'q_loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }
        
    def compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
       
        # Q-values actuales para las acciones tomadas
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values objetivo
        with torch.no_grad():
            # Mejor acción en siguiente estado según red objetivo
            next_q_values = self.target_network(next_states).max(1)[0]
            
            # Target: r + γ * max_a' Q(s', a') si no terminó
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # MSE loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        return loss
    
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    
    def save(self, filepath: str) -> None:
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.get_epsilon(),
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'hidden_dims': self.hidden_dims
        }
        
        torch.save(checkpoint, filepath)
        print(f"Agente guardado en: {filepath}")
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Cargar redes
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar parámetros
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'epsilon': self.get_epsilon(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory.capacity,
            'network_params': sum(p.numel() for p in self.q_network.parameters()),
            'device': str(self.device),
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }