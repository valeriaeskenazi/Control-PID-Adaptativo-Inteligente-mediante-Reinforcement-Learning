"""
Agente DQN simple para control PID
Implementación directa y fácil de entender
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy

from .model import DQN_Network
from .action_space import PIDActionSpace
from ..buffer_memory import SimpleReplayBuffer, Experience
from ..abstract_agent import AbstractValueBasedAgent


class DQN_Agent(AbstractValueBasedAgent):
    """
    Agente DQN simple para control PID
    
    Características:
    - Red neuronal principal (online)
    - Red neuronal objetivo (target) 
    - Experience replay
    - Epsilon-greedy exploration
    """
    
    def __init__(self, 
                 state_dim=6,
                 lr=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=32,
                 target_update_freq=100,
                 device='cpu'):
        """
        Inicializar agente DQN
        
        Args:
            state_dim: Dimensiones del estado
            lr: Learning rate
            gamma: Factor de descuento
            epsilon_start: Epsilon inicial
            epsilon_end: Epsilon mínimo
            epsilon_decay: Decaimiento de epsilon
            memory_size: Tamaño del buffer de experiencias
            batch_size: Tamaño del batch para entrenamiento
            target_update_freq: Frecuencia de actualización de red objetivo
            device: Dispositivo PyTorch
        """
        # Llamar al constructor padre (incluye epsilon management)
        super().__init__(
            state_dim=state_dim, 
            action_dim=3, 
            device=device, 
            seed=None,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_end,
            epsilon_decay=epsilon_decay
        )
        
        # Parámetros específicos de DQN
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Espacio de acciones PID
        self.action_space = PIDActionSpace()
        self.n_actions = self.action_space.n_actions
        
        # Redes neuronales
        self.q_network = DQN_Network(state_dim, self.n_actions)  # Red principal
        self.target_network = DQN_Network(state_dim, self.n_actions)  # Red objetivo
        
        # Copiar pesos a red objetivo
        self.update_target_network()
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Memory para experience replay 
        self.memory = SimpleReplayBuffer(capacity=memory_size, device=device)
        
        # Contadores
        self.steps_done = 0
        self.episodes_done = 0
        
        print(f" Agente DQN creado:")
        print(f"   Estados: {state_dim}")
        print(f"   Acciones: {self.n_actions}")
        print(f"   Learning rate: {lr}")
        print(f"   Gamma: {gamma}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Seleccionar acción usando epsilon-greedy (implementa método abstracto)
        
        Args:
            state: Estado actual del ambiente
            training: Si está en modo entrenamiento
            
        Returns:
            action: Parámetros PID [Kp, Ki, Kd]
        """
        # Usar epsilon del padre solo durante entrenamiento
        current_epsilon = self.get_epsilon() if training else 0.0
        
        # Usar el preprocesamiento del padre
        state_tensor = self.preprocess_state(state)
        
        # Seleccionar acción
        if np.random.random() < current_epsilon:
            # Acción aleatoria
            action_index = np.random.randint(0, self.n_actions)
        else:
            # Acción greedy
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax().item()
        
        # Convertir a parámetros PID
        pid_params = self.action_space.index_to_pid(action_index)
        
        # Actualizar contadores
        self.training_step += 1
        
        # Guardar último índice para store_experience
        self._last_action_index = action_index
        
        return pid_params
    
    def get_last_action_index(self):
        """Obtener el último índice de acción (helper para almacenar experiencias)"""
        return getattr(self, '_last_action_index', 0)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Almacenar experiencia en memoria
        
        Args:
            state: Estado actual
            action: Acción tomada (parámetros PID) - se convierte a índice internamente
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        # Convertir acción PID a índice para almacenamiento
        if hasattr(self, '_last_action_index'):
            action_index = self._last_action_index
        else:
            # Fallback: convertir PID a índice
            action_index = self.action_space.pid_to_index(action[0], action[1], action[2])
        
        experience = Experience(state, action_index, reward, next_state, done)
        self.memory.add(experience)
    
    def update(self, batch_data=None):
        """
        Actualizar agente (implementa método abstracto)
        
        Args:
            batch_data: No usado en DQN (usa su propio buffer)
        
        Returns:
            metrics: Diccionario con métricas de entrenamiento
        """
        # Verificar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return {}
        
        # Muestrear batch (ya viene como tensores)
        batch = self.memory.sample(self.batch_size)
        
        # Ya son tensores listos para usar
        states = batch['states']
        actions = batch['actions'].long()  # Convertir a LongTensor para indexing
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values objetivo
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcular pérdida
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Actualizar epsilon usando método del padre
        self.update_epsilon()
        
        # Actualizar red objetivo periódicamente
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Devolver métricas como diccionario (interfaz AbstractPIDAgent)
        return {
            'q_loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'training_step': self.training_step
        }
    
    def update_target_network(self):
        """Copiar pesos de red principal a red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"🔄 Red objetivo actualizada (step {self.training_step})")
    
    
    def compute_q_loss(self, states: torch.Tensor, actions: torch.Tensor,
                       rewards: torch.Tensor, next_states: torch.Tensor,
                       dones: torch.Tensor) -> torch.Tensor:
        """
        Implementa el método abstracto de AbstractValueBasedAgent
        """
        batch_size = states.shape[0]
        
        # Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values objetivo
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcular pérdida
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        return loss
    
    def save(self, filepath: str) -> None:
        """
        Implementa el método abstracto de AbstractPIDAgent
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.get_epsilon(),
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
        print(f"💾 Agente guardado en: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Implementa el método abstracto de AbstractPIDAgent
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar parámetros de epsilon del padre
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        print(f"📂 Agente cargado desde: {filepath}")
    
    def get_stats(self):
        """Obtener estadísticas del agente"""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'memory_size': len(self.memory),
            'network_params': sum(p.numel() for p in self.q_network.parameters())
        }

