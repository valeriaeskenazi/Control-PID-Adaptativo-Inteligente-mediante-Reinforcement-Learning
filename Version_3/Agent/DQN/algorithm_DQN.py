"""
Agente DQN para control PID con acciones discretas incrementales.
Compatible con DeltaPIDActionSpace (7 acciones: 0-6).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Union, Optional

from .model import DQN_Network
from ..replay_buffers import SimpleReplayBuffer, Experience
from ..abstract_agent import AbstractValueBasedAgent


class DQNAgent(AbstractValueBasedAgent):
    """
    Agente DQN para control PID con acciones discretas incrementales.
    
    Características:
    - Red neuronal principal (online)
    - Red neuronal objetivo (target)
    - Experience replay
    - Epsilon-greedy exploration
    - Compatible con DeltaPIDActionSpace (7 acciones)
    
    Args:
        state_dim: Dimensión del espacio de estados (default: 6)
        action_dim: Número de acciones discretas (default: 7 para DeltaPIDActionSpace)
        hidden_dims: Dimensiones de capas ocultas
        lr: Learning rate
        gamma: Factor de descuento
        epsilon_start: Epsilon inicial para exploración
        epsilon_min: Epsilon mínimo
        epsilon_decay: Factor de decaimiento de epsilon
        memory_size: Tamaño del buffer de replay
        batch_size: Tamaño del batch para entrenamiento
        target_update_freq: Frecuencia de actualización de red objetivo
        device: Dispositivo ('cpu' o 'cuda')
        seed: Semilla para reproducibilidad
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 7,  # 7 acciones para DeltaPIDActionSpace
        hidden_dims: tuple = (128, 128, 64),
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """Inicializar agente DQN."""
        # Llamar al constructor padre
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
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
        
        # Copiar pesos a red objetivo
        self.update_target_network()
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = SimpleReplayBuffer(capacity=memory_size, device=device)
        
        print("=" * 60)
        print("✅ DQN Agent creado")
        print(f"   Estado: {state_dim} dims")
        print(f"   Acciones: {action_dim} (DeltaPIDActionSpace)")
        print(f"   Hidden layers: {hidden_dims}")
        print(f"   Learning rate: {lr}")
        print(f"   Gamma: {gamma}")
        print(f"   Epsilon: {epsilon_start} → {epsilon_min} (decay: {epsilon_decay})")
        print(f"   Device: {device}")
        print("=" * 60)
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> int:
        """
        Seleccionar acción usando epsilon-greedy.
        
        Args:
            state: Estado actual [PV, SP, error, error_prev, error_int, error_der]
            training: Si está en modo entrenamiento (afecta epsilon)
        
        Returns:
            action_index: Índice de acción discreta (0-6)
                0: Kp ↑, 1: Ki ↑, 2: Kd ↑
                3: Kp ↓, 4: Ki ↓, 5: Kd ↓
                6: Mantener
        """
        # Usar epsilon solo en entrenamiento
        current_epsilon = self.get_epsilon() if training else 0.0
        
        # Preprocesar estado
        state_tensor = self.preprocess_state(state)
        
        # Epsilon-greedy
        if np.random.random() < current_epsilon:
            # Exploración: acción aleatoria
            action_index = np.random.randint(0, self.action_dim)
        else:
            # Explotación: acción greedy
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax(dim=1).item()
        
        return action_index
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Almacenar experiencia en el buffer.
        
        Args:
            state: Estado actual
            action: Índice de acción tomada (0-6)
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Actualizar redes del agente usando experience replay.
        
        Args:
            batch_data: No usado (DQN usa su propio buffer interno)
        
        Returns:
            metrics: Diccionario con métricas de entrenamiento
        """
        # Verificar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return {}
        
        # Muestrear batch del buffer
        batch = self.memory.sample(self.batch_size)
        
        states = batch['states']
        actions = batch['actions'].long()  # Convertir a LongTensor para indexing
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Calcular loss
        loss = self.compute_q_loss(states, actions, rewards, next_states, dones)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Actualizar epsilon
        self.update_epsilon()
        
        # Actualizar red objetivo periódicamente
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
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
        """
        Calcular pérdida Q-learning (implementa método abstracto).
        
        Args:
            states: Batch de estados
            actions: Batch de acciones
            rewards: Batch de recompensas
            next_states: Batch de siguientes estados
            dones: Batch de flags de terminación
        
        Returns:
            loss: Pérdida Q-learning
        """
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
        """Copiar pesos de red principal a red objetivo."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.training_step > 0:  # No imprimir en inicialización
            print(f"🔄 Red objetivo actualizada (step {self.training_step})")
    
    def save(self, filepath: str) -> None:
        """
        Guardar estado del agente (implementa método abstracto).
        
        Args:
            filepath: Ruta donde guardar el checkpoint
        """
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
        print(f"💾 Agente guardado en: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Cargar estado del agente (implementa método abstracto).
        
        Args:
            filepath: Ruta del checkpoint a cargar
        """
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
        
        print(f"📂 Agente cargado desde: {filepath}")
        print(f"   Training step: {self.training_step}")
        print(f"   Epsilon: {self.epsilon:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del agente."""
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