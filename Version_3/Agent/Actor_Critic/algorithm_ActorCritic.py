"""
Agente Actor-Critic simple (one-step) para acciones continuas.
Compatible con control PID y orquestador de setpoints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
from typing import Dict, Any, Optional

import sys
sys.path.append('..')

from .model_ac import ActorNetwork, CriticNetwork
from ..abstract_agent import AbstractActorCriticAgent



class ActorCriticAgent(AbstractActorCriticAgent):
    """
    Agente Actor-Critic simple (one-step) para acciones continuas.
    
    Características:
    - Actor: genera acciones continuas
    - Critic: estima valor de estados
    - Actualización one-step (sin replay buffer)
    - Acciones continuas en rango [-1, 1]
    
    Args:
        state_dim: Dimensión del espacio de estados
        action_dim: Dimensión del espacio de acciones (continuas)
        hidden_dims: Dimensiones de capas ocultas
        lr_actor: Learning rate del actor
        lr_critic: Learning rate del critic
        gamma: Factor de descuento
        device: Dispositivo ('cpu' o 'cuda')
        seed: Semilla para reproducibilidad
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple = (128, 128, 64),
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """Inicializar agente Actor-Critic."""
        # Llamar al constructor padre
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            seed=seed
        )
        
        # Parámetros específicos de Actor-Critic
        self.gamma = gamma
        self.hidden_dims = hidden_dims
        
        # Redes neuronales
        self.actor_net = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.critic_net = CriticNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Optimizadores
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        
        # Para almacenar última experiencia (one-step)
        self.last_state = None
        self.last_action = None
        self.last_log_prob = None
        
        print("=" * 60)
        print("Actor-Critic Agent creado")
        print(f"   Estado: {state_dim} dims")
        print(f"   Acciones: {action_dim} dims (continuas)")
        print(f"   Hidden layers: {hidden_dims}")
        print(f"   LR Actor: {lr_actor}")
        print(f"   LR Critic: {lr_critic}")
        print(f"   Gamma: {gamma}")
        print(f"   Device: {device}")
        print("=" * 60)
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        """
        Seleccionar acción usando la política del actor.
        
        Args:
            state: Estado actual
            training: Si está en modo entrenamiento (afecta muestreo)
        
        Returns:
            action: Acción continua [action_dim] en rango [-1, 1]
        """
        # Preprocesar estado
        state_tensor = self.preprocess_state(state)
        
        if training:
            # EN TRAINING: Mantener gradientes para backprop
            action_mean = self.actor_net(state_tensor)
            
            # Muestrear de distribución normal para exploración
            std = 0.1
            dist = distributions.Normal(action_mean, std)
            action_tensor = dist.sample()
            
            # Guardar log_prob para actualización (CON gradientes)
            self.last_log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        else:
            # EN EVALUACIÓN: Sin gradientes
            with torch.no_grad():
                action_mean = self.actor_net(state_tensor)
                action_tensor = action_mean
        
        # Clip a [-1, 1] por seguridad
        action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
        
        # Convertir a numpy
        action = action_tensor.squeeze().cpu().numpy()
        
        # Guardar para actualización
        self.last_state = state_tensor
        self.last_action = action_tensor
        
        return action
    
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Actualizar redes usando one-step Actor-Critic.
        
        Args:
            batch_data: Dict con 'reward', 'next_state', 'done'
        
        Returns:
            metrics: Diccionario con métricas de entrenamiento
        """
        if self.last_state is None:
            return {}
        
        # Extraer datos
        reward = batch_data['reward']
        next_state = batch_data['next_state']
        done = batch_data['done']
        
        # Preprocesar next_state
        next_state_tensor = self.preprocess_state(next_state)
        
        # Calcular valores
        current_value = self.critic_net(self.last_state)
        
        with torch.no_grad():
            if done:
                next_value = torch.tensor([[0.0]], device=self.device)
            else:
                next_value = self.critic_net(next_state_tensor)
        
        # Calcular ventaja (delta)
        reward_tensor = torch.tensor([[reward]], device=self.device, dtype=torch.float32)
        delta = reward_tensor + self.gamma * next_value - current_value
        
        # Actualizar Critic
        critic_loss = delta.pow(2).mean()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 1.0)
        self.optimizer_critic.step()
        
        # Actualizar Actor
        # Loss: -log_prob * delta (REINFORCE con baseline)
        actor_loss = -(self.last_log_prob * delta.detach()).mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 1.0)
        self.optimizer_actor.step()
        
        # Incrementar contador
        self.training_step += 1
        
        # Devolver métricas
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'delta': delta.item(),
            'training_step': self.training_step
        }
    
    def compute_actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular pérdida del actor (implementa método abstracto).
        
        Args:
            states: Batch de estados
            actions: Batch de acciones
            advantages: Batch de ventajas
        
        Returns:
            loss: Pérdida del actor
        """
        # Forward pass del actor
        action_means = self.actor_net(states)
        
        # Distribución normal
        std = 0.1
        dist = distributions.Normal(action_means, std)
        
        # Log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Loss: -log_prob * advantage
        loss = -(log_probs * advantages).mean()
        
        return loss
    
    def compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcular pérdida del critic (implementa método abstracto).
        
        Args:
            states: Batch de estados
            actions: Batch de acciones (no usado en state-value)
            rewards: Batch de recompensas
            next_states: Batch de siguientes estados
            dones: Batch de flags de terminación
        
        Returns:
            loss: Pérdida del critic
        """
        # Valores actuales
        current_values = self.critic_net(states)
        
        # Valores siguientes
        with torch.no_grad():
            next_values = self.critic_net(next_states)
            next_values[dones] = 0.0
        
        # Targets
        targets = rewards + self.gamma * next_values
        
        # MSE loss
        loss = nn.MSELoss()(current_values, targets)
        
        return loss
    
    def save(self, filepath: str) -> None:
        """
        Guardar estado del agente (implementa método abstracto).
        
        Args:
            filepath: Ruta donde guardar el checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor_net.state_dict(),
            'critic_state_dict': self.critic_net.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
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
        self.actor_net.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        
        # Restaurar parámetros
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        print(f"📂 Agente cargado desde: {filepath}")
        print(f"   Training step: {self.training_step}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del agente."""
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'actor_params': sum(p.numel() for p in self.actor_net.parameters()),
            'critic_params': sum(p.numel() for p in self.critic_net.parameters()),
            'device': str(self.device),
            'gamma': self.gamma
        }
