"""
OrchestratorAgent - Usa Actor-Critic para decidir setpoints de variables manipulables.
"""

from typing import List, Tuple, Optional
import numpy as np
from Agent.Actor_Critic.algorithm_ActorCritic import ActorCriticAgent

class OrchestratorAgent:
    """
    Agente orquestador que decide setpoints para variables manipulables.
    
    Usa Actor-Critic para mapear:
    - Entrada: [pv_target_1, ..., pv_target_k, sp_target_1, ..., sp_target_k]
    - Salida: [sp_manip_1, ..., sp_manip_n] en rango [-1, 1]
    
    Args:
        actor_critic_agent: Instancia de ActorCriticAgent
        n_manipulable_vars: Número de variables manipulables
        sp_ranges: Rangos operativos [(min, max), ...] para cada variable
    """
    
    def __init__(self,
                 actor_critic_agent,
                 n_manipulable_vars: int,
                 sp_ranges: List[Tuple[float, float]]):
        
        self.actor_critic_agent = actor_critic_agent
        self.n_manipulable_vars = n_manipulable_vars
        self.sp_ranges = sp_ranges
        
        # Estado y acción previos (para actualización)
        self.last_state = None
        self.last_action = None
        
        print(f"✅ OrchestratorAgent creado")
        print(f"   Variables manipulables: {n_manipulable_vars}")
        print(f"   Rangos SP: {sp_ranges}")
    
    def decide_setpoints(
        self,
        pv_targets: List[float],
        sp_targets: List[float]
    ) -> List[float]:
        """
        Decidir setpoints para variables manipulables.
        
        Args:
            pv_targets: Valores actuales de variables objetivo
            sp_targets: Setpoints deseados de variables objetivo
        
        Returns:
            sp_manipulables: Lista de setpoints para cada variable manipulable
        """
        # Construir estado: concatenar pv_targets y sp_targets
        state = np.array(pv_targets + sp_targets, dtype=np.float32)
        
        # Actor-Critic selecciona acción (setpoints normalizados en [-1, 1])
        action_normalized = self.actor_critic_agent.select_action(state, training=True)
        
        # Desnormalizar a rangos reales
        sp_manipulables = self._denormalize_setpoints(action_normalized)
        
        # Guardar para actualización posterior
        self.last_state = state
        self.last_action = action_normalized
        
        return sp_manipulables
    
    def update_policy(self, reward_global: float) -> None:
        """
        Actualizar política del orquestador con recompensa global.
        
        Args:
            reward_global: Recompensa global obtenida con los SP decididos
        """
        if self.last_state is None:
            return
        
        # Como es one-step y no tenemos next_state real,
        # asumimos mismo estado (política estacionaria)
        batch_data = {
            'reward': reward_global,
            'next_state': self.last_state,  # Simplificación
            'done': False
        }
        
        # Actualizar Actor-Critic
        metrics = self.actor_critic_agent.update(batch_data)
        
        if metrics:
            print(f"      Orquestador actualizado: "
                  f"actor_loss={metrics.get('actor_loss', 0):.4f}, "
                  f"critic_loss={metrics.get('critic_loss', 0):.4f}")
    
    def _denormalize_setpoints(self, action_normalized: np.ndarray) -> List[float]:
        """
        Convertir acciones normalizadas [-1, 1] a rangos reales de SP.
        
        Args:
            action_normalized: Acciones en [-1, 1]
        
        Returns:
            sp_denormalizados: Setpoints en rangos reales
        """
        sp_list = []
        
        for i in range(self.n_manipulable_vars):
            min_sp, max_sp = self.sp_ranges[i]
            
            # Desnormalizar de [-1, 1] a [min_sp, max_sp]
            sp = min_sp + (action_normalized[i] + 1.0) * (max_sp - min_sp) / 2.0
            
            # Clip por seguridad
            sp = np.clip(sp, min_sp, max_sp)
            
            sp_list.append(float(sp))
        
        return sp_list
    
    def reset_weights(self) -> None:
        """Resetear pesos (experimento desde cero)."""

        
        old_config = {
            'state_dim': self.actor_critic_agent.state_dim,
            'action_dim': self.actor_critic_agent.action_dim,
            'hidden_dims': self.actor_critic_agent.hidden_dims,
            'lr_actor': self.actor_critic_agent.optimizer_actor.param_groups[0]['lr'],
            'lr_critic': self.actor_critic_agent.optimizer_critic.param_groups[0]['lr'],
            'gamma': self.actor_critic_agent.gamma,
            'device': str(self.actor_critic_agent.device)
        }
        
        self.actor_critic_agent = ActorCriticAgent(**old_config)
        self.last_state = None
        self.last_action = None
    
    def clear_buffers(self) -> None:
        """Limpiar buffers (no aplica para Actor-Critic one-step)."""
        self.last_state = None
        self.last_action = None
