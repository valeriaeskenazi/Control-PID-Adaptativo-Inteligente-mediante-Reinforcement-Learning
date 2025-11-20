"""
ControllerAgent - Wrapper minimalista para usar DQNAgent.
"""

from typing import Tuple, Optional
import numpy as np
from Agent.DQN.algorithm_DQN import DQNAgent


class ControllerAgent:
    """Wrapper que usa DQNAgent para entrenar y retornar PIDs."""
    
    def __init__(self,
                 var_idx: int,
                 dqn_agent,
                 initial_pid: Tuple[float, float, float] = (1.0, 0.1, 0.05)):
        """
        Args:
            var_idx: Índice de la variable que controla
            dqn_agent: Instancia de tu DQNAgent
            initial_pid: PID inicial (Kp, Ki, Kd)
        """
        self.var_idx = var_idx
        self.dqn_agent = dqn_agent
        self.initial_pid = initial_pid
        self.pid_history = []
    
    def train(self,
          env,
          n_episodes: int,
          var_idx: int,
          setpoint: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Entrenar DQN y extraer PID.
        
        Args:
            env: Ambiente (puede ser single o multi-variable)
            n_episodes: Número de episodios de entrenamiento
            var_idx: Índice de la variable que controla este agente
            setpoint: Setpoint para la variable (opcional)
        
        Returns:
            Mejor PID (Kp, Ki, Kd)
        """
        # Configurar ambiente
        if setpoint is not None:
            env.set_setpoint(setpoint, var_idx=var_idx)
        
        # Detectar si es multi-variable
        is_multi_var = hasattr(env, 'n_variables') and env.n_variables > 1
        
        # Entrenar n episodios
        for episode in range(n_episodes):
            state, info = env.reset()
            done = False
            
            while not done:
                # Seleccionar acción para ESTA variable
                if is_multi_var:
                    # Extraer solo la observación de esta variable (6 dims por variable)
                    var_state = state[var_idx * 6:(var_idx + 1) * 6]
                else:
                    var_state = state
                
                action_var = self.dqn_agent.select_action(var_state, training=True)
                
                # Construir acción completa para el ambiente
                if is_multi_var:
                    # Acción "mantener" (índice 6) para otras variables
                    action = [6] * env.n_variables
                    action[var_idx] = action_var
                else:
                    action = action_var
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Extraer reward de esta variable si es multi-var
                #if is_multi_var and hasattr(env, 'last_rewards'):
                #    # Si el ambiente trackea rewards por variable
                #    reward_var = env.last_rewards[var_idx] if hasattr(env, 'last_rewards') else reward
                #else:
                #    reward_var = reward
                # Por ahora usar el reward total (simplificación)
                # Asegurar que el reward sea un escalar
                reward_var = float(reward) if not isinstance(reward, (list, np.ndarray)) else float(reward[0])      
                
                # Siguiente estado de esta variable
                if is_multi_var:
                    next_var_state = next_state[var_idx * 6:(var_idx + 1) * 6]
                else:
                    next_var_state = next_state
                
                self.dqn_agent.store_experience(
                    state=var_state, 
                    action=int(action_var), 
                    reward=float(reward_var),  # Asegurar que es float escalar
                    next_state=next_var_state, 
                    done=bool(done)
                )
                
                if len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
                    self.dqn_agent.update()
                
                state = next_state
        
        # Extraer PID del ambiente
        if is_multi_var:
            best_pid = env.pid_action_spaces[var_idx].get_current_pid()
        else:
            best_pid = env.pid_action_space.get_current_pid()
        
        self.pid_history.append(best_pid)
        
        return best_pid
    
    def has_previous_pid(self) -> bool:
        """¿Tiene PID previo?"""
        return len(self.pid_history) > 0
    
    def get_previous_pid(self) -> Tuple[float, float, float]:
        """Obtener último PID."""
        if not self.has_previous_pid():
            return self.initial_pid
        return self.pid_history[-1]
    
    def reset_weights(self) -> None:
        """Resetear pesos (experimento desde cero)."""
        
        old_config = {
            'state_dim': self.dqn_agent.state_dim,
            'action_dim': self.dqn_agent.action_dim,
            'hidden_dims': self.dqn_agent.hidden_dims,
            'lr': self.dqn_agent.optimizer.param_groups[0]['lr'],
            'gamma': self.dqn_agent.gamma,
            'device': str(self.dqn_agent.device)
        }
        
        self.dqn_agent = DQNAgent(**old_config)
        self.pid_history = []
    
    def clear_buffers(self) -> None:
        """Limpiar replay buffer (transfer learning)."""
        self.dqn_agent.memory.clear()


