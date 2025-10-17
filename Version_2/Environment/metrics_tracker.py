from typing import Dict, Any, Optional
from collections import deque
import numpy as np


class EpisodeMetricsTracker:
    """
    Rastrea métricas de rendimiento durante episodios.
    
    Métricas rastreadas:
    - Total de episodios
    - Episodios exitosos
    - Tiempo de asentamiento promedio
    - Error de estado estacionario promedio
    - Historial de recompensas
    
    Args:
        history_size: Tamaño del historial de métricas recientes
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        
        # Métricas acumuladas
        self.total_episodes = 0
        self.successful_episodes = 0
        
        # Promedios móviles
        self.avg_settling_time = 0.0
        self.avg_steady_state_error = 0.0
        self.avg_episode_reward = 0.0
        
        # Historial reciente
        self.reward_history = deque(maxlen=history_size)
        self.settling_time_history = deque(maxlen=history_size)
        self.error_history = deque(maxlen=history_size)
        
        # Episodio actual
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
    
    def start_episode(self) -> None:
        """Iniciar tracking de nuevo episodio."""
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
    
    def update_step(self, reward: float) -> None:
        """
        Actualizar métricas en cada step.
        
        Args:
            reward: Recompensa obtenida en el step
        """
        self.current_episode_reward += reward
        self.current_episode_steps += 1
    
    def end_episode(self,
                    settling_time: Optional[float] = None,
                    steady_state_error: Optional[float] = None,
                    success: bool = False) -> Dict[str, Any]:
        """
        Finalizar episodio y actualizar métricas.
        
        Args:
            settling_time: Tiempo de asentamiento del episodio [s]
            steady_state_error: Error de estado estacionario
            success: Si el episodio fue exitoso
        
        Returns:
            Resumen de métricas del episodio
        """
        self.total_episodes += 1
        
        if success:
            self.successful_episodes += 1
        
        # Actualizar historial
        self.reward_history.append(self.current_episode_reward)
        
        if settling_time is not None:
            self.settling_time_history.append(settling_time)
            self.avg_settling_time = np.mean(self.settling_time_history)
        
        if steady_state_error is not None:
            self.error_history.append(steady_state_error)
            self.avg_steady_state_error = np.mean(self.error_history)
        
        # Promedio de recompensas
        if len(self.reward_history) > 0:
            self.avg_episode_reward = np.mean(self.reward_history)
        
        # Resumen del episodio
        episode_summary = {
            'episode_number': self.total_episodes,
            'episode_reward': self.current_episode_reward,
            'episode_steps': self.current_episode_steps,
            'success': success,
            'settling_time': settling_time,
            'steady_state_error': steady_state_error
        }
        
        return episode_summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Obtener todas las métricas actuales.
        
        Returns:
            Diccionario con métricas
        """
        success_rate = (
            self.successful_episodes / self.total_episodes
            if self.total_episodes > 0 else 0.0
        )
        
        return {
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'success_rate': success_rate,
            'avg_settling_time': self.avg_settling_time,
            'avg_steady_state_error': self.avg_steady_state_error,
            'avg_episode_reward': self.avg_episode_reward,
            'recent_rewards': list(self.reward_history),
            'recent_settling_times': list(self.settling_time_history),
            'recent_errors': list(self.error_history)
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas resumidas (sin historial completo).
        
        Returns:
            Diccionario con estadísticas resumidas
        """
        metrics = self.get_metrics()
        
        # Remover historiales largos
        summary = {
            'total_episodes': metrics['total_episodes'],
            'successful_episodes': metrics['successful_episodes'],
            'success_rate': metrics['success_rate'],
            'avg_settling_time': metrics['avg_settling_time'],
            'avg_steady_state_error': metrics['avg_steady_state_error'],
            'avg_episode_reward': metrics['avg_episode_reward']
        }
        
        # Agregar estadísticas recientes (últimos 10)
        if len(self.reward_history) > 0:
            recent_10 = list(self.reward_history)[-10:]
            summary['last_10_avg_reward'] = np.mean(recent_10)
            summary['last_10_std_reward'] = np.std(recent_10)
        
        return summary
    
    def reset(self) -> None:
        """Resetear todas las métricas."""
        self.total_episodes = 0
        self.successful_episodes = 0
        self.avg_settling_time = 0.0
        self.avg_steady_state_error = 0.0
        self.avg_episode_reward = 0.0
        
        self.reward_history.clear()
        self.settling_time_history.clear()
        self.error_history.clear()
        
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0