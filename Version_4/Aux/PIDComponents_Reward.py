import numpy as np
from typing import List


class RewardCalculator:    
    def __init__(self, 
                 weights=None,
                 manipulable_ranges=None,
                 dead_band=0.02):

        # Pesos por defecto
        if weights is None:
            self.weights = {
                'error': 1.0, # Que tan lejo del setpoint
                'tiempo': 0.01, # Que tan rápido responde el sistema
                'overshoot': 0.5, # Que tanto sobrepasa el setpoint
                'energy': 0.1 # Que tanto esfuerzo se necesita
            }
        else:
            self.weights = weights
        
        self.manipulable_ranges = manipulable_ranges
        self.dead_band = dead_band
    
    def calculate(self, 
                  errors: List[float],
                  tiempos_respuesta: List[float],
                  overshoots: List[float],
                  energy_step: float,
                  pvs: List[float],
                  setpoints: List[float],
                  terminated: bool,
                  truncated: bool) -> float:
        
        # REWARD INTERMEDIO (durante el episodio)
        if not terminated and not truncated:
            return self._calculate_step_reward(errors, tiempos_respuesta, overshoots, energy_step)
        
        # REWARD FINAL (episodio terminó)
        else:
            return self._calculate_episode_reward(errors, tiempos_respuesta, overshoots, 
                                                   energy_step, pvs, setpoints, terminated)
    
    def _calculate_step_reward(self, 
                               errors: List[float],
                               tiempos: List[float],
                               overshoots: List[float],
                               energy: float) -> float:

        # Promedios
        mean_error = sum(errors) / len(errors) if errors else 0
        mean_tiempo = sum(tiempos) / len(tiempos) if tiempos else 0
        mean_overshoot = sum(overshoots) / len(overshoots) if overshoots else 0
        
        # Reward (todos los componentes son penalizaciones)
        reward = (
            -self.weights['error'] * mean_error +
            -self.weights['tiempo'] * mean_tiempo +
            -self.weights['overshoot'] * mean_overshoot +
            -self.weights['energy'] * energy
        )
        
        return reward
    
    def _calculate_episode_reward(self, 
                                  errors: List[float],
                                  tiempos: List[float],
                                  overshoots: List[float],
                                  energy: float,
                                  pvs: List[float],
                                  setpoints: List[float],
                                  terminated: bool) -> float:

        # Base: reward del step
        step_reward = self._calculate_step_reward(errors, tiempos, overshoots, energy)
        
        # Verificar si todas las variables llegaron al objetivo (dentro del dead_band)
        success = all(
            abs(pv - sp) / abs(sp) < self.dead_band if sp != 0 else abs(pv) < self.dead_band
            for pv, sp in zip(pvs, setpoints)
        )
        
        # Amplificar reward según resultado
        if terminated and success:
            # Éxito: duplica el reward (si step_reward es negativo, lo hace menos negativo)
            return step_reward * 2.0
        elif terminated and not success:
            # Fallo: duplica la penalización (hace el reward más negativo)
            return step_reward * 2.0
        else:  # truncated
            # Truncado: 20% más penalización por no terminar
            return step_reward * 1.2
    
    def update_weights(self, new_weights: dict):
        """Actualizar pesos de los componentes."""
        self.weights.update(new_weights)
    
    def get_weights(self) -> dict:
        """Obtener pesos actuales."""
        return self.weights.copy()