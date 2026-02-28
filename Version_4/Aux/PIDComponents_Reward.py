import numpy as np
from typing import List, Dict, Optional
from .PIDComponentes_StabilityCriteria import StabilityCriteria


class RewardCalculator:    
    def __init__(self, 
                 weights: Optional[Dict] = None,
                 manipulable_ranges: Optional[List] = None, # Lista de tuplas (min, max) para cada variable controlada, usada para normalizar el error
                 dead_band: float = 0.02, # Porcentaje de error relativo al setpoint para considerar que se llegó al objetivo
                 max_time: float = 1800.0, # Tiempo máximo esperado para alcanzar el setpoint (en segundos), usado para normalizar el tiempo de respuesta
                 stability_config: Optional[Dict] = None): # Dict con parámetros para StabilityCriteria (opcional). Si None, usa valores por defecto
        
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
        
        self.manipulable_ranges = manipulable_ranges or [(0.0, 100.0)]
        self.dead_band = dead_band
        self.max_time = max_time

        # Max error posible por variable (para normalizar)
        self.max_errors = [r[1] - r[0] for r in self.manipulable_ranges]

        # Componente de estabilidad
        sc = stability_config or {}
        self.stability_checker = StabilityCriteria(
            error_increase_tolerance=sc.get('error_increase_tolerance', 1.5),
            max_sign_changes_ratio=sc.get('max_sign_changes_ratio', 0.2),
            max_abrupt_change_ratio=sc.get('max_abrupt_change_ratio', 0.05),
            abrupt_change_threshold=sc.get('abrupt_change_threshold', 0.3)
        )

    def calculate(self,
                  errors: List[float], # Error absoluto por variable [|e1|, |e2|, ...]
                  tiempos_respuesta: List[float], # Tiempo de respuesta por variable [t1, t2, ...]
                  overshoots: List[float], # Overshoot relativo por variable [(pv1 - sp1)/sp1, (pv2 - sp2)/sp2, ...]
                  energy_step: float, # Energía consumida en el step actual (normalizada)
                  pvs: List[float],
                  setpoints: List[float],
                  terminated: bool, #True si el episodio terminó (éxito o fallo)
                  truncated: bool, # True si se alcanzó max_steps
                  trajs_pv: Optional[List[List[float]]] = None, #Trayectorias de PV del ResponseTimeDetector
                  trajs_control: Optional[List[List[float]]] = None) -> float: #Trayectorias de control del ResponseTimeDetector
        
        # Evaluar estabilidad si se proporcionan trayectorias
        stability = None
        if trajs_pv is not None and trajs_control is not None:
            stability = self.stability_checker.check_all(trajs_pv, trajs_control, setpoints)

        if not terminated and not truncated:
            return self._calculate_step_reward(errors, tiempos_respuesta, overshoots,
                                               energy_step, stability)
        else:
            return self._calculate_episode_reward(errors, tiempos_respuesta, overshoots,
                                                  energy_step, pvs, setpoints,
                                                  terminated, stability)

    
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