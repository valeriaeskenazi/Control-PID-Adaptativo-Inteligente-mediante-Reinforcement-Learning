import numpy as np
from typing import Dict, Any


class AdaptiveRewardCalculator:
    """
    Calcula recompensas adaptativas según dificultad del proceso.
    
    Componentes de la recompensa:
    - Proporcional: Error actual
    - Integral: Error acumulado
    - Derivativo: Tasa de cambio del error
    - Energía: Penalización por esfuerzo de control excesivo
    
    Args:
        upper_range: Rango superior del proceso
        lower_range: Rango inferior del proceso
        dead_band: Banda muerta aceptable
    """
    
    def __init__(self, 
                 upper_range: float,
                 lower_range: float,
                 dead_band: float):
        
        self.upper_range = upper_range
        self.lower_range = lower_range
        self.dead_band = dead_band
        
        # Parámetros según dificultad
        self.difficulty_params = {
            "EASY": {
                'max_reward': 1.0,
                'min_positive': 0.1,
                'negative_reward': -2.0,
                'tolerance_factor': 1.0
            },
            "MEDIUM": {
                'max_reward': 1.0,
                'min_positive': 0.2,
                'negative_reward': -1.5,
                'tolerance_factor': 1.5
            },
            "DIFFICULT": {
                'max_reward': 1.0,
                'min_positive': 0.3,
                'negative_reward': -1.0,
                'tolerance_factor': 2.0
            },
            "UNKNOWN": {
                'max_reward': 1.0,
                'min_positive': 0.2,
                'negative_reward': -1.0,
                'tolerance_factor': 1.5
            }
        }
    
    def calculate(self,
                  pv: float,
                  setpoint: float,
                  error: float,
                  error_integral: float,
                  error_derivative: float,
                  control_output: float,
                  process_difficulty: str) -> float:
        """
        Calcular recompensa adaptativa.
        
        Args:
            pv: Variable de proceso actual
            setpoint: Punto de ajuste
            error: Error actual (setpoint - pv)
            error_integral: Error integral acumulado
            error_derivative: Derivada del error
            control_output: Señal de control aplicada
            process_difficulty: Dificultad del proceso ('EASY', 'MEDIUM', 'DIFFICULT', 'UNKNOWN')
        
        Returns:
            Recompensa calculada
        """
        error_abs = abs(error)
        
        # Componentes tipo PID
        proportional_component = -error_abs
        integral_component = -abs(error_integral) * 0.001
        derivative_component = -abs(error_derivative) * 0.1
        energy_penalty = -abs(control_output) * 0.05
        
        # Obtener parámetros según dificultad
        params = self.difficulty_params[process_difficulty]
        adjusted_dead_band = self.dead_band * params['tolerance_factor']
        
        # Calcular recompensa base
        base_reward = self._calculate_base_reward(
            pv, setpoint, error_abs, adjusted_dead_band, params
        )
        
        # Combinar todos los componentes
        total_reward = (
            base_reward +
            proportional_component * 0.1 +
            integral_component +
            derivative_component +
            energy_penalty
        )
        
        return float(np.clip(
            total_reward,
            params['negative_reward'],
            params['max_reward']
        ))
    
    def _calculate_base_reward(self,
                               pv: float,
                               setpoint: float,
                               error_abs: float,
                               adjusted_dead_band: float,
                               params: Dict[str, float]) -> float:
        """
        Calcular recompensa base adaptativa.
        
        Args:
            pv: Variable de proceso
            setpoint: Punto de ajuste
            error_abs: Error absoluto
            adjusted_dead_band: Banda muerta ajustada por dificultad
            params: Parámetros de recompensa
        
        Returns:
            Recompensa base
        """
        # Dentro de banda muerta: recompensa graduada
        if error_abs <= adjusted_dead_band:
            if error_abs == 0.0:
                precision_factor = 1.0
            else:
                precision_factor = 1.0 - (error_abs / adjusted_dead_band) * 0.2
            return params['max_reward'] * precision_factor
        
        # Fuera de rango: penalización fuerte
        if pv < self.lower_range or pv > self.upper_range:
            return params['negative_reward']
        
        # En rango pero fuera de banda muerta: interpolación lineal
        max_error_in_range = max(
            abs(setpoint - self.lower_range),
            abs(setpoint - self.upper_range)
        )
        
        if max_error_in_range <= adjusted_dead_band:
            return params['max_reward']
        
        normalized_error = (
            (error_abs - adjusted_dead_band) /
            (max_error_in_range - adjusted_dead_band)
        )
        normalized_error = np.clip(normalized_error, 0, 1)
        
        base_reward = (
            params['max_reward'] -
            normalized_error * (params['max_reward'] - params['min_positive'])
        )
        
        return max(base_reward, params['min_positive'])
    
    def get_difficulty_params(self, difficulty: str) -> Dict[str, float]:
        """
        Obtener parámetros para una dificultad específica.
        
        Args:
            difficulty: Dificultad del proceso
        
        Returns:
            Diccionario con parámetros
        """
        return self.difficulty_params.get(difficulty, self.difficulty_params["UNKNOWN"])