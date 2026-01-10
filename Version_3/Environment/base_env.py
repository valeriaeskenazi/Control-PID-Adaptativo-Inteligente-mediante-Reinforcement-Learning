import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List, Union
from collections import deque
from abc import ABC, abstractmethod
import logging

import sys
sys.path.append('.')

from .pid_components import ResponseTimeDetector
from .difficulty_classifier import ProcessDifficultyClassifier
from .reward_system import AdaptiveRewardCalculator
from .metrics_tracker import EpisodeMetricsTracker



class BasePIDControlEnv(gym.Env, ABC):
    """
    Clase base abstracta para ambientes de control PID.
    
    Soporta:
    - Single-agent: n_variables=1 (comportamiento original)
    - Multi-agent: n_variables>1 (N agentes cooperativos)
    
    Implementa la estructura de Gymnasium y delega:
    - Clasificación de dificultad -> ProcessDifficultyClassifier (N instancias)
    - Cálculo de recompensas -> AdaptiveRewardCalculator (N instancias)
    - Tracking de métricas -> EpisodeMetricsTracker (N instancias)
    - Detección de respuesta -> ResponseTimeDetector (N instancias)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Configuración por defecto
        default_config = {
            'n_variables': 1,  #número de variables a controlar
            'upper_range': 100.0,
            'lower_range': 0.0,
            'setpoint': 75.0,
            'dead_band': 2.0,
            'max_episode_steps': 1000,
            'dt': 1.0,
            'enable_logging': False,
            'log_level': 'INFO',
            'cooperation_bonus': 0.5  # bonus cuando todos alcanzan objetivo
        }
        
        # Merge configuración
        self.config = {**default_config, **(config or {})}
        
        # Número de variables (agentes)
        self.n_variables = self.config['n_variables']
        
        # Validar y normalizar configuración
        self._normalize_config()
        self._validate_config(self.config)
        
        # Extraer parámetros principales (ahora son listas)
        self.upper_ranges = self.config['upper_range']
        self.lower_ranges = self.config['lower_range']
        self.setpoints = self.config['setpoint']
        self.dead_bands = self.config['dead_band']
        self.max_episode_steps = self.config['max_episode_steps']
        self.dt = self.config['dt']
        self.cooperation_bonus = self.config['cooperation_bonus']
        
        # Setup observation space (común para todos)
        self._setup_observation_space()
        
        # Variables de estado del proceso (ahora son listas)
        self.pvs = [0.0] * self.n_variables
        self.error_prevs = [0.0] * self.n_variables
        self.step_count = 0
        self.error_integrals = [0.0] * self.n_variables
        self.error_derivatives = [0.0] * self.n_variables
        self.error_histories = [deque(maxlen=10) for _ in range(self.n_variables)]
        self.out_of_range_counts = [0] * self.n_variables
        
        # Módulos especializados (N instancias, una por variable)
        self.response_detectors = [
            ResponseTimeDetector() 
            for _ in range(self.n_variables)
        ]
        
        self.difficulty_classifiers = [
            ProcessDifficultyClassifier()
            for _ in range(self.n_variables)
        ]
        
        self.reward_calculators = [
            AdaptiveRewardCalculator(
                upper_range=self.upper_ranges[i],
                lower_range=self.lower_ranges[i],
                dead_band=self.dead_bands[i]
            )
            for i in range(self.n_variables)
        ]
        
        self.metrics_trackers = [
            EpisodeMetricsTracker()
            for _ in range(self.n_variables)
        ]
        
        # Logging
        self.logger = None
        if self.config['enable_logging']:
            self._setup_logging(self.config['log_level'])
        
        # Info de configuración
        if self.n_variables == 1:
            print(f"Configurado como SINGLE-AGENT (1 variable)")
        else:
            print(f"Configurado como MULTI-AGENT ({self.n_variables} variables)")
    
    def _normalize_config(self) -> None:
        """
        Normalizar parámetros: si son escalares, replicarlos para N variables.
        
        Ejemplos:
        - Escalar: 100.0 → [100.0, 100.0, ..., 100.0]
        - Lista: [100.0, 90.0, 80.0] → se valida que tenga n_variables elementos
        """
        params_to_normalize = ['upper_range', 'lower_range', 'setpoint', 'dead_band']
        
        for param in params_to_normalize:
            value = self.config[param]
            
            # Si es escalar, replicar para todas las variables
            if isinstance(value, (int, float)):
                self.config[param] = [float(value)] * self.n_variables
            
            # Si es lista, validar longitud
            elif isinstance(value, (list, tuple)):
                if len(value) != self.n_variables:
                    raise ValueError(
                        f"{param} debe ser escalar o lista de longitud {self.n_variables}, "
                        f"recibido lista de longitud {len(value)}"
                    )
                self.config[param] = list(value)
            else:
                raise TypeError(
                    f"{param} debe ser escalar o lista, recibido tipo {type(value)}"
                )
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validar parámetros de configuración."""
        required_keys = [
            'n_variables', 'upper_range', 'lower_range', 'dead_band',
            'max_episode_steps', 'dt'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta parámetro requerido: {key}")
        
        if config['n_variables'] < 1:
            raise ValueError("n_variables debe ser >= 1")
        
        # Validar rangos para cada variable
        for i in range(self.n_variables):
            if config['upper_range'][i] <= config['lower_range'][i]:
                raise ValueError(
                    f"Variable {i}: upper_range debe ser mayor que lower_range"
                )
            
            if config['dead_band'][i] < 0:
                raise ValueError(f"Variable {i}: dead_band debe ser no negativo")
        
        if config['max_episode_steps'] <= 0:
            raise ValueError("max_episode_steps debe ser positivo")
        
        if config['dt'] <= 0:
            raise ValueError("dt debe ser positivo")
    
    def _setup_observation_space(self) -> None:
        """
        Definir espacio de observaciones.
        
        Single-agent (n_variables=1): 6 dimensiones [pv, sp, error, error_prev, integral, derivative]
        Multi-agent (n_variables>1): 6*N dimensiones (concatenación de todas las variables)
        """
        low_bounds = []
        high_bounds = []
        
        for i in range(self.n_variables):
            range_span = self.upper_ranges[i] - self.lower_ranges[i]
            
            low_bounds.extend([
                self.lower_ranges[i] - range_span * 0.2,  # PV
                self.lower_ranges[i],                      # SP
                -range_span,                               # error
                -range_span,                               # error_prev
                -range_span * 100,                         # error_integral
                -range_span * 10                           # error_derivative
            ])
            
            high_bounds.extend([
                self.upper_ranges[i] + range_span * 0.2,
                self.upper_ranges[i],
                range_span,
                range_span,
                range_span * 100,
                range_span * 10
            ])
        
        self.observation_space = spaces.Box(
            low=np.array(low_bounds),
            high=np.array(high_bounds),
            dtype=np.float32
        )
    
    @abstractmethod
    def _setup_action_space(self) -> None:
        """Definir espacio de acciones (implementar en clases hijas)."""
        pass
    
    @abstractmethod
    def _apply_control(self, action) -> Tuple[List[Optional[float]], List[Optional[Tuple]]]:
        """
        Aplicar acción al proceso (implementar en clases hijas).
        
        Args:
            action: Acción del agente (puede ser lista de acciones en multi-agent)
        
        Returns:
            Tuple con:
            - control_outputs: Lista de N control outputs [control_1, ..., control_N]
            - pid_params_list: Lista de N tuplas de PID params [(kp_1,ki_1,kd_1), ..., (kp_N,ki_N,kd_N)]
        """
        pass
    
    @abstractmethod
    def _update_process(self, 
                        control_outputs: List[Optional[float]],
                        pid_params_list: List[Optional[Tuple]]) -> List[float]:
        """
        Actualizar el proceso físico (implementar en clases hijas).
        
        Args:
            control_outputs: Lista de N señales de control
            pid_params_list: Lista de N tuplas de parámetros PID
        
        Returns:
            Lista de nuevos valores de PV [pv_1, ..., pv_N]
        """
        pass
    
    def step(self, action) -> Tuple[np.ndarray, Union[float, List[float]], bool, bool, Dict[str, Any]]:
        """
        Ejecutar un paso en el ambiente.
        
        Args:
            action: Acción del agente (escalar para single-agent, lista para multi-agent)
        
        Returns:
            Tuple con (observación, recompensa, terminated, truncated, info)
            - recompensa: float (single-agent) o List[float] (multi-agent)
        """
        # 1. Aplicar control
        control_outputs, pid_params_list = self._apply_control(action)
        
        # 2. Actualizar proceso
        self.pvs = self._update_process(control_outputs, pid_params_list)
        
        # 3. Calcular errores para cada variable
        errors = [self.setpoints[i] - self.pvs[i] for i in range(self.n_variables)]
        
        # 4. Actualizar tracking de errores
        for i in range(self.n_variables):
            self.error_histories[i].append(errors[i])
            self.error_integrals[i] += errors[i] * self.dt
            
            # Calcular derivada
            if len(self.error_histories[i]) >= 2:
                self.error_derivatives[i] = (
                    (self.error_histories[i][-1] - self.error_histories[i][-2]) / self.dt
                )
            else:
                self.error_derivatives[i] = 0.0
        
        # 5. Detectar tiempo de respuesta y clasificar dificultad para cada variable
        estimated_response_times = []
        process_difficulties = []
        
        for i in range(self.n_variables):
            control_out = control_outputs[i] if control_outputs[i] is not None else 0.0
            
            estimated_rt = self.response_detectors[i].update(
                control_out, self.pvs[i], self.setpoints[i], self.dt
            )
            estimated_response_times.append(estimated_rt)
            
            difficulty = self.difficulty_classifiers[i].classify(estimated_rt)
            process_difficulties.append(difficulty)
        
        # 6. Calcular recompensas individuales para cada variable
        individual_rewards = []
        
        for i in range(self.n_variables):
            control_out = control_outputs[i] if control_outputs[i] is not None else 0.0
            
            reward_i = self.reward_calculators[i].calculate(
                pv=self.pvs[i],
                setpoint=self.setpoints[i],
                error=errors[i],
                error_integral=self.error_integrals[i],
                error_derivative=self.error_derivatives[i],
                control_output=control_out,
                process_difficulty=process_difficulties[i]
            )
            individual_rewards.append(reward_i)
            
            # Actualizar métricas por variable
            self.metrics_trackers[i].update_step(reward_i)
        
        # 7. Calcular bonus de cooperación (si todas las variables están en banda muerta)
        all_in_deadband = all(
            abs(errors[i]) <= self.dead_bands[i] 
            for i in range(self.n_variables)
        )
        
        if all_in_deadband and self.n_variables > 1:
            cooperation_bonus = self.cooperation_bonus
        else:
            cooperation_bonus = 0.0
        
        # 8. Recompensas finales (individuales + bonus cooperativo)
        final_rewards = [r + cooperation_bonus for r in individual_rewards]
        
        # 9. Actualizar estado
        self.error_prevs = errors.copy()
        self.step_count += 1
        
        # 10. Condiciones de término
        terminated = self.step_count >= self.max_episode_steps
        
        # Truncar si alguna variable se sale de control
        truncated = any(
            self._check_truncation(i, errors[i], process_difficulties[i])
            for i in range(self.n_variables)
        )
        
        # 11. Preparar info
        info = {
            'process_difficulties': process_difficulties,
            'estimated_response_times': estimated_response_times,
            'step_count': self.step_count,
            'cooperation_bonus': cooperation_bonus,
            'all_in_deadband': all_in_deadband,
            'individual_rewards': individual_rewards,
            'current_pvs': self.pvs.copy(),  # ✅ AGREGAR ESTA LÍNEA
            'setpoints': self.setpoints.copy()  # ✅ Y ESTA
        }
        
        # Agregar info específico si aplica
        if any(p is not None for p in pid_params_list):
            info['pid_params_list'] = pid_params_list
        if any(c is not None for c in control_outputs):
            info['control_outputs'] = [float(c) if c is not None else None 
                                       for c in control_outputs]
        
        # Formato de recompensa según modo
        if self.n_variables == 1:
            reward_output = final_rewards[0]  # Escalar para single-agent
        else:
            reward_output = final_rewards  # Lista para multi-agent
        
        return self._get_observation(), reward_output, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resetear el ambiente.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
        
        Returns:
            Tuple con (observación inicial, info)
        """
        super().reset(seed=seed)
        
        # Reset variables de estado (todas las variables)
        self.error_prevs = [0.0] * self.n_variables
        self.step_count = 0
        self.error_integrals = [0.0] * self.n_variables
        self.error_derivatives = [0.0] * self.n_variables
        self.out_of_range_counts = [0] * self.n_variables
        
        for history in self.error_histories:
            history.clear()
        
        # Reset módulos especializados (todos)
        for i in range(self.n_variables):
            self.response_detectors[i].reset()
            self.difficulty_classifiers[i].reset()
            self.metrics_trackers[i].start_episode()
        
        if self.logger:
            total_episodes = sum(
                tracker.total_episodes for tracker in self.metrics_trackers
            ) // self.n_variables
            self.logger.debug(f"Environment reset - Episode {total_episodes + 1}")
        
        return self._get_observation(), {}
    
    def _check_truncation(self, var_idx: int, error: float, 
                         process_difficulty: str) -> bool:
        """
        Verificar criterios de truncamiento adaptativos para una variable.
        
        Args:
            var_idx: Índice de la variable
            error: Error actual
            process_difficulty: Dificultad del proceso
        
        Returns:
            True si se debe truncar el episodio
        """
        threshold_multiplier = {
            "EASY": 0.5,
            "MEDIUM": 0.7,
            "DIFFICULT": 1.0,
            "UNKNOWN": 0.6
        }
        
        multiplier = threshold_multiplier.get(process_difficulty, 0.6)
        range_span = self.upper_ranges[var_idx] - self.lower_ranges[var_idx]
        max_allowed_error = range_span * multiplier
        
        if abs(error) > max_allowed_error:
            self.out_of_range_counts[var_idx] += 1
            
            # Paciencia según dificultad
            patience = {
                "EASY": 20,
                "MEDIUM": 50,
                "DIFFICULT": 100,
                "UNKNOWN": 30
            }
            max_patience = patience.get(process_difficulty, 30)
            
            return self.out_of_range_counts[var_idx] > max_patience
        else:
            self.out_of_range_counts[var_idx] = 0
            return False
    
    def _get_observation(self) -> np.ndarray:
        """
        Obtener observación actual del proceso.
        
        Single-agent: [pv, sp, error, error_prev, integral, derivative]
        Multi-agent: concatenación de todas las variables
        """
        obs = []
        
        for i in range(self.n_variables):
            error = self.setpoints[i] - self.pvs[i]
            
            obs.extend([
                self.pvs[i],
                self.setpoints[i],
                error,
                self.error_prevs[i],
                self.error_integrals[i],
                self.error_derivatives[i]
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def set_setpoint(self, new_setpoint: Union[float, List[float]], 
                     var_idx: Optional[int] = None) -> None:
        """
        Cambiar el setpoint del proceso.
        
        Args:
            new_setpoint: Nuevo valor objetivo (escalar o lista)
            var_idx: Índice de variable (si None, aplica a todas en single-agent)
        """
        # Si es escalar y single-agent
        if isinstance(new_setpoint, (int, float)) and self.n_variables == 1:
            new_setpoint = float(new_setpoint)
            if not (self.lower_ranges[0] <= new_setpoint <= self.upper_ranges[0]):
                raise ValueError(
                    f"Setpoint {new_setpoint} debe estar entre "
                    f"{self.lower_ranges[0]} y {self.upper_ranges[0]}"
                )
            self.setpoints[0] = new_setpoint
        
        # Si es escalar y se especifica índice
        elif isinstance(new_setpoint, (int, float)) and var_idx is not None:
            new_setpoint = float(new_setpoint)
            if not (self.lower_ranges[var_idx] <= new_setpoint <= self.upper_ranges[var_idx]):
                raise ValueError(
                    f"Setpoint {new_setpoint} para variable {var_idx} debe estar entre "
                    f"{self.lower_ranges[var_idx]} y {self.upper_ranges[var_idx]}"
                )
            self.setpoints[var_idx] = new_setpoint
        
        # Si es lista
        elif isinstance(new_setpoint, (list, tuple)):
            if len(new_setpoint) != self.n_variables:
                raise ValueError(
                    f"new_setpoint debe tener {self.n_variables} elementos, "
                    f"recibido {len(new_setpoint)}"
                )
            
            for i, sp in enumerate(new_setpoint):
                if not (self.lower_ranges[i] <= sp <= self.upper_ranges[i]):
                    raise ValueError(
                        f"Setpoint {sp} para variable {i} debe estar entre "
                        f"{self.lower_ranges[i]} y {self.upper_ranges[i]}"
                    )
            
            self.setpoints = list(new_setpoint)
        
        else:
            raise ValueError(
                "new_setpoint debe ser escalar (con var_idx) o lista de longitud n_variables"
            )
        
        if self.logger:
            self.logger.info(f"Setpoint(s) cambiado(s) a: {self.setpoints}")
    
    def render(self, mode: str = 'human') -> None:
        """Renderizar estado del ambiente."""
        if mode == 'human':
            difficulties = [clf.get_difficulty() for clf in self.difficulty_classifiers]
            
            print(f"\n{'='*80}")
            print(f"Step: {self.step_count:4d}")
            print(f"{'='*80}")
            
            for i in range(self.n_variables):
                error = self.setpoints[i] - self.pvs[i]
                print(
                    f"Var {i}: "
                    f"PV={self.pvs[i]:6.2f} | "
                    f"SP={self.setpoints[i]:6.2f} | "
                    f"Error={error:6.2f} | "
                    f"I={self.error_integrals[i]:6.2f} | "
                    f"D={self.error_derivatives[i]:6.2f} | "
                    f"Diff={difficulties[i]}"
                )
            print(f"{'='*80}\n")
    
    def get_metrics(self, var_idx: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Obtener métricas del tracker.
        
        Args:
            var_idx: Índice de variable (si None, retorna todas)
        
        Returns:
            Métricas de una variable o lista de todas
        """
        if var_idx is not None:
            return self.metrics_trackers[var_idx].get_metrics()
        else:
            return [tracker.get_metrics() for tracker in self.metrics_trackers]
    
    def get_summary_stats(self, var_idx: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Obtener estadísticas resumidas.
        
        Args:
            var_idx: Índice de variable (si None, retorna todas)
        
        Returns:
            Estadísticas de una variable o lista de todas
        """
        if var_idx is not None:
            return self.metrics_trackers[var_idx].get_summary_stats()
        else:
            return [tracker.get_summary_stats() for tracker in self.metrics_trackers]
    
    def _setup_logging(self, log_level: str = 'INFO') -> None:
        """Configurar logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
