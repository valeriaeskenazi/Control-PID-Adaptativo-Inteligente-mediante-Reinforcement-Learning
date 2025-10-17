
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from collections import deque
from abc import ABC, abstractmethod
import logging

from .pid_components import ResponseTimeDetector
from .difficulty_classifier import ProcessDifficultyClassifier
from .reward_system import AdaptiveRewardCalculator
from .metrics_tracker import EpisodeMetricsTracker


class BasePIDControlEnv(gym.Env, ABC):
    """
    Clase base abstracta para ambientes de control PID.
    
    Implementa la estructura de Gymnasium y delega:
    - Clasificación de dificultad -> ProcessDifficultyClassifier
    - Cálculo de recompensas -> AdaptiveRewardCalculator
    - Tracking de métricas -> EpisodeMetricsTracker
    - Detección de respuesta -> ResponseTimeDetector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Configuración por defecto
        default_config = {
            'upper_range': 100.0,
            'lower_range': 0.0,
            'setpoint': 75.0,
            'dead_band': 2.0,
            'max_episode_steps': 1000,
            'dt': 1.0,
            'enable_logging': False,
            'log_level': 'INFO'
        }
        
        # Merge configuración
        self.config = {**default_config, **(config or {})}
        self._validate_config(self.config)
        
        # Extraer parámetros principales
        self.upper_range = self.config['upper_range']
        self.lower_range = self.config['lower_range']
        self.setpoint = self.config['setpoint']
        self.dead_band = self.config['dead_band']
        self.max_episode_steps = self.config['max_episode_steps']
        self.dt = self.config['dt']
        
        # Setup observation space (común para todos)
        self._setup_observation_space()
        
        # Variables de estado del proceso
        self.pv = 0.0
        self.error_prev = 0.0
        self.step_count = 0
        self.error_integral = 0.0
        self.error_derivative = 0.0
        self.error_history = deque(maxlen=10)
        self.out_of_range_count = 0
        
        # Módulos especializados
        self.response_detector = ResponseTimeDetector()
        self.difficulty_classifier = ProcessDifficultyClassifier()
        self.reward_calculator = AdaptiveRewardCalculator(
            upper_range=self.upper_range,
            lower_range=self.lower_range,
            dead_band=self.dead_band
        )
        self.metrics_tracker = EpisodeMetricsTracker()
        
        # Logging
        self.logger = None
        if self.config['enable_logging']:
            self._setup_logging(self.config['log_level'])
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validar parámetros de configuración."""
        required_keys = [
            'upper_range', 'lower_range', 'dead_band',
            'max_episode_steps', 'dt'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Falta parámetro requerido: {key}")
        
        if config['upper_range'] <= config['lower_range']:
            raise ValueError("upper_range debe ser mayor que lower_range")
        
        if config['dead_band'] < 0:
            raise ValueError("dead_band debe ser no negativo")
        
        if config['max_episode_steps'] <= 0:
            raise ValueError("max_episode_steps debe ser positivo")
        
        if config['dt'] <= 0:
            raise ValueError("dt debe ser positivo")
    
    def _setup_observation_space(self) -> None:
        """Definir espacio de observaciones (6 dimensiones)."""
        range_span = self.upper_range - self.lower_range
        
        self.observation_space = spaces.Box(
            low=np.array([
                self.lower_range - range_span * 0.2,  # PV
                self.lower_range,                      # SP
                -range_span,                           # error
                -range_span,                           # error_prev
                -range_span * 100,                     # error_integral
                -range_span * 10                       # error_derivative
            ]),
            high=np.array([
                self.upper_range + range_span * 0.2,
                self.upper_range,
                range_span,
                range_span,
                range_span * 100,
                range_span * 10
            ]),
            dtype=np.float32
        )
    
    @abstractmethod
    def _setup_action_space(self) -> None:
        """Definir espacio de acciones (implementar en clases hijas)."""
        pass
    
    @abstractmethod
    def _apply_control(self, action) -> Tuple[Optional[float], Optional[Tuple]]:
        """
        Aplicar acción al proceso (implementar en clases hijas).
        
        Args:
            action: Acción del agente
        
        Returns:
            Tuple con (control_output, pid_params)
        """
        pass
    
    @abstractmethod
    def _update_process(self, control_output: Optional[float],
                        pid_params: Optional[Tuple]) -> float:
        """
        Actualizar el proceso físico (implementar en clases hijas).
        
        Args:
            control_output: Señal de control (puede ser None en PLC real)
            pid_params: Parámetros PID (puede ser None en control directo)
        
        Returns:
            Nuevo valor de PV
        """
        pass
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecutar un paso en el ambiente.
        
        Args:
            action: Acción del agente
        
        Returns:
            Tuple con (observación, recompensa, terminated, truncated, info)
        """
        # 1. Aplicar control
        control_output, pid_params = self._apply_control(action)
        
        # 2. Actualizar proceso
        self.pv = self._update_process(control_output, pid_params)
        
        # 3. Calcular error
        error = self.setpoint - self.pv
        
        # 4. Actualizar tracking de error
        self.error_history.append(error)
        self.error_integral += error * self.dt
        
        # 5. Calcular derivada
        if len(self.error_history) >= 2:
            self.error_derivative = (
                (self.error_history[-1] - self.error_history[-2]) / self.dt
            )
        else:
            self.error_derivative = 0.0
        
        # 6. Detectar tiempo de respuesta y clasificar dificultad
        if control_output is not None:
            estimated_response_time = self.response_detector.update(
                control_output, self.pv, self.setpoint, self.dt
            )
            process_difficulty = self.difficulty_classifier.classify(
                estimated_response_time
            )
        else:
            estimated_response_time = None
            process_difficulty = self.difficulty_classifier.get_difficulty()
        
        # 7. Calcular recompensa usando módulo especializado
        reward = self.reward_calculator.calculate(
            pv=self.pv,
            setpoint=self.setpoint,
            error=error,
            error_integral=self.error_integral,
            error_derivative=self.error_derivative,
            control_output=control_output or 0.0,
            process_difficulty=process_difficulty
        )
        
        # 8. Actualizar métricas
        self.metrics_tracker.update_step(reward)
        
        # 9. Actualizar estado
        self.error_prev = error
        self.step_count += 1
        
        # 10. Condiciones de término
        terminated = self.step_count >= self.max_episode_steps
        truncated = self._check_truncation(error, process_difficulty)
        
        # 11. Preparar info
        info = {
            'process_difficulty': process_difficulty,
            'estimated_response_time': estimated_response_time,
            'step_count': self.step_count
        }
        
        # Agregar info específico si aplica
        if pid_params is not None:
            info['pid_params'] = pid_params
        if control_output is not None:
            info['control_output'] = float(control_output)
        
        return self._get_observation(), reward, terminated, truncated, info
    
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
        
        # Reset variables de estado
        self.error_prev = 0.0
        self.step_count = 0
        self.error_integral = 0.0
        self.error_derivative = 0.0
        self.error_history.clear()
        self.out_of_range_count = 0
        
        # Reset módulos especializados
        self.response_detector.reset()
        self.difficulty_classifier.reset()
        
        # Iniciar tracking de episodio
        self.metrics_tracker.start_episode()
        
        if self.logger:
            self.logger.debug(
                f"Environment reset - Episode "
                f"{self.metrics_tracker.total_episodes + 1}"
            )
        
        return self._get_observation(), {}
    
    def _check_truncation(self, error: float, process_difficulty: str) -> bool:
        """
        Verificar criterios de truncamiento adaptativos.
        
        Args:
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
        max_allowed_error = (self.upper_range - self.lower_range) * multiplier
        
        if abs(error) > max_allowed_error:
            self.out_of_range_count += 1
            
            # Paciencia según dificultad
            patience = {
                "EASY": 20,
                "MEDIUM": 50,
                "DIFFICULT": 100,
                "UNKNOWN": 30
            }
            max_patience = patience.get(process_difficulty, 30)
            
            return self.out_of_range_count > max_patience
        else:
            self.out_of_range_count = 0
            return False
    
    def _get_observation(self) -> np.ndarray:
        """Obtener observación actual del proceso."""
        error = self.setpoint - self.pv
        
        return np.array([
            self.pv,
            self.setpoint,
            error,
            self.error_prev,
            self.error_integral,
            self.error_derivative
        ], dtype=np.float32)
    
    def set_setpoint(self, new_setpoint: float) -> None:
        """
        Cambiar el setpoint del proceso.
        
        Args:
            new_setpoint: Nuevo valor objetivo
        """
        if not (self.lower_range <= new_setpoint <= self.upper_range):
            raise ValueError(
                f"Setpoint {new_setpoint} debe estar entre "
                f"{self.lower_range} y {self.upper_range}"
            )
        
        self.setpoint = new_setpoint
        
        if self.logger:
            self.logger.info(f"Setpoint cambiado a: {new_setpoint}")
    
    def render(self, mode: str = 'human') -> None:
        """Renderizar estado del ambiente."""
        if mode == 'human':
            error = self.setpoint - self.pv
            difficulty = self.difficulty_classifier.get_difficulty()
            
            print(
                f"Step: {self.step_count:4d} | "
                f"PV: {self.pv:6.2f} | "
                f"SP: {self.setpoint:6.2f} | "
                f"Error: {error:6.2f} | "
                f"Integral: {self.error_integral:6.2f} | "
                f"Derivative: {self.error_derivative:6.2f} | "
                f"Difficulty: {difficulty}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del tracker."""
        return self.metrics_tracker.get_metrics()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas resumidas."""
        return self.metrics_tracker.get_summary_stats()
    
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