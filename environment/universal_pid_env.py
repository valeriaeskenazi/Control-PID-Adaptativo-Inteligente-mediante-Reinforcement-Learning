import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, Union
from collections import deque
import logging

class UniversalPIDControlEnv(gym.Env):
    """
    Universal adaptive environment for PID control with reinforcement learning.
    
    Automatically adapts difficulty based on process response time:
    - EASY: response_time < 60s (linear, fast processes)  
    - MEDIUM: 60s <= response_time < 1800s (moderate deadtime, slight non-linearity)
    - DIFFICULT: response_time >= 1800s (large deadtime, highly non-linear)
    
    Features:
    - Automatic process difficulty detection
    - Adaptive reward functions
    - External process integration
    - Comprehensive logging and metrics
    - Performance optimizations
    
    Args:
        config: Configuration dictionary with environment parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Default configuration
        default_config = {
            'upper_range': 100.0, #Limite fisico de seguridad superior del proceso
            'lower_range': 0.0, #Limite fisico de seguridad inferior del proceso
            'setpoint': 75.0, # Valor objetivo del proceso (target)
            'dead_band': 2.0, # Banda de trabajo alrededor del setpoint
            'max_episode_steps': 1000,
            'dt': 1.0,  # Time step [s]
            'enable_logging': False,
            'log_level': 'INFO'
        }
        
        if config is None:
            config = default_config
        else:
            # Validate and merge with defaults
            config = {**default_config, **config}
            self._validate_config(config)
        
        self.upper_range = config['upper_range']
        self.lower_range = config['lower_range'] 
        self.setpoint = config['setpoint']
        self.dead_band = config['dead_band']
        self.max_episode_steps = config['max_episode_steps']
        self.dt = config['dt']
        
        # Umbrales para clasificación automática
        self.EASY_THRESHOLD = 60.0      # < 60s = proceso fácil
        self.DIFFICULT_THRESHOLD = 1800.0  # >= 30min = proceso difícil
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32 # Control output normalized (-1 to 1)
        )
        
        range_span = self.upper_range - self.lower_range
        # Enhanced observation space with 6 dimensions
        self.observation_space = spaces.Box(
            low=np.array([
                self.lower_range - range_span*0.2,  # Límite inferior posible para PV
                self.lower_range,                    # Límite inferior posible para setpoint
                -range_span,                         # Límite inferior para error
                -range_span,                         # Límite inferior para error anterior
                -range_span*100,                     # Límite inferior para error integral
                -range_span*10                       # Límite inferior para error derivativo
            ]),
            high=np.array([
                self.upper_range + range_span*0.2,  # Límite superior posible para PV
                self.upper_range,                    # Límite superior posible para setpoint
                range_span,                          # Límite superior para error
                range_span,                          # Límite superior para error anterior
                range_span*100,                      # Límite superior para error integral
                range_span*10                        # Límite superior para error derivativo
            ]),
            dtype=np.float32
        )
        
        # Variables para detección de tiempo de respuesta
        self.response_time_detector = ResponseTimeDetector()
        self.process_difficulty = "UNKNOWN"  # EASY, MEDIUM, DIFFICULT
        
        # Initialize logging if enabled
        self.logger = None
        if config['enable_logging']:
            self._setup_logging(config['log_level'])
        
        # Performance tracking
        self.episode_metrics = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'avg_settling_time': 0.0,
            'avg_steady_state_error': 0.0
        }
        
        # Integral and derivative tracking for enhanced rewards
        self.error_integral = 0.0
        self.error_derivative = 0.0
        self.error_history = deque(maxlen=10)  # For derivative calculation
        
        self.external_process = None
        self.reset()
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required_keys = ['upper_range', 'lower_range', 'dead_band', 'max_episode_steps', 'dt']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        if config['upper_range'] <= config['lower_range']:
            raise ValueError("upper_range must be greater than lower_range")
        
        if config['dead_band'] < 0:
            raise ValueError("dead_band must be non-negative")
        
        if config['max_episode_steps'] <= 0:
            raise ValueError("max_episode_steps must be positive")
        
        if config['dt'] <= 0:
            raise ValueError("dt must be positive")
    
    def _setup_logging(self, log_level: str = 'INFO') -> None:
        """Setup logging for the environment"""
        self.logger = logging.getLogger('UniversalPIDEnv')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def connect_external_process(self, process_simulator) -> None:
        """Connect external process simulator with validation"""
        required_methods = ['step', 'get_initial_pv']
        for method in required_methods:
            if not hasattr(process_simulator, method):
                raise ValueError(
                    f"Process simulator must implement '{method}' method"
                )
        
        self.external_process = process_simulator
        if self.logger:
            self.logger.info("External process connected successfully")
        
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Setpoint se toma de la configuración inicial (no se modifica en reset)        
        # El PV inicial debe venir del proceso real/simulado, no inventarse
        if self.external_process is not None:
            self.pv = self.external_process.get_initial_pv()
        else:
            # Solo para testing sin proceso conectado
            self.pv = self.setpoint + np.random.uniform(-5, 5)
            print("WARNING: Proceso no conectado, usando PV dummy para testing")
            
        self.error_prev = 0.0
        self.step_count = 0
        
        # Reset enhanced tracking variables
        self.error_integral = 0.0
        self.error_derivative = 0.0
        self.error_history.clear()
        
        # Reset response time detector
        self.response_time_detector.reset()
        self.process_difficulty = "UNKNOWN"
        
        # Reset out of range counter
        self.out_of_range_count = 0
        
        # Update episode metrics
        self.episode_metrics['total_episodes'] += 1
        
        if self.logger:
            self.logger.debug(f"Environment reset - Episode {self.episode_metrics['total_episodes']}")
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        control_output = action[0]
        
        # Obtener nueva PV del proceso externo
        if self.external_process is not None:
            self.pv = self.external_process.step(control_output, self.setpoint)
        else:
            # Simulador dummy
            self.pv += control_output * 0.5 + np.random.normal(0, 0.1)
            
        error = self.setpoint - self.pv
        
        # Update error tracking for enhanced rewards
        self.error_history.append(error)
        self.error_integral += error * self.dt
        
        # Calculate error derivative
        if len(self.error_history) >= 2:
            self.error_derivative = (self.error_history[-1] - self.error_history[-2]) / self.dt
        else:
            self.error_derivative = 0.0
        
        # Detect process response time with noise filtering
        estimated_response_time = self.response_time_detector.update(
            control_output, self.pv, self.setpoint, self.dt
        )
        
        # Clasificar dificultad del proceso basado en tiempo de respuesta
        if estimated_response_time is not None:
            self.process_difficulty = self._classify_difficulty(estimated_response_time)
        
        # Calculate enhanced adaptive reward
        reward = self._calculate_enhanced_reward(
            self.pv, self.setpoint, error, control_output
        )
        
        # Actualizar estado
        self.error_prev = error
        self.step_count += 1
        
        # Condiciones de término
        terminated = self.step_count >= self.max_episode_steps
        truncated = self._check_truncation(error)
            
        return self._get_observation(), reward, terminated, truncated, {
            'process_difficulty': self.process_difficulty,
            'estimated_response_time': estimated_response_time
        }
    
    def _classify_difficulty(self, response_time):
        """Clasificar dificultad según tiempo de respuesta"""
        if response_time < self.EASY_THRESHOLD:
            return "EASY"
        elif response_time < self.DIFFICULT_THRESHOLD:
            return "MEDIUM" 
        else:
            return "DIFFICULT"
    
    def _calculate_enhanced_reward(self, pv: float, setpoint: float, 
                                  error: float, control_output: float) -> float:
        """
        Enhanced reward function with PID-like terms and energy penalty.
        
        Components:
        - Proportional: Current error
        - Integral: Accumulated error over time
        - Derivative: Rate of error change
        - Energy: Penalty for excessive control effort
        """
        error_abs = abs(error)
        
        # PID-like reward components
        proportional_component = -error_abs
        integral_component = -abs(self.error_integral) * 0.001  # Small weight to prevent windup
        derivative_component = -abs(self.error_derivative) * 0.1  # Penalize rapid changes
        energy_penalty = -abs(control_output) * 0.05  # Penalize excessive control effort
        
        # Definir parámetros según dificultad
        if self.process_difficulty == "EASY":
            max_reward = 1.0
            min_positive = 0.1
            negative_reward = -2.0
            tolerance_factor = 1.0
            
        elif self.process_difficulty == "MEDIUM":
            max_reward = 1.0
            min_positive = 0.2  # Más tolerante al error
            negative_reward = -1.5
            tolerance_factor = 1.5  # Banda muerta más grande
            
        elif self.process_difficulty == "DIFFICULT":
            max_reward = 1.0
            min_positive = 0.3  # Muy tolerante al error
            negative_reward = -1.0
            tolerance_factor = 2.0  # Banda muerta aún más grande
            
        else:  # UNKNOWN - usar configuración conservadora
            max_reward = 1.0
            min_positive = 0.2
            negative_reward = -1.0
            tolerance_factor = 1.5
        
        adjusted_dead_band = self.dead_band * tolerance_factor
        
        # Base adaptive reward logic with graduated precision within dead band
        base_reward = 0.0
        if error_abs <= adjusted_dead_band:
            # Graduated reward: maximum at exact setpoint, decreasing within dead band
            if error_abs == 0.0:
                precision_factor = 1.0  # Perfect setpoint matching
            else:
                # Linear decrease from 1.0 to 0.8 as error approaches dead band limit
                precision_factor = 1.0 - (error_abs / adjusted_dead_band) * 0.2
            base_reward = max_reward * precision_factor
            
        elif pv < self.lower_range or pv > self.upper_range:
            base_reward = negative_reward
            
        else:
            # Linear interpolation within range
            max_error_in_range = max(
                abs(setpoint - self.lower_range),
                abs(setpoint - self.upper_range)
            )
            
            if max_error_in_range <= adjusted_dead_band:
                base_reward = max_reward
            else:
                normalized_error = (error_abs - adjusted_dead_band) / (max_error_in_range - adjusted_dead_band)
                normalized_error = np.clip(normalized_error, 0, 1)
                base_reward = max_reward - normalized_error * (max_reward - min_positive)
                base_reward = max(base_reward, min_positive)
        
        # Combine all reward components
        total_reward = (
            base_reward + 
            proportional_component * 0.1 +  # Small weight for proportional
            integral_component + 
            derivative_component + 
            energy_penalty
        )
        
        return float(np.clip(total_reward, negative_reward, max_reward))
    
    
    def _check_truncation(self, error: float) -> bool:
        """Criterios de truncamiento adaptativos según dificultad"""
        threshold_multiplier = {
            "EASY": 0.5,
            "MEDIUM": 0.7,
            "DIFFICULT": 1.0,
            "UNKNOWN": 0.6
        }
        
        multiplier = threshold_multiplier.get(self.process_difficulty, 0.6)
        max_allowed_error = (self.upper_range - self.lower_range) * multiplier
        
        if abs(error) > max_allowed_error:
            if not hasattr(self, 'out_of_range_count'):
                self.out_of_range_count = 0
            self.out_of_range_count += 1
            
            # Procesos difíciles tienen más paciencia
            patience = {"EASY": 20, "MEDIUM": 50, "DIFFICULT": 100, "UNKNOWN": 30}
            max_patience = patience.get(self.process_difficulty, 30)
            
            return self.out_of_range_count > max_patience
        else:
            self.out_of_range_count = 0
            return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with enhanced state information"""
        error = self.setpoint - self.pv
        # Enhanced observation includes integral and derivative terms
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
        Cambiar el setpoint del proceso después del entrenamiento.
        
        Args:
            new_setpoint: Nuevo valor objetivo para el proceso
        """
        if not (self.lower_range <= new_setpoint <= self.upper_range):
            raise ValueError(f"Setpoint {new_setpoint} debe estar entre {self.lower_range} y {self.upper_range}")
        
        self.setpoint = new_setpoint
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Setpoint cambiado a: {new_setpoint}")
    
    def render(self, mode: str = 'human') -> None:
        """Render environment state with enhanced information"""
        if mode == 'human':
            error = self.setpoint - self.pv
            print(f"Step: {self.step_count:4d} | PV: {self.pv:6.2f} | SP: {self.setpoint:6.2f} | "
                  f"Error: {error:6.2f} | Integral: {self.error_integral:6.2f} | "
                  f"Derivative: {self.error_derivative:6.2f} | Difficulty: {self.process_difficulty}")
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get comprehensive episode metrics"""
        return self.episode_metrics.copy()
    
    def update_metrics(self, reward: float, terminated: bool) -> None:
        """Update performance metrics"""
        if terminated and reward > 0.8:  # Consider successful if high final reward
            self.episode_metrics['successful_episodes'] += 1
        
        # Update other metrics as needed
        if self.logger:
            self.logger.debug(f"Episode metrics updated: {self.episode_metrics}")


class ResponseTimeDetector:
    """
    Detector simple de tiempo de respuesta del proceso.
    Estima cuánto tarda el proceso en responder a cambios de control.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.control_changes = []
        self.pv_responses = []
        self.time_stamps = []
        self.current_time = 0
        self.last_significant_control_change = None
        self.response_time_estimates = []
        
    def update(self, control_output, pv, setpoint, dt):
        self.current_time += dt
        
        # Detectar cambio significativo en control
        if len(self.control_changes) == 0:
            significant_change = True
        else:
            change = abs(control_output - self.control_changes[-1])
            significant_change = change > 0.1  # Umbral de cambio significativo
            
        if significant_change:
            self.last_significant_control_change = {
                'time': self.current_time,
                'control': control_output,
                'pv_start': pv
            }
            
        # Detectar respuesta del proceso si hubo cambio reciente
        if self.last_significant_control_change is not None:
            time_since_change = self.current_time - self.last_significant_control_change['time']
            pv_change = abs(pv - self.last_significant_control_change['pv_start'])
            
            # Si el PV cambió significativamente, calcular tiempo de respuesta
            if pv_change > abs(setpoint - self.last_significant_control_change['pv_start']) * 0.1:
                if time_since_change > 0:
                    self.response_time_estimates.append(time_since_change * 3)  # ~3τ para 95% respuesta
                    self.last_significant_control_change = None
        
        # Registrar datos
        self.control_changes.append(control_output)
        self.pv_responses.append(pv)
        self.time_stamps.append(self.current_time)
        
        # Retornar estimación actual si hay suficientes datos
        if len(self.response_time_estimates) >= 2:
            return np.median(self.response_time_estimates[-5:])  # Mediana de últimas 5 estimaciones
        
        return None


# Ejemplo de uso con diferentes procesos
if __name__ == "__main__":
    # Crear ambiente universal
    env = UniversalPIDControlEnv()
    
    # Simular proceso rápido (fácil)
    class FastProcess:
        def __init__(self):
            self.pv = 50.0
            
        def get_initial_pv(self):
            return self.pv
            
        def step(self, control, setpoint):
            self.pv += control * 5.0  # Respuesta rápida
            return self.pv + np.random.normal(0, 0.1)
    
    fast_process = FastProcess()
    env.connect_external_process(fast_process)
    
    # Test
    obs, _ = env.reset(options={'setpoint': 60.0})
    print(f"Setpoint: {env.setpoint:.2f}")
    
    for step in range(100):
        error = obs[2]
        action = [np.clip(error * 0.1, -1, 1)]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            env.render()
            print(f"  Reward: {reward:.3f}, Difficulty: {info['process_difficulty']}")
            
        if terminated or truncated:
            break
    
    print(f"\nProceso clasificado como: {env.process_difficulty}")
    if info['estimated_response_time']:
        print(f"Tiempo de respuesta estimado: {info['estimated_response_time']:.1f} segundos")
