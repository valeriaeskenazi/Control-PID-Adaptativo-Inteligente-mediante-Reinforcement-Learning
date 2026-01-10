"""
Ambiente de simulación para control PID.
Calcula control internamente (no requiere hardware real) para entrenamientos.

Soporta:
- Single-agent: n_variables=1 (comportamiento original)
- Multi-agent: n_variables>1 (N agentes cooperativos)
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List

from Environment.base_env import BasePIDControlEnv
from Environment.pid_components import PIDController, DeltaPIDActionSpace


class SimulationPIDEnv(BasePIDControlEnv):
    """
    Ambiente de simulación que calcula PID internamente.
    
    Modos disponibles (control_mode):
    - 'direct': Control directo (acción continua → control output directo)
    - 'pid_tuning': Tuning de parámetros PID (acción discreta → ajusta Kp, Ki, Kd)
    
    Soporta single-agent (n_variables=1) y multi-agent (n_variables>1).
    
    Args:
        config: Configuración del ambiente (debe incluir 'n_variables')
        control_mode: Modo de control ('direct' o 'pid_tuning')
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 control_mode: str = None):
        
        # Inicializar clase base
        super().__init__(config)
        
        # El control_mode puede venir como argumento o en el config
        if control_mode is not None:
            self.control_mode = control_mode
        elif config is not None and 'control_mode' in config:
            self.control_mode = config['control_mode']
        else:
            self.control_mode = 'direct'  # Default
        
        self._setup_action_space()
        
        # Proceso externo (simulador)
        self.external_process = None
    
    def _setup_action_space(self) -> None:
        """Definir espacio de acciones según modo y número de variables."""
        
        if self.control_mode == 'direct':
            n_params = self.n_variables * 3  # 3 parámetros (Kp, Ki, Kd) por variable
            
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0] * self.n_variables),
                high=np.array([100.0, 1.0, 1.0] * self.n_variables),
                shape=(n_params,),  
                dtype=np.float32
            )
            
            # Crear N controladores PID
            self.pid_controllers = [
                PIDController(
                    kp=1.0, ki=0.1, kd=0.05,
                    dt=self.dt,
                    output_limits=(-1.0, 1.0)
                )
                for _ in range(self.n_variables)
            ]
    
            self.pid_action_spaces = None  # No se usa en modo directo
            
        elif self.control_mode == 'pid_tuning':
            # Modo tuning PID: acciones discretas
            # Crear N espacios de acción PID (uno por variable)
            self.pid_action_spaces = [
                DeltaPIDActionSpace(
                    initial_pid=(1.0, 0.1, 0.05),
                    delta_percent=0.2
                )
                for _ in range(self.n_variables)
            ]
            
            # Crear N controladores PID (uno por variable)
            self.pid_controllers = [
                PIDController(
                    kp=1.0,
                    ki=0.1,
                    kd=0.05,
                    dt=self.dt,
                    output_limits=(-1.0, 1.0)
                )
                for _ in range(self.n_variables)
            ]
            
            # Espacio de acciones: MultiDiscrete o Tuple de Discrete
            if self.n_variables == 1:
                # Single-agent: Discrete simple
                self.action_space = spaces.Discrete(
                    self.pid_action_spaces[0].n_actions
                )
            else:
                # Multi-agent: MultiDiscrete (cada agente elige una acción discreta)
                self.action_space = spaces.MultiDiscrete(
                    [self.pid_action_spaces[i].n_actions 
                     for i in range(self.n_variables)]
                )
            
            print("=" * 60)
            print("✅ Modo: PID Tuning (Simulación)")
            print(f"   N Variables: {self.n_variables}")
            print(f"   Acciones por variable: {self.pid_action_spaces[0].n_actions}")
            if self.n_variables == 1:
                print(f"   Espacio: Discrete({self.pid_action_spaces[0].n_actions})")
            else:
                print(f"   Espacio: MultiDiscrete([7, 7, ..., 7]) x{self.n_variables}")
            print(f"   PID inicial: {self.pid_action_spaces[0].get_current_pid()}")
            print(f"   PIDControllers: {self.n_variables} activos")
            print("=" * 60)
            
        else:
            raise ValueError(
                f"control_mode debe ser 'direct' o 'pid_tuning', "
                f"recibido: '{self.control_mode}'"
            )
    
    def _apply_control(self, action):
        """
        Aplicar acción de control y retornar outputs.
        
        Args:
            action: Acción del agente (formato depende del modo)
                   - direct: array continuo de shape (n_variables,)
                   - pid_tuning: int (1 var) o array de ints (múltiples vars)
        
        Returns:
            control_outputs: Lista de salidas de control
            pid_params_list: Lista de parámetros PID actuales (o None en modo direct)
        """
        # Verificar que esté inicializado
        if self.setpoints is None or self.pvs is None:
            raise RuntimeError("Ambiente no inicializado. Llama a reset() primero.")
        
        if action is None:
            raise ValueError(
                f"action no puede ser None. "
                f"Modo: {self.control_mode}"
            )
        
        control_outputs = []
        pid_params_list = []
        
        if self.control_mode == 'direct':
        # Modo directo: acción son los PIDs [Kp, Ki, Kd, Kp, Ki, Kd, ...]
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            
            control_outputs = []
            pid_params_list = []
            
            for i in range(self.n_variables):
                # Extraer [Kp, Ki, Kd] de esta variable
                kp = float(action[i * 3])
                ki = float(action[i * 3 + 1])
                kd = float(action[i * 3 + 2])
                
                # Actualizar PID
                self.pid_controllers[i].update_gains(kp, ki, kd)
                
                # Calcular control output
                control_output = self.pid_controllers[i].compute(
                    setpoint=self.setpoints[i],
                    process_value=self.pvs[i]
                )
                
                control_outputs.append(control_output)
                pid_params_list.append((kp, ki, kd))
            
        elif self.control_mode == 'pid_tuning':
            # Convertir acción a lista si es escalar
            if isinstance(action, (int, np.integer)):
                action_list = [action]
            elif isinstance(action, np.ndarray):
                action_list = [int(a) for a in action]
            else:
                action_list = [int(a) for a in action]
            
            # Aplicar cada acción PID
            for i, pid_controller in enumerate(self.pid_controllers):
                # Aplicar acción (actualización de PID)
                new_pid = self.pid_action_spaces[i].apply_action(action_list[i])
                pid_controller.update_gains(*new_pid)
                
                # Calcular salida de control
                control_output = pid_controller.compute(
                    setpoint=self.setpoints[i],
                    process_value=self.pvs[i]
                )
                
                control_outputs.append(control_output)
                pid_params_list.append(pid_controller.get_params())
        
        return control_outputs, pid_params_list
    
    def _update_process(self, 
                        control_outputs: List[Optional[float]],
                        pid_params_list: List[Optional[Tuple]]) -> List[float]:
        """
        Actualizar proceso simulado.
        
        Args:
            control_outputs: Lista de N señales de control
            pid_params_list: Lista de N tuplas de parámetros PID (no usado en simulación)
        
        Returns:
            Lista de nuevos valores de PV [pv_1, ..., pv_N]
        """
        if self.external_process is not None:
            # Usar simulador externo
            new_pvs = self.external_process.step(control_outputs, self.setpoints)
            
            if len(new_pvs) != self.n_variables:
                raise ValueError(
                    f"Simulador retornó {len(new_pvs)} PVs pero se esperaban {self.n_variables}"
                )
        else:
            # Simulador dummy para testing (variables independientes)
            new_pvs = []
            for i in range(self.n_variables):
                self.pvs[i] += control_outputs[i] * 0.5 + np.random.normal(0, 0.1)
                new_pvs.append(self.pvs[i])
            
            if self.step_count == 0:
                print(" WARNING: Proceso no conectado, usando simulador dummy")
        
        return new_pvs
    
    def connect_external_process(self, process_simulator) -> None:
        """
        Conectar simulador de proceso externo.
        
        El simulador debe implementar:
        - Para single-agent (n_variables=1):
            * step(control_output: float, setpoint: float) -> float (retrocompatible)
            * get_initial_pv() -> float
          O la interfaz multi-variable (también funciona)
        
        - Para multi-agent (n_variables>1):
            * step(control_outputs: List[float], setpoints: List[float]) -> List[float]
            * get_initial_pvs() -> List[float]
            * get_n_variables() -> int
        
        Args:
            process_simulator: Objeto simulador que implementa la interfaz requerida
        """
        # Detectar qué interfaz tiene el simulador
        has_multi_interface = (
            hasattr(process_simulator, 'get_n_variables') and
            hasattr(process_simulator, 'get_initial_pvs')
        )
        
        has_single_interface = (
            hasattr(process_simulator, 'get_initial_pv')
        )
        
        # Validar según número de variables
        if self.n_variables == 1:
            # Single-agent: aceptar cualquier interfaz
            if not has_multi_interface and not has_single_interface:
                raise ValueError(
                    "Para n_variables=1, el simulador debe implementar:\n"
                    "  - step(control, setpoint) -> pv\n"
                    "  - get_initial_pv() -> pv\n"
                    "O la interfaz MultiVariableProcessInterface"
                )
            
            # Validar método step
            if not hasattr(process_simulator, 'step'):
                raise ValueError("Process simulator debe implementar método: 'step'")
            
        else:
            # Multi-agent: requiere interfaz multi-variable
            if not has_multi_interface:
                raise ValueError(
                    f"Para n_variables={self.n_variables}, el simulador debe implementar "
                    "MultiVariableProcessInterface:\n"
                    "  - step(controls: List, setpoints: List) -> List[pvs]\n"
                    "  - get_initial_pvs() -> List[pvs]\n"
                    "  - get_n_variables() -> int\n"
                    "  - reset() -> None"
                )
            
            # Validar que el número de variables coincida
            simulator_n_vars = process_simulator.get_n_variables()
            if simulator_n_vars != self.n_variables:
                raise ValueError(
                    f"Simulador tiene {simulator_n_vars} variables pero el ambiente "
                    f"está configurado para {self.n_variables}"
                )
        
        self.external_process = process_simulator
        
        if self.logger:
            self.logger.info(
                f"Proceso simulado conectado exitosamente ({self.n_variables} variable(s))"
            )
        else:
            print(f"✅ Proceso simulado conectado ({self.n_variables} variable(s))")
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resetear ambiente de simulación.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
        
        Returns:
            Tuple con (observación inicial, info)
        """
        # Reset de la clase base
        obs, info = super().reset(seed=seed, options=options)
        
        # Obtener PVs iniciales
        if self.external_process is not None:
            # Detectar interfaz del simulador
            if hasattr(self.external_process, 'get_initial_pvs'):
                # Interfaz multi-variable
                self.pvs = self.external_process.get_initial_pvs()
            elif hasattr(self.external_process, 'get_initial_pv') and self.n_variables == 1:
                # Interfaz single-variable (retrocompatible)
                self.pvs = [self.external_process.get_initial_pv()]
            else:
                raise RuntimeError(
                    "No se pudo obtener PVs iniciales del simulador"
                )
        else:
            # Dummy: PVs cerca de los setpoints con ruido
            self.pvs = [
                self.setpoints[i] + np.random.uniform(-5, 5)
                for i in range(self.n_variables)
            ]
        
        # Reset específico del modo PID tuning
        if self.control_mode == 'pid_tuning':
            for i in range(self.n_variables):
                self.pid_action_spaces[i].reset()
                self.pid_controllers[i].reset()
                pid_params = self.pid_action_spaces[i].get_current_pid()
                self.pid_controllers[i].update_gains(*pid_params)
        
        return self._get_observation(), info
    
    def get_process_state(self) -> List[float]:
        """
        Obtener estado completo del proceso conectado.
        
        Returns:
            Estado del proceso externo o PVs actuales si no hay proceso conectado
        """
        if self.external_process is not None:
            if hasattr(self.external_process, 'get_state'):
                return self.external_process.get_state()
            else:
                raise AttributeError(
                    "El proceso externo debe implementar método 'get_state()'"
                )
        else:
            # Fallback: retornar PVs actuales
            return self.pvs

    def reset_process(self) -> None:
        """Resetear el proceso externo si está conectado."""
        if self.external_process is not None:
            if hasattr(self.external_process, 'reset'):
                self.external_process.reset()
            else:
                raise AttributeError(
                    "El proceso externo debe implementar método 'reset()'"
                )
