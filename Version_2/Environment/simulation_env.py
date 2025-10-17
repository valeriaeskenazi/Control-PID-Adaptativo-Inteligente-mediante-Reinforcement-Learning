"""
Ambiente de simulación para control PID.
Calcula control internamente (no requiere hardware real).
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from .base_env import BasePIDControlEnv
from .pid_components import PIDController, DeltaPIDActionSpace


class SimulationPIDEnv(BasePIDControlEnv):
    """
    Ambiente de simulación que calcula PID internamente.
    
    Modos disponibles:
    - 'direct': Control directo (acción continua)
    - 'pid_tuning': Tuning de parámetros PID (acción discreta)
    
    Args:
        config: Configuración del ambiente
        control_mode: Modo de control ('direct' o 'pid_tuning')
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 control_mode: str = 'direct'):
        
        # Inicializar clase base
        super().__init__(config)
        
        self.control_mode = control_mode
        self._setup_action_space()
        
        # Proceso externo (simulador)
        self.external_process = None
    
    def _setup_action_space(self) -> None:
        """Definir espacio de acciones según modo."""
        if self.control_mode == 'direct':
            # Modo control directo: acción continua
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            self.pid_action_space = None
            self.pid_controller = None
            
            print("=" * 60)
            print("✅ Modo: Control Directo (Simulación)")
            print("   Acción: Continua [-1.0, 1.0]")
            print("=" * 60)
            
        elif self.control_mode == 'pid_tuning':
            # Modo tuning PID: acciones discretas
            self.pid_action_space = DeltaPIDActionSpace(
                initial_pid=(1.0, 0.1, 0.05),
                delta_percent=0.2
            )
            self.pid_controller = PIDController(
                kp=1.0,
                ki=0.1,
                kd=0.05,  # ← CORREGIDO: eliminado duplicado
                dt=self.dt,
                output_limits=(-1.0, 1.0)
            )
            self.action_space = spaces.Discrete(self.pid_action_space.n_actions)
            
            print("=" * 60)
            print("✅ Modo: PID Tuning (Simulación)")
            print(f"   Acciones disponibles: {self.pid_action_space.n_actions}")
            print(f"   PID inicial: {self.pid_action_space.get_current_pid()}")
            print(f"   PIDController interno: Activo")
            print("=" * 60)
            
        else:
            raise ValueError(
                f"control_mode debe ser 'direct' o 'pid_tuning', "
                f"recibido: '{self.control_mode}'"
            )
    
    def _apply_control(self, action) -> Tuple[Optional[float], Optional[Tuple]]:
        """
        Aplicar acción y calcular control_output.
        
        Args:
            action: Acción del agente (continua o discreta según modo)
        
        Returns:
            Tuple con (control_output, pid_params)
        """
        if self.control_mode == 'direct':
            # Control directo: la acción ES el control_output
            control_output = float(action[0])
            pid_params = None
            
        elif self.control_mode == 'pid_tuning':
            # Tuning PID: traducir índice a parámetros
            pid_params = self.pid_action_space.apply_action(action)
            
            # Actualizar ganancias del controlador
            self.pid_controller.update_gains(*pid_params)
            
            # Calcular control_output usando PID
            error = self.setpoint - self.pv
            control_output = self.pid_controller.compute(error)
        
        return control_output, pid_params
    
    def _update_process(self, control_output: Optional[float],
                        pid_params: Optional[Tuple]) -> float:
        """
        Actualizar proceso simulado.
        
        Args:
            control_output: Señal de control
            pid_params: No usado en simulación
        
        Returns:
            Nuevo valor de PV
        """
        if self.external_process is not None:
            # Usar simulador externo
            new_pv = self.external_process.step(control_output, self.setpoint)
        else:
            # Simulador dummy para testing
            self.pv += control_output * 0.5 + np.random.normal(0, 0.1)
            new_pv = self.pv
            
            if self.step_count == 0:
                print("⚠️  WARNING: Proceso no conectado, usando simulador dummy")
        
        return new_pv
    
    def connect_external_process(self, process_simulator) -> None:
        """
        Conectar simulador de proceso externo.
        
        Args:
            process_simulator: Objeto con métodos:
                - step(control_output, setpoint) -> pv
                - get_initial_pv() -> pv
        """
        required_methods = ['step', 'get_initial_pv']
        for method in required_methods:
            if not hasattr(process_simulator, method):
                raise ValueError(
                    f"Process simulator debe implementar método: '{method}'"
                )
        
        self.external_process = process_simulator
        
        if self.logger:
            self.logger.info("Proceso simulado conectado exitosamente")
        else:
            print("✅ Proceso simulado conectado")
    
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
        
        # Obtener PV inicial
        if self.external_process is not None:
            self.pv = self.external_process.get_initial_pv()
        else:
            # Dummy: PV cerca del setpoint con ruido
            self.pv = self.setpoint + np.random.uniform(-5, 5)
        
        # Reset específico del modo PID tuning
        if self.control_mode == 'pid_tuning':
            self.pid_action_space.reset()
            self.pid_controller.reset()
            pid_params = self.pid_action_space.get_current_pid()
            self.pid_controller.update_gains(*pid_params)
        
        return self._get_observation(), info