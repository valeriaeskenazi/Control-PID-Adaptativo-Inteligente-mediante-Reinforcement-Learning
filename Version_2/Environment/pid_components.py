"""
Componentes PID reutilizables para control y reinforcement learning.

Contiene:
- PIDController: Calcula el Output en base a los Kp,Ki,Kd y el error
- DeltaPIDActionSpace: Codificador de acciones discretas
- ResponseTimeDetector: Detector de tiempo de respuesta del proceso
"""

import numpy as np
from typing import Tuple, List, Optional


class PIDController:
    """    
    Calcula la señal de control usando los términos proporcional,
    integral y derivativo del error.
    
    Args:
        kp: Ganancia proporcional
        ki: Ganancia integral
        kd: Ganancia derivativa
        dt: Paso de tiempo [s]
        output_limits: Límites de salida (min, max)
    """
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, dt=1.0, 
                 output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        
        # Estado interno
        self.integral = 0.0
        self.prev_error = 0.0
    
    def update_gains(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
    
    def compute(self, error: float) -> float:
        """
        Calcular salida PID con anti-windup.
        
        Args:
            error: Error actual (setpoint - pv)
        
        Returns:
            control_output: Señal de control dentro de output_limits
        """
        # Término proporcional
        P = self.kp * error
        
        # Término integral
        self.integral += error * self.dt
        I = self.ki * self.integral
        
        # Término derivativo
        derivative = (error - self.prev_error) / self.dt
        D = self.kd * derivative
        
        # Salida total
        output = P + I + D
        
        # Aplicar límites
        output_clipped = np.clip(output, *self.output_limits)
        
        # Anti-windup: si hay saturación, no acumular integral
        if output != output_clipped:
            self.integral -= error * self.dt
        
        # Guardar para próxima iteración
        self.prev_error = error
        
        return float(output_clipped)
    
    def reset(self) -> None:
        """Resetear estado interno del controlador."""
        self.integral = 0.0
        self.prev_error = 0.0
    
    def get_state(self) -> dict:
        """Obtener estado interno del controlador."""
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self.integral,
            'prev_error': self.prev_error
        }


class DeltaPIDActionSpace:
    """    
    En vez de valores absolutos, las acciones representan cambios
    relativos a los parámetros PID actuales.
    
    Acciones disponibles:
        0: Kp ↑ (aumentar Kp)
        1: Ki ↑ (aumentar Ki)
        2: Kd ↑ (aumentar Kd)
        3: Kp ↓ (reducir Kp)
        4: Ki ↓ (reducir Ki)
        5: Kd ↓ (reducir Kd)
        6: Mantener (no cambiar)
    
    Args:
        initial_pid: PID inicial (Kp, Ki, Kd)
        delta_percent: Porcentaje de cambio (0.2 = 20%)
        limits: Límites para cada parámetro [(kp_min, kp_max), ...]
    """
    
    def __init__(self, 
                 initial_pid: Tuple[float, float, float] = (1.0, 0.1, 0.05),
                 delta_percent: float = 0.2,
                 limits: Optional[List[Tuple[float, float]]] = None):
        
        self.initial_pid = np.array(initial_pid, dtype=np.float32)
        self.current_pid = self.initial_pid.copy()
        self.delta = delta_percent
        
        # Límites por defecto
        if limits is None:
            self.limits = [
                (0.01, 100.0),  # Kp
                (0.0, 10.0),     # Ki
                (0.0, 10.0)      # Kd
            ]
        else:
            self.limits = limits
        
        # Mapeo de acciones: (nombre, param_idx, dirección)
        self.action_map = {
            0: ('Kp', 0, +1),
            1: ('Ki', 1, +1),
            2: ('Kd', 2, +1),
            3: ('Kp', 0, -1),
            4: ('Ki', 1, -1),
            5: ('Kd', 2, -1),
            6: ('None', -1, 0)
        }
        
        self.n_actions = len(self.action_map)
        self.param_names = ['Kp', 'Ki', 'Kd']
    
    def apply_action(self, action_idx: int) -> Tuple[float, float, float]:
        """
        Aplicar acción y retornar PID actualizado.
        
        Args:
            action_idx: Índice de acción (0-6)
        
        Returns:
            Tuple con (Kp, Ki, Kd) actualizado
        """
        if action_idx < 0 or action_idx >= self.n_actions:
            raise ValueError(
                f"action_idx debe estar en [0, {self.n_actions-1}], "
                f"recibido: {action_idx}"
            )
        
        param_name, param_idx, direction = self.action_map[action_idx]
        
        if param_idx >= 0:  # Modificar parámetro
            multiplier = 1.0 + (direction * self.delta)
            self.current_pid[param_idx] *= multiplier
            
            # Aplicar límites
            min_val, max_val = self.limits[param_idx]
            self.current_pid[param_idx] = np.clip(
                self.current_pid[param_idx],
                min_val,
                max_val
            )
        
        return tuple(self.current_pid)
    
    def reset(self, pid: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Resetear al PID inicial.
        
        Args:
            pid: PID personalizado, si None usa initial_pid
        """
        if pid is None:
            self.current_pid = self.initial_pid.copy()
        else:
            self.current_pid = np.array(pid, dtype=np.float32)
    
    def get_current_pid(self) -> Tuple[float, float, float]:
        """Obtener PID actual."""
        return tuple(self.current_pid)
    
    def get_action_description(self, action_idx: int) -> str:
        """
        Obtener descripción legible de una acción.
        
        Args:
            action_idx: Índice de acción
        
        Returns:
            Descripción de la acción
        """
        param_name, _, direction = self.action_map[action_idx]
        
        if direction > 0:
            return f"{param_name} ↑ {self.delta*100:.0f}%"
        elif direction < 0:
            return f"{param_name} ↓ {self.delta*100:.0f}%"
        else:
            return "Mantener PID"


class ResponseTimeDetector:
    """
    Detector de tiempo de respuesta del proceso.
    
    Estima cuánto tiempo tarda el proceso en responder a cambios
    significativos en la señal de control.

    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Resetear detector."""
        self.control_changes = []
        self.pv_responses = []
        self.time_stamps = []
        self.current_time = 0
        self.last_significant_control_change = None
        self.response_time_estimates = []
    
    def update(self, control_output: float, pv: float, 
               setpoint: float, dt: float) -> Optional[float]:
        """
        Actualizar detector con nueva medición.
        
        Args:
            control_output: Señal de control aplicada
            pv: Variable de proceso actual
            setpoint: Punto de ajuste
            dt: Paso de tiempo
        
        Returns:
            Estimación del tiempo de respuesta o None si no hay suficientes datos
        """
        self.current_time += dt
        
        # Detectar cambio significativo en control
        if len(self.control_changes) == 0:
            significant_change = True
        else:
            change = abs(control_output - self.control_changes[-1])
            significant_change = change > 0.1
        
        if significant_change:
            self.last_significant_control_change = {
                'time': self.current_time,
                'control': control_output,
                'pv_start': pv
            }
        
        # Detectar respuesta del proceso
        if self.last_significant_control_change is not None:
            time_since_change = (
                self.current_time - 
                self.last_significant_control_change['time']
            )
            pv_change = abs(pv - self.last_significant_control_change['pv_start'])
            
            # Si el PV cambió significativamente
            expected_change = abs(
                setpoint - self.last_significant_control_change['pv_start']
            )
            
            if pv_change > expected_change * 0.1 and time_since_change > 0:
                # Estimar tiempo de respuesta (~3τ para 95% respuesta)
                self.response_time_estimates.append(time_since_change * 3)
                self.last_significant_control_change = None
        
        # Registrar datos
        self.control_changes.append(control_output)
        self.pv_responses.append(pv)
        self.time_stamps.append(self.current_time)
        
        # Retornar estimación si hay suficientes datos
        if len(self.response_time_estimates) >= 2:
            return float(np.median(self.response_time_estimates[-5:]))
        
        return None