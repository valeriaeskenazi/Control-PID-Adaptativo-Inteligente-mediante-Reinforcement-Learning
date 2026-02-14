import numpy as np
from typing import Tuple, Optional


class TankSimulator:
    """
    Simulador de tanque cilíndrico simple con válvula de salida.
    
    Dinámica:
        dh/dt = (Qin - Qout) / A
        Qout = Cv * sqrt(h)
    
    Args:
        area: Área de la sección transversal del tanque [m²]
        cv: Coeficiente de descarga de la válvula [m^2.5/s]
        max_height: Altura máxima del tanque [m]
        max_flow_in: Caudal máximo de entrada [m³/s]
        dt: Paso de tiempo para integración [s]
    """
    
    def __init__(
        self,
        area: float = 1.0,
        cv: float = 0.1,
        max_height: float = 10.0,
        max_flow_in: float = 0.5,
        dt: float = 1.0
    ):
        """Inicializar simulador de tanque."""
        self.area = area
        self.cv = cv
        self.max_height = max_height
        self.max_flow_in = max_flow_in
        self.dt = dt
        
        # Estado actual
        self.height = 0.0
        self.flow_in = 0.0
    
    def step(self, control_output: float, setpoint: float) -> float:
        # Convertir control_output (-1, 1) a caudal de entrada
        # control_output = 0 → 50% del caudal máximo (punto medio)
        # control_output = 1 → 100% del caudal máximo
        # control_output = -1 → 0% del caudal máximo
        self.flow_in = self.max_flow_in * (0.5 + 0.5 * control_output)
        self.flow_in = np.clip(self.flow_in, 0.0, self.max_flow_in)
        
        # Caudal de salida (depende del nivel por gravedad)
        if self.height > 0:
            flow_out = self.cv * np.sqrt(self.height)
        else:
            flow_out = 0.0
        
        # Integración Euler: dh/dt = (Qin - Qout) / A
        dh_dt = (self.flow_in - flow_out) / self.area
        self.height += dh_dt * self.dt
        
        # Limitar altura entre 0 y max_height
        self.height = np.clip(self.height, 0.0, self.max_height)
        
        return self.height
    
    def simulate_step(
        self,
        control_output: float,
        variable_index: int,
        dt: float
    ) -> float:

        # Guardar dt original
        dt_original = self.dt
        self.dt = dt
        
        # Aplicar límites al control_output
        self.flow_in = self.max_flow_in * (0.5 + 0.5 * control_output)
        self.flow_in = np.clip(self.flow_in, 0.0, self.max_flow_in)
        
        # Ejecutar step del simulador (actualiza self.state internamente)
        result = self.step(control_output=self.flow_in, setpoint=None)
        
        # Restaurar dt original
        self.dt = dt_original
        
        # Retornar el NUEVO PV
        return result[variable_index]
    
    def get_initial_pv(self) -> float:
        # Inicializar en un nivel aleatorio entre 20% y 80% de la altura máxima
        self.height = np.random.uniform(0.2 * self.max_height, 0.8 * self.max_height)
        self.flow_in = 0.0
        
        return self.height
    
    def reset(self, initial_height: Optional[float] = None) -> float:
        if initial_height is not None:
            self.height = np.clip(initial_height, 0.0, self.max_height)
        else:
            self.height = self.get_initial_pv()
        
        self.flow_in = 0.0
        
        return self.height

    def get_state(self) -> dict:
        flow_out = self.cv * np.sqrt(self.height) if self.height > 0 else 0.0
        
        return {
            'height': self.height,
            'flow_in': self.flow_in,
            'flow_out': flow_out,
            'area': self.area,
            'cv': self.cv
        }