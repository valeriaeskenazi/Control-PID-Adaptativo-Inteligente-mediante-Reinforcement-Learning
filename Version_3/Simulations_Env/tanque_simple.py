"""
Simulador de tanque simple para control de nivel.
"""

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
        
        print("=" * 60)
        print("🚰 Simulador de Tanque creado")
        print(f"   Área: {area} m²")
        print(f"   Coeficiente descarga: {cv} m^2.5/s")
        print(f"   Altura máxima: {max_height} m")
        print(f"   Caudal máximo entrada: {max_flow_in} m³/s")
        print(f"   Paso de tiempo: {dt} s")
        print("=" * 60)
    
    def step(self, control_output: float, setpoint: float) -> float:
        """
        Simular un paso de tiempo.
        
        Args:
            control_output: Señal de control PID (normalizada -1 a 1)
            setpoint: Nivel deseado [m]
        
        Returns:
            height: Nuevo nivel del tanque [m]
        """
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
    
    def get_initial_pv(self) -> float:
        """
        Obtener nivel inicial del tanque.
        
        Returns:
            height: Nivel inicial [m]
        """
        # Inicializar en un nivel aleatorio entre 20% y 80% de la altura máxima
        self.height = np.random.uniform(0.2 * self.max_height, 0.8 * self.max_height)
        self.flow_in = 0.0
        
        return self.height
    
    def reset(self, initial_height: Optional[float] = None) -> float:
        """
        Resetear simulador.
        
        Args:
            initial_height: Altura inicial opcional [m]
        
        Returns:
            height: Nivel inicial [m]
        """
        if initial_height is not None:
            self.height = np.clip(initial_height, 0.0, self.max_height)
        else:
            self.height = self.get_initial_pv()
        
        self.flow_in = 0.0
        
        return self.height
    
    def get_state(self) -> dict:
        """Obtener estado actual del simulador."""
        flow_out = self.cv * np.sqrt(self.height) if self.height > 0 else 0.0
        
        return {
            'height': self.height,
            'flow_in': self.flow_in,
            'flow_out': flow_out,
            'area': self.area,
            'cv': self.cv
        }