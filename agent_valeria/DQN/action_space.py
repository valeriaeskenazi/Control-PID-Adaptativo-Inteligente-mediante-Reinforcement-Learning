"""
Espacio de acciones para PID
Convierte entre índices discretos y valores PID continuos
"""
import numpy as np

class PIDActionSpace:
    """
    Manejo del espacio de acciones para PID
    Convierte entre índices discretos y valores PID continuos
    """
    
    def __init__(self, 
                 kp_range=(0.1, 10.0), 
                 ki_range=(0.01, 5.0), 
                 kd_range=(0.001, 2.0),
                 n_discrete_per_param=4):
        """
        Inicializar espacio de acciones
        
        Args:
            kp_range: Rango para Kp (min, max)
            ki_range: Rango para Ki (min, max)  
            kd_range: Rango para Kd (min, max)
            n_discrete_per_param: Niveles de discretización por parámetro
        """
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        self.n_discrete = n_discrete_per_param
        
        # Crear valores discretos para cada parámetro
        self.kp_values = np.linspace(kp_range[0], kp_range[1], n_discrete_per_param)
        self.ki_values = np.linspace(ki_range[0], ki_range[1], n_discrete_per_param)
        self.kd_values = np.linspace(kd_range[0], kd_range[1], n_discrete_per_param)
        
        # Total de acciones = 4 * 4 * 4 = 64
        self.n_actions = n_discrete_per_param ** 3
        
        print(f"Espacio de acciones PID creado:")
        print(f"  Kp: {self.kp_values}")
        print(f"  Ki: {self.ki_values}")
        print(f"  Kd: {self.kd_values}")
        print(f"  Total acciones: {self.n_actions}")
    
    def index_to_pid(self, action_index):
        """
        Convertir índice de acción a valores PID
        
        Args:
            action_index: Índice de la acción (0 a n_actions-1)
            
        Returns:
            (kp, ki, kd): Array con valores PID [Kp, Ki, Kd]
        """
        # Convertir índice 1D a índices 3D
        kd_idx = action_index % self.n_discrete
        ki_idx = (action_index // self.n_discrete) % self.n_discrete
        kp_idx = (action_index // (self.n_discrete ** 2)) % self.n_discrete
        
        # Obtener valores reales
        kp = self.kp_values[kp_idx]
        ki = self.ki_values[ki_idx]
        kd = self.kd_values[kd_idx]
        
        return np.array([kp, ki, kd])
    
    def pid_to_index(self, kp, ki, kd):
        """
        Convertir valores PID a índice de acción (aproximado)
        
        Args:
            kp, ki, kd: Valores PID
            
        Returns:
            action_index: Índice más cercano
        """
        # Encontrar índices más cercanos
        kp_idx = np.argmin(np.abs(self.kp_values - kp))
        ki_idx = np.argmin(np.abs(self.ki_values - ki))
        kd_idx = np.argmin(np.abs(self.kd_values - kd))
        
        # Convertir a índice 1D
        return kp_idx * (self.n_discrete ** 2) + ki_idx * self.n_discrete + kd_idx
