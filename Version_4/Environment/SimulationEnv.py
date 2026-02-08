import numpy as np
from typing import Optional, Dict, Any


class SimulationPIDEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuración básica
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', 
                                             [(0.0, 100.0)] * self.n_manipulable_vars)
        self.dt_sim = config.get('dt_simulation', 1.0)
    
        # Estado actual del proceso (se inicializa en reset)
        self.manipulable_pvs = None
        
        # Simulador externo (se conecta con connect_external_process)
        self.external_process = None
    
    
    def simulate_step(self, control_output: float, variable_index: int, dt: float) -> float:
        # Delegar al simulador externo
        new_pv = self.external_process.simulate_step(control_output, variable_index, dt)
        
        # Clipear a rangos físicos
        min_val, max_val = self.manipulable_ranges[variable_index]
        new_pv = np.clip(new_pv, min_val, max_val)
        
        # Actualizar estado interno
        if self.manipulable_pvs is not None:
            self.manipulable_pvs[variable_index] = new_pv
        
        return float(new_pv)
    
    def connect_external_process(self, process_simulator) -> None:
        self.external_process = process_simulator
    
    def reset(self, initial_pvs: Optional[list] = None) -> list:
        if initial_pvs is not None:
            self.manipulable_pvs = list(initial_pvs)
        else:
            self.manipulable_pvs = [
                np.random.uniform(rango[0], rango[1])
                for rango in self.manipulable_ranges
            ]
        
        if self.external_process is not None and hasattr(self.external_process, 'reset'):
            self.external_process.reset()
        
        return self.manipulable_pvs.copy()

    def get_state(self) -> list:
        if self.manipulable_pvs is None:
            raise RuntimeError("Ambiente no inicializado. Llama a reset() primero.")
        return self.manipulable_pvs.copy()