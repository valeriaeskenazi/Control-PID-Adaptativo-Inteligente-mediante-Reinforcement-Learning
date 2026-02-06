import time
import numpy as np

class ResponseTimeDetector:
    
    def __init__(self, proceso, variable_index, env_type='simulation'):

        self.proceso = proceso
        self.variable_index = variable_index
        self.env_type = env_type # Varable definida en el entorno
    
    def estimate(self, pv_inicial, sp, pid_controller, max_time=1800, tolerance=0.05):
        if self.env_type == 'simulation':
            return self._estimate_from_simulation(pv_inicial, sp, pid_controller, max_time, tolerance)
        elif self.env_type == 'real':
            return self._estimate_online(pv_inicial, sp, pid_controller, max_time, tolerance)
    
    def _estimate_from_simulation(self, pv_inicial, sp, pid_controller, max_time, tolerance):

        # Resetear el PID
        pid_controller.reset()
        
        pv = pv_inicial
        t = 0
        dt_sim = 1.0  # Paso de simulación
        
        # Error objetivo para convergencia
        dead_band = tolerance * abs(sp - pv_inicial)
        
        # Simular hasta convergencia
        while abs(sp - pv) > dead_band:
            # PID calcula control
            control_output = pid_controller.compute(
                setpoint=sp,
                process_value=pv
            )
            
            # Simular un paso
            pv = self.proceso.simulate_step(
                control_output=control_output,
                variable_index=self.variable_index,
                dt=dt_sim
            )
            
            t += dt_sim
            
            # Timeout
            if t > max_time:
                t=max_time
                break
        
        return t    
    
    def _estimate_online(self, pv_inicial, sp, pid_controller, max_time, tolerance):
        # Resetear PID
        pid_controller.reset()
        
        t = 0
        dt_sample = 1.0
        
        dead_band = tolerance * abs(sp - pv_inicial)
        
        # Leer PV inicial
        pv_actual = self.proceso.read_pv(self.variable_index)
        
        # Medir en tiempo real
        while abs(sp - pv_actual) > dead_band:
            # Calcular control
            control_output = pid_controller.compute(
                setpoint=sp,
                process_value=pv_actual
            )
            
            # Escribir control al proceso real
            self.proceso.write_control(
                control_output=control_output,
                variable_index=self.variable_index
            )
            
            # Esperar tiempo REAL
            time.sleep(dt_sample)
            t += dt_sample
            
            # Leer nuevo PV
            pv_actual = self.proceso.read_pv(self.variable_index)
            
            # Timeout
            if t >= max_time:
                return max_time
    
        return t