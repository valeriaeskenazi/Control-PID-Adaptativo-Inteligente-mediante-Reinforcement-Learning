import time
import numpy as np

class ResponseTimeDetector:
    
    def __init__(self, proceso, variable_index, env_type='simulation', dt=1.0, tolerance=0.05):
        self.proceso = proceso
        self.variable_index = variable_index
        self.env_type = env_type
        self.dt = dt
        self.tolerance = tolerance

    def estimate(self, pv_inicial, sp, pid_controller, max_time=1800):
        if self.env_type == 'simulation':
            return self._estimate_from_simulation(pv_inicial, sp, pid_controller, max_time)
        elif self.env_type == 'real':
            return self._estimate_online(pv_inicial, sp, pid_controller, max_time)

    def _estimate_from_simulation(self, pv_inicial, sp, pid_controller, max_time):
        # Resetear el PID
        pid_controller.reset()
        
        # Inicializar resultado
        resultado = {
            'tiempo': 0,
            'trayectoria_pv': [pv_inicial],
            'trayectoria_control': [],
            'pv_final': pv_inicial,
            'converged': False
        }
        
        pv = pv_inicial
        t = 0
        dead_band = self.tolerance * abs(sp)
        
        # Simular hasta convergencia
        while abs(sp - pv) > dead_band:
            # PID calcula control
            control_output = pid_controller.compute(
                setpoint=sp,
                process_value=pv
            )
            resultado['trayectoria_control'].append(control_output)
            
            # Simular un paso
            pv = self.proceso.simulate_step(
                control_output=control_output,
                variable_index=self.variable_index,
                dt=self.dt
            )
            resultado['trayectoria_pv'].append(pv)

            t += self.dt

            # Timeout
            if t >= max_time:
                resultado['tiempo'] = max_time
                resultado['pv_final'] = pv
                resultado['converged'] = False
                return resultado
        
        # Convergencia exitosa
        resultado['tiempo'] = t
        resultado['pv_final'] = pv
        resultado['converged'] = True
        
        return resultado

    def _estimate_online(self, pv_inicial, sp, pid_controller, max_time):
        # Resetear PID
        pid_controller.reset()
        
        # Inicializar resultado
        resultado = {
            'tiempo': 0,
            'trayectoria_pv': [],
            'trayectoria_control': [],
            'pv_final': pv_inicial,
            'converged': False
        }
        
        t = 0
        dead_band = self.tolerance * abs(sp)

        # Leer PV inicial
        pv_actual = self.proceso.read_pv(self.variable_index)
        resultado['trayectoria_pv'].append(pv_actual)
        
        # Medir en tiempo real
        while abs(sp - pv_actual) > dead_band:
            # Calcular control
            control_output = pid_controller.compute(
                setpoint=sp,
                process_value=pv_actual
            )
            resultado['trayectoria_control'].append(control_output)
            
            # Escribir control al proceso real
            self.proceso.write_control(
                control_output=control_output,
                variable_index=self.variable_index
            )
            
            # Esperar tiempo REAL
            time.sleep(self.dt)
            t += self.dt

            # Leer nuevo PV
            pv_actual = self.proceso.read_pv(self.variable_index)
            resultado['trayectoria_pv'].append(pv_actual)
            
            # Timeout
            if t >= max_time:
                resultado['tiempo'] = max_time
                resultado['pv_final'] = pv_actual
                resultado['converged'] = False
                return resultado
        
        # Convergencia exitosa
        resultado['tiempo'] = t
        resultado['pv_final'] = pv_actual
        resultado['converged'] = True
        
        return resultado