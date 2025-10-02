"""
Simulador de Control de Nivel de Tanque
Simulador físico simple para entrenar agentes de control PID
Compatible con interface OpenAI Gym (sin dependencia directa)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


class TankLevelSimulator:
    """
    Simulador de control de nivel en tanque cilíndrico
    
    Física del proceso:
    - Tanque cilíndrico con área constante
    - Caudal de entrada controlado por válvula (variable manipulada)  
    - Caudal de salida por gravedad: Qout = C * √h
    - Ecuación: A * dh/dt = Qin - Qout
    """
    
    def __init__(self,
                 tank_area: float = 2.0,        # m² - Área del tanque
                 max_height: float = 5.0,       # m - Altura máxima del tanque
                 max_inflow: float = 10.0,      # L/s - Caudal máximo de entrada
                 outflow_coeff: float = 2.0,    # Coeficiente de salida
                 dt: float = 1.0,               # s - Paso de simulación  
                 noise_level: float = 0.01,     # Nivel de ruido en medición
                 initial_level: float = 2.5,    # m - Nivel inicial
                 render_mode: Optional[str] = None):  # Interface compatible con Gym
        
        # Parámetros del tanque
        self.tank_area = tank_area
        self.max_height = max_height  
        self.max_inflow = max_inflow
        self.outflow_coeff = outflow_coeff
        self.dt = dt
        self.noise_level = noise_level
        self.render_mode = render_mode
        
        # Estado del proceso
        self.level = initial_level  # m - Nivel actual (PV)
        self.setpoint = 2.5         # m - Setpoint deseado
        self.time = 0.0             # s - Tiempo de simulación
        
        # Límites físicos
        self.min_level = 0.0
        self.max_level = max_height
        self.min_inflow = 0.0
        
        # Para cálculo de derivadas e integrales
        self.prev_error = 0.0
        self.integral_error = 0.0
        
        # Historia para gráficos
        self.history = {
            'time': [],
            'level': [],
            'setpoint': [],
            'inflow': [],
            'outflow': []
        }
        
        # Definir espacios de observación y acción (compatible con Gym)
        # Observation space: [level, setpoint, error, prev_error, integral, derivative]
        self.observation_space = {
            'low': np.array([-max_height, 0.0, -max_height, -max_height, -np.inf, -np.inf]),
            'high': np.array([max_height, max_height, max_height, max_height, np.inf, np.inf]),
            'shape': (6,),
            'dtype': np.float32
        }
        
        # Action space: Caudal de entrada (0 a max_inflow L/s)
        self.action_space = {
            'low': 0.0,
            'high': max_inflow,
            'shape': (1,),
            'dtype': np.float32
        }
        
        # Metadata compatible con Gym
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 4,
        }
        
        print(f"🏗️ Simulador de Tanque Creado (Gym Env):")
        print(f"   Área: {tank_area} m²")
        print(f"   Altura máxima: {max_height} m")
        print(f"   Caudal máximo: {max_inflow} L/s")
        print(f"   Nivel inicial: {initial_level} m")
        print(f"   Observation space: {self.observation_space}")
        print(f"   Action space: {self.action_space}")
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecutar un paso de simulación (Gym interface)
        
        Args:
            action: Acción del agente (caudal de entrada o array)
        
        Returns:
            observation: Estado actual [level, setpoint, error, prev_error, integral, derivative]
            reward: Recompensa para el agente
            terminated: Si el episodio terminó por condición terminal
            truncated: Si el episodio terminó por límite de tiempo
            info: Información adicional
        """
        # Convertir acción a escalar si es array
        if hasattr(action, '__len__'):
            control_signal = float(action[0])
        else:
            control_signal = float(action)
        # Limitar caudal de entrada
        inflow = np.clip(control_signal, self.min_inflow, self.max_inflow)
        
        # Calcular caudal de salida (por gravedad)
        # Qout = C * √h (solo si hay líquido)
        if self.level > 0:
            outflow = self.outflow_coeff * np.sqrt(self.level)
        else:
            outflow = 0.0
        
        # Ecuación diferencial: A * dh/dt = Qin - Qout
        # Convertir L/s a m³/s: dividir por 1000
        dhdt = (inflow/1000 - outflow/1000) / self.tank_area
        
        # Integrar usando Euler
        self.level += dhdt * self.dt
        
        # Aplicar límites físicos
        self.level = np.clip(self.level, self.min_level, self.max_level)
        
        # Avanzar tiempo
        self.time += self.dt
        
        # Obtener estado con ruido
        measured_level = self.level + np.random.normal(0, self.noise_level)
        
        # Calcular errores para PID
        error = self.setpoint - measured_level
        self.integral_error += error * self.dt
        derivative_error = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        
        # Estado para el agente
        observation = np.array([
            measured_level,          # PV - Nivel medido  
            self.setpoint,           # SP - Setpoint
            error,                   # Error actual
            self.prev_error,         # Error anterior
            self.integral_error,     # Integral del error
            derivative_error         # Derivada del error
        ], dtype=np.float32)
        
        # Calcular recompensa
        reward = self._calculate_reward(error)
        
        # Verificar condiciones de terminación
        terminated = self._check_terminated()
        truncated = False  # Por ahora no usamos truncated
        
        # Información adicional
        info = {
            'level': self.level,
            'setpoint': self.setpoint,
            'error': error,
            'inflow': inflow,
            'outflow': outflow,
            'time': self.time
        }
        
        # Guardar historia
        self._update_history(inflow, outflow)
        
        # Actualizar error anterior
        self.prev_error = error
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, error: float) -> float:
        """
        Calcular recompensa basada en el error
        
        Función de recompensa:
        - Máxima recompensa cuando error = 0
        - Penaliza errores grandes exponencialmente
        - Penaliza niveles peligrosos (muy altos/bajos)
        """
        # Recompensa principal: exponencial del error absoluto
        error_reward = np.exp(-abs(error) * 2.0)
        
        # Penalización por niveles peligrosos
        safety_penalty = 0.0
        if self.level < 0.5:  # Nivel muy bajo
            safety_penalty = -2.0 * (0.5 - self.level)
        elif self.level > self.max_height * 0.9:  # Nivel muy alto
            safety_penalty = -2.0 * (self.level - self.max_height * 0.9)
        
        return error_reward + safety_penalty
    
    def _check_terminated(self) -> bool:
        """
        Verificar si la simulación debe terminar por condiciones terminales
        
        Condiciones de terminación:
        - Nivel fuera de límites seguros
        """
        # Terminar si nivel es peligroso
        if self.level <= 0.0 or self.level >= self.max_height:
            return True
        
        return False
    
    def _update_history(self, inflow: float, outflow: float):
        """Actualizar historia para gráficos"""
        self.history['time'].append(self.time)
        self.history['level'].append(self.level)
        self.history['setpoint'].append(self.setpoint)
        self.history['inflow'].append(inflow)
        self.history['outflow'].append(outflow)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reiniciar simulación (Gym interface)
        
        Args:
            seed: Semilla para números aleatorios
            options: Opciones adicionales (puede incluir initial_level, setpoint)
        
        Returns:
            observation: Estado inicial
            info: Información adicional
        """
        # Configurar semilla
        if seed is not None:
            np.random.seed(seed)
        
        # Extraer opciones
        if options is None:
            options = {}
        initial_level = options.get('initial_level', None)
        setpoint = options.get('setpoint', None)
        # Nivel inicial
        if initial_level is None:
            # Nivel aleatorio entre 20% y 80% de la altura
            self.level = np.random.uniform(
                self.max_height * 0.2, 
                self.max_height * 0.8
            )
        else:
            self.level = np.clip(initial_level, self.min_level, self.max_level)
        
        # Setpoint
        if setpoint is not None:
            self.setpoint = np.clip(setpoint, self.min_level, self.max_level)
        
        # Reiniciar variables de control
        self.time = 0.0
        self.prev_error = 0.0
        self.integral_error = 0.0
        
        # Limpiar historia
        self.history = {key: [] for key in self.history.keys()}
        
        # Estado inicial
        error = self.setpoint - self.level
        observation = np.array([
            self.level,      # PV
            self.setpoint,   # SP  
            error,           # Error
            0.0,            # Error anterior
            0.0,            # Integral
            0.0             # Derivada
        ], dtype=np.float32)
        
        # Información inicial
        info = {
            'level': self.level,
            'setpoint': self.setpoint,
            'error': error,
            'time': self.time
        }
        
        return observation, info
    
    def set_setpoint(self, new_setpoint: float):
        """Cambiar setpoint durante la simulación"""
        self.setpoint = np.clip(new_setpoint, self.min_level, self.max_level)
    
    def get_tank_info(self) -> dict:
        """Obtener información actual del tanque"""
        outflow = self.outflow_coeff * np.sqrt(max(self.level, 0)) if self.level > 0 else 0
        
        return {
            'level': self.level,
            'setpoint': self.setpoint,
            'time': self.time,
            'outflow': outflow,
            'error': self.setpoint - self.level,
            'tank_capacity': self.tank_area * self.max_height
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Graficar resultados de la simulación
        
        Args:
            save_path: Ruta para guardar gráfico (opcional)
        """
        if not self.history['time']:
            print("⚠️ No hay datos para graficar")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico 1: Nivel vs Setpoint
        ax1.plot(self.history['time'], self.history['level'], 'b-', 
                label='Nivel Real', linewidth=2)
        ax1.plot(self.history['time'], self.history['setpoint'], 'r--', 
                label='Setpoint', linewidth=2)
        ax1.set_ylabel('Nivel (m)')
        ax1.set_title('Control de Nivel de Tanque')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Caudales
        ax2.plot(self.history['time'], self.history['inflow'], 'g-', 
                label='Caudal Entrada', linewidth=2)
        ax2.plot(self.history['time'], self.history['outflow'], 'm-', 
                label='Caudal Salida', linewidth=2)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Caudal (L/s)')
        ax2.set_title('Caudales del Sistema')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def render(self, mode: str = "human"):
        """
        Renderizar el ambiente (Gym interface)
        
        Args:
            mode: Modo de renderizado ("human" o "rgb_array")
        """
        if mode == "human":
            # Mostrar información básica
            print(f"🏗️ Tank Level: {self.level:.2f}m | Setpoint: {self.setpoint:.2f}m | Error: {self.setpoint - self.level:.3f}m")
        elif mode == "rgb_array":
            # Para el modo rgb_array necesitarías devolver una imagen
            # Por simplicidad, devolvemos None por ahora
            return None
    
    def close(self):
        """Cerrar el ambiente (Gym interface)"""
        # Limpiar recursos si es necesario
        pass


# Función de utilidad para pruebas rápidas
def test_tank_simulator():
    """Prueba rápida del simulador con interface Gym"""
    tank = TankLevelSimulator()
    
    print("\n🧪 Prueba del simulador (Gym interface)...")
    observation, info = tank.reset(options={'initial_level': 1.0, 'setpoint': 3.0})
    print(f"Observación inicial: {observation}")
    print(f"Info inicial: {info}")
    
    # Simular algunos pasos con control constante
    for i in range(50):
        # Control simple proporcional
        error = observation[2]  # Error está en posición 2
        control = 5.0 + 2.0 * error  # Control proporcional simple
        
        observation, reward, terminated, truncated, info = tank.step(control)
        
        if i % 10 == 0:
            print(f"Paso {i}: Nivel={info['level']:.2f}m, "
                  f"Error={info['error']:.2f}m, "
                  f"Recompensa={reward:.3f}")
        
        if terminated or truncated:
            break
    
    print("✅ Prueba completada")
    return tank


if __name__ == "__main__":
    # Ejecutar prueba si se llama directamente
    simulator = test_tank_simulator()
    simulator.plot_results()