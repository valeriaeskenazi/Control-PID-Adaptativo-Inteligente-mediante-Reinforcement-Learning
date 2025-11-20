"""
Simulador de reactor CSTR para control de concentración y volumen.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional, Dict, List, Union


class CSTRSimulator:
    """
    Simulador de reactor CSTR (Continuously Stirred Tank Reactor) con control
    de concentración y volumen.
    
    Dinámica del reactor:
        - Estados: Ca, Cb, Cc (concentraciones), T (temperatura), V (volumen)
        - Controles: Tc (temperatura de enfriamiento), F (flujo de entrada)
        - Reacciones: A -> B -> C (exotérmicas)
    
    Args:
        dt: Paso de tiempo para integración [s]
        control_limits: Límites de las variables de control [(Tc_min, Tc_max), (F_min, F_max)]
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        control_limits: Tuple[Tuple[float, float], Tuple[float, float]] = ((290, 450), (99, 105))
    ):
        """Inicializar simulador de CSTR."""
        self.dt = dt
        self.Tc_min, self.Tc_max = control_limits[0]
        self.F_min, self.F_max = control_limits[1]
        
        # Parámetros del proceso
        self.Tf = 350.0  # Temperatura de alimentación (K)
        self.Caf = 1.0   # Concentración de A en alimentación (mol/m³)
        self.Fout = 100.0  # Flujo volumétrico de salida (m³/s)
        self.rho = 1000.0  # Densidad (kg/m³)
        self.Cp = 0.239    # Capacidad calorífica (J/kg-K)
        self.UA = 5e4      # Coeficiente de transferencia de calor (W/m²-K)
        
        # Parámetros de reacción A->B
        self.mdelH_AB = 5e3      # Calor de reacción (J/mol)
        self.EoverR_AB = 8750.0  # Energía de activación / R (K)
        self.k0_AB = 7.2e10      # Factor pre-exponencial (1/s)
        
        # Parámetros de reacción B->C
        self.mdelH_BC = 4e3      # Calor de reacción (J/mol)
        self.EoverR_BC = 10750.0 # Energía de activación / R (K)
        self.k0_BC = 8.2e10      # Factor pre-exponencial (1/s)
        
        # Estado actual [Ca, Cb, Cc, T, V]
        self.state = np.zeros(5)
        
        # Condiciones de estado estacionario
        self.Ca_ss = 0.80
        self.Cb_ss = 0.0
        self.Cc_ss = 0.0
        self.T_ss = 327.0
        self.V_ss = 102.0
        
        # Valores iniciales de variables manipulables
        self.Tc_initial = 327.0
        self.F_initial = 100.0
        
        print("=" * 60)
        print("⚗️  Simulador de Reactor CSTR creado")
        print(f"   Temperatura alimentación: {self.Tf} K")
        print(f"   Concentración alimentación: {self.Caf} mol/m³")
        print(f"   Límites Tc: [{self.Tc_min}, {self.Tc_max}] K")
        print(f"   Límites F: [{self.F_min}, {self.F_max}] m³/s")
        print(f"   Paso de tiempo: {dt} s")
        print(f"   Modo: Multi-variable (2 variables manipulables)")
        print("=" * 60)
    
    def get_n_variables(self) -> int:
        """
        Número de variables manipulables.
        
        Returns:
            int: Número de variables (Tc y F) = 2
        """
        return 2
    
    def get_initial_pvs(self) -> List[float]:
        """
        Valores iniciales de variables de proceso.
        
        Returns:
            List[float]: [Tc_inicial, F_inicial]
        """
        return [self.Tc_initial, self.F_initial]
    
    def _reactor_dynamics(self, x: np.ndarray, t: float, u: np.ndarray) -> np.ndarray:
        """
        Ecuaciones diferenciales del reactor CSTR.
        """
        # Desempacar estados
        Ca, Cb, Cc, T, V = x
        
        # ✅ PROTEGER VALORES ANTES de usarlos
        Ca = np.clip(Ca, 0.0, 2.0)
        Cb = np.clip(Cb, 0.0, 2.0)
        Cc = np.clip(Cc, 0.0, 2.0)
        T = np.clip(T, 50.0, 500.0)   # Evitar T=0
        V = np.clip(V, 10.0, 200.0)   # Evitar V=0
        
        # Desempacar controles
        Tc, Fin = u
        
        # Resto del código SIN CAMBIOS...
        rA = self.k0_AB * np.exp(-self.EoverR_AB / T) * Ca
        rB = self.k0_BC * np.exp(-self.EoverR_BC / T) * Cb
        
        dCadt = (Fin * self.Caf - self.Fout * Ca) / V - rA
        dCbdt = rA - rB - self.Fout * Cb / V
        dCcdt = rB - self.Fout * Cc / V
        
        dTdt = (
            Fin / V * (self.Tf - T)
            + self.mdelH_AB / (self.rho * self.Cp) * rA
            + self.mdelH_BC / (self.rho * self.Cp) * rB
            + self.UA / V / self.rho / self.Cp * (Tc - T)
        )
        
        dVdt = Fin - self.Fout
        
        return np.array([dCadt, dCbdt, dCcdt, dTdt, dVdt])
    
    def step(
        self,
        control_output: Union[np.ndarray, List[float]],
        setpoint: Optional[Union[np.ndarray, List[float]]] = None
    ) -> Union[List[float], Dict[str, float]]:
        """
        Simular un paso de tiempo.
        
        Args:
            control_output: Vector de control [Tc, F] o normalizado [-1, 1]
                           Puede ser lista o array
            setpoint: Setpoints deseados (opcional, ignorado en multi-variable)
        
        Returns:
            Para modo multi-variable: List[float] con [T_actual, F_actual]
            Para modo single-variable: Dict con el estado completo
        """
        # Determinar si es modo multi-variable (lista) o single-variable
        is_multi_variable = isinstance(control_output, (list, np.ndarray)) and len(control_output) >= 2
        
        if is_multi_variable:
            # Modo MULTI-VARIABLE
            Tc = float(control_output[0])
            F = float(control_output[1])
            
            # Si el control viene normalizado (-1, 1), desnormalizar
            if abs(Tc) <= 1.0 and abs(F) <= 1.0:
                Tc = self._denormalize(Tc, self.Tc_min, self.Tc_max)
                F = self._denormalize(F, self.F_min, self.F_max)
            
            # Aplicar límites
            Tc = np.clip(Tc, self.Tc_min, self.Tc_max)
            F = np.clip(F, self.F_min, self.F_max)
            
            u = np.array([Tc, F])
            
            # Integrar ecuaciones diferenciales
            t_span = [0, self.dt]
            solution = odeint(self._reactor_dynamics, self.state, t_span, args=(u,))

            # Actualizar estado (última fila de la solución)
            self.state = solution[-1]

            # Proteger contra valores físicamente imposibles (DESPUÉS de actualizar)
            self.state[0] = np.clip(self.state[0], 0.0, 2.0)    # Ca
            self.state[1] = np.clip(self.state[1], 0.0, 2.0)    # Cb  
            self.state[2] = np.clip(self.state[2], 0.0, 2.0)    # Cc
            self.state[3] = np.clip(self.state[3], 50.0, 500.0) # T
            self.state[4] = np.clip(self.state[4], 10.0, 200.0) # V
            
            # Retornar PVs actuales: [T_actual, F_actual]
            T_meas = self.state[3] + np.random.uniform(-0.1, 0.1)
            F_meas = F + np.random.uniform(-0.01, 0.01)
            
            return [float(T_meas), float(F_meas)]
        
        else:
            # Modo SINGLE-VARIABLE (retrocompatibilidad)
            # Si el control viene normalizado (-1, 1), desnormalizar
            if np.all(np.abs(control_output) <= 1.0):
                Tc = self._denormalize(control_output[0], self.Tc_min, self.Tc_max)
                F = self._denormalize(control_output[1], self.F_min, self.F_max)
            else:
                Tc, F = control_output
            
            # Aplicar límites
            Tc = np.clip(Tc, self.Tc_min, self.Tc_max)
            F = np.clip(F, self.F_min, self.F_max)
            
            u = np.array([Tc, F])
            
            # Integrar ecuaciones diferenciales
            t_span = [0, self.dt]
            solution = odeint(self._reactor_dynamics, self.state, t_span, args=(u,))
            
            # Actualizar estado (última fila de la solución)
            self.state = solution[-1]
            
            # Agregar ruido de medición (como en el código original)
            Ca_meas = self.state[0] + np.random.uniform(-0.001, 0.001)
            Cb_meas = self.state[1] + np.random.uniform(-0.001, 0.001)
            Cc_meas = self.state[2] + np.random.uniform(-0.001, 0.001)
            T_meas = self.state[3] + np.random.uniform(-0.1, 0.1)
            V_meas = self.state[4] + np.random.uniform(-0.01, 0.01)
            
            return {
                'Ca': Ca_meas,
                'Cb': Cb_meas,
                'Cc': Cc_meas,
                'T': T_meas,
                'V': V_meas,
                'Tc': Tc,
                'F': F
            }
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        randomize: bool = False
    ) -> Union[List[float], Dict[str, float]]:
        """
        Resetear simulador a condiciones iniciales.
        
        Args:
            initial_state: Estado inicial [Ca, Cb, Cc, T, V] (opcional)
            randomize: Si True, inicializa con valores aleatorios cercanos al ss
        
        Returns:
            Para multi-variable: List[float] con [T_inicial, F_inicial]
            Para single-variable: Dict con estado inicial del reactor
        """
        if initial_state is not None:
            self.state = np.array(initial_state)
        elif randomize:
            # Inicializar cerca del estado estacionario con variación
            self.state = np.array([
                self.Ca_ss + np.random.uniform(-0.1, 0.1),
                self.Cb_ss + np.random.uniform(-0.01, 0.01),
                self.Cc_ss + np.random.uniform(-0.01, 0.01),
                self.T_ss + np.random.uniform(-5, 5),
                self.V_ss + np.random.uniform(-2, 2)
            ])
        else:
            # Estado estacionario por defecto
            self.state = np.array([
                self.Ca_ss,
                self.Cb_ss,
                self.Cc_ss,
                self.T_ss,
                self.V_ss
            ])
        
        # Retornar valores iniciales de PVs para multi-variable
        return self.get_initial_pvs()
    
    def get_state(self) -> List[float]:
        """
        Obtener estado actual del simulador.
        
        Returns:
            Lista con [Cb, T, Cc, Ca, V] para acceso por índice
        """
        return [
            float(self.state[1]),  # Cb - índice 0 del resultado (variable objetivo)
            float(self.state[3]),  # T  - índice 1 del resultado (manipulable)
            float(self.state[2]),  # Cc - índice 2
            float(self.state[0]),  # Ca - índice 3
            float(self.state[4])   # V  - índice 4
        ]
    
    def get_measurements(self) -> Dict[str, float]:
        """
        Obtener mediciones con ruido (simula sensores reales).
        
        Returns:
            measurements: Mediciones ruidosas del reactor
        """
        return {
            'Ca': self.state[0] + np.random.uniform(-0.001, 0.001),
            'Cb': self.state[1] + np.random.uniform(-0.001, 0.001),
            'Cc': self.state[2] + np.random.uniform(-0.001, 0.001),
            'T': self.state[3] + np.random.uniform(-0.1, 0.1),
            'V': self.state[4] + np.random.uniform(-0.01, 0.01)
        }
    
    def set_disturbance(self, Caf: Optional[float] = None, Tf: Optional[float] = None):
        """
        Introducir perturbaciones en las condiciones de alimentación.
        
        Args:
            Caf: Nueva concentración de alimentación (mol/m³)
            Tf: Nueva temperatura de alimentación (K)
        """
        if Caf is not None:
            self.Caf = Caf
            print(f"🔀 Perturbación aplicada: Caf = {Caf} mol/m³")
        
        if Tf is not None:
            self.Tf = Tf
            print(f"🔀 Perturbación aplicada: Tf = {Tf} K")
    
    @staticmethod
    def _denormalize(value: float, min_val: float, max_val: float) -> float:
        """Convertir de [-1, 1] a rango real."""
        return ((value + 1) / 2) * (max_val - min_val) + min_val
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Convertir de rango real a [-1, 1]."""
        return 2 * (value - min_val) / (max_val - min_val) - 1