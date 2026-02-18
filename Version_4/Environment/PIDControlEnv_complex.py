import gymnasium as gym
import numpy as np
import random
from abc import ABC
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List, Union

from Aux.PIDComponents_PID import PIDController 
from Aux.PIDComponents_time import ResponseTimeDetector
from Aux.PIDComponentes_translate import ApplyAction
from Aux.PIDComponents_Reward import RewardCalculator
from .Simulation_Env.SimulationEnv import SimulationPIDEnv

class PIDControlEnv_Complex(gym.Env, ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        #CONFIGURACION DEL AMBIENTE

        ##Arquitectura
        self.architecture = config.get('architecture', 'jerarquica')  # 'simple' o 'jerairquica'

        ##Tipo de entorno
        env_type = config.get('env_type', 'simulation')
        if env_type == 'simulation':
            self.proceso = SimulationPIDEnv(config.get('env_type_config', {}))
        #elif env_type == 'real':
        #    self.proceso = RealPIDEnv(config.get('env_type_config', {}))

        
        ##Variables del proceso
        ###Control
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', [(0.0, 100.0), (0.0, 100.0)]) #Rangos de las variables manipulables 
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]
       
        ###Target
        self.n_target_vars = config.get('n_target_vars', 1)
        self.target_ranges = config.get('target_ranges', [(0.0, 1.0)])
        self.target_setpoints = config.get('target_setpoints', [0.2])
        self.target_working_ranges = config.get('target_working_ranges', [(0.0, 5.0)])
        self.target_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.target_working_ranges
        ]

        #CONFIGURACION DEL AGENTE

        ##Configuracion de los Agentes según arquitectura
        self.agente_orch = config.get('agent_orchestrator_config', {})
        self.agente_ctrl = None
        self.action_type_ctrl = None

        ## Estado interno
        ###errores
        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        ### tiempo de respuesta y dt (Detectores de tiempo solo para dinamicas controlables (PID))
        self.dt_sim = config.get('dt_usuario', 1.0)  

        self.response_time_detectors = [
            ResponseTimeDetector(
                proceso=self.proceso,
                variable_index=i,
                env_type=env_type,
                dt=self.dt_sim,
                tolerance=0.02  
            )
            for i in range(self.n_target_vars)
        ]

        ### Dinamica del ambiente (PIDs para variables manipulables)
        self.pid_controllers = [
            PIDController(kp=1.0, ki=0.1, kd=0.01, dt=self.dt_sim)  
            for _ in range(self.n_manipulable_vars)
        ]

        #### Valor dummy iniciales (se calculan en el primer step)
        self.tiempo_respuesta = [0.0] * self.n_target_vars


        #ESPACIO DE OBSERVACIONES

        self.obs_structure = ['pv', # Dónde estoy
                              'sp', # Dónde quiero estar
                              'error', # Cuánto me falta?
                              'error_integral', # Hay offset acumulado? (offset es una diferencia constante entre pv y sp que no permite llegar a sp)
                              'error_derivative' # Voy muy rápido/lento?
                              ]
        self.obs_size = len(self.obs_structure) 
        # Según arquitectura, cuántas variables ve cada uno para cumplir con la estructura de observación
        n_obs_ctrl = self.obs_size * self.n_manipulable_vars
        n_obs_orch = self.obs_size * self.n_target_vars

        self.observation_space = spaces.Dict({
            'orch': spaces.Box(
                low=np.full(n_obs_orch, -np.inf, dtype=np.float32),
                high=np.full(n_obs_orch, np.inf, dtype=np.float32),
                dtype=np.float32
            ),
            'ctrl': spaces.Box(
                low=np.full(n_obs_ctrl, -np.inf, dtype=np.float32),
                high=np.full(n_obs_ctrl, np.inf, dtype=np.float32),
                dtype=np.float32
            )
        })
        
        # ESPACIO DE ACCIONES

        # El espacio de acciones es continuo, ya que da numeros, pero se puede manejar tambien como discreto si se usan indices para seleccionar acciones predefinidas
            # Orch
        if self.agente_orch.get('agent_type', 'continuous') == 'continuous':
            self.action_space_orch = spaces.Box(
                low=np.array([-r[1] for r in self.manipulable_ranges], dtype=np.float32),
                high=np.array([r[1] for r in self.manipulable_ranges], dtype=np.float32),
                dtype=np.float32
            )
        elif self.agente_orch.get('agent_type', 'discrete') == 'discrete':
            self.action_space_orch = spaces.MultiDiscrete([3] * self.n_manipulable_vars)

            #Ctrl
        if self.agente_ctrl.get('agent_type', 'continuous') == 'continuous':
            self.action_space_ctrl = spaces.Box(
                low=np.tile(np.array([-100, -10, -1]), self.n_manipulable_vars).astype(np.float32),
                high=np.tile(np.array([100, 10, 1]), self.n_manipulable_vars).astype(np.float32),
                dtype=np.float32
            )
        elif self.agente_ctrl.get('agent_type', 'discrete') == 'discrete':
            self.action_space_ctrl = spaces.MultiDiscrete([7] * self.n_manipulable_vars)

        # Combinar en Dict
        self.action_space = spaces.Dict({
            'orch': self.action_space_orch,
            'ctrl': self.action_space_ctrl
        })    

        ## Mapeo de acciones discretas
        if self.agente_orch.get('agent_type', 'continuous') == 'discrete':
            self.ACTION_MAP_ORCH = {
                0: +1,  # Aumentar SP
                1: -1,  # Disminuir SP
                2: 0    # Mantener
            }       

        ## Componente para traducir acciones a parámetros de control
        self.apply_action_orch = ApplyAction(
            delta_percent_orch=config.get('delta_percent_orch', 0.05),
            manipulable_ranges=self.manipulable_ranges
        )

        self.apply_action_ctrl = ApplyAction(
            delta_percent_ctrl=config.get('delta_percent_ctrl', 0.2),
            pid_limits=config.get('pid_limits', None),
            manipulable_ranges=self.manipulable_ranges
        )

        # ENTRENAMIENTO
        self.max_steps = config.get('max_steps', 20)
        self.current_step = 0

        ## Recompensa
        self.reward_calculator = RewardCalculator(
            weights=config.get('reward_weights', None),
            manipulable_ranges=self.manipulable_ranges,
            dead_band=config.get('reward_dead_band', 0.02)
        )
            
    def _get_observation(self):            
        obs_orch = []
        for j in range(self.n_target_vars):
            obs_orch.extend([
                self.target_pvs[j],
                self.target_setpoints[j],
                self.error_target[j],
                self.error_integral_target[j],
                self.error_derivative_target[j]
            ])
        
        obs_ctrl = []
        for i in range(self.n_manipulable_vars):
            # PV actual, SP deseado (definido por ORCH), errores
            sp_desired = self.new_SP[i] if hasattr(self, 'new_SP') else self.manipulable_pvs[i]
            error = sp_desired - self.manipulable_pvs[i]
            
            obs_ctrl.extend([
                self.manipulable_pvs[i],
                sp_desired,
                error,
                0.0,  # error_integral (simplificado)
                0.0   # error_derivative (simplificado)
            ])
        
        return {
            'orch': np.array(obs_orch, dtype=np.float32),
            'ctrl': np.array(obs_ctrl, dtype=np.float32)
        }

    def _get_info(self):
        return {
            'target_pvs': self.target_pvs.copy(),
            'manipulable_pvs': self.manipulable_pvs.copy(),
            'energy': self.energy_accumulated,
            'new_SP': self.new_SP.copy() if hasattr(self, 'new_SP') else []
        }

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # VARIABES DEL ENTORNO A RESETEAR
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges    
        ]

        self.target_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.target_working_ranges
        ]
        self.target_setpoints = [
            random.uniform(rango[0], rango[1])
            for rango in self.target_ranges
        ]

        self.new_SP = self.manipulable_pvs.copy()    

        # ERRORES
        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        #TIEMPO
        self.tiempo_respuesta = [0.0] * self.n_target_vars

        #VARIABLES DE INFO
        self.trajectory_target = [[] for _ in range(self.n_target_vars)]
        self.energy_accumulated = 0.0
        self.overshoot_target = [0.0] * self.n_target_vars
        self.accumulated_error_target = [0.0] * self.n_target_vars

        # VARIABLES DE ENTRENAMIENTO
        self.current_step = 0

        # OBSERVACION E INFO
        observation = self._get_observation()
        info = self._get_info() 

        return observation, info

    def step(self, action):        
        # 1. EXTRAER ACCIÓN DE ORCH
        if isinstance(action, dict):
            action_orch = action['orch']
        else:
            action_orch = action
        
        # 2. TRADUCIR ACCIÓN ORCH A NUEVOS SETPOINTS
        self.action_type_orch = self.agente_orch.get('agent_type', 'continuous')
        self.new_SP = self.apply_action_orch.translate(
            action=action_orch,
            agent_type='orch',
            action_type=self.action_type_orch,
            current_values=self.manipulable_pvs
        )
        
        # 3. CTRL DECIDE PARÁMETROS PID PARA ALCANZAR ESOS SP
        obs_ctrl = self._get_observation()['ctrl']
        action_ctrl = self.agente_ctrl.select_action(obs_ctrl, training=False)
        
        # 4. TRADUCIR ACCIÓN CTRL A PARÁMETROS PID
        pid_params = self.apply_action_ctrl.translate(
            action=action_ctrl,
            agent_type='ctrl',
            action_type=self.action_type_ctrl,
            current_values=[(c.kp, c.ki, c.kd) for c in self.pid_controllers]
        )
        
        # 5. ACTUALIZAR PARÁMETROS PID
        for i, (kp, ki, kd) in enumerate(pid_params):
            self.pid_controllers[i].kp = kp
            self.pid_controllers[i].ki = ki
            self.pid_controllers[i].kd = kd
        
        # 6. SIMULAR VARIABLES MANIPULABLES (PIDs → nuevos SP)
        energy_step = 0.0
        
        for i in range(self.n_manipulable_vars):
            resultado = self.response_time_detectors[i].estimate(
                pv_inicial=self.manipulable_pvs[i],
                sp=self.new_SP[i],  #  Nuevo SP definido por ORCH
                pid_controller=self.pid_controllers[i],
                max_time=1800
            )
            
            # Guardar resultados
            self.manipulable_pvs[i] = resultado['pv_final']
            
            # Acumular energía
            if 'trayectoria_control' in resultado:
                energy_step += sum(abs(u) for u in resultado['trayectoria_control']) * self.dt_sim
        
        # 7. ACTUALIZAR VARIABLES OBJETIVO (dinámica del proceso)
        # El proceso calcula target_pvs basado en manipulable_pvs
        self.target_pvs = self.proceso.get_target_values(self.manipulable_pvs)
        
        # 8. ACTUALIZAR ERRORES (solo de variables objetivo)
        self._update_errors()
        
        # 9. CALCULAR REWARD (basado en variables objetivo)
        errors = [abs(pv - sp) for pv, sp in zip(self.target_pvs, self.target_setpoints)]
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        reward = self.reward_calculator.calculate(
            errors=errors,
            tiempos_respuesta=[0.0] * self.n_target_vars,  # No aplicable aquí
            overshoots=[0.0] * self.n_target_vars,
            energy_step=energy_step,
            pvs=self.target_pvs,
            setpoints=self.target_setpoints,
            terminated=terminated,
            truncated=truncated
        )
        
        # 10. OBTENER OBSERVACIÓN E INFO
        observation = self._get_observation()
        info = self._get_info()
        
        # 11. INCREMENTAR STEP
        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def _update_errors(self):

        self.dt = self.dt_sim

        # Actualizar errores para variables objetivo
        for i in range(self.n_target_vars):
            error = self.target_setpoints[i] - self.target_pvs[i]
            self.error_target[i] = error
            self.error_integral_target[i] += error * self.dt
            self.error_derivative_target[i] = (error - self.error_prevs_target[i]) / self.dt if self.dt > 0 else 0.0
            self.error_prevs_target[i] = error

        return {
            'error_target': self.error_target,
            'error_integral_target': self.error_integral_target,
            'error_derivative_target': self.error_derivative_target,
            'error_prevs_target': self.error_prevs_target
        }
    
    def _check_truncated(self) -> bool:
        # Episodio se trunca si alcanza max_steps
        return self.current_step >= self.max_steps
    
    def _check_terminated(self) -> bool:
        threshold = 0.02  # 2% de error relativo
        
        # Éxito: todas las variables dentro del threshold
        errors_relativo = [
            abs(pv - sp) / abs(sp) if sp != 0 else abs(pv - sp)
            for pv, sp in zip(self.target_pvs, self.target_setpoints)
        ]
        success = all(error < threshold for error in errors_relativo)
        
        # Fallo: alguna variable fuera de rango físico
        failure = any(
            pv < rango[0] or pv > rango[1]
            for pv, rango in zip(self.target_pvs, self.target_ranges)
        )
        
        return success or failure

    def _calculate_variable_metrics(self, var_idx: int, resultado: dict):
        trayectoria = resultado['trayectoria_pv']
        sp = self.target_setpoints[var_idx]
        
        # 1. Overshoot (máximo pico sobre SP, en porcentaje)
        max_pv = max(trayectoria)
        if max_pv > sp:
            self.overshoot_target[var_idx] = (max_pv - sp) / sp * 100
        else:
            self.overshoot_target[var_idx] = 0.0
        
        # 2. Error acumulado (integral del error absoluto)
        accumulated_error = sum(abs(pv - sp) for pv in trayectoria) * self.dt_sim
        self.accumulated_error_target[var_idx] = accumulated_error
        
        # 3. Energía (esfuerzo de control)
        if 'trayectoria_control' in resultado:
            energy = sum(abs(u) for u in resultado['trayectoria_control']) * self.dt_sim
            self.energy_accumulated += energy    

    def _calculate_reward(self, energy_step, terminated, truncated) -> float:

        errors = [abs(pv - sp) for pv, sp in zip(self.target_pvs, self.target_setpoints)]
        
        return self.reward_calculator.calculate(
            errors=errors,
            tiempos_respuesta=self.tiempo_respuesta,
            overshoots=self.overshoot_target,
            energy_step=energy_step,
            pvs=self.target_pvs,
            setpoints=self.target_setpoints,
            terminated=terminated,
            truncated=truncated
        )  