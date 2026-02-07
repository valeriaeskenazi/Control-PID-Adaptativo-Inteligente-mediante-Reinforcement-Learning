import gymnasium as gym
import numpy as np
import random
from abc import ABC, abstractmethod
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List, Union

from ..Aux.pid_components import PIDController
from ..Aux.pid_components_time import ResponseTimeDetector
from .real_env import RealPIDEnv
from .simulation_env import SimulationPIDEnv

class BasePIDControlEnv(gym.Env, ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        #CONFIGURACION DEL AMBIENTE

        ##Arquitectura
        self.architecture = config.get('architecture', 'jerarquica')  # 'simple' o 'jerairquica'

        ##Tipo de entorno
        env_type = config.get('env_type', 'simulation')
        if env_type == 'simulation':
            self.proceso = SimulationPIDEnv(config.get('env_type_config', {}))
        elif env_type == 'real':
            self.proceso = RealPIDEnv(config.get('env_type_config', {}))

        ## Dinamica del ambiente
        self.pid_controllers = [
            PIDController(kp=1.0, ki=0.1, kd=0.01, dt=1.0)  # dt dummy, se actualiza en reset
            for _ in range(self.n_manipulable_vars)
        ]    
        
        ##Variables del proceso
        ###Control
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', [(0.0, 100.0), (0.0, 100.0)]) #Rangos de las variables manipulables 
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]
        self.manipulable_setpoints = config.get('manipulable_setpoints')
        if self.manipulable_setpoints is None:
            self.manipulable_setpoints = [
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
        if self.architecture == 'simple':
            self.agente_orch = False
            self.agente_ctrl = config.get('agent_controller_config', {})
        elif self.architecture == 'jerarquica':
            self.agente_orch = config.get('agent_orchestrator_config', {})
            self.agente_ctrl = config.get('agent_controller_config', {})

        ## Estado interno
        ###errores
        self.error_integral_manipulable = [0.0] * self.n_manipulable_vars
        self.error_derivative_manipulable = [0.0] * self.n_manipulable_vars
        self.error_manipulable = [0.0] * self.n_manipulable_vars
        self.error_prevs_manipulable = [0.0] * self.n_manipulable_vars

        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        ### tiempo de respuesta y dt (Detectores de tiempo solo para dinamicas controlables (PID))
        self.dt_sim = config.get('dt_simulation', 1.0)  

        self.response_time_detectors = [
            ResponseTimeDetector(
                proceso=self.proceso,
                variable_index=i,
                env_type=env_type,
                dt=self.dt_sim  
            )
            for i in range(self.n_manipulable_vars)
        ]

        #### Valores dummy iniciales (se calculan en el primer step)
        self.tiempo_respuesta = None
        self.step_duration = None


        #ESPACIO DE OBSERVACIONES

        self.obs_structure = ['pv', # Dónde estoy
                              'sp', # Dónde quiero estar
                              'error', # Cuánto me falta?
                              'error_integral', # Hay offset acumulado? (offset es una diferencia constante entre pv y sp que no permite llegar a sp)
                              'error_derivative' # Voy muy rápido/lento?
                              ]
        self.obs_size = len(self.obs_structure) 
        # Según arquitectura, cuántas variables ve cada uno para cumplir con la estructura de observación
        if self.architecture == 'simple':
            n_obs_total = self.obs_size * self.n_manipulable_vars
            
            self.observation_space = spaces.Box(
                low=np.full(n_obs_total, -np.inf, dtype=np.float32),
                high=np.full(n_obs_total, np.inf, dtype=np.float32),
                dtype=np.float32
            )

        elif self.architecture == 'jerarquica':
            n_obs_ctrl = self.obs_size * self.n_manipulable_vars
            n_obs_orch = self.obs_size * self.n_target_vars
            
            self.observation_space = {
                'ctrl': spaces.Box(
                    low=np.full(n_obs_ctrl, -np.inf, dtype=np.float32),
                    high=np.full(n_obs_ctrl, np.inf, dtype=np.float32),
                    dtype=np.float32
                ),
                'orch': spaces.Box(
                    low=np.full(n_obs_orch, -np.inf, dtype=np.float32),
                    high=np.full(n_obs_orch, np.inf, dtype=np.float32),
                    dtype=np.float32
                )
            }
        
        # ESPACIO DE ACCIONES

        # El espacio de acciones es continuo, ya que da numeros, pero se puede manejar tambien como discreto si se usan indices para seleccionar acciones predefinidas
        if self.architecture == 'simple':
            if self.agente_ctrl.get('agent_type', 'continuous') == 'continuous':
                self.action_space = spaces.Box(
                    low=np.tile(np.array([-100, -10, -1]), self.n_manipulable_vars).astype(np.float32),
                    high=np.tile(np.array([100, 10, 1]), self.n_manipulable_vars).astype(np.float32),
                    dtype=np.float32
                )
            elif self.agente_ctrl.get('agent_type', 'discrete') == 'discrete':
                self.action_space = spaces.MultiDiscrete([7] * self.n_manipulable_vars)

        elif self.architecture == 'jerarquica':
            # Ctrl
            if self.agente_ctrl.get('agent_type', 'continuous') == 'continuous':
                self.action_space_ctrl = spaces.Box(
                    low=np.tile(np.array([-100, -10, -1]), self.n_manipulable_vars).astype(np.float32),
                    high=np.tile(np.array([100, 10, 1]), self.n_manipulable_vars).astype(np.float32),
                    dtype=np.float32
                )
            elif self.agente_ctrl.get('agent_type', 'discrete') == 'discrete':
                self.action_space_ctrl = spaces.MultiDiscrete([7] * self.n_manipulable_vars)
            
            # Orch
            if self.agente_orch.get('agent_type', 'continuous') == 'continuous':
                self.action_space_orch = spaces.Box(
                    low=np.array([-r[1] for r in self.manipulable_ranges], dtype=np.float32),
                    high=np.array([r[1] for r in self.manipulable_ranges], dtype=np.float32),
                    dtype=np.float32
                )
            elif self.agente_orch.get('agent_type', 'discrete') == 'discrete':
                self.action_space_orch = spaces.MultiDiscrete([3] * self.n_manipulable_vars)

        ## Mapeo de acciones discretas
        if self.agente_ctrl.get('agent_type', 'continuous') == 'discrete':
            self.ACTION_MAP_CTRL = {
                0: ('Kp', +1),  # Kp ↑
                1: ('Ki', +1),  # Ki ↑
                2: ('Kd', +1),  # Kd ↑
                3: ('Kp', -1),  # Kp ↓
                4: ('Ki', -1),  # Ki ↓
                5: ('Kd', -1),  # Kd ↓
                6: ('mantener', 0)
            }

        if self.agente_orch.get('agent_type', 'continuous') == 'discrete':
            self.ACTION_MAP_ORCH = {
                0: +1,  # Aumentar SP
                1: -1,  # Disminuir SP
                2: 0    # Mantener
            }       


    def _get_observation(self):
                
        if self.architecture == 'simple':
            obs = []
            for i in range(self.n_manipulable_vars):
                obs.extend([
                    self.manipulable_pvs[i],
                    self.manipulable_setpoints[i],
                    self.error_manipulable[i],
                    self.error_integral_manipulable[i],
                    self.error_derivative_manipulable[i]
                ])
            return np.array(obs, dtype=np.float32)
        
        elif self.architecture == 'jerarquica':
            obs_ctrl = []
            for i in range(self.n_manipulable_vars):
                obs_ctrl.extend([
                    self.manipulable_pvs[i],
                    self.manipulable_setpoints[i],
                    self.error_manipulable[i],
                    self.error_integral_manipulable[i],
                    self.error_derivative_manipulable[i]
                ])
            
            obs_orch = []
            for j in range(self.n_target_vars):
                obs_orch.extend([
                    self.target_pvs[j],
                    self.target_setpoints[j],
                    self.error_target[j],
                    self.error_integral_target[j],
                    self.error_derivative_target[j]
                ])
            
            return {
                'ctrl': np.array(obs_ctrl, dtype=np.float32),
                'orch': np.array(obs_orch, dtype=np.float32)
            }

    def _get_info(self):
        
        info = {
            # Trayectorias completas durante el step
            'trajectory_manipulable': self.trajectory_manipulable,  # Lista de listas [[pv1_t0, pv1_t1, ...], [pv2_t0, pv2_t1, ...]]
            'trajectory_target': self.trajectory_target,            # Idem pero para el orchestrador
            
            # Energía acumulada (esfuerzo de control)
            'energy': self.energy_accumulated,  # Suma de |control_output| * dt
            
            # Overshoot (máximo pico sobre SP)
            'overshoot_manipulable': self.overshoot_manipulable,  # Lista [overshoot_var1, overshoot_var2, ...]
            'overshoot_target': self.overshoot_target,            # Idem pero para el orchestrador
            
            # Error acumulado absoluto
            'accumulated_error_manipulable': self.accumulated_error_manipulable,  
            'accumulated_error_target': self.accumulated_error_target,            
            
        }
        
        return info

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # VARIABES DEL ENTORNO A RESETEAR
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges    
        ]
        self.manipulable_setpoints = [
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

        # DINAMICA DEL AMBIENTE
        self.pid_controllers = [
            PIDController(kp=1.0, ki=0.1, kd=0.01, dt=1.0)  # dt dummy, se actualiza en reset
            for _ in range(self.n_manipulable_vars)
        ]

        # ERRORES
        self.error_integral_manipulable = [0.0] * self.n_manipulable_vars
        self.error_derivative_manipulable = [0.0] * self.n_manipulable_vars
        self.error_manipulable = [0.0] * self.n_manipulable_vars
        self.error_prevs_manipulable = [0.0] * self.n_manipulable_vars

        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        #TIEMPO
        self.tiempo_respuesta = None
        self.dt = None
        self.step_duration = None

        #VARIABLES DE INFO
        self.trajectory_manipulable = [[] for _ in range(self.n_manipulable_vars)]
        self.trajectory_target = [[] for _ in range(self.n_target_vars)]
        self.energy_accumulated = 0.0
        self.overshoot_manipulable = [0.0] * self.n_manipulable_vars
        self.overshoot_target = [0.0] * self.n_target_vars
        self.accumulated_error_manipulable = [0.0] * self.n_manipulable_vars
        self.accumulated_error_target = [0.0] * self.n_target_vars

        # OBSERVACION E INFO
        observation = self._get_observation()
        info = self._get_info() 

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, Union[float, List[float]], bool, bool, Dict[str, Any]]:

        # 1. Aplicar control
        control_outputs, pid_params_list = self._apply_control(action)
        
        # 2. Actualizar proceso
        self.pvs = self._update_process(control_outputs, pid_params_list)
        
        # 3. Calcular errores para cada variable
        errors = [self.setpoints[i] - self.pvs[i] for i in range(self.n_variables)]
        
        # 4. Actualizar tracking de errores
        for i in range(self.n_variables):
            self.error_histories[i].append(errors[i])
            self.error_integrals[i] += errors[i] * self.dt
            
            # Calcular derivada
            if len(self.error_histories[i]) >= 2:
                self.error_derivatives[i] = (
                    (self.error_histories[i][-1] - self.error_histories[i][-2]) / self.dt
                )
            else:
                self.error_derivatives[i] = 0.0
        
        # 5. Detectar tiempo de respuesta y clasificar dificultad para cada variable
        estimated_response_times = []
        process_difficulties = []
        
        for i in range(self.n_variables):
            control_out = control_outputs[i] if control_outputs[i] is not None else 0.0
            
            estimated_rt = self.response_detectors[i].update(
                control_out, self.pvs[i], self.setpoints[i], self.dt
            )
            estimated_response_times.append(estimated_rt)
            
            difficulty = self.difficulty_classifiers[i].classify(estimated_rt)
            process_difficulties.append(difficulty)
        
        # 6. Calcular recompensas individuales para cada variable
        individual_rewards = []
        
        for i in range(self.n_variables):
            control_out = control_outputs[i] if control_outputs[i] is not None else 0.0
            
            reward_i = self.reward_calculators[i].calculate(
                pv=self.pvs[i],
                setpoint=self.setpoints[i],
                error=errors[i],
                error_integral=self.error_integrals[i],
                error_derivative=self.error_derivatives[i],
                control_output=control_out,
                process_difficulty=process_difficulties[i]
            )
            individual_rewards.append(reward_i)
            
            # Actualizar métricas por variable
            self.metrics_trackers[i].update_step(reward_i)
        
        # 7. Calcular bonus de cooperación (si todas las variables están en banda muerta)
        all_in_deadband = all(
            abs(errors[i]) <= self.dead_bands[i] 
            for i in range(self.n_variables)
        )
        
        if all_in_deadband and self.n_variables > 1:
            cooperation_bonus = self.cooperation_bonus
        else:
            cooperation_bonus = 0.0
        
        # 8. Recompensas finales (individuales + bonus cooperativo)
        final_rewards = [r + cooperation_bonus for r in individual_rewards]
        
        # 9. Actualizar estado
        self.error_prevs = errors.copy()
        self.step_count += 1
        
        # 10. Condiciones de término
        terminated = self.step_count >= self.max_episode_steps
        
        # Truncar si alguna variable se sale de control
        truncated = any(
            self._check_truncation(i, errors[i], process_difficulties[i])
            for i in range(self.n_variables)
        )
        
        # 11. Preparar info
        info = {
            'process_difficulties': process_difficulties,
            'estimated_response_times': estimated_response_times,
            'step_count': self.step_count,
            'cooperation_bonus': cooperation_bonus,
            'all_in_deadband': all_in_deadband,
            'individual_rewards': individual_rewards,
            'current_pvs': self.pvs.copy(),  # ✅ AGREGAR ESTA LÍNEA
            'setpoints': self.setpoints.copy()  # ✅ Y ESTA
        }
        
        # Agregar info específico si aplica
        if any(p is not None for p in pid_params_list):
            info['pid_params_list'] = pid_params_list
        if any(c is not None for c in control_outputs):
            info['control_outputs'] = [float(c) if c is not None else None 
                                       for c in control_outputs]
        
        # Formato de recompensa según modo
        if self.n_variables == 1:
            reward_output = final_rewards[0]  # Escalar para single-agent
        else:
            reward_output = final_rewards  # Lista para multi-agent
        
        return self._get_observation(), reward_output, terminated, truncated, info    