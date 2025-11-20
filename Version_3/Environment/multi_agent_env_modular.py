
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Tuple
import torch

# Imports del proyecto
from Environment.simulation_env import SimulationPIDEnv
from Environment.base_env import BasePIDControlEnv
from Environment.pid_components import PIDController
from Entrenamiento.pid_trainer import PIDTrainer
from Entrenamiento.stability_criteria import StabilityCriteria
from Entrenamiento.controller_agent import ControllerAgent
from Entrenamiento.orchestrator_agent import OrchestratorAgent
from Agent.DQN.algorithm_DQN import DQNAgent # Para los controladores
from Agent.Actor_Critic.algorithm_ActorCritic import ActorCriticAgent # Para el orquestador
from Environment.reward_system import AdaptiveRewardCalculator

class MultiAgentPIDEnv:
    """
    Ambiente Multi-Agente para control PID jerárquico.
    
    Nomenclatura importante:
    - self.architecture: 'direct' o 'indirect'
        → Define la ARQUITECTURA del sistema (nivel usuario)
        → 'direct': Entrenar controladores directamente sin orquestador
        → 'indirect': Usar orquestador que ajusta setpoints para controladores
    
    - control_mode (para base_env): 'direct' o 'pid_tuning'
      → Define el TIPO DE ACCIÓN del agente (nivel implementación)
      → 'direct': Acción continua (control output directo)
      → 'pid_tuning': Acción discreta (ajuste de parámetros PID)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # No hereda de clase base

        # Logger para métricas
        self.deadband = config.get('deadband', 0.01)

        # ARQUITECTURA: ¿Hay orquestador? ('direct' o 'indirect')
        self.architecture = config.get('architecture', 'indirect')
        self.target_indices = config.get('target_indices', [0])
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.n_target_vars = config.get('n_target_vars', 1)
        self.target_ranges = config.get('target_ranges', [(0.0, 1.0)])
        self.target_setpoints = config.get('target_setpoints', [0.2])
        self.target_pvs = list(self.target_setpoints)

        self.sp_ranges = config.get('sp_ranges', [(290.0, 450.0), (99.0, 105.0)]) # Rangos de los setpoints para cada variable manipulable
        self.orchestrator_iterations = config.get('r_orchestrator_iterations', 10)
        self.j_max_retries = config.get('j_max_retries', 3)
        self.n_episodes = config.get('n_episodes', 100)
        self.max_episode_steps = config.get('max_episode_steps', 200)

        # Configuración para los controladores DQN
        self.agent_lr = config.get('agent_lr', 0.001)
        self.agent_gamma = config.get('agent_gamma', 0.99)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.controller_hidden_dims = config.get('hidden_dims', (64, 64))
        self.n_actions_per_var = config.get('n_actions_per_var', 7)
        self.initial_pid = config.get('initial_pid', (1.0, 0.1, 0.05))

        # Configuración para el orquestador Actor-Critic
        self.orch_lr_actor = config.get('orch_lr_actor', 0.0001)
        self.orch_lr_critic = config.get('orch_lr_critic', 0.001)
        self.orch_gamma = config.get('orch_gamma', 0.99)
        self.orch_hidden_dims = config.get('orch_hidden_dims', (64, 64))

        # Contador de iteraciones del entrenamiento
        self.current_iteration = 0

        self.device = config.get('device', 'cpu')
        self.config = config  # Guardar config para callbacks

        base_env_config = {
            'control_mode': config.get('control_mode', 'pid_tuning'),
            'n_variables': self.n_manipulable_vars,
            'n_actions': self.n_actions_per_var,
            'initial_pid': self.initial_pid,
            'dt': config.get('dt', 1.0),
            'max_episode_steps': self.max_episode_steps,
            'device': self.device,
            'setpoint': config.get('setpoint', [370.0, 102.0]),
            'upper_range': config.get('upper_range', [450.0, 105.0]),
            'lower_range': config.get('lower_range', [290.0, 99.0]),
            'dead_band': config.get('dead_band', [0.01, 0.01])
        }

        # Crear ambiente según tipo
        env_type = config.get('env_type', 'simulation')

        if env_type == 'simulation':
            self.base_env = SimulationPIDEnv(base_env_config)
        elif env_type == 'real':
            from Environment.real_env import RealPIDEnv
            self.base_env = RealPIDEnv(base_env_config)
        else:
            raise ValueError(f"env_type '{env_type}' no soportado. Use 'simulation' o 'real'")

        # Inicializar calculador de recompensas
        self.reward_calculator = AdaptiveRewardCalculator(
            upper_range=config.get('upper_range', [450.0, 105.0])[0],  # Usar primer valor como referencia
            lower_range=config.get('lower_range', [290.0, 99.0])[0],
            dead_band=config.get('dead_band', [0.01, 0.01])[0]
)

        # Inicializar el PIDTrainer
        self.stability_criteria = StabilityCriteria()
        self.pid_trainer = PIDTrainer(
            stability_criteria=self.stability_criteria
        )

        # Inicializar el Orquestador (Actor-Critic)
        state_dim_orch = self.n_target_vars * 2
        action_dim_orch = self.n_manipulable_vars
        
        
        # Opción 1: Recibir orquestador ya creado (MÁS GENÉRICO)
        if 'orchestrator_agent' in config:
            self.orchestrator = config['orchestrator_agent']

        # Opción 2: Crearlo automáticamente (BACKWARD COMPATIBLE)
        else:
            actor_critic = ActorCriticAgent(
                state_dim=state_dim_orch,
                action_dim=action_dim_orch,
                hidden_dims=tuple(config.get('orch_hidden_dims', (128, 128, 64))),
                lr_actor=config.get('orch_lr_actor', 0.0001),
                lr_critic=config.get('orch_lr_critic', 0.001),
                gamma=config.get('orch_gamma', 0.99),
                device=config.get('device', 'cpu')
            )
            
            self.orchestrator = OrchestratorAgent(
                actor_critic_agent=actor_critic,
                n_manipulable_vars=self.n_manipulable_vars,
                sp_ranges=self.sp_ranges
            )

        # Crear agentes controladores (uno por variable manipulable)
        self.controller_agents = []

        # Opción 1: Recibir agentes ya creados (MÁS GENÉRICO)
        if 'controller_agents' in config:
            self.controller_agents = config['controller_agents']
            
        # Opción 2: Crearlos automáticamente con DQN (BACKWARD COMPATIBLE)
        else:
            control_mode = config.get('control_mode', 'pid_tuning')
            
            for i in range(self.n_manipulable_vars):
                # Crear agente según control_mode
                if control_mode == 'pid_tuning':
                    # Agente discreto (DQN)
                    rl_agent = DQNAgent(
                        state_dim=6,
                        action_dim=config.get('n_actions_per_var', 7),
                        hidden_dims=tuple(config.get('hidden_dims', (128, 128, 64))),
                        lr=config.get('agent_lr', 0.001),
                        gamma=config.get('agent_gamma', 0.99),
                        epsilon_start=config.get('epsilon_start', 1.0),
                        epsilon_min=config.get('epsilon_min', 0.01),
                        epsilon_decay=config.get('epsilon_decay', 0.995),
                        batch_size=config.get('batch_size', 32),
                        memory_size=config.get('memory_size', 10000),
                        device=config.get('device', 'cpu')
                    )
                elif control_mode == 'direct':
                    # Agente continuo (Actor-Critic)
                    rl_agent = ActorCriticAgent(
                        state_dim=6,
                        action_dim=3,  # [Kp, Ki, Kd]
                        hidden_dims=tuple(config.get('hidden_dims', (128, 128, 64))),
                        lr_actor=config.get('agent_lr', 0.001),
                        lr_critic=config.get('agent_lr', 0.001) * 10,
                        gamma=config.get('agent_gamma', 0.99),
                        device=config.get('device', 'cpu')
                    )
                else:
                    raise ValueError(f"control_mode '{control_mode}' no soportado")
                
                # Crear wrapper ControllerAgent
                controller = ControllerAgent(
                    var_idx=i,
                    dqn_agent=rl_agent,  # Funciona con cualquier agente que implemente AbstractPIDAgent
                    initial_pid=config.get('initial_pid', (1.0, 0.1, 0.05))
                )
                self.controller_agents.append(controller)

        print("=" * 60)
        print(f"MultiAgentPIDEnv inicializado")
        print(f"Arquitectura: {self.architecture}")
        print(f"Variables manipulables: {self.n_manipulable_vars}")
        print(f"Base_env control_mode: {config.get('control_mode', 'pid_tuning')}")
        print("=" * 60)


    def _get_observation(self) -> np.ndarray:
        """Obtener observación para el orquestador."""
        state = self.base_env.get_process_state()
        pv_targets = [state[idx] for idx in self.target_indices]
        
        obs = np.array(pv_targets + self.target_setpoints, dtype=np.float32)
        
        # Proteger contra NaN/inf
        if not np.all(np.isfinite(obs)):
            print(f"Observación con NaN/inf: {obs}, usando valores default")
            obs = np.zeros_like(obs)
        
        return obs

    def _calculate_reward(self, observation: np.ndarray, action: Any) -> float:
        """Calcular recompensa global."""
        return self.orchestrator.calculate_reward(observation)

    def _check_termination(self, observation: np.ndarray) -> bool:
        """Verificar terminación."""
        return self.current_iteration >= self.orchestrator_iterations

    def _check_truncation(self, observation: np.ndarray) -> bool:
        """Verificar truncamiento."""
        return False

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> tuple:
        """Resetear ambiente."""
        if seed is not None:
            np.random.seed(seed)
        
        # Resetear contador
        self.current_iteration = 0
        
        # Limpiar buffers del orquestador (no "reset" que no existe)
        self.orchestrator.clear_buffers()
        
        # Resetear ambiente base
        self.base_env.reset()

        # Obtener observación inicial (usar método propio)
        observation = self._get_observation()
        
        # Inicializar current_state
        self.current_state = observation

        info = {}
        return observation, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecutar un paso del ambiente multi-agente.
        
        Nota: 'action' es None porque el orquestador decide internamente
        los setpoints basándose en el estado actual.
        """
        # El orquestador elige nuevos setpoints
        new_setpoints = self.orchestrator.decide_setpoints(
            self.target_pvs, 
            self.target_setpoints
        )

        # Entrenar controladores con nuevos setpoints
        best_pids, errors = self._train_controllers_with_new_setpoints(new_setpoints)

        # Ejecutar simulación con PIDs optimizados
        global_reward = self._execute_with_pids(best_pids, new_setpoints)

        # Orquestador aprende de la recompensa global
        if np.isfinite(global_reward):
            self.orchestrator.update_policy(global_reward)

            #  AGREGAR LOGGING
            state = self.base_env.get_process_state()
            target_pv = state[self.target_indices[0]]
            # Extraer losses si están disponibles
            if hasattr(self.orchestrator, 'actor_critic_agent'):
                # Los losses se pueden obtener del último update
                # Por ahora usar valores dummy
                actor_loss = None  # TODO: capturar del update_policy
                critic_loss = None
            
        else:
            print(f" Reward inválido ({global_reward}), saltando actualización del orquestador")

        # Actualizar observación para próximo step
        next_state = self._get_observation()

        self.current_iteration += 1

        terminated = self._check_termination(next_state)
        truncated = self._check_truncation(next_state)

        self.current_state = next_state

        info = {
            "best_pids": best_pids,
            "setpoints": new_setpoints,
            "controller_errors": errors,
            "global_reward": global_reward
        }

        # Loguear en W&B
        if self.config.get('wandb_log_callback') is not None:
            trainer_stats = self.pid_trainer.get_statistics()
            self.config['wandb_log_callback'](
                iteration=self.current_iteration,
                global_reward=global_reward,
                best_sp=new_setpoints,
                pids=best_pids,
                errors=errors,
                stats={
                    'pid_trainer': trainer_stats,
                    'current_pvs': info.get('current_pvs', new_setpoints),  
                    'target_pv': target_pv,  
                    'target_sp': self.target_setpoints[0]  
                }
            )

        return next_state, global_reward, terminated, truncated, info

    def _train_controllers_with_new_setpoints(self, new_setpoints):
        """Entrenar controladores con nuevos setpoints."""
        optimized_pids = []
        errors = []
        
        for i in range(len(new_setpoints)):
            print(f"🔧 Entrenando controlador {i} con SP={new_setpoints[i]}")  # 🔍
            
            pid, error = self.pid_trainer.find_best_pid(
                agent=self.controller_agents[i],
                env=self.base_env,
                var_idx=i,
                setpoint=new_setpoints[i],
                n_episodes=self.n_episodes,
                j_max_retries=self.j_max_retries,
                verbose=False
            )
            
            print(f"   Resultado: PID={pid}, error={error}")  
            
            optimized_pids.append(pid)
            errors.append(error)
        
        return optimized_pids, errors

    def _execute_with_pids(self, pids: List[Tuple[float, float, float]], 
                      setpoints: List[float]) -> float:
        """
        Ejecutar simulación con PIDs y setpoints dados.
        
        Retorna la recompensa global basada en el tracking de las variables objetivo.
        """
        # Configurar PIDs en el base_env
        for i in range(self.n_manipulable_vars):
            # Si PID es None, usar valores default
            if pids[i] is None:
                print(f"⚠️ Usando PID default para variable {i}")
                pids[i] = self.initial_pid  # (1.0, 0.1, 0.05)
            
            self.base_env.pid_controllers[i].update_gains(*pids[i])
            self.base_env.setpoints[i] = setpoints[i]

        # Resetear proceso
        self.base_env.reset_process()
        
        # Inicializar acumuladores
        total_reward = 0.0
        step_count = 0
        
        # Ejecutar episodio
        for step in range(self.max_episode_steps):
            # Acción de "mantener" PIDs (índice 3 = MAINTAIN)
            maintain_action = [3] * self.n_manipulable_vars
            
            # Step del ambiente (retorna lista de rewards)
            state, rewards_list, done, truncated, info = self.base_env.step(maintain_action)
            
            # Sumar reward (promedio de todas las variables)
            step_reward = np.mean(rewards_list) if isinstance(rewards_list, list) else rewards_list
            total_reward += step_reward
            step_count += 1
            
            # LOGGING: Capturar datos de este step
            current_pvs = info.get('current_pvs', setpoints)
            current_pv_targets = [state[idx] for idx in self.target_indices]
            
            if done or truncated:
                break
        
        # Retornar reward promedio
        return total_reward / step_count if step_count > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del entrenamiento."""
        stats = {
            'pid_trainer': self.pid_trainer.get_statistics()
        }
        return stats

    def train(self) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """Entrenar el sistema multi-agente."""
        if self.architecture == 'indirect':
            return self._train_indirect_mode()
        elif self.architecture == 'direct':
            return self._train_direct_mode()
        else:
            raise ValueError(f"Arquitectura '{self.architecture}' no soportada")

    def _train_direct_mode(self) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """
        Modo directo: Entrenar controladores sin orquestador.
        
        Cada controlador se entrena independientemente con su setpoint fijo.
        """
        best_pids = []
        errors = []
        
        # Setpoints iniciales del config
        initial_setpoints = self.config.get('setpoint', [370.0, 102.0])
        
        for i in range(self.n_manipulable_vars):
            pid, error = self.pid_trainer.find_best_pid(
                agent=self.controller_agents[i],
                env=self.base_env,
                var_idx=i,
                setpoint=initial_setpoints[i],
                n_episodes=self.n_episodes,
                j_max_retries=self.j_max_retries,
                verbose=True
            )
            best_pids.append(pid)
            errors.append(error)
        
        return best_pids, errors

    def _train_indirect_mode(self) -> Tuple[List[Tuple[float, float, float]], List[float]]:
        """
        Modo indirecto: Entrenar con orquestador.
        
        El orquestador ajusta setpoints dinámicamente para optimizar 
        la variable objetivo.
        """
        best_global_reward = -float('inf')
        best_pids_overall = []
        best_setpoints_overall = []

        for iteration in range(self.orchestrator_iterations):
            # El step() ejecuta: orquestador decide → entrena controladores → evalúa
            obs, global_reward, terminated, truncated, info = self.step(None)

            if global_reward > best_global_reward:
                best_global_reward = global_reward
                best_pids_overall = info["best_pids"]
                best_setpoints_overall = info["setpoints"]

            if terminated:
                break

        return best_pids_overall, best_setpoints_overall
