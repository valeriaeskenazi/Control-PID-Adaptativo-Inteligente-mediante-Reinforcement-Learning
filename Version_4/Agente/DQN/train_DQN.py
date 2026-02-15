import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path
#import wandb
from datetime import datetime


from ...Environment import PIDControlEnv_simple , PIDControlEnv_complex
from .algorithm_DQN import DQNAgent
from ..memory import Experience, SimpleReplayBuffer, PriorityReplayBuffer


class DQNTrainer:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architecture = config['env_config']['architecture']  # 'simple' o 'jerarquica'
        
        # AMBIENTE
        if self.architecture == 'simple':
            self.env = PIDControlEnv_simple(config['env_config'])
        
        elif self.architecture == 'jerarquica':
            self.env = PIDControlEnv_complex(config['env_config'])
        
        
        # AGENTES
        if self.architecture == 'simple':
            # Crear agente CTRL desde cero
            self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')
            self.agent_role = 'ctrl'
            self.agent_orch = None
        
        elif self.architecture == 'jerarquica':
            # CTRL: Cargar modelo pre-entrenado
            ctrl_checkpoint = config.get('ctrl_checkpoint_path', None)
            
            if ctrl_checkpoint:
                print(f"Cargando agente CTRL pre-entrenado desde: {ctrl_checkpoint}")
                self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')
                self.agent_ctrl.load(ctrl_checkpoint)
                
                # FREEZEAR agente CTRL (no entrenar)
                self.agent_ctrl.training = False  # Flag para no actualizar
            else:
                raise ValueError(
                    "Arquitectura jerárquica requiere 'ctrl_checkpoint_path' "
                    "con el modelo CTRL pre-entrenado"
                )
            # ORCH: Crear desde cero (este SÍ se entrena)
            self.agent_orch = self._create_agent(config['agent_orch_config'], 'orch')
            self.agent_role = 'orch'
        
        # ENTRENAMIENTO
        self.n_episodes = config.get('n_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 200)
        self.eval_freq = config.get('eval_frequency', 50)
        self.save_freq = config.get('save_frequency', 100)
        self.log_freq = config.get('log_frequency', 10)
        
        # Directorios
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # W&B logging
        #self.use_wandb = config.get('use_wandb', False)
        #if self.use_wandb:
        #    wandb.init(
        #        project=config.get('wandb_project', 'pid-dqn'),
        #        name=config.get('run_name', f"dqn_{self.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        #        config=config
        #    )

        # ESTADÍSTICAS
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
    
    def _create_agent(self, agent_config: Dict[str, Any], agent_type: str) -> DQNAgent:
        
        # Crear replay buffer según configuración
        buffer_type = agent_config.get('buffer_type', 'simple')
        buffer_size = agent_config.get('buffer_size', 10000)
        device = agent_config.get('device', 'cpu')
        
        if buffer_type == 'simple':
            replay_buffer = SimpleReplayBuffer(capacity=buffer_size, device=device)
        elif buffer_type == 'priority':
            replay_buffer = PriorityReplayBuffer(
                capacity=buffer_size,
                alpha=agent_config.get('priority_alpha', 0.6),
                beta=agent_config.get('priority_beta', 0.4),
                total_training_steps=self.n_episodes * self.max_steps_per_episode,
                device=device
            )
        
        # Crear agente
        agent = DQNAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            agent_role= self.agent_role,
            hidden_dims=agent_config.get('hidden_dims', (128, 128, 64)),
            lr=agent_config.get('lr', 0.001),
            gamma=agent_config.get('gamma', 0.99),
            epsilon_start=agent_config.get('epsilon_start', 1.0),
            epsilon_min=agent_config.get('epsilon_min', 0.01),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            batch_size=agent_config.get('batch_size', 32),
            target_update_freq=agent_config.get('target_update_freq', 100),
            device=device,
            seed=agent_config.get('seed', None),
            replay_buffer=replay_buffer
        )
        
        return agent
    
    def train(self):

        for episode in range(self.n_episodes):
            episode_reward, episode_length, episode_metrics = self._run_episode(episode, training=True)
            
            # Guardar estadísticas
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Logging
            if episode % self.log_freq == 0:
                self._log_episode(episode, episode_reward, episode_length, episode_metrics)
            
            # Evaluación
            if episode % self.eval_freq == 0 and episode > 0:
                eval_reward = self._evaluate()
                
                #if self.use_wandb:
                #    wandb.log({'eval_reward': eval_reward}, step=episode)

                # Guardar mejor modelo
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_checkpoint(episode, best=True)
            
            # Checkpoint periódico
            if episode % self.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode, best=False)

        #if self.use_wandb:
        #    wandb.finish()

        

    def _run_episode(self, episode: int, training: bool = True) -> tuple:
        # Reset ambiente
        if self.architecture == 'simple':
            state = self.env.reset()[0]
        else:  # jerarquica
            obs = self.env.reset()[0]
            state_ctrl = obs['ctrl']
            state_orch = obs['orch']
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Métricas acumuladas
        ctrl_losses = []
        orch_losses = []
        
        while not done and episode_length < self.max_steps_per_episode:
            
            # ARQUITECTURA SIMPLE
            if self.architecture == 'simple':
                # Seleccionar acción CTRL
                action = self.agent_ctrl.select_action(state, training=training)
                
                # Ejecutar en ambiente
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Almacenar experiencia
                if training:
                    experience = Experience(state, action, reward, next_state, done)
                    self.agent_ctrl.memory.add(experience)
                    
                    # Actualizar agente
                    metrics = self.agent_ctrl.update()
                    if metrics:
                        ctrl_losses.append(metrics.get('q_loss', 0))
                
                state = next_state
            
            # ARQUITECTURA JERÁRQUICA
            else:
                # 1. ORCH decide setpoints
                action_orch = self.agent_orch.select_action(state_orch, training=training)
                
                # 2. CTRL ajusta parámetros PID para alcanzar setpoints
                action_ctrl = self.agent_ctrl.select_action(state_ctrl, training=training)
                
                # 3. Combinar acciones 
                action = {
                    'ctrl': action_ctrl,
                    'orch': action_orch
                }
                
                # Ejecutar en ambiente
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                next_state_ctrl = next_obs['ctrl']
                next_state_orch = next_obs['orch']
                
                # Almacenar experiencias
                if training:
                    # Experiencia ORCH
                    exp_orch = Experience(state_orch, action_orch, reward, next_state_orch, done)
                    self.agent_orch.memory.add(exp_orch)
                    
                    # Actualizar agentes
                    metrics_ctrl = self.agent_ctrl.update()
                    metrics_orch = self.agent_orch.update()
                    
                    if metrics_ctrl:
                        ctrl_losses.append(metrics_ctrl.get('q_loss', 0))
                    if metrics_orch:
                        orch_losses.append(metrics_orch.get('q_loss', 0))
                
                state_ctrl = next_state_ctrl
                state_orch = next_state_orch
            
            episode_reward += reward
            episode_length += 1
        
        # Compilar métricas del episodio
        episode_metrics = {
            'ctrl_loss': np.mean(ctrl_losses) if ctrl_losses else 0,
            'ctrl_epsilon': self.agent_ctrl.get_epsilon(),
        }
        
        if self.architecture == 'jerarquica':
            episode_metrics.update({
                'orch_loss': np.mean(orch_losses) if orch_losses else 0,
                'orch_epsilon': self.agent_orch.get_epsilon(),
            })
        
        return episode_reward, episode_length, episode_metrics
    
    def _evaluate(self, n_eval_episodes: int = 5) -> float:
        """Evaluar agente sin exploración."""
        eval_rewards = []
        
        for _ in range(n_eval_episodes):
            episode_reward, _, _ = self._run_episode(episode=-1, training=False)
            eval_rewards.append(episode_reward)
        
        mean_reward = np.mean(eval_rewards)
        print(f"Evaluación: Reward promedio = {mean_reward:.2f}")
        
        return mean_reward
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        """Logging de episodio."""
        print(f"\nEpisodio {episode}/{self.n_episodes}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Length: {length}")
        print(f"  CTRL Loss: {metrics['ctrl_loss']:.4f}")
        print(f"  CTRL Epsilon: {metrics['ctrl_epsilon']:.4f}")
        
        if self.architecture == 'jerarquica':
            print(f"  ORCH Loss: {metrics['orch_loss']:.4f}")
            print(f"  ORCH Epsilon: {metrics['orch_epsilon']:.4f}")
        
        #if self.use_wandb:
        #    log_dict = {
        #        'episode': episode,
        #        'reward': reward,
        #        'episode_length': length,
        #        **metrics
        #    }
            #wandb.log(log_dict, step=episode)

    def _save_checkpoint(self, episode: int, best: bool = False):
        """Guardar checkpoint."""
        suffix = 'best' if best else f'ep{episode}'
        
        # Guardar CTRL
        ctrl_path = self.checkpoint_dir / f'agent_ctrl_{suffix}.pt'
        self.agent_ctrl.save(str(ctrl_path))
        
        # Guardar ORCH si existe
        if self.agent_orch is not None:
            orch_path = self.checkpoint_dir / f'agent_orch_{suffix}.pt'
            self.agent_orch.save(str(orch_path))
        
        print(f"Checkpoint guardado: {suffix}")

