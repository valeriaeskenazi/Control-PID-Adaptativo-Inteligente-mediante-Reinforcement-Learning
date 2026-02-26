import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path

from Environment import PIDControlEnv_simple, PIDControlEnv_complex
from .algorithm_PPO import PPOAgent
from ..memory import SimpleReplayBuffer, PriorityReplayBuffer
from ..DQN.algorithm_DQN import DQNAgent
from ..DDPG.algorithm_DDPG import DDPGAgent
from ..Actor_Critic.algorithm_AC import ACAgent
from .algorithm_PPO import PPOAgent


class PPOTrainer:
    """
    Diferencia clave en el loop de entrenamiento respecto a AC/DDPG:
    PPO acumula experiencia en el rollout buffer durante rollout_steps pasos
    (potencialmente cruzando múltiples episodios) y solo actualiza cuando
    el buffer está lleno. Esto es on-policy: las experiencias se descartan
    tras cada update.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architecture = config['env_config']['architecture']

        # AMBIENTE
        if self.architecture == 'simple':
            self.env = PIDControlEnv_simple.PIDControlEnv_Simple(config['env_config'])
        elif self.architecture == 'jerarquica':
            self.env = PIDControlEnv_complex.PIDControlEnv_Complex(config['env_config'])

        # AGENTES
        if self.architecture == 'simple':
            self.agent_role = 'ctrl'
            self.agent_orch = None
            self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')

        elif self.architecture == 'jerarquica':
            ctrl_checkpoint = config.get('ctrl_checkpoint_path', None)
            if not ctrl_checkpoint:
                raise ValueError("Arquitectura jerárquica requiere ctrl_checkpoint_path")

            print(f"Cargando agente CTRL pre-entrenado desde: {ctrl_checkpoint}")
            self.agent_role = 'ctrl'

            ctrl_algo = config['agent_ctrl_config'].get('algorithm', 'dqn')
            if ctrl_algo == 'dqn':
                device = config['agent_ctrl_config'].get('device', 'cpu')
                buffer_type = config['agent_ctrl_config'].get('buffer_type', 'simple')
                buffer_size = config['agent_ctrl_config'].get('buffer_size', 10000)
                rb = PriorityReplayBuffer(capacity=buffer_size, device=device) \
                    if buffer_type == 'priority' else SimpleReplayBuffer(capacity=buffer_size, device=device)
                self.agent_ctrl = DQNAgent(
                    state_dim=config['agent_ctrl_config']['state_dim'],
                    action_dim=config['agent_ctrl_config']['action_dim'],
                    agent_role='ctrl',
                    n_vars=config['agent_ctrl_config']['n_vars'],
                    hidden_dims=config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                    device=device,
                    replay_buffer=rb
                )
            elif ctrl_algo == 'ddpg':
                self.agent_ctrl = DDPGAgent(
                    state_dim=config['agent_ctrl_config']['state_dim'],
                    action_dim=config['agent_ctrl_config']['action_dim'],
                    agent_role='ctrl',
                    n_vars=config['agent_ctrl_config']['n_vars'],
                    hidden_dims=config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                    device=config['agent_ctrl_config'].get('device', 'cpu')
                )
            elif ctrl_algo == 'ac':
                self.agent_ctrl = ACAgent(
                    state_dim=config['agent_ctrl_config']['state_dim'],
                    action_dim=config['agent_ctrl_config']['action_dim'],
                    agent_role='ctrl',
                    n_vars=config['agent_ctrl_config']['n_vars'],
                    hidden_dims=config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                    device=config['agent_ctrl_config'].get('device', 'cpu')
                )
            elif ctrl_algo == 'ppo':
                self.agent_ctrl = PPOAgent(
                    state_dim=config['agent_ctrl_config']['state_dim'],
                    action_dim=config['agent_ctrl_config']['action_dim'],
                    agent_role='ctrl',
                    n_vars=config['agent_ctrl_config']['n_vars'],
                    hidden_dims=config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                    device=config['agent_ctrl_config'].get('device', 'cpu')
    )    

            self.agent_ctrl.load(ctrl_checkpoint)
            self.env.agente_ctrl = self.agent_ctrl
            self.env.action_type_ctrl = config['agent_ctrl_config'].get('action_type', 'discrete')

            # ORCH PPO
            self.agent_role = 'orch'
            self.agent_orch = self._create_agent(config['agent_orch_config'], 'orch')

        # ENTRENAMIENTO
        self.n_episodes = config.get('n_episodes', 1000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 200)
        self.eval_freq = config.get('eval_frequency', 50)
        self.save_freq = config.get('save_frequency', 100)
        self.log_freq = config.get('log_frequency', 10)

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # ESTADÍSTICAS
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energies = []
        self.episode_max_overshoots = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.clip_fractions = []  # métrica diagnóstica exclusiva de PPO
        self.kp_history = []
        self.ki_history = []
        self.kd_history = []

        self.best_reward = -float('inf')

    def _create_agent(self, agent_config: Dict[str, Any], agent_role: str) -> PPOAgent:
        agent = PPOAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            agent_role=agent_role,
            n_vars=agent_config['n_vars'],
            hidden_dims=agent_config.get('hidden_dims', (128, 128, 64)),
            lr_actor=agent_config.get('lr_actor', 0.0003),
            lr_critic=agent_config.get('lr_critic', 0.001),
            gamma=agent_config.get('gamma', 0.99),
            gae_lambda=agent_config.get('gae_lambda', 0.95),
            clip_epsilon=agent_config.get('clip_epsilon', 0.2),
            ppo_epochs=agent_config.get('ppo_epochs', 10),
            rollout_steps=agent_config.get('rollout_steps', 2048),
            mini_batch_size=agent_config.get('mini_batch_size', 64),
            entropy_coef=agent_config.get('entropy_coef', 0.01),
            value_coef=agent_config.get('value_coef', 0.5),
            max_grad_norm=agent_config.get('max_grad_norm', 0.5),
            device=agent_config.get('device', 'cpu'),
            seed=agent_config.get('seed', None)
        )
        return agent

    def train(self):
        for episode in range(self.n_episodes):
            episode_reward, episode_length, episode_metrics = self._run_episode(episode, training=True)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_energies.append(episode_metrics.get('energy', 0))
            self.episode_max_overshoots.append(episode_metrics.get('max_overshoot', 0))
            self.actor_losses.append(episode_metrics.get('actor_loss', 0))
            self.critic_losses.append(episode_metrics.get('critic_loss', 0))
            self.entropies.append(episode_metrics.get('entropy', 0))
            self.clip_fractions.append(episode_metrics.get('clip_fraction', 0))

            if self.architecture == 'simple':
                self.kp_history.append(episode_metrics.get('kp', 1.0))
                self.ki_history.append(episode_metrics.get('ki', 0.1))
                self.kd_history.append(episode_metrics.get('kd', 0.01))

            if episode % self.log_freq == 0:
                self._log_episode(episode, episode_reward, episode_length, episode_metrics)

            if episode % self.eval_freq == 0 and episode > 0:
                eval_reward = self._evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_checkpoint(episode, best=True)

            if episode % self.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode, best=False)

    def _run_episode(self, episode: int, training: bool = True) -> tuple:
        if self.architecture == 'simple':
            state = self.env.reset()[0]
        else:
            obs = self.env.reset()[0]
            state_ctrl = obs['ctrl']
            state_orch = obs['orch']

        episode_reward = 0
        episode_length = 0
        done = False
        episode_metrics_list = []
        episode_energy = 0

        while not done and episode_length < self.max_steps_per_episode:

            if self.architecture == 'simple':
                action = self.agent_ctrl.select_action(state, training=training)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if training:
                    # PPO: guardar transición en buffer (incluye log_prob y value del select_action)
                    self.agent_ctrl.store_transition(state, action, reward, next_state, done)
                    # Intentar update (solo actualiza cuando buffer llega a rollout_steps)
                    metrics = self.agent_ctrl.update()
                    if metrics:
                        episode_metrics_list.append(metrics)

                episode_energy += info.get('energy', 0)
                state = next_state

            else:
                action_orch = self.agent_orch.select_action(state_orch, training=training)
                action_ctrl = self.agent_ctrl.select_action(state_ctrl, training=False)

                action = {'ctrl': action_ctrl, 'orch': action_orch}
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state_ctrl = next_obs['ctrl']
                next_state_orch = next_obs['orch']

                if training:
                    self.agent_orch.store_transition(state_orch, action_orch, reward, next_state_orch, done)
                    metrics = self.agent_orch.update()
                    if metrics:
                        episode_metrics_list.append(metrics)

                state_ctrl = next_state_ctrl
                state_orch = next_state_orch

            episode_reward += reward
            episode_length += 1

        # Agregar métricas PID finales del episodio
        episode_metrics = {
            'actor_loss':    np.mean([m.get('actor_loss', 0) for m in episode_metrics_list]) if episode_metrics_list else 0,
            'critic_loss':   np.mean([m.get('critic_loss', 0) for m in episode_metrics_list]) if episode_metrics_list else 0,
            'entropy':       np.mean([m.get('entropy', 0) for m in episode_metrics_list]) if episode_metrics_list else 0,
            'clip_fraction': np.mean([m.get('clip_fraction', 0) for m in episode_metrics_list]) if episode_metrics_list else 0,
            'energy':        episode_energy,
            'max_overshoot': 0,
        }

        if self.architecture == 'simple':
            pid_params_0 = self.env.pid_controllers[0].get_params()
            pid_params_1 = self.env.pid_controllers[1].get_params()
            episode_metrics['kp'] = (pid_params_0[0] + pid_params_1[0]) / 2
            episode_metrics['ki'] = (pid_params_0[1] + pid_params_1[1]) / 2
            episode_metrics['kd'] = (pid_params_0[2] + pid_params_1[2]) / 2

        return episode_reward, episode_length, episode_metrics

    def _evaluate(self, n_eval_episodes: int = 5) -> float:
        eval_rewards = []
        for _ in range(n_eval_episodes):
            episode_reward, _, _ = self._run_episode(episode=-1, training=False)
            eval_rewards.append(episode_reward)
        mean_reward = np.mean(eval_rewards)
        print(f"Evaluación: Reward promedio = {mean_reward:.2f}")
        return mean_reward

    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        agent = self.agent_orch if self.architecture == 'jerarquica' else self.agent_ctrl
        buffer_size = len(agent.buffer)
        rollout_steps = agent.rollout_steps

        print(f"\nEpisodio {episode}/{self.n_episodes}")
        print(f"  Reward: {reward:.2f} | Length: {length}")
        print(f"  Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f} | Clip Fraction: {metrics['clip_fraction']:.4f}")
        print(f"  Buffer: {buffer_size}/{rollout_steps} steps")

    def _save_checkpoint(self, episode: int, best: bool = False):
        suffix = 'best' if best else f'ep{episode}'
        if self.agent_ctrl is not None:
            self.agent_ctrl.save(str(self.checkpoint_dir / f'agent_ctrl_{suffix}.pt'))
        if self.agent_orch is not None:
            self.agent_orch.save(str(self.checkpoint_dir / f'agent_orch_{suffix}.pt'))
        print(f"Checkpoint guardado: {suffix}")
