import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class SimplePlotter:
    
    def __init__(self, save_dir: Optional[str] = None):
        # Estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#F18F01',
            'success': '#06A77D',
            'danger': '#C73E1D',
            'gray': '#5A5A5A'
        }
    
    def plot_training_overview(
        self,
        episode_rewards: List[float],
        episode_energies: List[float],
        episode_max_overshoots: List[float],
        epsilons: List[float],
        window: int = 20
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        episodes = np.arange(len(episode_rewards))
        
        # 1. REWARDS 
        ax = axes[0, 0]
        ax.plot(episodes, episode_rewards, alpha=0.3, color=self.colors['gray'], label='Raw')
        if len(episode_rewards) >= window:
            ma = self._moving_average(episode_rewards, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['primary'], 
                   linewidth=2.5, label=f'MA({window})')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Total Reward', fontsize=11)
        ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 2. ENERGY (ESFUERZO DE CONTROL)
        ax = axes[0, 1]
        ax.plot(episodes, episode_energies, alpha=0.3, color=self.colors['gray'], label='Raw')
        if len(episode_energies) >= window:
            ma = self._moving_average(episode_energies, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['secondary'], 
                   linewidth=2.5, label=f'MA({window})')
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title('Control Effort (Energy)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. MAX OVERSHOOT
        ax = axes[1, 0]
        ax.plot(episodes, episode_max_overshoots, alpha=0.3, color=self.colors['gray'], label='Raw')
        if len(episode_max_overshoots) >= window:
            ma = self._moving_average(episode_max_overshoots, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['danger'], 
                   linewidth=2.5, label=f'MA({window})')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Max Overshoot (%)', fontsize=11)
        ax.set_title('Maximum Overshoot', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 4. EPSILON (EXPLORACIÓN)
        ax = axes[1, 1]
        ax.plot(episodes, epsilons, color=self.colors['success'], linewidth=2.5)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Epsilon (ε)', fontsize=11)
        ax.set_title('Exploration Rate', fontsize=12, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_action_distribution(
        self,
        action_counts: Dict[int, int],
        action_labels: Optional[List[str]] = None
    ):
        if action_labels is None:
            action_labels = [
                'Kp ↑', 'Ki ↑', 'Kd ↑',
                'Kp ↓', 'Ki ↓', 'Kd ↓',
                'Mantener'
            ]
        
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        labels = [action_labels[a] if a < len(action_labels) else f'Action {a}' 
                 for a in actions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(actions)), counts, color=self.colors['primary'], alpha=0.8)
        
        # Destacar acción más usada
        max_idx = counts.index(max(counts))
        bars[max_idx].set_color(self.colors['success'])
        
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(' Action Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Mostrar porcentajes
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        plt.show()
    
    def plot_best_episode(
        self,
        pv_trajectory: List[float],
        sp_trajectory: List[float],
        control_trajectory: Optional[List[float]] = None,
        title: str = "Best Episode"
    ):
        # Graficar el mejor episodio (PV vs SP).
        n_plots = 2 if control_trajectory else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        steps = np.arange(len(pv_trajectory))

        # PLOT 1: PV vs SP
        ax = axes[0]
        ax.plot(steps, pv_trajectory, label='Process Value (PV)', 
               color=self.colors['primary'], linewidth=2.5)
        ax.plot(steps, sp_trajectory, label='Setpoint (SP)', 
               color=self.colors['danger'], linestyle='--', linewidth=2)
        
        # Banda de ±2%
        sp_mean = np.mean(sp_trajectory)
        ax.fill_between(steps, 
                        sp_mean * 0.98, sp_mean * 1.02,
                        alpha=0.15, color=self.colors['success'], 
                        label='±2% tolerance')
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f' {title} - Tracking Performance', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # PLOT 2: CONTROL SIGNAL 
        if control_trajectory:
            ax = axes[1]
            ax.plot(steps, control_trajectory, color=self.colors['secondary'], linewidth=2)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Control Output', fontsize=11)
            ax.set_title('Control Signal', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Step', fontsize=11)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pid_evolution(
        self,
        kp_history: List[float],
        ki_history: List[float],
        kd_history: List[float],
        var_name: str = "Variable"
    ):
        # Evolución de parámetros PID durante entrenamiento.
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        episodes = np.arange(len(kp_history))
        
        # Kp
        axes[0].plot(episodes, kp_history, color='#1f77b4', linewidth=2)
        axes[0].set_ylabel('Kp', fontsize=11)
        axes[0].set_title(f' PID Parameters Evolution - {var_name}', 
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Ki
        axes[1].plot(episodes, ki_history, color='#2ca02c', linewidth=2)
        axes[1].set_ylabel('Ki', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Kd
        axes[2].plot(episodes, kd_history, color='#d62728', linewidth=2)
        axes[2].set_xlabel('Episode', fontsize=11)
        axes[2].set_ylabel('Kd', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        """Calcular promedio móvil."""
        return np.convolve(data, np.ones(window)/window, mode='valid')


# FUNCIÓN DE RESUMEN

def print_summary(
    episode_rewards: List[float],
    episode_energies: List[float],
    episode_max_overshoots: List[float],
    best_episode_idx: int
):
    n_episodes = len(episode_rewards)
    last_10 = slice(-10, None)
    print("RESUMEN DE ENTRENAMIENTO")
    print(f"\n{'Métrica':<30} {'Último':<12} {'Promedio':<12} {'Mejor':<12}")
    
    # Rewards
    print(f"{'Reward':<30} {episode_rewards[-1]:>11.2f} {np.mean(episode_rewards[last_10]):>11.2f} "
          f"{max(episode_rewards):>11.2f}")
    
    # Energy
    print(f"{'Energy':<30} {episode_energies[-1]:>11.2f} {np.mean(episode_energies[last_10]):>11.2f} "
          f"{min(episode_energies):>11.2f}")
    
    # Overshoot
    print(f"{'Max Overshoot (%)':<30} {episode_max_overshoots[-1]:>11.2f} "
          f"{np.mean(episode_max_overshoots[last_10]):>11.2f} "
          f"{min(episode_max_overshoots):>11.2f}")

    print(f"\n Mejor episodio: #{best_episode_idx} (Reward: {episode_rewards[best_episode_idx]:.2f})")
    print(f"Mejora total: {((episode_rewards[-1] - episode_rewards[0]) / abs(episode_rewards[0]) * 100):.1f}%")
    print(f"Total episodios: {n_episodes}")