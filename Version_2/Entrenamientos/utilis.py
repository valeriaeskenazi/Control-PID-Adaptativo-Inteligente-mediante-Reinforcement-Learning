
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

import sys
sys.path.append('..')  # Subir un nivel para acceder a las carpetas

from Environment.simulation_env import SimulationPIDEnv  
from Simuladores.tanque_simple import TankSimulator
from Agentes.DQN.algorithm_DQN import DQNAgent

def train_DQN(n_episodes: int = 10, 
                           env_config: Dict = None, 
                           agent: DQNAgent = None,
                           save_path: str = 'dqn_tank_training_results.png',
                           log_callback=None):  #Para W&B logs
    """Entrenar agente DQN y visualizar resultados."""
    print("\n" + "="*60)
    print(f"ENTRENAMIENTO: {n_episodes} episodios")
    print("="*60 + "\n")
    
    # Crear ambiente
    env = SimulationPIDEnv(config=env_config, control_mode='pid_tuning')
    tank = TankSimulator(area=1.0, cv=0.1, max_height=10.0, max_flow_in=0.5, dt=1.0)
    env.connect_external_process(tank)
    
    # Métricas adicionales para RL
    losses = []
    epsilons = []
    
    # Entrenamiento
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_losses = [] 
        
        # Para graficar último episodio
        if episode == n_episodes - 1:
            trajectory = {'pv': [], 'sp': [], 'error': [], 'actions': []}
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                metrics = agent.update()
                if metrics:
                    losses.append(metrics['q_loss'])  # ← Para gráfico general
                    episode_losses.append(metrics['q_loss'])  # ← Para este episodio        
            
            # Guardar trayectoria último episodio
            if episode == n_episodes - 1:
                trajectory['pv'].append(state[0])
                trajectory['sp'].append(state[1])
                trajectory['error'].append(state[2])
                trajectory['actions'].append(action)
            
            state = next_state
        
        final_error = abs(next_state[2])  # Error final (siempre disponible)
        success = final_error <= env_config.get('dead_band', 0.2)
        
        env.metrics_tracker.end_episode(
            settling_time=None,
            steady_state_error=final_error,
            success=success
        )
        
        epsilons.append(agent.get_epsilon())
        
        # Callback para logging externo (W&B, etc.)
        if log_callback is not None:
            tracker_metrics = env.get_metrics()
            avg_loss = np.mean(episode_losses) if episode_losses else None
            
            log_callback(
                episode=episode + 1,
                episode_reward=tracker_metrics['recent_rewards'][-1],
                avg_reward=tracker_metrics['avg_episode_reward'],
                success_rate=tracker_metrics['success_rate'],
                epsilon=agent.get_epsilon(),
                q_loss=avg_loss,
                steady_state_error=final_error
            )

        # Imprimir progreso
        if (episode + 1) % 10 == 0 or episode == 0:
            tracker_metrics = env.get_metrics()
            print(f"Ep {episode+1:3d}/{n_episodes} | "
                  f"Reward: {tracker_metrics['recent_rewards'][-1]:7.2f} | "
                  f"Avg: {tracker_metrics['avg_episode_reward']:7.2f} | "
                  f"ε: {agent.get_epsilon():.3f} | "
                  f"Success: {tracker_metrics['success_rate']:.2%}")
    
    # ========== OBTENER MÉTRICAS DEL TRACKER ==========
    tracker_summary = env.get_summary_stats()
    tracker_full = env.get_metrics()
    
    # ========== VISUALIZACIÓN ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Rewards por episodio (DEL TRACKER)
    episode_rewards = tracker_full['recent_rewards']
    axes[0, 0].plot(episode_rewards, 'b-', alpha=0.6, label='Reward')
    if len(episode_rewards) >= 10:
        smooth = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(episode_rewards)), smooth, 
                       'r-', linewidth=2, label='Media móvil (10)')
    axes[0, 0].axhline(y=tracker_summary['avg_episode_reward'], 
                      color='green', linestyle='--', alpha=0.5, 
                      label=f"Promedio: {tracker_summary['avg_episode_reward']:.2f}")
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Reward Total')
    axes[0, 0].set_title('Evolución de Recompensas')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Losses (RL específico)
    if losses:
        axes[0, 1].plot(losses, 'g-', alpha=0.6)
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Q-Loss')
        axes[0, 1].set_title('Pérdida durante Entrenamiento')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Epsilon decay
    axes[0, 2].plot(epsilons, 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Episodio')
    axes[0, 2].set_ylabel('Epsilon')
    axes[0, 2].set_title('Decaimiento de Exploración')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Trayectoria último episodio
    axes[1, 0].plot(trajectory['pv'], 'b-', label='PV (Nivel)', linewidth=2)
    axes[1, 0].plot(trajectory['sp'], 'r--', label='Setpoint', linewidth=2)
    dead_band = env_config.get('dead_band', 0.2)
    axes[1, 0].fill_between(range(len(trajectory['pv'])),
                            [s - dead_band for s in trajectory['sp']],
                            [s + dead_band for s in trajectory['sp']],
                            alpha=0.2, color='green', label='Dead Band')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Nivel [m]')
    axes[1, 0].set_title(f'Trayectoria - Episodio {n_episodes}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Error
    axes[1, 1].plot(trajectory['error'], 'r-', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 1].axhline(y=dead_band, color='g', linestyle=':', alpha=0.5)
    axes[1, 1].axhline(y=-dead_band, color='g', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Error [m]')
    axes[1, 1].set_title('Error de Control')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribución de acciones
    action_counts = np.bincount(trajectory['actions'], minlength=7)
    axes[1, 2].bar(range(7), action_counts, color='steelblue', edgecolor='black')
    axes[1, 2].set_xlabel('Acción')
    axes[1, 2].set_ylabel('Frecuencia')
    axes[1, 2].set_title(f'Acciones Tomadas (Ep {n_episodes})')
    axes[1, 2].set_xticks(range(7))
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráficos guardados en: {save_path}")
    plt.show()
    
    # ========== ESTADÍSTICAS FINALES ==========
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print("\n📊 MÉTRICAS DEL AMBIENTE (MetricsTracker):")
    print(f"  Total episodios:          {tracker_summary['total_episodes']}")
    print(f"  Episodios exitosos:       {tracker_summary['successful_episodes']}")
    print(f"  Tasa de éxito:            {tracker_summary['success_rate']:.2%}")
    print(f"  Reward promedio (global): {tracker_summary['avg_episode_reward']:.2f}")
    print(f"  Error estado estacionario: {tracker_summary['avg_steady_state_error']:.4f}")
    
    if 'last_10_avg_reward' in tracker_summary:
        print(f"\n  Últimos 10 episodios:")
        print(f"    Reward promedio:        {tracker_summary['last_10_avg_reward']:.2f}")
        print(f"    Desviación estándar:    {tracker_summary['last_10_std_reward']:.2f}")
    
    print(f"\n🤖 MÉTRICAS DEL AGENTE:")
    print(f"  Epsilon final:            {agent.get_epsilon():.4f}")
    print(f"  Experiencias en buffer:   {len(agent.memory)}")
    print(f"  Updates realizados:       {len(losses)}")
    print("="*60 + "\n")
    
    # ========== RETORNAR TODO ==========
    return {
        'tracker_metrics': tracker_full,
        'tracker_summary': tracker_summary,
        'losses': losses,
        'epsilons': epsilons,
        'trajectory': trajectory,
        'agent_epsilon': agent.get_epsilon(),
        'buffer_size': len(agent.memory)
    }