"""
Script de prueba: DQN Agent + Ambiente Simulado + Tanque

Prueba la integración completa del sistema:
- SimulationPIDEnv en modo 'pid_tuning'
- DQNAgent con acciones discretas
- TankSimulator como proceso
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Imports del proyecto

import sys
sys.path.append('..')  # Subir un nivel para acceder a las carpetas

from Environment.simulation_env import SimulationPIDEnv  
from Simuladores.tanque_simple import TankSimulator
from Agentes.DQN.algorithm_DQN import DQNAgent


def test_integration():
    """Probar integración básica sin entrenamiento."""
    print("\n" + "="*60)
    print("PRUEBA 1: Integración básica")
    print("="*60 + "\n")
    
    # 1. Crear ambiente
    env_config = {
        'upper_range': 10.0,      # Altura máxima tanque
        'lower_range': 0.0,       # Altura mínima
        'setpoint': 5.0,          # Nivel deseado: 5m
        'dead_band': 0.2,         # Banda muerta: ±0.2m
        'max_episode_steps': 200,
        'dt': 1.0,                # 1 segundo por step
        'enable_logging': False
    }
    
    env = SimulationPIDEnv(config=env_config, control_mode='pid_tuning')
    
    # 2. Crear simulador de tanque
    tank = TankSimulator(
        area=1.0,           # 1 m²
        cv=0.1,             # Coeficiente descarga
        max_height=10.0,    # 10 m
        max_flow_in=0.5,    # 0.5 m³/s
        dt=1.0              # 1 segundo
    )
    
    # 3. Conectar tanque al ambiente
    env.connect_external_process(tank)
    
    # 4. Crear agente DQN
    agent = DQNAgent(
        state_dim=6,
        action_dim=7,
        hidden_dims=(64, 64),
        lr=0.001,
        epsilon_start=1.0,
        memory_size=1000,
        batch_size=32,
        device='cpu'
    )
    
    # 5. Ejecutar un episodio de prueba
    state, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print("\nEjecutando episodio de prueba...")
    print(f"Estado inicial - PV: {state[0]:.2f}, SP: {state[1]:.2f}, Error: {state[2]:.2f}")
    
    while not done and step < 50:
        # Seleccionar acción
        action = agent.select_action(state, training=True)
        
        # Ejecutar acción en ambiente
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Almacenar experiencia
        agent.store_experience(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        step += 1
        
        # Imprimir cada 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d} | PV: {state[0]:5.2f} | Error: {state[2]:6.2f} | Reward: {reward:6.2f}")
    
    print(f"\n✅ Episodio completado:")
    print(f"   Steps: {step}")
    print(f"   Reward total: {total_reward:.2f}")
    print(f"   Experiencias en buffer: {len(agent.memory)}")


def test_training_loop(n_episodes: int = 5):
    """Probar loop de entrenamiento básico."""
    print("\n" + "="*60)
    print(f"PRUEBA 2: Loop de entrenamiento ({n_episodes} episodios)")
    print("="*60 + "\n")
    
    # Configurar ambiente
    env_config = {
        'upper_range': 10.0,
        'lower_range': 0.0,
        'setpoint': 5.0,
        'dead_band': 0.2,
        'max_episode_steps': 200,
        'dt': 1.0
    }
    
    env = SimulationPIDEnv(config=env_config, control_mode='pid_tuning')
    
    tank = TankSimulator(area=1.0, cv=0.1, max_height=10.0, max_flow_in=0.5, dt=1.0)
    env.connect_external_process(tank)
    
    # Crear agente
    agent = DQNAgent(
        state_dim=6,
        action_dim=7,
        hidden_dims=(64, 64),
        lr=0.001,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        memory_size=5000,
        batch_size=32,
        target_update_freq=50,
        device='cpu'
    )
    
    # Métricas
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Loop de entrenamiento
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Seleccionar acción
            action = agent.select_action(state, training=True)
            
            # Step en ambiente
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Almacenar experiencia
            agent.store_experience(state, action, reward, next_state, done)
            
            # Entrenar agente
            if len(agent.memory) >= agent.batch_size:
                metrics = agent.update()
                if metrics:
                    losses.append(metrics['q_loss'])
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Imprimir progreso
        print(f"Episodio {episode+1:2d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {episode_length:3d} | "
              f"Epsilon: {agent.get_epsilon():.3f} | "
              f"Buffer: {len(agent.memory):4d}")
    
    # Resumen
    print(f"\n✅ Entrenamiento completado:")
    print(f"   Reward promedio: {np.mean(episode_rewards):.2f}")
    print(f"   Steps promedio: {np.mean(episode_lengths):.1f}")
    print(f"   Epsilon final: {agent.get_epsilon():.3f}")
    print(f"   Losses registrados: {len(losses)}")
    
    return episode_rewards, episode_lengths, losses


def test_with_visualization(n_episodes: int = 10):
    """Probar y visualizar resultados."""
    print("\n" + "="*60)
    print(f"PRUEBA 3: Entrenamiento con visualización ({n_episodes} episodios)")
    print("="*60 + "\n")
    
    # Configurar ambiente
    env_config = {
        'upper_range': 10.0,
        'lower_range': 0.0,
        'setpoint': 5.0,
        'dead_band': 0.2,
        'max_episode_steps': 200,
        'dt': 1.0
    }
    
    env = SimulationPIDEnv(config=env_config, control_mode='pid_tuning')
    tank = TankSimulator(area=1.0, cv=0.1, max_height=10.0, max_flow_in=0.5, dt=1.0)
    env.connect_external_process(tank)
    
    agent = DQNAgent(
        state_dim=6,
        action_dim=7,
        hidden_dims=(128, 64),
        lr=0.001,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        memory_size=10000,
        batch_size=64,
        target_update_freq=100,
        device='cpu'
    )
    
    # Métricas para graficar
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Entrenamiento
    for episode in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # Para graficar último episodio
        if episode == n_episodes - 1:
            trajectory = {'pv': [], 'sp': [], 'error': []}
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                metrics = agent.update()
                if metrics:
                    losses.append(metrics['q_loss'])
            
            episode_reward += reward
            episode_length += 1
            
            # Guardar trayectoria último episodio
            if episode == n_episodes - 1:
                trajectory['pv'].append(state[0])
                trajectory['sp'].append(state[1])
                trajectory['error'].append(state[2])
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Ep {episode+1:3d} | Reward: {episode_reward:7.2f} | "
              f"Steps: {episode_length:3d} | ε: {agent.get_epsilon():.3f}")
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Rewards por episodio
    axes[0, 0].plot(episode_rewards, 'b-', alpha=0.6)
    axes[0, 0].plot(np.convolve(episode_rewards, np.ones(3)/3, mode='valid'), 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Episodio')
    axes[0, 0].set_ylabel('Reward Total')
    axes[0, 0].set_title('Reward por Episodio')
    axes[0, 0].grid(True)
    
    # 2. Losses
    if losses:
        axes[0, 1].plot(losses, 'g-', alpha=0.6)
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Q-Loss')
        axes[0, 1].set_title('Pérdida durante Entrenamiento')
        axes[0, 1].grid(True)
    
    # 3. Trayectoria último episodio
    axes[1, 0].plot(trajectory['pv'], 'b-', label='PV (Nivel)')
    axes[1, 0].plot(trajectory['sp'], 'r--', label='Setpoint')
    axes[1, 0].fill_between(range(len(trajectory['pv'])),
                            [s - 0.2 for s in trajectory['sp']],
                            [s + 0.2 for s in trajectory['sp']],
                            alpha=0.2, color='green', label='Dead Band')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Nivel [m]')
    axes[1, 0].set_title(f'Trayectoria - Episodio {n_episodes}')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Error
    axes[1, 1].plot(trajectory['error'], 'r-')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Error [m]')
    axes[1, 1].set_title('Error de Control')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_tank_training_results.png', dpi=150)
    print(f"\n📊 Gráficos guardados en: dqn_tank_training_results.png")
    plt.show()
    
    print(f"\n✅ Entrenamiento con visualización completado")


if __name__ == "__main__":
    print("\n🧪 PRUEBAS DE INTEGRACIÓN DQN + TANQUE\n")
    
    # Prueba 1: Integración básica
    test_integration()
    
    # Prueba 2: Loop de entrenamiento
    episode_rewards, episode_lengths, losses = test_training_loop(n_episodes=5)
    
    # Prueba 3: Con visualización
    test_with_visualization(n_episodes=10)
    
    print("\n" + "="*60)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("="*60 + "\n")