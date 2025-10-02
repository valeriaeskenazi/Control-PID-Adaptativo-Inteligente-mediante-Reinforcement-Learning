"""
Entrenamiento del agente DQN para control de nivel de tanque
Conecta el simulador de tanque con el agente DQN
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import time

# Agregar paths para importar módulos del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from simulations.tanque_nivel.tanque_simulator import TankLevelSimulator
from agent_valeria.DQN.dqn_agent import DQN_Agent


class TankDQNTrainer:
    """
    Entrenador para agente DQN en control de nivel de tanque
    
    Maneja:
    - Episodios de entrenamiento
    - Métricas y logging
    - Guardado de modelos
    - Visualización de resultados
    """
    
    def __init__(self,
                 # Parámetros del simulador
                 tank_area: float = 2.0,
                 max_height: float = 5.0,
                 max_inflow: float = 10.0,
                 
                 # Parámetros del agente DQN
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 
                 # Parámetros de entrenamiento
                 max_episode_steps: int = 200,
                 device: str = 'cpu'):
        
        # Crear simulador
        self.simulator = TankLevelSimulator(
            tank_area=tank_area,
            max_height=max_height,
            max_inflow=max_inflow,
            dt=1.0,  # 1 segundo por step
            noise_level=0.02  # 2% de ruido
        )
        
        # Crear agente DQN
        self.agent = DQN_Agent(
            state_dim=6,  # [level, setpoint, error, prev_error, integral, derivative]
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device
        )
        
        self.max_episode_steps = max_episode_steps
        
        # Métricas de entrenamiento
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'average_errors': [],
            'epsilon_values': [],
            'q_losses': []
        }
        
        print(f"🚀 Entrenador DQN-Tanque creado:")
        print(f"   Máximo steps por episodio: {max_episode_steps}")
        print(f"   Dispositivo: {device}")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Entrenar un episodio completo
        
        Returns:
            metrics: Métricas del episodio
        """
        # Reiniciar simulador con condiciones aleatorias
        setpoint = np.random.uniform(1.5, 4.0)  # Setpoint aleatorio
        observation, info = self.simulator.reset(options={'setpoint': setpoint})
        
        episode_reward = 0.0
        episode_errors = []
        step_count = 0
        
        for step in range(self.max_episode_steps):
            # Seleccionar acción usando DQN
            pid_params = self.agent.select_action(observation, training=True)
            
            # Convertir parámetros PID a señal de control
            # Control PID: u = Kp*e + Ki*∫e + Kd*de/dt
            kp, ki, kd = pid_params
            error = observation[2]  # Error actual
            integral = observation[4]  # Integral del error
            derivative = observation[5]  # Derivada del error
            
            # Calcular señal de control PID
            control_signal = kp * error + ki * integral + kd * derivative
            
            # Aplicar límites y offset (caudal base)
            base_flow = 4.0  # Caudal base para equilibrio aproximado
            control_signal = base_flow + control_signal
            control_signal = np.clip(control_signal, 0.0, self.simulator.max_inflow)
            
            # Ejecutar paso en simulador
            next_observation, reward, terminated, truncated, info = self.simulator.step(control_signal)
            done = terminated or truncated
            
            # Almacenar experiencia
            self.agent.store_experience(observation, pid_params, reward, next_observation, done)
            
            # Actualizar agente (si hay suficientes experiencias)
            update_metrics = self.agent.update()
            
            # Acumular métricas
            episode_reward += reward
            episode_errors.append(abs(error))
            step_count += 1
            
            # Actualizar estado
            observation = next_observation
            
            # Terminar si el simulador dice que terminó
            if done:
                break
        
        # Calcular métricas del episodio
        avg_error = np.mean(episode_errors) if episode_errors else 0.0
        
        episode_metrics = {
            'reward': episode_reward,
            'length': step_count,
            'avg_error': avg_error,
            'epsilon': self.agent.get_epsilon(),
            'final_level': self.simulator.level,
            'setpoint': self.simulator.setpoint
        }
        
        return episode_metrics
    
    def train(self, 
              num_episodes: int = 1000,
              save_freq: int = 100,
              plot_freq: int = 50,
              save_dir: str = "./models"):
        """
        Entrenar el agente DQN
        
        Args:
            num_episodes: Número de episodios de entrenamiento
            save_freq: Frecuencia de guardado del modelo
            plot_freq: Frecuencia de plots de progreso
            save_dir: Directorio para guardar modelos
        """
        # Crear directorio de guardado
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n🎯 Iniciando entrenamiento DQN: {num_episodes} episodios")
        print(f"💾 Modelos se guardarán cada {save_freq} episodios en: {save_dir}")
        
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Entrenar episodio
            metrics = self.train_episode()
            
            # Guardar métricas
            self.training_metrics['episode_rewards'].append(metrics['reward'])
            self.training_metrics['episode_lengths'].append(metrics['length'])
            self.training_metrics['average_errors'].append(metrics['avg_error'])
            self.training_metrics['epsilon_values'].append(metrics['epsilon'])
            
            # Log progreso
            if episode % 10 == 0:
                recent_rewards = self.training_metrics['episode_rewards'][-10:]
                avg_reward = np.mean(recent_rewards)
                
                print(f"Episodio {episode:4d} | "
                      f"Recompensa: {metrics['reward']:6.2f} | "
                      f"Promedio(10): {avg_reward:6.2f} | "
                      f"Error: {metrics['avg_error']:5.3f} | "
                      f"Epsilon: {metrics['epsilon']:5.3f} | "
                      f"Nivel final: {metrics['final_level']:4.2f}m")
            
            # Guardar modelo periódicamente
            if episode % save_freq == 0:
                model_path = os.path.join(save_dir, f"dqn_tank_ep{episode}.pth")
                self.agent.save(model_path)
                print(f"💾 Modelo guardado: {model_path}")
            
            # Mostrar progreso gráfico
            if episode % plot_freq == 0:
                self.plot_training_progress(show=False, save_path=os.path.join(save_dir, f"progress_ep{episode}.png"))
        
        # Guardar modelo final
        final_model_path = os.path.join(save_dir, "dqn_tank_final.pth")
        self.agent.save(final_model_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Entrenamiento completado en {elapsed_time:.1f} segundos")
        print(f"📊 Modelo final guardado: {final_model_path}")
        
        # Plot final
        self.plot_training_progress(show=True, save_path=os.path.join(save_dir, "training_final.png"))
    
    def plot_training_progress(self, show: bool = True, save_path: str = None):
        """
        Graficar progreso del entrenamiento
        
        Args:
            show: Mostrar gráfico
            save_path: Ruta para guardar (opcional)
        """
        if not self.training_metrics['episode_rewards']:
            print("⚠️ No hay métricas para graficar")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(self.training_metrics['episode_rewards']) + 1)
        
        # Recompensas por episodio
        ax1.plot(episodes, self.training_metrics['episode_rewards'], 'b-', alpha=0.7)
        # Promedio móvil
        if len(self.training_metrics['episode_rewards']) > 20:
            window = 20
            moving_avg = np.convolve(self.training_metrics['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(episodes)+1), moving_avg, 'r-', linewidth=2, label='Promedio móvil (20)')
            ax1.legend()
        ax1.set_title('Recompensa por Episodio')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Recompensa')
        ax1.grid(True, alpha=0.3)
        
        # Error promedio por episodio
        ax2.plot(episodes, self.training_metrics['average_errors'], 'g-')
        ax2.set_title('Error Promedio por Episodio')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Error Absoluto Promedio')
        ax2.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax3.plot(episodes, self.training_metrics['epsilon_values'], 'm-')
        ax3.set_title('Decaimiento de Epsilon (Exploración)')
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
        
        # Longitud de episodios
        ax4.plot(episodes, self.training_metrics['episode_lengths'], 'c-')
        ax4.set_title('Duración de Episodios')
        ax4.set_xlabel('Episodio')
        ax4.set_ylabel('Steps por Episodio')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Progreso de Entrenamiento DQN - Control de Tanque', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Progreso guardado: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def test_trained_agent(self, model_path: str, num_test_episodes: int = 5):
        """
        Probar agente entrenado
        
        Args:
            model_path: Ruta del modelo entrenado
            num_test_episodes: Número de episodios de prueba
        """
        print(f"\n🧪 Probando agente entrenado: {model_path}")
        
        # Cargar modelo
        self.agent.load(model_path)
        
        for episode in range(1, num_test_episodes + 1):
            print(f"\n--- Episodio de prueba {episode} ---")
            
            # Condiciones de prueba
            setpoint = 3.0  # Setpoint fijo para comparar
            observation, info = self.simulator.reset(options={'initial_level': 1.0, 'setpoint': setpoint})
            
            total_reward = 0
            
            for step in range(self.max_episode_steps):
                # Usar agente sin exploración
                pid_params = self.agent.select_action(observation, training=False)
                
                # Control PID
                kp, ki, kd = pid_params
                error = observation[2]
                integral = observation[4]
                derivative = observation[5]
                
                control_signal = kp * error + ki * integral + kd * derivative
                control_signal = 4.0 + control_signal  # Base flow
                control_signal = np.clip(control_signal, 0.0, self.simulator.max_inflow)
                
                # Ejecutar paso
                next_observation, reward, terminated, truncated, info = self.simulator.step(control_signal)
                done = terminated or truncated
                total_reward += reward
                
                # Log cada 20 steps
                if step % 20 == 0:
                    print(f"  Step {step:3d}: Nivel={info['level']:.2f}m, "
                          f"Error={info['error']:.3f}m, "
                          f"PID=[{kp:.2f}, {ki:.2f}, {kd:.2f}]")
                
                observation = next_observation
                
                if done:
                    break
            
            print(f"  Recompensa total: {total_reward:.2f}")
            
            # Mostrar gráfico del último episodio
            if episode == num_test_episodes:
                self.simulator.plot_results(save_path=f"test_episode_{episode}.png")


def main():
    """Función principal para entrenamiento"""
    # Crear entrenador
    trainer = TankDQNTrainer(
        # Parámetros del tanque
        tank_area=2.0,
        max_height=5.0,
        max_inflow=10.0,
        
        # Parámetros del agente
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=100,
        
        # Entrenamiento
        max_episode_steps=200,
        device='cpu'
    )
    
    # Entrenar
    trainer.train(
        num_episodes=500,  # Empezar con pocos episodios
        save_freq=50,
        plot_freq=25,
        save_dir="./models_tank_dqn"
    )
    
    # Probar modelo final
    trainer.test_trained_agent(
        model_path="./models_tank_dqn/dqn_tank_final.pth",
        num_test_episodes=3
    )


if __name__ == "__main__":
    main()