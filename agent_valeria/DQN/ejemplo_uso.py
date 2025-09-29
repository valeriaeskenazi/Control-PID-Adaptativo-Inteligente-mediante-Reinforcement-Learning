"""
Ejemplo de uso del agente DQN simple
Muestra cómo entrenar y usar el agente
"""
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQN_Agent

def simulador_proceso_simple(pid_params, estado_actual):
    """
    Simulador muy simple de un proceso para probar
    
    Args:
        pid_params: [Kp, Ki, Kd]
        estado_actual: [PV, SP, error, error_prev, error_int, error_der]
    
    Returns:
        nuevo_estado, reward, done
    """
    pv_actual = estado_actual[0]
    setpoint = estado_actual[1]
    error_prev = estado_actual[2]
    
    # Control PID simple
    kp, ki, kd = pid_params
    error = setpoint - pv_actual
    
    # Simular respuesta del proceso (muy simplificado)
    control_output = kp * error + ki * sum([error, error_prev]) + kd * (error - error_prev)
    
    # Nuevo PV (proceso de primer orden simplificado)
    nuevo_pv = pv_actual + control_output * 0.1 + np.random.normal(0, 0.1)
    
    # Nuevo estado
    nuevo_error = setpoint - nuevo_pv
    error_integral = sum([error, error_prev, nuevo_error])  # Simplificado
    error_derivative = nuevo_error - error
    
    nuevo_estado = np.array([
        nuevo_pv,           # PV
        setpoint,           # SP (fijo)
        nuevo_error,        # Error
        error,              # Error anterior  
        error_integral,     # Error integral
        error_derivative    # Error derivativo
    ])
    
    # Reward simple: negativo del error absoluto
    reward = -abs(nuevo_error)
    if abs(nuevo_error) < 2.0:  # Dentro de dead band
        reward += 5.0
    
    # Episode termina si está muy lejos o muy cerca
    done = abs(nuevo_error) < 1.0 or abs(nuevo_error) > 50.0
    
    return nuevo_estado, reward, done


def entrenar_agente(episodios=100):
    """Entrenar el agente DQN"""
    
    print("🏋️ Entrenando agente DQN...")
    
    # Crear agente
    agent = DQN_Agent(
        state_dim=6,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=5000,
        batch_size=32
    )
    
    # Métricas de entrenamiento
    rewards_por_episodio = []
    losses = []
    
    for episodio in range(episodios):
        # Estado inicial (PV alejado del setpoint)
        estado = np.array([30.0, 75.0, 45.0, 40.0, 0.0, 0.0])  # PV=30, SP=75
        
        reward_total = 0
        pasos = 0
        max_pasos = 100
        
        while pasos < max_pasos:
            # Seleccionar acción
            action_idx, pid_params = agent.select_action(estado, training=True)
            
            # Simular ambiente
            nuevo_estado, reward, done = simulador_proceso_simple(pid_params, estado)
            
            # Almacenar experiencia
            agent.store_experience(estado, action_idx, reward, nuevo_estado, done)
            
            # Entrenar
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            # Actualizar
            estado = nuevo_estado
            reward_total += reward
            pasos += 1
            
            if done:
                break
        
        rewards_por_episodio.append(reward_total)
        agent.episodes_done += 1
        
        # Log cada 10 episodios
        if episodio % 10 == 0:
            stats = agent.get_stats()
            print(f"Episodio {episodio:3d}: Reward={reward_total:6.1f}, "
                  f"Epsilon={stats['epsilon']:.3f}, "
                  f"Steps={pasos:2d}, "
                  f"Memory={stats['memory_size']:4d}")
    
    # Guardar agente entrenado
    agent.save_agent("dqn_pid_agent.pth")
    
    return agent, rewards_por_episodio, losses


def probar_agente_entrenado():
    """Probar agente ya entrenado"""
    
    print("🧪 Probando agente entrenado...")
    
    # Crear y cargar agente
    agent = DQN_Agent()
    agent.load_agent("dqn_pid_agent.pth")
    
    # Estado inicial
    estado = np.array([20.0, 75.0, 55.0, 50.0, 0.0, 0.0])  # PV muy lejos de SP
    
    print(f"Estado inicial: PV={estado[0]:.1f}, SP={estado[1]:.1f}")
    
    # Simular control
    historico_pv = [estado[0]]
    historico_reward = []
    
    for paso in range(50):
        # Acción sin exploración
        action_idx, pid_params = agent.select_action(estado, training=False)
        
        # Simular
        nuevo_estado, reward, done = simulador_proceso_simple(pid_params, estado)
        
        historico_pv.append(nuevo_estado[0])
        historico_reward.append(reward)
        
        print(f"Paso {paso:2d}: PV={nuevo_estado[0]:5.1f}, "
              f"PID=[{pid_params[0]:.2f}, {pid_params[1]:.2f}, {pid_params[2]:.2f}], "
              f"Reward={reward:5.1f}")
        
        estado = nuevo_estado
        
        if done:
            print(f"✅ Convergió en {paso+1} pasos!")
            break
    
    # Graficar resultado
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(historico_pv, 'b-', label='PV')
    plt.axhline(y=75.0, color='r', linestyle='--', label='Setpoint')
    plt.axhline(y=73.0, color='g', linestyle=':', alpha=0.7, label='Dead band')
    plt.axhline(y=77.0, color='g', linestyle=':', alpha=0.7)
    plt.xlabel('Pasos')
    plt.ylabel('Valor')
    plt.title('Control PID con DQN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(historico_reward, 'orange')
    plt.xlabel('Pasos')
    plt.ylabel('Reward')
    plt.title('Recompensa por paso')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_control_result.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("🤖 Ejemplo de uso DQN para control PID")
    print("="*50)
    
    # Entrenar
    agent, rewards, losses = entrenar_agente(episodios=50)
    
    print("\n" + "="*50)
    
    # Probar
    probar_agente_entrenado()
    
    print("\n🎉 ¡Ejemplo completado!")
    print("📁 Archivo guardado: dqn_pid_agent.pth")
    print("📊 Gráfico guardado: dqn_control_result.png")