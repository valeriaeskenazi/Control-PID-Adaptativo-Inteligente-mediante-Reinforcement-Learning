#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que la arquitectura funciona.
Configuración mínima: 2 iteraciones del orquestador, 5 episodios por controlador.
"""

import sys
import numpy as np

# Configurar path
sys.path.insert(0, '.')

print("=" * 80)
print("🧪 PRUEBA RÁPIDA - ARQUITECTURA MULTI-AGENTE")
print("=" * 80)

# ============================================================================
# 1. IMPORTS
# ============================================================================
print("\n📦 Importando módulos...")

try:
    from Environment.multi_agent_env_modular import MultiAgentPIDEnv
    from Simulations_Env.reactor_CSTR import CSTRSimulator
    print("✅ Imports exitosos")
except Exception as e:
    print(f"❌ Error en imports: {e}")
    sys.exit(1)

# ============================================================================
# 2. CONFIGURACIÓN MÍNIMA
# ============================================================================
print("\n⚙️  Configurando ambiente...")

config = {
    # Arquitectura
    'architecture': 'indirect',  # Modo con orquestador
    
    # Variables
    'n_manipulable_vars': 2,  # T y Tc
    'n_target_vars': 1,       # CB (concentración)
    'target_indices': [0],    # CB está en índice 0 del state
    
    # Rangos de las variables manipulables
    'sp_ranges': [(290.0, 450.0), (99.0, 105.0)],  # Rangos de T y Tc
    
    # Variables objetivo
    'target_ranges': [(0.0, 1.0)],      # Rango válido de CB
    'target_setpoints': [0.2],           # CB deseado = 0.2
    
    # Configuración base_env
    'setpoint': [370.0, 102.0],          # Setpoints iniciales [T, Tc]
    'upper_range': [450.0, 105.0],       # Límites superiores
    'lower_range': [290.0, 99.0],        # Límites inferiores
    'dead_band': [5.0, 0.5],             # Bandas muertas
    
    # Entrenamiento (REDUCIDO PARA PRUEBA RÁPIDA)
    'n_episodes': 20,                     # ⚡ Solo 5 episodios por controlador
    'max_episode_steps': 20,             # ⚡ Solo 20 pasos por episodio
    'orchestrator_iterations': 2,        # ⚡ Solo 2 iteraciones del orquestador
    'j_max_retries': 1,                  # ⚡ Solo 1 reintento
    
    # Configuración de agentes (reducida)
    'agent_lr': 0.01,                    # Learning rate alto para aprender rápido
    'hidden_dims': (32, 32),             # ⚡ Red pequeña
    'orch_hidden_dims': (32, 32),        # ⚡ Red pequeña
    'batch_size': 8,                     # ⚡ Batch pequeño para pocas experiencias
    'memory_size': 500,                  # ⚡ Buffer pequeño

    # Otros
    'dt': 1.0,
    'device': 'cpu'
}

print("✅ Configuración lista")
print(f"   - Arquitectura: {config['architecture']}")
print(f"   - Variables manipulables: {config['n_manipulable_vars']}")
print(f"   - Variables objetivo: {config['n_target_vars']}")
print(f"   - Iteraciones orquestador: {config['orchestrator_iterations']}")
print(f"   - Episodios por controlador: {config['n_episodes']}")

# ============================================================================
# 3. CREAR AMBIENTE
# ============================================================================
print("\n🏗️  Creando MultiAgentPIDEnv...")

try:
    env = MultiAgentPIDEnv(config)
    print("✅ Ambiente creado exitosamente")
except Exception as e:
    print(f"❌ Error creando ambiente: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. CONECTAR SIMULADOR
# ============================================================================
print("\n🔌 Conectando simulador CSTR...")

try:
    reactor = CSTRSimulator()
    env.base_env.connect_external_process(reactor)
    print("✅ Simulador conectado")
except Exception as e:
    print(f"❌ Error conectando simulador: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 5. EJECUTAR ENTRENAMIENTO
# ============================================================================
print("\n🚀 Iniciando entrenamiento rápido...")
print("   (Esto puede tomar 1-2 minutos)")
print("-" * 80)

try:
    best_pids, best_setpoints = env.train()
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 80)
    
    print("\n📊 RESULTADOS:")
    print("\nMejores PIDs encontrados:")
    for i, pid in enumerate(best_pids):
        print(f"   Variable {i}: Kp={pid[0]:.4f}, Ki={pid[1]:.4f}, Kd={pid[2]:.4f}")
    
    print("\nMejores Setpoints:")
    for i, sp in enumerate(best_setpoints):
        print(f"   Variable {i}: SP={sp:.2f}")
    
    # Estadísticas
    stats = env.get_statistics()
    print("\n📈 ESTADÍSTICAS:")
    print(f"   PID Trainer stats: {stats}")
    
except Exception as e:
    print(f"\n❌ Error durante entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 6. PRUEBA DE RESET
# ============================================================================
print("\n🔄 Probando reset del ambiente...")

try:
    obs, info = env.reset()
    print(f"✅ Reset exitoso - Observación shape: {obs.shape}")
except Exception as e:
    print(f"❌ Error en reset: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("🎉 PRUEBA COMPLETADA EXITOSAMENTE")
print("=" * 80)
print("\n✅ La arquitectura funciona correctamente")
print("✅ Próximo paso: Entrenar con más iteraciones y episodios")
print("\nPara entrenamiento completo, modifica en config:")
print("   - orchestrator_iterations: 10-50")
print("   - n_episodes: 50-200")
print("   - max_episode_steps: 100-500")
print("=" * 80)

# Al final del script
print("\n📊 Generando gráficos...")
env.logger.plot_results(
    save_dir='./results/test_quick',
    show=False  # True para ver interactivamente
)