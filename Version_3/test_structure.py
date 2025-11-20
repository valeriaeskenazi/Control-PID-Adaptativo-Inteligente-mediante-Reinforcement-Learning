#!/usr/bin/env python3
"""
Script de prueba rápida - VERSIÓN SIMPLIFICADA
Solo verifica que los imports y la estructura funcionan correctamente.
"""

import sys
sys.path.insert(0, '.')

print("=" * 80)
print("🧪 VERIFICACIÓN DE ARQUITECTURA")
print("=" * 80)

# ============================================================================
# VERIFICAR IMPORTS
# ============================================================================
print("\n1️⃣  Verificando imports de módulos principales...")

errors = []

# Agent
try:
    from Agent.abstract_agent import AbstractPIDAgent
    print("   ✅ Agent/abstract_agent.py")
except Exception as e:
    errors.append(f"Agent/abstract_agent.py: {e}")
    print(f"   ❌ Agent/abstract_agent.py: {e}")

try:
    from Agent.DQN.algorithm_DQN import DQNAgent
    print("   ✅ Agent/DQN/algorithm_DQN.py")
except Exception as e:
    errors.append(f"Agent/DQN: {e}")
    print(f"   ❌ Agent/DQN: {e}")

try:
    from Agent.Actor_Critic.algorithm_ActorCritic import ActorCriticAgent
    print("   ✅ Agent/Actor_Critic/algorithm_ActorCritic.py")
except Exception as e:
    errors.append(f"Agent/Actor_Critic: {e}")
    print(f"   ❌ Agent/Actor_Critic: {e}")

# Environment
try:
    from Environment.base_env import BasePIDControlEnv
    print("   ✅ Environment/base_env.py")
except Exception as e:
    errors.append(f"Environment/base_env: {e}")
    print(f"   ❌ Environment/base_env: {e}")

try:
    from Environment.simulation_env import SimulationPIDEnv
    print("   ✅ Environment/simulation_env.py")
except Exception as e:
    errors.append(f"Environment/simulation_env: {e}")
    print(f"   ❌ Environment/simulation_env: {e}")

try:
    from Environment.multi_agent_env_modular import MultiAgentPIDEnv
    print("   ✅ Environment/multi_agent_env_modular.py")
except Exception as e:
    errors.append(f"Environment/multi_agent_env_modular: {e}")
    print(f"   ❌ Environment/multi_agent_env_modular: {e}")

# Entrenamiento
try:
    from Entrenamiento.controller_agent import ControllerAgent
    print("   ✅ Entrenamiento/controller_agent.py")
except Exception as e:
    errors.append(f"Entrenamiento/controller_agent: {e}")
    print(f"   ❌ Entrenamiento/controller_agent: {e}")

try:
    from Entrenamiento.orchestrator_agent import OrchestratorAgent
    print("   ✅ Entrenamiento/orchestrator_agent.py")
except Exception as e:
    errors.append(f"Entrenamiento/orchestrator_agent: {e}")
    print(f"   ❌ Entrenamiento/orchestrator_agent: {e}")

try:
    from Entrenamiento.pid_trainer import PIDTrainer
    print("   ✅ Entrenamiento/pid_trainer.py")
except Exception as e:
    errors.append(f"Entrenamiento/pid_trainer: {e}")
    print(f"   ❌ Entrenamiento/pid_trainer: {e}")

# Simuladores
try:
    from Simulations_Env.reactor_CSTR import CSTRSimulator
    print("   ✅ Simulations_Env/reactor_CSTR.py")
except Exception as e:
    errors.append(f"Simulations_Env/reactor_CSTR: {e}")
    print(f"   ❌ Simulations_Env/reactor_CSTR: {e}")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 80)
if len(errors) == 0:
    print("✅ TODOS LOS MÓDULOS SE IMPORTAN CORRECTAMENTE")
    print("=" * 80)
    print("\n🎉 Tu arquitectura está bien estructurada!")
    print("\n📝 PRÓXIMOS PASOS:")
    print("   1. Asegúrate de tener instalado:")
    print("      - pip install gymnasium torch numpy scipy")
    print("   2. Ejecuta el entrenamiento completo con test_quick.py")
else:
    print(f"❌ ERRORES ENCONTRADOS: {len(errors)}")
    print("=" * 80)
    print("\n🔧 Errores a resolver:")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print("\n💡 Posible causa: Falta instalar dependencias")
    print("   Ejecuta: pip install gymnasium torch numpy scipy")

print("=" * 80)
