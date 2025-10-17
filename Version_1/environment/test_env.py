# test_env.py
import numpy as np
from universal_pid_env import UniversalPIDControlEnv

print("=" * 70)
print("PRUEBA 1: Modo DIRECT (control continuo)")
print("=" * 70)

env_direct = UniversalPIDControlEnv(control_mode='direct')
print(f"✅ Ambiente creado")
print(f"   Action space: {env_direct.action_space}")
print(f"   Observation space: {env_direct.observation_space.shape}")

obs, info = env_direct.reset()
print(f"\n✅ Reset exitoso")
print(f"   Observación inicial: {obs}")

# Probar 5 steps
for i in range(5):
    action = np.array([0.3], dtype=np.float32)  # Acción continua
    obs, reward, terminated, truncated, info = env_direct.step(action)
    print(f"   Step {i+1}: reward={reward:.3f}, pv={obs[0]:.2f}")

print("\n" + "=" * 70)
print("PRUEBA 2: Modo PID_TUNING (acciones discretas)")
print("=" * 70)

env_pid = UniversalPIDControlEnv(control_mode='pid_tuning')
print(f"✅ Ambiente creado")
print(f"   Action space: {env_pid.action_space}")
print(f"   Número de acciones: {env_pid.action_space.n}")

obs, info = env_pid.reset()
print(f"\n✅ Reset exitoso")
print(f"   Observación inicial: {obs}")
print(f"   PID inicial: {env_pid.pid_action_space.get_current_pid()}")

# Probar diferentes acciones
acciones_prueba = [
    (0, "Kp ↑"),
    (1, "Ki ↑"),
    (2, "Kd ↑"),
    (6, "Mantener"),
    (3, "Kp ↓")
]

for action_idx, descripcion in acciones_prueba:
    obs, reward, terminated, truncated, info = env_pid.step(action_idx)
    print(f"\n   Acción {action_idx} ({descripcion}):")
    print(f"      PID: {info['pid_params']}")
    print(f"      Control output: {info['control_output']:.4f}")
    print(f"      Reward: {reward:.3f}")
    print(f"      PV: {obs[0]:.2f}, Error: {obs[2]:.2f}")

print("\n" + "=" * 70)
print("✅ TODAS LAS PRUEBAS EXITOSAS")
print("=" * 70)