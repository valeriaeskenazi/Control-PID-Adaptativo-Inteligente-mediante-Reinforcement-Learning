# 🏗️ Simulación de Control de Nivel de Tanque

Simulador físico simple para entrenar y probar agentes de control PID usando Deep Q-Network (DQN).

## 📁 Estructura

```
tanque_nivel/
├── tanque_simulator.py    # Simulador físico del tanque
├── train_dqn.py          # Script de entrenamiento DQN
└── README.md             # Este archivo
```

## 🎯 Objetivo

Entrenar un agente DQN para controlar el nivel de líquido en un tanque cilíndrico mediante la manipulación del caudal de entrada.

## ⚙️ Física del Proceso

### Sistema:
- **Tanque cilíndrico** con área constante (2 m²)
- **Caudal de entrada**: Controlado por válvula (0-10 L/s)
- **Caudal de salida**: Por gravedad `Qout = C * √h`
- **Variable controlada**: Nivel del líquido (0-5 m)

### Ecuación diferencial:
```
A * dh/dt = Qin - Qout
```

### Características:
- **Respuesta**: No-lineal (salida depende de √h)
- **Estabilidad**: Naturalmente estable
- **Tiempo de respuesta**: ~60-120 segundos

## 🧠 Agente DQN

### Estado (6 dimensiones):
```python
state = [
    nivel_actual,      # PV (Process Variable)
    setpoint,          # SP (Set Point) 
    error,             # SP - PV
    error_anterior,    # Para cálculo de derivada
    integral_error,    # Para cálculo de integral
    derivada_error     # Derivada del error
]
```

### Acción:
- **Espacio discreto**: 64 combinaciones de parámetros PID
- **Conversión**: Índice → [Kp, Ki, Kd] → Señal de control
- **Control PID**: `u = Kp*e + Ki*∫e + Kd*de/dt`

### Recompensa:
```python
# Recompensa principal: exponencial del error
reward = exp(-|error| * 2.0)

# Penalizaciones por niveles peligrosos
if nivel < 0.5m: penalty = -2.0 * (0.5 - nivel)
if nivel > 4.5m: penalty = -2.0 * (nivel - 4.5)
```

## 🚀 Uso

### Entrenamiento:
```bash
cd simulations/tanque_nivel
python train_dqn.py
```

### Prueba rápida del simulador:
```bash
python tanque_simulator.py
```

### Parámetros principales:

**Simulador:**
- `tank_area`: Área del tanque (m²)
- `max_height`: Altura máxima (m)
- `max_inflow`: Caudal máximo entrada (L/s)
- `noise_level`: Ruido en medición (%)

**Agente DQN:**
- `lr`: Learning rate (0.001)
- `gamma`: Factor descuento (0.99)
- `epsilon_start/end`: Exploración inicial/final
- `memory_size`: Tamaño buffer replay (10000)
- `batch_size`: Tamaño batch entrenamiento (32)

**Entrenamiento:**
- `num_episodes`: Episodios totales (500)
- `max_episode_steps`: Steps máximos por episodio (200)

## 📊 Resultados Esperados

### Durante entrenamiento:
1. **Epsilon decay**: De 1.0 → 0.01 (exploración → explotación)
2. **Recompensa**: Incremento gradual conforme aprende
3. **Error promedio**: Disminución progresiva
4. **Convergencia**: ~300-500 episodios

### Agente entrenado:
- **Error estacionario**: < 0.1m
- **Tiempo de establecimiento**: < 100 segundos
- **Sobrepico**: < 10%
- **Robustez**: Maneja perturbaciones y ruido

## 🔧 Personalización

### Cambiar complejidad del proceso:
```python
# Más ruido
TankLevelSimulator(noise_level=0.05)  # 5% ruido

# Tanque más rápido
TankLevelSimulator(tank_area=1.0)  # Área menor

# Mayor no-linealidad
TankLevelSimulator(outflow_coeff=3.0)  # Salida más agresiva
```

### Ajustar DQN:
```python
# Exploración más conservadora
DQN_Agent(epsilon_decay=0.999)

# Red más grande
DQN_Network(hidden_size=256)

# Memoria más grande
DQN_Agent(memory_size=50000)
```

## 📈 Métricas de Evaluación

El entrenador guarda automáticamente:
- **Gráficos de progreso**: Cada 25 episodios
- **Modelos**: Cada 50 episodios
- **Métricas**: Recompensa, error, epsilon, duración

### Archivos generados:
```
models_tank_dqn/
├── dqn_tank_ep50.pth
├── dqn_tank_ep100.pth
├── ...
├── dqn_tank_final.pth
├── progress_ep25.png
├── training_final.png
└── test_episode_3.png
```

## 🎯 Próximos Pasos

1. **Validar funcionamiento básico**: 500 episodios
2. **Optimizar hiperparámetros**: Grid search
3. **Agregar complejidad**: Perturbaciones, delays
4. **Comparar algoritmos**: PPO, SAC, etc.
5. **Implementar en proceso real**: Hardware-in-the-loop

## 🔍 Debugging

### Problemas comunes:

**No converge:**
- Reducir learning rate
- Aumentar epsilon_decay (explorar más tiempo)
- Verificar recompensa (debe ser positiva en promedio)

**Oscilaciones:**
- Ajustar parámetros PID del espacio de acciones
- Reducir ruido en simulador
- Aumentar target_update_freq

**Memoria insuficiente:**
- Reducir batch_size
- Reducir memory_size
- Usar CPU en lugar de GPU para pruebas

**Simulador inestable:**
- Verificar límites físicos
- Ajustar paso de simulación (dt)
- Revisar ecuación diferencial