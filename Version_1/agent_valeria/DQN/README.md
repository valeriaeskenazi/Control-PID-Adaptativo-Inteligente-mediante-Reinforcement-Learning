# DQN para Control PID - Explicación

Este documento explica cómo funciona el algoritmo DQN (Deep Q-Network) aplicado al control PID, específicamente el manejo de **acciones discretas** vs **valores continuos**.

## 🎯 Problema Principal

En control PID necesitamos determinar 3 parámetros continuos:
- **Kp** (Ganancia Proporcional): 0.1 a 10.0
- **Ki** (Ganancia Integral): 0.01 a 5.0  
- **Kd** (Ganancia Derivativa): 0.001 a 2.0

Pero **DQN clásico** solo maneja **acciones discretas**.

## 🔄 Solución: Discretización

### Paso 1: Crear valores discretos para cada parámetro

En lugar de valores continuos, creamos **4 opciones** para cada parámetro:

```python
Kp_opciones = [0.1, 3.4, 6.7, 10.0]     # 4 valores discretos
Ki_opciones = [0.01, 1.67, 3.33, 5.0]   # 4 valores discretos
Kd_opciones = [0.001, 0.67, 1.33, 2.0]  # 4 valores discretos
```

### Paso 2: Calcular todas las combinaciones posibles

```
Total combinaciones = 4 (Kp) × 4 (Ki) × 4 (Kd) = 64 acciones
```

### Paso 3: Enumerar todas las combinaciones

| Índice | Kp   | Ki   | Kd    | Descripción |
|--------|------|------|-------|-------------|
| 0      | 0.1  | 0.01 | 0.001 | Control muy suave |
| 1      | 0.1  | 0.01 | 0.67  | Control suave con derivativa |
| 2      | 0.1  | 0.01 | 1.33  | Control suave con más derivativa |
| ...    | ...  | ...  | ...   | ... |
| 31     | 3.4  | 1.67 | 2.0   | Control moderado |
| ...    | ...  | ...  | ...   | ... |
| 63     | 10.0 | 5.0  | 2.0   | Control muy agresivo |

## 🧠 Arquitectura de la Red Neuronal

```
Estado del proceso (6 dimensiones)
        ↓
[PV, SP, error, error_prev, error_int, error_der]
        ↓
    Red DQN
        ↓
64 Q-values (uno por cada combinación PID)
```

### Estructura de capas:
```python
fc1: Linear(6 → 128)    # Capa de entrada
fc2: Linear(128 → 128)  # Capa oculta 1  
fc3: Linear(128 → 64)   # Capa oculta 2
fc4: Linear(64 → 64)    # Capa de salida (Q-values)
```

### Inicialización inteligente de pesos:

**¿Por qué es importante?**
- **Problema**: Pesos mal inicializados pueden causar gradientes que "explotan" o "se desvanecen"
- **Solución**: Inicialización Kaiming Normal + bias pequeño

```python
# Kaiming Normal para pesos
nn.init.kaiming_normal_(layer.weight)
# - Calcula varianza óptima: std = sqrt(2.0 / fan_in)
# - Para capa 128→64: std = sqrt(2.0/128) ≈ 0.125
# - Inicializa pesos con distribución Normal(mean=0, std=0.125)

# Bias constante pequeño  
nn.init.constant_(layer.bias, 0.01)
# - Evita neuronas "muertas" (siempre hay pequeña activación)
# - Mejor que bias=0.0 para el inicio del entrenamiento
```

**Ejemplo práctico:**
```python
# Sin inicialización inteligente:
output = [1000000, -1000000, ...]  # ← EXPLOTA!
# o:    [0.00001, 0.00001, ...]   # ← SE DESVANECE!

# Con Kaiming + bias=0.01:
output = [0.23, -0.87, 1.45, ...]  # ← VALORES RAZONABLES
```

### Arquitectura de dos redes (Q-Network + Target Network):

**¿Por qué DQN necesita dos redes idénticas?**

DQN utiliza dos redes neuronales con la **misma arquitectura** pero **roles diferentes**:

#### 1. **Red Principal (Q-Network)**
```python
self.q_network = DQN_Network(state_dim, n_actions)  # Red principal
```
- **Función**: Aprender y generar Q-values para seleccionar acciones
- **Se entrena**: SÍ, se actualiza constantemente con backpropagation
- **Uso**: `q_values = self.q_network(state)` → Seleccionar mejores acciones

#### 2. **Red Objetivo (Target Network)**  
```python
self.target_network = DQN_Network(state_dim, n_actions)  # Red objetivo
```
- **Función**: Generar Q-values "estables" para calcular targets de entrenamiento
- **Se entrena**: NO, se copia periódicamente de la red principal
- **Uso**: `target_q = self.target_network(next_state)` → Calcular objetivos estables

#### **Problema sin Target Network:**
```python
# ❌ INESTABLE: Red persiguiendo su propia cola
current_q = q_network(state)           # Red da Q-values actuales
target_q = q_network(next_state)       # ¡LA MISMA RED da el target!
# La red trata de alcanzar un objetivo que ella misma genera
# y que cambia constantemente → Aprendizaje inestable
```

#### **Solución con Target Network:**
```python
# ✅ ESTABLE: Objetivos fijos temporalmente
current_q = q_network(state)           # Red principal da Q-values
target_q = target_network(next_state)  # Red DIFERENTE da targets estables
# Los targets permanecen fijos por períodos → Aprendizaje estable
```

#### **Actualización periódica:**
```python
# Cada 1000 pasos (configurable):
if self.training_step % self.target_update_freq == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
    # ↑ Copiar pesos de red principal a red objetivo
```

#### **Analogía del Basketball:**
- **Sin Target Network**: Intentas encestar en un aro que se mueve cada vez que tiras
- **Con Target Network**: El aro permanece fijo mientras practicas, luego se mueve a una nueva posición

Este mecanismo es fundamental para la estabilidad del algoritmo DQN y evita divergencias durante el entrenamiento.

### Ejemplo concreto:

```python
# Estado actual
estado = [PV=50.0, SP=75.0, error=25.0, error_prev=20.0, error_int=100.0, error_der=5.0]

# Red neuronal procesa el estado
q_values = dqn_network(estado)
# Resultado: [0.2, 0.8, -0.1, 0.9, 0.3, ..., 0.1]  (64 valores)
#             ↑    ↑    ↑    ↑
#          Q(a0) Q(a1) Q(a2) Q(a3) ...

# Cada Q-value representa: "¿Qué tan buena es esta combinación PID para este estado?"
```

## 🎯 Selección de Acción (Epsilon-Greedy)

```python
if random() < epsilon:
    # Exploración: acción aleatoria
    accion_indice = random.randint(0, 63)
else:
    # Explotación: mejor acción según Q-values
    accion_indice = argmax(q_values)  # Ejemplo: índice 31

# Convertir índice a parámetros PID reales
pid_params = action_space.index_to_pid(31)
# Resultado: [Kp=3.4, Ki=1.67, Kd=2.0]
```

## 🔄 Conversión entre Índices y Parámetros PID

### El proceso completo de conversión:

La conversión de índice único a parámetros PID es un proceso de **3 pasos fundamentales**:

#### **Paso 1: Crear las "tablas de lookup" (en `__init__`)**

```python
# Crear valores uniformemente espaciados para cada parámetro
self.kp_values = np.linspace(0.1, 10.0, 4)   # [0.1, 3.4, 6.7, 10.0]
self.ki_values = np.linspace(0.01, 5.0, 4)   # [0.01, 1.67, 3.33, 5.0]  
self.kd_values = np.linspace(0.001, 2.0, 4)  # [0.001, 0.67, 1.33, 2.0]
#                                              ↑     ↑     ↑     ↑
#                                           idx=0  idx=1 idx=2 idx=3
```

#### **Paso 2: Convertir índice 1D → coordenadas 3D**

**Problema**: La red neuronal devuelve un índice único (ej: 37), pero necesitamos coordenadas (Kp, Ki, Kd).

**Analogía del edificio de apartamentos 4×4×4:**
- Tenemos apartamento #37
- Necesitamos saber: ¿qué piso, fila y columna?

```python
def index_to_pid(action_index):
    # Ejemplo: action_index = 37, n_discrete = 4
    
    # ¿En qué "columna" Kd está?
    kd_idx = action_index % 4 = 37 % 4 = 1
    
    # ¿En qué "fila" Ki está?  
    ki_idx = (action_index // 4) % 4 = (37 // 4) % 4 = 9 % 4 = 1
    
    # ¿En qué "piso" Kp está?
    kp_idx = (action_index // 16) % 4 = (37 // 16) % 4 = 2 % 4 = 2
    
    # Resultado: coordenadas [piso=2, fila=1, columna=1]
```

#### **Paso 3: Coordenadas → Valores PID reales**

**Usar las coordenadas como índices en las tablas pre-calculadas:**

```python
    # Buscar en las "tablas de lookup"
    kp = self.kp_values[kp_idx]  # kp_values[2] = 6.7
    ki = self.ki_values[ki_idx]  # ki_values[1] = 1.67
    kd = self.kd_values[kd_idx]  # kd_values[1] = 0.67
    
    return np.array([6.7, 1.67, 0.67])  # ¡Parámetros PID listos!
```

### **Analogía completa del menú de restaurante:**

```
Configuración del "menú":
Kp Menu: [0] Suave (0.1)   [1] Medio (3.4)   [2] Fuerte (6.7)   [3] Extra (10.0)
Ki Menu: [0] Bajo (0.01)   [1] Medio (1.67)  [2] Alto (3.33)    [3] Max (5.0)
Kd Menu: [0] Min (0.001)   [1] Poco (0.67)   [2] Medio (1.33)   [3] Max (2.0)

Red neuronal dice: "Orden #37"
↓ Conversión matemática ↓
Coordenadas: Kp[2], Ki[1], Kd[1]
↓ Lookup en el menú ↓  
Pedido real: "Fuerte (6.7), Medio (1.67), Poco (0.67)"
```

### **¿Por qué este proceso en 3 pasos?**

1. **Red neuronal**: Solo puede trabajar con números enteros (índices)
2. **Ambiente PID**: Necesita valores continuos reales
3. **Discretización**: Permite que DQN funcione con acciones "casi-continuas"

### Ejemplo numérico completo:

```python
# Entrada: Índice de la red neuronal
action_index = 37

# Paso 1: Ya tenemos las tablas (creadas en __init__)
kp_values = [0.1, 3.4, 6.7, 10.0]
ki_values = [0.01, 1.67, 3.33, 5.0]  
kd_values = [0.001, 0.67, 1.33, 2.0]

# Paso 2: Matemática de coordenadas
kd_idx = 37 % 4 = 1
ki_idx = (37 // 4) % 4 = 1  
kp_idx = (37 // 16) % 4 = 2

# Paso 3: Lookup de valores reales
kp = kp_values[2] = 6.7
ki = ki_values[1] = 1.67
kd = kd_values[1] = 0.67

# Resultado final
pid_params = [6.7, 1.67, 0.67]  # ¡Listo para controlar el proceso!
```

## 📊 Ventajas y Desventajas

### ✅ Ventajas:
- **Compatible con DQN clásico**: Usa la teoría estándar de Q-learning
- **Exploración estructurada**: Explora combinaciones coherentes de PID
- **Estable**: Evita valores PID extremos o inválidos
- **Interpretable**: Cada acción tiene significado físico claro

### ❌ Desventajas:
- **Resolución limitada**: Solo 4 valores por parámetro
- **Crecimiento exponencial**: 4³ = 64, pero 10³ = 1000 acciones
- **No encuentra valores "entre" opciones**: Si el óptimo es Kp=2.5, nunca lo encuentra

## 🚀 Alternativas

### 1. **Más discretización**
```python
# 8 valores por parámetro = 8³ = 512 acciones
# Más precisión, pero red más grande y entrenamiento más lento
```

### 2. **Algoritmos continuos**
```python
# DDPG, TD3, SAC: Directamente [Kp, Ki, Kd] continuos
# Más complejo pero más flexible
```

### 3. **DQN multi-head**
```python
# 3 redes separadas: una para Kp, otra para Ki, otra para Kd
# Más experimental
```

## 💻 Uso en el Código

```python
from DQN import DQN_Agent

# Crear agente
agent = DQN_Agent(state_dim=6)

# Estado del proceso
estado = [pv, sp, error, error_prev, error_int, error_der]

# Obtener acción
action_idx, pid_params = agent.select_action(estado)

# pid_params contiene [Kp, Ki, Kd] listos para usar en el controlador
```

## 🎯 Resumen

**DQN discretiza el espacio continuo PID en 64 combinaciones fijas, donde cada "acción" es una tupla (Kp, Ki, Kd). La red neuronal aprende a evaluar qué combinación es mejor para cada estado del proceso.**

Esto permite usar DQN estándar para un problema inherentemente continuo, sacrificando algo de resolución por simplicidad y estabilidad del algoritmo.