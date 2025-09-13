# Diario de charla sobre mi tesis - 6 sept 2025

## Lo que charlamos hoy sobre RL y PID

Estuvimos viendo el paper de CIRL y comparándolo con mi idea de tesis. Resultó súper útil porque me di cuenta de varias cosas.

### El paper CIRL - qué hacen ellos

Los tipos del paper crearon un agente RL que ajusta parámetros PID, pero está súper atado a un reactor químico específico (CSTR). Básicamente:
- Hicieron un simulador completo del reactor con 5 ecuaciones diferenciales
- Solo funciona para ese proceso en particular
- No probaron nada real, solo simulación
- Usan Gymnasium + SciPy para resolver las matemáticas

### Mi idea (que sigue siendo diferente)

Yo quiero algo mucho más general - un agente que pueda sintonizar PID para cualquier proceso, no solo uno específico. La filosofía es:

- Que el ambiente sea súper simple: le das setpoint, rango de operación, y te devuelve recompensas
- No necesito saber las ecuaciones del proceso de fondo
- Separar totalmente el entrenamiento de la simulación específica
- Imitar lo que hace la gente real: probar valores, ver qué pasa, ajustar

### La idea clave de aprendizaje

Mi concepto es que el agente entrene en montón de procesos diferentes y desarrolle como una "memoria" para reconocer patrones:
- "Ah, este proceso responde lento, debe ser como aquel tanque de nivel que vi antes"
- "Este tiene oscilaciones, mejor uso una estrategia más conservadora"
- "Este deadtime me recuerda al intercambiador, voy a aplicar lo que funcionó allá"

### Los problemas que me señalaste (y que son reales)

**1. ¿Cómo va a saber qué proceso es?**
Sin información, el agente tiene que adivinar solo mirando la respuesta. Un proceso lento puede parecer roto hasta que junta suficientes datos.

**2. El tema del "olvido catastrófico"**
Las redes neuronales son medio tontas - cuando aprenden algo nuevo, se olvidan lo viejo. Necesitaría técnicas fancy como:
- Elastic Weight Consolidation (no tengo idea qué es, pero suena complicado)
- Progressive Neural Networks
- Experience Replay con datos de procesos viejos

**3. Escalas de tiempo súper diferentes**
Un lazo de nivel puede tardar minutos en responder, uno de temperatura segundos. ¿Cómo normalizo las recompensas?

**4. Los procesos son re diversos**
Hay procesos lineales, no lineales, con deadtime, sin deadtime, estables, inestables... es un quilombo bárbaro.

**5. Va a necesitar muchísimas muestras**
A diferencia de CIRL que tiene el conocimiento PID embebido, mi agente tiene que aprender todo desde cero.

### Posibles soluciones que surgieron

**1. Familias de procesos**
En lugar de un agente universal, tal vez entrenar varios especializados (térmicos, de nivel, de flujo) y un meta-agente que elija cuál usar.

**2. Features mínimas**
Aunque quiera ser agnóstico, tal vez necesite algunas pistas básicas:
- ¿Es rápido o lento?
- ¿Tiene deadtime?
- ¿Es integrativo o autorregulado?

**3. Entrenamiento progresivo**
Empezar con procesos simples (primer orden) e ir subiendo la complejidad gradualmente.

### Mi reflexión después de la charla

La idea sigue siendo buena, pero es más compleja de lo que pensaba inicialmente. No es simplemente "entrenar en todo y que funcione". Necesito ser más inteligente sobre cómo estructurar el problema.

Tal vez debería empezar con una familia chica de procesos para probar el concepto antes de ir por la generalización total. O buscar un punto medio donde el agente sea más general que CIRL pero no tan ambicioso como "funcionar para todo".

### Próximos pasos que me van quedando (versión mejorada)

**1. Mapear el universo de procesos industriales**
Hacer una lista completa de escenarios industriales típicos y clasificarlos por dificultad:

**Fácil:**
- Control de nivel en tanque (1er orden, sin deadtime)
- Control de flujo (respuesta rápida, lineal)
- Control de presión en sistema simple

**Medio:**
- Control de temperatura con intercambiador (deadtime moderado)
- Control de pH (no-lineal pero predecible)
- Control de densidad en mezcla

**Difícil:**
- Control de temperatura en horno (deadtime grande, no-lineal)
- Control de composición en destilación (multivariable)
- Control de viscosidad en reactor batch

**2. Estrategia de transfer learning escalonada**
- Probar transfer learning DENTRO de cada nivel (fácil→fácil, medio→medio)
- Si funciona dentro del nivel, probar ENTRE niveles (fácil→medio→difícil)
- Ver en qué punto se rompe la transferibilidad

**3. Desarrollar ambientes modulares para cada proceso**
Crear simuladores simples pero representativos para cada uno de la lista.

**4. Métricas claras de transferibilidad**
- ¿Cuántas muestras necesita en el proceso nuevo vs entrenar desde cero?
- ¿Converge más rápido? ¿A mejor performance?

La charla me ayudó a bajar un poco a tierra y ser más realista sobre la complejidad, pero sigo pensando que vale la pena intentarlo. Tu sugerencia de mapear todo primero es clave - sin eso estoy disparando a ciegas.

---

## Documentación del Ambiente Universal PID

### ¿Qué devuelve el ambiente?

El ambiente implementa la interfaz estándar de Gymnasium y devuelve en cada step:
- **Observación**: Estado actual del sistema
- **Recompensa**: Evaluación de qué tan bien está funcionando el control  
- **Terminado**: Si el episodio finalizó exitosamente
- **Truncado**: Si el episodio fue cortado por mal desempeño
- **Info**: Metadata adicional sobre el proceso

### Las Observaciones (6 dimensiones)

El agente recibe un vector de 6 valores en cada step:

```python
observacion = [pv, setpoint, error, error_prev, error_integral, error_derivative]
```

**¿Por qué estos valores?**
1. **`pv` (Process Variable)**: El valor actual del proceso - lo que estamos midiendo
2. **`setpoint`**: El valor objetivo que queremos alcanzar
3. **`error`**: Diferencia entre setpoint y pv - el error actual
4. **`error_prev`**: Error del step anterior - para calcular tendencias
5. **`error_integral`**: Acumulación del error en el tiempo - detecta bias persistente
6. **`error_derivative`**: Velocidad de cambio del error - detecta si mejora/empeora

**Filosofía**: Le damos al agente la misma información que tendría un operador humano viendo los instrumentos - valor actual, objetivo, y el historial reciente para entender la tendencia.

### El Sistema de Recompensas (Multinivel)

La recompensa no es simplemente "llegaste al setpoint = +1". Es una combinación sofisticada:

#### 1. Recompensa Base (Por proximidad al setpoint)
```
Setpoint exacto (error = 0): +1.0 (máxima recompensa)
Dentro de banda muerta: +1.0 a +0.8 (graduada según precisión)
Fuera de rango operativo: -2.0 a -1.0 (penalización severa)  
En el medio: interpolación lineal según qué tan cerca esté
```

**Actualización importante**: Cambié la lógica para que la recompensa sea graduada dentro de la banda muerta. Ahora:
- **Error = 0** (setpoint exacto): Recompensa máxima (1.0)
- **Dentro de dead band**: Recompensa decrece linealmente de 1.0 a 0.8 según qué tan lejos del setpoint exacto esté
- **Motivación**: Evitar que el agente se "conforme" con estar cerca del setpoint sin intentar ser preciso

#### 2. Componentes tipo PID (Refinamiento)
- **Proporcional**: `-abs(error) × 0.1` → Penaliza error actual
- **Integral**: `-abs(error_integral) × 0.001` → Penaliza acumulación de error
- **Derivativo**: `-abs(error_derivative) × 0.1` → Penaliza cambios bruscos  
- **Energía**: `-abs(control_output) × 0.05` → Penaliza esfuerzo excesivo

**¿Por qué esta complejidad?** 
- No queremos solo llegar al setpoint, sino llegar de forma **estable** y **eficiente**
- Un control que oscila mucho puede technically alcanzar el setpoint pero ser horrible en la práctica
- Penalizamos el "gasto energético" porque en la industria la eficiencia importa

#### 3. Adaptación por Dificultad del Proceso

El ambiente ajusta la recompensa según qué tan difícil es el proceso:

**EASY** (ej: control de flujo):
- Banda muerta normal
- Penalización alta (-2.0) por salirse del rango
- Paciencia baja (20 steps) antes de truncar

**MEDIUM** (ej: control de temperatura):  
- Banda muerta 1.5x más grande (más tolerante)
- Penalización moderada (-1.5)
- Paciencia media (50 steps)

**DIFFICULT** (ej: control de composición):
- Banda muerta 2x más grande (muy tolerante)  
- Penalización baja (-1.0)
- Paciencia alta (100 steps)

**¿Por qué adaptar?** Porque un error de 0.1°C en temperatura puede ser excelente, pero 0.1% en composición puede ser terrible. El contexto importa.

### Criterios de Finalización

**Terminado = True (Éxito)**:
- El agente mantiene el error dentro de la banda muerta por suficiente tiempo
- Se considera que "dominó" el proceso

**Truncado = True (Fracaso)**:
- El proceso se sale del rango operativo por mucho tiempo
- El agente "perdió el control" y hay que cortarlo

**¿Por qué esta distinción?** En la industria real, salirse de rango puede ser peligroso/costoso. Un algoritmo que causa esto es peor que uno que simplemente no optimiza bien.

### Información Adicional (Info Dict)

En cada step también devuelve metadata útil:
```python
{
    'process_difficulty': 'EASY'|'MEDIUM'|'DIFFICULT'|'UNKNOWN',
    'dead_band': valor_actual_banda_muerta,  
    'time_constant': constante_tiempo_proceso,
    'step_count': pasos_transcurridos
}
```

Esta info no está disponible para el agente durante entrenamiento (sería trampa), pero es útil para análisis posterior y debugging.

### Resumen: ¿Por qué está diseñado así?

**Objetivo**: Crear un agente que no solo alcance el setpoint, sino que lo haga como lo haría un buen operador industrial:
- **Rápido pero sin sobrepasar** (penalización por oscilaciones)
- **Estable una vez que llega** (recompensa por mantenerse en banda muerta)  
- **Eficiente energéticamente** (penalización por control agresivo)
- **Adaptado al tipo de proceso** (tolerancias diferentes según dificultad)

La complejidad de la recompensa refleja la complejidad del problema real: en control industrial no basta con "funcionar", hay que funcionar bien.

---

## Mejora en el Sistema de Recompensas - 13 Sept 2025

### El Problema Detectado

Mientras revisaba la implementación de recompensas, me di cuenta de un issue importante en la lógica original:

**Problema**: La recompensa máxima (+1.0) se otorgaba por estar **dentro de la banda muerta**, no necesariamente en el setpoint exacto.

**Ejemplo problemático:**
- Setpoint: 50°C
- Dead band: ±2°C  
- PV = 48°C (error = 2°C, pero dentro de banda muerta)
- **Resultado**: Recompensa máxima (+1.0)

**¿Por qué es problemático?**
1. **El agente puede "conformarse"** con estar en 48°C en lugar de optimizar hacia 50°C
2. **No desarrolla precisión** - cualquier valor dentro del rango le da la misma recompensa
3. **Puede generar bias sistemático** - quedarse siempre 1-2°C alejado del setpoint
4. **Contradice el objetivo** de control preciso que buscamos

### La Solución Implementada

Cambié de una **recompensa binaria** (dentro/fuera) a una **recompensa graduada** dentro de la banda muerta:

**Código anterior:**
```python
if error_abs <= adjusted_dead_band:
    base_reward = max_reward  # Siempre +1.0
```

**Código nuevo:**
```python
if error_abs <= adjusted_dead_band:
    if error_abs == 0.0:
        precision_factor = 1.0  # Setpoint exacto
    else:
        # Decrece linealmente de 1.0 a 0.8
        precision_factor = 1.0 - (error_abs / adjusted_dead_band) * 0.2
    base_reward = max_reward * precision_factor
```

### Cómo Funciona la Nueva Lógica

**Ejemplos con dead_band = 2.0:**

| Error Absoluto | Factor Precisión | Recompensa Base | Comentario |
|----------------|------------------|-----------------|-------------|
| 0.0°C | 1.0 | 1.0 | Setpoint perfecto |
| 0.5°C | 0.95 | 0.95 | Muy cerca, excelente |
| 1.0°C | 0.90 | 0.90 | Bueno, pero puede mejorar |
| 1.5°C | 0.85 | 0.85 | Aceptable, necesita ajuste |
| 2.0°C | 0.80 | 0.80 | Límite de banda muerta |

### Beneficios Esperados

1. **Incentiva precisión**: El agente siempre tiene motivación para acercarse más al setpoint
2. **Evita conformismo**: No se queda satisfecho con "estar cerca"
3. **Mantiene tolerancia industrial**: Sigue siendo muy positiva dentro de la banda muerta (0.8-1.0)
4. **Graduación suave**: No hay saltos bruscos que confundan al agente

### Filosofía de Diseño

**Balance entre realismo industrial y optimización ML:**

- **Realismo**: En la industria, estar dentro de banda muerta ES aceptable
- **Optimización**: Para ML, queremos que siempre busque mejorar
- **Solución**: Recompensa alta pero graduada - satisface ambos objetivos

**Factor de graduación (0.2):**
- No muy agresivo (sigue siendo positiva la recompensa)
- Suficiente para crear gradiente de optimización
- Preserva estabilidad de entrenamiento

### Impacto en el Entrenamiento

**Expectativas:**
1. **Convergencia más lenta inicialmente** - el agente ya no se conforma fácil
2. **Mejor precisión final** - aprenderá a ser más exacto
3. **Menos variabilidad en steady state** - incentivo para mantenerse cerca del setpoint
4. **Mayor robustez** - no depende de "casualidades" dentro de banda muerta

Este cambio alinea mejor el ambiente con los objetivos reales de control: no solo estabilidad, sino también precisión.

---

## Reestructuración Arquitectura Modular - 13 Sept 2025

### El Problema con la Arquitectura Original

Cuando empecé a implementar las redes neuronales, creé `network.py` con todo mezclado en un solo archivo:
- PIDActorNetwork (específico para Actor-Critic)
- PIDCriticNetwork (específico para Actor-Critic) 
- SharedPIDNetwork (específico para PPO)

**Problema**: Esto funcionaría solo para PPO. Mi idea original era tener un `abstract_agent` que permitiera experimentar con diferentes algoritmos de RL.

### La Nueva Arquitectura Modular

Reestructuré todo el código siguiendo el patrón de **Abstract Agent + Componentes Modulares**:

```
agent/
├── base_agent.py              # Contratos/interfaces abstractas
├── networks/
│   ├── base_networks.py       # Piezas LEGO reutilizables
│   ├── actor_critic.py        # Redes para policy gradient
│   ├── q_networks.py          # Redes para value-based (pendiente)
│   └── policy_networks.py     # Redes para actor-critic (pendiente)
├── algorithms/
│   ├── ppo_agent.py           # Implementación PPO (pendiente)
│   ├── dqn_agent.py           # Implementación DQN (pendiente)
│   └── sac_agent.py           # Implementación SAC (pendiente)
```

### Las Clases Abstractas (El Contrato)

**`AbstractPIDAgent`** define la interfaz que TODOS los agentes deben implementar:

```python
@abstractmethod
def select_action(state) -> action:
    # "Dame estado, devuelve PID params"
    pass

@abstractmethod  
def update(batch_data) -> metrics:
    # "Dame experiencias, mejora parámetros"
    pass
```

**Especializaciones por familia de algoritmos:**

```python
# Para PPO, A2C, REINFORCE
class AbstractPolicyGradientAgent(AbstractPIDAgent):
    @abstractmethod
    def compute_policy_loss(...)
    def compute_value_loss(...)

# Para DQN, DDQN (con discretización)
class AbstractValueBasedAgent(AbstractPIDAgent):
    @abstractmethod  
    def compute_q_loss(...)
    def get_epsilon(...)

# Para DDPG, TD3, SAC
class AbstractActorCriticAgent(AbstractPIDAgent):
    @abstractmethod
    def compute_actor_loss(...)
    def compute_critic_loss(...)
```

### Componentes de Red Modulares (Piezas LEGO)

En lugar de redes monolíticas, ahora tengo **componentes reutilizables**:

#### **Base Networks (base_networks.py):**
```python
# 🧩 Extractor de características genérico
FeatureExtractor(
    input_dim=6,           # Estado proceso
    hidden_dims=[128, 64], # Arquitectura
    dropout_rate=0.1       # Regularización
)

# 🧩 Salida específica para PID (garantiza rangos válidos)
PIDOutputLayer(
    kp_range=(0.1, 10.0),  # Rango Kp
    ki_range=(0.01, 5.0),  # Rango Ki  
    kd_range=(0.001, 2.0)  # Rango Kd
)

# 🧩 Cabezas especializadas
ValueHead()      # Para críticos
QValueHead()     # Para Q-learning  
DuelingHead()    # Para Dueling DQN
```

#### **Actor-Critic Networks (actor_critic.py):**
```python
# Actores y críticos combinando piezas base
ActorNetwork = FeatureExtractor + PIDOutputLayer
CriticNetwork = FeatureExtractor + ValueHead
SharedActorCritic = FeatureExtractor_compartido + 2_cabezas

# Actores estocásticos para PPO
StochasticActor = FeatureExtractor + PIDOutputLayer + LogStdHead
```

### Factory Pattern y Configuración

**PIDAgentConfig**: Configuración unificada para todos los algoritmos
```python
config = PIDAgentConfig(
    # Red
    hidden_dims=[128, 64],
    dropout_rate=0.1,
    
    # PID ranges
    kp_range=(0.1, 10.0),
    
    # Training
    learning_rate=3e-4,
    batch_size=64,
    
    # Algoritmo específico (kwargs)
    ppo_epochs=10,      # Para PPO
    clip_ratio=0.2,     # Para PPO
    epsilon=0.1         # Para DQN
)
```

**Factory function**: Crear cualquier agente fácilmente
```python
ppo_agent = create_agent('ppo', config)
dqn_agent = create_agent('dqn', config)  
sac_agent = create_agent('sac', config)
# Todos implementan la misma interfaz!
```

### Ventajas de la Nueva Arquitectura

#### **1. Reutilización de Componentes**
```python
# Mismo FeatureExtractor para PPO, DQN, SAC
features = FeatureExtractor(input_dim=6, hidden_dims=[128, 64])

# Mismo PIDOutputLayer pero con rangos diferentes
conservative_output = PIDOutputLayer(kp_range=(0.1, 2.0))
aggressive_output = PIDOutputLayer(kp_range=(1.0, 20.0))
```

#### **2. Intercambiabilidad de Algoritmos**
```python
# Cambio de algoritmo sin cambiar resto del código
agent = create_agent('ppo', config)    # Empieza con PPO
# Si no funciona bien...
agent = create_agent('sac', config)    # Prueba SAC
# Misma interfaz: select_action(), update(), save(), load()
```

#### **3. Facilidad para Experimentar**
```python
# Probar diferentes arquitecturas
config_small = PIDAgentConfig(hidden_dims=[64, 32])
config_large = PIDAgentConfig(hidden_dims=[256, 128, 64])

# Probar diferentes rangos PID
config_conservative = PIDAgentConfig(kp_range=(0.1, 2.0))
config_aggressive = PIDAgentConfig(kp_range=(1.0, 15.0))
```

#### **4. Testabilidad**
```python
# Cada componente se prueba por separado
feature_extractor = FeatureExtractor()
test_input = torch.randn(32, 6)
features = feature_extractor(test_input)  # ✅ 

pid_output = PIDOutputLayer()
pid_params = pid_output(features)         # ✅
```

### Cómo Implementar un Nuevo Algoritmo

**Ejemplo: PPOAgent**
```python
class PPOAgent(AbstractPolicyGradientAgent):
    def __init__(self, config: PIDAgentConfig):
        super().__init__()
        
        # Usar componentes modulares
        self.actor = StochasticActor(
            kp_range=config.kp_range,
            ki_range=config.ki_range,
            kd_range=config.kd_range
        )
        self.critic = CriticNetwork()
        
        # PPO específico
        self.clip_ratio = config.clip_ratio
        self.ppo_epochs = config.ppo_epochs
    
    def select_action(self, state):
        # Implementación específica PPO
        action, log_prob = self.actor.get_action_and_log_prob(state)
        return self.postprocess_action(action)
    
    def update(self, batch_data):
        # Algoritmo PPO con clipping, etc.
        policy_loss = self.compute_policy_loss(...)
        value_loss = self.compute_value_loss(...)
        # ...
```

### Flujo de Trabajo Simplificado

```python
# 1. Configurar experimento
config = PIDAgentConfig(
    learning_rate=0.001,
    batch_size=64,
    kp_range=(0.1, 5.0),
    ppo_epochs=10  # PPO específico
)

# 2. Crear agente
agent = create_agent('ppo', config)

# 3. Entrenar (interfaz idéntica para todos los algoritmos)
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Mismo método para todos los agentes
        action = agent.select_action(state)  
        next_state, reward, done, info = env.step(action)
        
        # Mismo método para todos los agentes
        batch_data = collect_batch(...)
        metrics = agent.update(batch_data)
```

### Comparación: Antes vs Ahora

**Antes (Monolítico):**
```python
network.py  # 300+ líneas, solo PPO, difícil de extender
```

**Ahora (Modular):**
```python
# Código organizado, extensible, reutilizable
base_agent.py (200 líneas)      # Contratos
base_networks.py (300 líneas)   # Componentes
actor_critic.py (200 líneas)    # Policy gradient específico
+ algorithms/ (por implementar) # Algoritmos específicos
```

### Próximos Pasos

1. **Implementar PPOAgent** como primer algoritmo concreto
2. **Crear algorithms/ppo_agent.py** usando los componentes modulares
3. **Probar end-to-end** training con el ambiente universal
4. **Implementar DQNAgent y SACAgent** para comparar algoritmos
5. **Crear sistema de benchmarking** para evaluar transferibilidad

Esta arquitectura me da la flexibilidad que necesitaba para experimentar con diferentes algoritmos de RL mientras mantengo código limpio y reutilizable. Es la base sólida para mi investigación de transfer learning entre procesos industriales.
