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
