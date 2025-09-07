# Clasificación de Procesos Industriales para Transfer Learning

## Tabla de Procesos con Características de Control

| Proceso | Tiempo de Respuesta | Tipo de Comportamiento | Tiempo Muerto | Calificación |
|---------|-------------------|----------------------|---------------|-------------|
| **NIVEL FÁCIL** |
| Control de velocidad de motor DC | < 1 segundo | Lineal, 1er orden | Despreciable | Fácil |
| Control de posición de servo | < 0.5 segundos | Lineal, 2do orden | Despreciable | Fácil |
| Control de torque en motor | < 0.2 segundos | Lineal, respuesta inmediata | Despreciable | Fácil |
| Control de flujo con válvula | < 5 segundos | Aproximadamente lineal | < 2 segundos | Fácil |
| Control de flujo en bomba centrífuga | 5-15 segundos | Lineal en rango operativo | < 5 segundos | Fácil |
| Control de caudal en sistema hidráulico | 2-8 segundos | Lineal | < 3 segundos | Fácil |
| Control de temperatura en resistencia eléctrica | 30-60 segundos | 1er orden puro | Despreciable | Fácil |
| Calentamiento de pequeño volumen de agua | 1-3 minutos | 1er orden | < 10 segundos | Fácil |
| Control de presión en compresor | 2-10 segundos | Lineal | < 5 segundos | Fácil |
| Control de presión en sistema neumático | 1-5 segundos | Lineal | < 2 segundos | Fácil |
| **NIVEL MEDIO** |
| Control de temperatura en intercambiador | 5-15 minutos | 1er orden + deadtime | 30-120 segundos | Medio |
| Control de temperatura en tanque agitado | 10-30 minutos | 1er orden, no-lineal leve | 1-3 minutos | Medio |
| Control de temperatura en serpentín | 5-20 minutos | 1er orden + deadtime | 2-8 minutos | Medio |
| Climatización HVAC | 15-45 minutos | Múltiples τ, interacciones | 2-10 minutos | Medio |
| Control de pH con ácido/base | 30 segundos - 2 minutos | Fuertemente no-lineal | 10-30 segundos | Medio |
| Control de conductividad | 1-5 minutos | Moderadamente no-lineal | 20-60 segundos | Medio |
| Control de concentración por dilución | 2-10 minutos | No-lineal, dinámicas mezcla | 30 segundos - 3 minutos | Medio |
| Control de nivel en tanque con salida variable | 5-30 minutos | Integrativo + perturbaciones | Variable | Medio |
| Control de nivel en múltiples tanques | 10-60 minutos | Integrativo, interacciones | 1-5 minutos | Medio |
| Control de interface en separador trifásico | 5-20 minutos | Complejo, múltiples fases | 2-8 minutos | Medio |
| Control de tensión en bobinado | 1-10 segundos | No-lineal por elasticidad | < 1 segundo | Medio |
| Control de espesor en laminado | 10-60 segundos | No-lineal, deadtime transporte | 5-30 segundos | Medio |
| **NIVEL DIFÍCIL** |
| Control de temperatura en horno industrial | 30-120 minutos | Fuertemente no-lineal | 10-30 minutos | Difícil |
| Control de temperatura en reactor exotérmico | 10-60 minutos | No-lineal, riesgo runaway | 2-15 minutos | Difícil |
| Control de temperatura en intercambiador multi-paso | 20-80 minutos | Múltiples τ, interacciones | 5-20 minutos | Difícil |
| Temple de metales | 30 minutos - 2 horas | Complejo, control de rampa | 5-30 minutos | Difícil |
| Control de presión de vapor en caldera | 5-30 minutos | Multivariable acoplado | 1-10 minutos | Difícil |
| Control de composición en destilación | 30 minutos - 2 horas | Multivariable, interacciones | 15-60 minutos | Difícil |
| Control de pureza en cristalizador | 1-6 horas | Extremadamente complejo | 10-45 minutos | Difícil |
| Control de concentración en evaporador | 20-90 minutos | Múltiples efectos | 5-25 minutos | Difícil |
| Control de conversión en reactor continuo | 15 minutos - 2 horas | Cinética compleja | 5-30 minutos | Difícil |
| Control de peso molecular en polimerización | 1-8 horas | Extremadamente no-lineal | 20 minutos - 2 horas | Difícil |
| Control de selectividad en reactor catalítico | 30 minutos - 4 horas | Múltiples variables | 10-60 minutos | Difícil |
| Control de pH en bioreactor | 1-15 minutos | Dinámicas biológicas | 30 segundos - 5 minutos | Difícil |
| Control de oxígeno disuelto en fermentación | 30 segundos - 5 minutos | No-lineal, acoplado | 10 segundos - 2 minutos | Difícil |
| Control de biomasa en cultivo celular | 2-24 horas | Cinéticas complejas | 30 minutos - 4 horas | Difícil |
| Control de temperatura en colada continua | 5-45 minutos | Dinámicas complejas | 3-20 minutos | Difícil |
| Control de composición en horno acería | 30 minutos - 3 horas | Múltiples variables | 10-45 minutos | Difícil |
| Control de espesor en laminación caliente | 1-15 segundos | Acoplamiento térmico-mecánico | 0.5-5 segundos | Difícil |
| Control de octanaje en reforming catalítico | 2-12 horas | Extremadamente complejo | 30 minutos - 3 horas | Difícil |
| Control de conversión en cracking catalítico | 1-6 horas | Desactivación catalizador | 15 minutos - 2 horas | Difícil |

## Criterios de Clasificación

**Fácil:** Deadtime < 30s, comportamiento lineal, constante de tiempo única, sin interacciones
**Medio:** Deadtime 30s-10min, no-linealidades moderadas, múltiples constantes de tiempo
**Difícil:** Deadtime > 10min, fuertemente no-lineal, multivariable acoplado, cinéticas complejas

## Estrategia de Selección

Elegir 3 procesos de cada nivel con:
1. Documentación abundante en literatura
2. Modelos matemáticos conocidos  
3. Representatividad industrial
4. Diversidad de dinámicas dentro del mismo nivel
