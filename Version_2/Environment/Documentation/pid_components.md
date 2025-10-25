"""
DESCRIPCIÓN DEL MÓDULO:
Contiene componentes reutilizables para implementar control PID y su integración 
con reinforcement learning. Incluye un controlador PID clásico con anti-windup, 
un codificador de acciones discretas para ajuste incremental de parámetros, y 
un detector automático de tiempo de respuesta del proceso basado en observación 
de cambios control-PV.
"""


class PIDController:
    """
    DESCRIPCIÓN DE LA CLASE:
    Implementa controlador PID clásico con tres términos (proporcional, integral, 
    derivativo) y mecanismo anti-windup para evitar saturación del integrador. 
    Calcula señal de control con límites configurables y mantiene estado interno 
    (integral acumulada, error previo) para cálculo del término derivativo.
    """
    
    # ====== MÉTODO PRINCIPAL ======
    
    __init__(kp, ki, kd, dt, output_limits):
        """
        Inicializador: Configura ganancias PID (kp, ki, kd), paso de tiempo dt, 
        límites de saturación output_limits (min, max) e inicializa estado interno 
        (integral=0, prev_error=0) para tracking entre iteraciones.
        """
    
    # ====== MÉTODOS DE CONTROL ======
    
    compute(error):
        """
        Calcula salida PID: Computa término proporcional (P = kp*error), integral 
        (I = ki*integral acumulada), derivativo (D = kd*(error-prev_error)/dt), 
        suma términos (P+I+D), aplica límites con clip, implementa anti-windup 
        (revierte acumulación si hay saturación), actualiza prev_error y retorna 
        control_output saturado.
        """
    
    update_gains(kp, ki, kd):
        """
        Actualiza ganancias PID: Modifica parámetros kp, ki, kd en tiempo real 
        sin resetear estado interno. Útil para ajuste online de parámetros por 
        agente RL o adaptación manual.
        """
    
    # ====== MÉTODOS AUXILIARES ======
    
    reset():
        """
        Reinicia estado interno: Resetea integral acumulada y prev_error a cero. 
        Se invoca al inicio de cada episodio para limpiar historial y evitar 
        carry-over entre experimentos.
        """
    
    get_state():
        """
        Obtiene estado del controlador: Retorna diccionario con ganancias actuales 
        (kp, ki, kd) y variables internas (integral, prev_error). Útil para 
        debugging, logging o serialización del controlador.
        """


class DeltaPIDActionSpace:
    """
    DESCRIPCIÓN DE LA CLASE:
    Codificador de acciones discretas para ajuste incremental de parámetros PID. 
    En lugar de valores absolutos, las acciones representan cambios relativos 
    (±delta_percent) sobre valores actuales. Define 7 acciones discretas (aumentar/ 
    disminuir Kp/Ki/Kd o mantener) con límites configurables por parámetro para 
    exploración segura.
    """
    
    # ====== MÉTODO PRINCIPAL ======
    
    __init__(initial_pid, delta_percent, limits):
        """
        Inicializador: Configura PID inicial (Kp, Ki, Kd), porcentaje de cambio 
        delta_percent (ej: 0.2 = 20%), límites por parámetro (min, max), crea 
        mapeo de 7 acciones discretas (0-2: aumentar Kp/Ki/Kd, 3-5: disminuir, 
        6: mantener) y copia initial_pid a current_pid para tracking.
        """
    
    # ====== MÉTODOS DE ACCIÓN ======
    
    apply_action(action_idx):
        """
        Aplica acción discreta: Decodifica action_idx (0-6) a modificación de 
        parámetro específico, calcula multiplicador (1±delta), actualiza current_pid, 
        aplica límites con clip para evitar valores inválidos y retorna tupla 
        (Kp, Ki, Kd) actualizada.
        """
    
    get_action_description(action_idx):
        """
        Obtiene descripción legible: Convierte action_idx numérico a string 
        descriptivo (ej: "Kp ↑ 20%", "Ki ↓ 20%", "Mantener PID"). Útil para 
        logging, debugging y explicabilidad del comportamiento del agente.
        """
    
    # ====== MÉTODOS AUXILIARES ======
    
    reset(pid):
        """
        Reinicia parámetros PID: Resetea current_pid a initial_pid por defecto, 
        o a tupla personalizada si se proporciona. Se invoca al inicio de episodios 
        para partir desde configuración conocida.
        """
    
    get_current_pid():
        """
        Obtiene PID actual: Retorna tupla (Kp, Ki, Kd) con valores actuales del 
        controlador. Permite consultar estado sin modificarlo, útil para 
        observaciones o logging.
        """


class ResponseTimeDetector:
    """
    DESCRIPCIÓN DE LA CLASE:
    Detector automático de tiempo de respuesta del proceso mediante observación 
    de relación temporal entre cambios en señal de control y respuesta en PV. 
    Identifica cambios significativos en control, mide tiempo hasta respuesta 
    observable en PV (>10% del cambio esperado), estima constante de tiempo (τ) 
    usando aproximación 3τ≈95% respuesta, y retorna mediana de últimas 5 
    estimaciones para robustez ante ruido.
    """
    
    # ====== MÉTODO PRINCIPAL ======
    
    __init__():
        """
        Inicializador: Invoca reset() para inicializar todas las estructuras de 
        datos vacías (listas de cambios de control, respuestas PV, timestamps, 
        estimaciones) y variables de tracking (current_time, 
        last_significant_control_change).
        """
    
    # ====== MÉTODOS DE DETECCIÓN ======
    
    update(control_output, pv, setpoint, dt):
        """
        Actualiza detector con medición: Incrementa tiempo acumulado, detecta 
        cambios significativos en control (>0.1), registra timestamp y valor inicial 
        de PV, calcula tiempo transcurrido desde último cambio, detecta respuesta 
        en PV (>10% del cambio esperado hacia setpoint), estima tiempo de respuesta 
        (~3τ), agrega a historial, registra datos en listas y retorna mediana de 
        últimas 5 estimaciones o None si hay <2 mediciones.
        """
    
    # ====== MÉTODO AUXILIAR ======
    
    reset():
        """
        Reinicia detector: Limpia todas las listas de historial (control_changes, 
        pv_responses, time_stamps, response_time_estimates), resetea current_time 
        a cero y limpia last_significant_control_change. Se invoca al inicio de 
        cada episodio.
        """