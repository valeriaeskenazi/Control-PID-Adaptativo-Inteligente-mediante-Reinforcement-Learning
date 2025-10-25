class BasePIDControlEnv(gym.Env, ABC):
    """
    DESCRIPCIÓN DE LA CLASE:
    Clase base abstracta que implementa la estructura principal de un ambiente 
    Gymnasium para control PID con aprendizaje por refuerzo. Gestiona el ciclo 
    de vida del ambiente (reset, step, render) y delega responsabilidades 
    específicas a módulos especializados: clasificación de dificultad, cálculo 
    de recompensas adaptativas, tracking de métricas y detección de tiempo de 
    respuesta. Las clases hijas deben implementar el espacio de acciones y la 
    dinámica específica del proceso a controlar.
    """
    
    # ====== MÉTODOS PRINCIPALES ======
    
    __init__(config):
        """
        Inicializador: Define configuración física por defecto (rangos, setpoint, 
        dt), valida parámetros, inicializa variables de estado del proceso (PV, 
        errores, integrales), configura espacio de observaciones (6D), e instancia 
        módulos especializados (ResponseTimeDetector, ProcessDifficultyClassifier, 
        AdaptiveRewardCalculator, EpisodeMetricsTracker) y sistema de logging.
        """
    
    step(action):
        """
        Ejecuta un paso de simulación: Aplica acción de control, actualiza 
        dinámica del proceso, calcula error y sus componentes (integral, derivada), 
        detecta tiempo de respuesta, clasifica dificultad del proceso, calcula 
        recompensa adaptativa, actualiza métricas, verifica condiciones de 
        terminación (truncated/terminated) y retorna tupla Gymnasium 
        (obs, reward, terminated, truncated, info).
        """
    
    reset(seed, options):
        """
        Reinicia el ambiente: Resetea variables de estado (errores, integrales, 
        contadores), limpia historial, reinicia módulos especializados 
        (detector, clasificador), inicia tracking de nuevo episodio y retorna 
        observación inicial con info dict vacío.
        """
    
    render(mode):
        """
        Visualización del estado: Imprime en consola (mode='human') información 
        del paso actual incluyendo step count, PV, setpoint, error, integral, 
        derivada y dificultad del proceso. Útil para debugging y monitoreo manual.
        """
    
    # ====== MÉTODOS ABSTRACTOS (implementar en clases hijas) ======
    
    _setup_action_space():
        """
        Define espacio de acciones: Método abstracto que debe implementar cada 
        clase hija para especificar el tipo de acciones (continuas, discretas, 
        multi-discretas) según la estrategia de control (ajuste directo, 
        parámetros PID, señal de control).
        """
    
    _apply_control(action):
        """
        Aplica acción al proceso: Método abstracto que traduce la acción del 
        agente en señal de control o parámetros PID. Retorna tupla 
        (control_output, pid_params) donde alguno puede ser None según el 
        modo de control implementado.
        """
    
    _update_process(control_output, pid_params):
        """
        Actualiza dinámica del proceso: Método abstracto que simula el 
        comportamiento físico del sistema controlado. Recibe señal de control 
        y/o parámetros PID, aplica ecuaciones diferenciales o modelo del proceso 
        y retorna nuevo valor de PV (process variable).
        """
    
    # ====== MÉTODOS DE CONFIGURACIÓN Y VALIDACIÓN ======
    
    _validate_config(config):
        """
        Valida parámetros: Verifica que el diccionario de configuración contenga 
        todas las claves requeridas y que los valores cumplan restricciones 
        lógicas (upper > lower, dead_band >= 0, steps > 0, dt > 0). 
        Lanza ValueError si hay inconsistencias.
        """
    
    _setup_observation_space():
        """
        Define espacio de observaciones: Configura Box space de 6 dimensiones 
        (PV, setpoint, error, error_prev, error_integral, error_derivative) 
        con límites adaptativos basados en rangos físicos del proceso. 
        Incluye márgenes de seguridad (±20% en PV).
        """
    
    _setup_logging(log_level):
        """
        Configura sistema de logs: Crea logger específico para la clase, 
        establece nivel de verbosidad (INFO, DEBUG, etc.), configura handler 
        de consola con formato timestamp y evita duplicación de handlers.
        """
    
    # ====== MÉTODOS AUXILIARES ======
    
    _get_observation():
        """
        Construye observación actual: Calcula error como (setpoint - PV) y 
        empaqueta vector numpy de 6 elementos con estado completo del proceso 
        (PV, SP, error, error_prev, integral, derivada) en formato float32.
        """
    
    _check_truncation(error, process_difficulty):
        """
        Verifica criterios de truncamiento: Evalúa si el error excede umbrales 
        adaptativos según dificultad del proceso (EASY: 50%, MEDIUM: 70%, 
        DIFFICULT: 100% del rango). Implementa sistema de paciencia (20-100 steps) 
        antes de truncar episodio por inestabilidad.
        """
    
    # ====== MÉTODOS PÚBLICOS DE UTILIDAD ======
    
    set_setpoint(new_setpoint):
        """
        Cambia setpoint dinámicamente: Valida que el nuevo valor objetivo esté 
        dentro de rangos físicos permitidos, actualiza self.setpoint y registra 
        cambio en logs. Útil para simular perturbaciones o cambios de referencia.
        """
    
    get_metrics():
        """
        Obtiene métricas del tracker: Retorna diccionario con estadísticas del 
        episodio actual desde EpisodeMetricsTracker (recompensas acumuladas, 
        promedios, errores, etc.).
        """
    
    get_summary_stats():
        """
        Obtiene estadísticas resumidas: Retorna diccionario con métricas agregadas 
        de todos los episodios ejecutados (promedios globales, mejor/peor episodio, 
        tendencias) desde EpisodeMetricsTracker.
        """