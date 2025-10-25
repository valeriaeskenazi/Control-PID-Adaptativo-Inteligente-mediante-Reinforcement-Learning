class EpisodeMetricsTracker:
    """
    DESCRIPCIÓN DE LA CLASE:
    Rastrea y gestiona métricas de rendimiento a lo largo de múltiples episodios 
    de entrenamiento. Mantiene estadísticas acumuladas (total episodios, tasa de 
    éxito) y promedios móviles (settling time, error estacionario, recompensas) 
    usando historial configurable. Calcula estadísticas resumidas y proporciona 
    tracking en tiempo real del episodio actual.
    """
    
    # ====== MÉTODO PRINCIPAL ======
    
    __init__(history_size):
        """
        Inicializador: Configura tamaño del historial de métricas recientes 
        (default 100 episodios), inicializa contadores acumulados (episodios 
        totales/exitosos), promedios móviles (settling time, error, recompensa), 
        crea deques para historial reciente (recompensas, tiempos, errores) y 
        variables para tracking del episodio actual (recompensa, steps).
        """
    
    # ====== MÉTODOS DE GESTIÓN DE EPISODIO ======
    
    start_episode():
        """
        Inicia tracking de episodio: Resetea acumuladores del episodio actual 
        (current_episode_reward y current_episode_steps a cero). Se invoca al 
        inicio de cada reset() del ambiente.
        """
    
    update_step(reward):
        """
        Actualiza métricas por step: Acumula recompensa del step actual en 
        current_episode_reward e incrementa contador de steps. Se invoca en 
        cada step() del ambiente para mantener tracking en tiempo real.
        """
    
    end_episode(settling_time, steady_state_error, success):
        """
        Finaliza episodio y actualiza estadísticas: Incrementa contador de 
        episodios totales/exitosos, agrega métricas al historial (recompensas, 
        settling time, errores), recalcula promedios móviles con numpy.mean() 
        y retorna diccionario resumen con datos del episodio terminado.
        """
    
    # ====== MÉTODOS DE CONSULTA ======
    
    get_metrics():
        """
        Obtiene todas las métricas actuales: Retorna diccionario completo con 
        contadores (episodios totales/exitosos), tasa de éxito calculada, 
        promedios móviles (settling time, error, recompensa) e historiales 
        completos recientes convertidos a listas. Útil para análisis detallado.
        """
    
    get_summary_stats():
        """
        Obtiene estadísticas resumidas: Retorna diccionario compacto sin 
        historiales largos, solo con promedios globales y estadísticas de 
        últimos 10 episodios (media y desviación estándar de recompensas). 
        Optimizado para logging y monitoreo rápido.
        """
    
    # ====== MÉTODO DE RESET ======
    
    reset():
        """
        Reinicia todas las métricas: Resetea contadores acumulados a cero, 
        limpia promedios móviles, vacía deques de historial y resetea variables 
        de episodio actual. Se usa para comenzar nuevo experimento desde cero, 
        no entre episodios normales.
        """