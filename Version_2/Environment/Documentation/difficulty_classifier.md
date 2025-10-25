class ProcessDifficultyClassifier:
    """
    DESCRIPCIÓN DE LA CLASE:
    Clasifica la dificultad del proceso de control según su tiempo de respuesta 
    estimado. Categoriza en 4 niveles (EASY <60s, MEDIUM 60-1800s, DIFFICULT ≥1800s, 
    UNKNOWN sin info) para adaptar estrategias de control y recompensas. Utiliza 
    umbrales configurables que reflejan la velocidad de respuesta del sistema físico.
    """
    
    # ====== CONSTANTES DE CLASE ======
    
    EASY_THRESHOLD = 60.0:
        """
        Umbral para procesos fáciles: Define límite superior (en segundos) para 
        clasificar procesos como rápidos. Por defecto 60s (1 minuto). Procesos 
        con respuesta más rápida son más fáciles de controlar.
        """
    
    DIFFICULT_THRESHOLD = 1800.0:
        """
        Umbral para procesos difíciles: Define límite inferior (en segundos) para 
        clasificar procesos como lentos. Por defecto 1800s (30 minutos). Procesos 
        con respuesta más lenta (ej: temperatura) requieren mayor paciencia.
        """
    
    # ====== MÉTODOS PRINCIPALES ======
    
    __init__(easy_threshold, difficult_threshold):
        """
        Inicializador: Configura umbrales personalizables para clasificación, 
        valida que easy_threshold < difficult_threshold para evitar inconsistencias, 
        e inicializa dificultad actual en estado "UNKNOWN" hasta tener mediciones.
        """
    
    classify(response_time):
        """
        Clasifica dificultad del proceso: Evalúa tiempo de respuesta estimado 
        contra umbrales configurados y retorna categoría correspondiente 
        ('EASY', 'MEDIUM', 'DIFFICULT', 'UNKNOWN'). Actualiza estado interno 
        current_difficulty. Si response_time es None, retorna 'UNKNOWN'.
        """
    
    get_difficulty():
        """
        Obtiene dificultad actual: Retorna la última categoría clasificada sin 
        realizar nuevos cálculos. Útil para consultar estado entre llamadas a 
        classify() o cuando no hay nuevo response_time disponible.
        """
    
    reset():
        """
        Reinicia clasificador: Resetea dificultad actual a estado "UNKNOWN". 
        Se invoca al inicio de cada episodio para limpiar información del 
        episodio anterior y partir desde cero.
        """