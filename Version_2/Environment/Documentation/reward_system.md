class AdaptiveRewardCalculator:
    """
    DESCRIPCIÓN DE LA CLASE:
    Calcula recompensas adaptativas que se ajustan según la dificultad del proceso 
    controlado. Combina cuatro componentes (proporcional al error, integral del 
    error acumulado, derivativo de tasa de cambio, penalización de energía de 
    control) con parámetros específicos por dificultad (EASY/MEDIUM/DIFFICULT/UNKNOWN) 
    que ajustan tolerancias, recompensas máximas/mínimas y penalizaciones para 
    promover aprendizaje apropiado según velocidad de respuesta del proceso.
    """
    
    # ====== MÉTODO PRINCIPAL ======
    
    __init__(upper_range, lower_range, dead_band):
        """
        Inicializador: Almacena límites físicos del proceso (upper_range, lower_range) 
        y banda muerta aceptable (dead_band), define diccionario difficulty_params 
        con 4 conjuntos de parámetros (max_reward, min_positive, negative_reward, 
        tolerance_factor) específicos para cada nivel de dificultad del proceso.
        """
    
    # ====== MÉTODO DE CÁLCULO PRINCIPAL ======
    
    calculate(pv, setpoint, error, error_integral, error_derivative, control_output, process_difficulty):
        """
        Calcula recompensa adaptativa completa: Obtiene parámetros según 
        process_difficulty, ajusta dead_band con tolerance_factor, calcula 
        base_reward mediante _calculate_base_reward(), suma componentes PID 
        ponderados (proporcional -error*0.1, integral -integral*0.001, derivativo 
        -derivative*0.1) y penalización energética (-control*0.05), aplica clip 
        entre negative_reward y max_reward y retorna float total.
        """
    
    # ====== MÉTODO AUXILIAR ======
    
    _calculate_base_reward(pv, setpoint, error_abs, adjusted_dead_band, params):
        """
        Calcula recompensa base graduada: Si error dentro de dead_band ajustada, 
        retorna max_reward con factor de precisión (1.0 - error/dead_band*0.2). 
        Si PV fuera de rango físico, retorna negative_reward. Si en rango pero 
        fuera de dead_band, interpola linealmente entre max_reward y min_positive 
        según normalized_error respecto a distancia máxima al límite del rango.
        """
    
    # ====== MÉTODO DE CONSULTA ======
    
    get_difficulty_params(difficulty):
        """
        Obtiene parámetros de dificultad: Retorna diccionario con parámetros 
        específicos (max_reward, min_positive, negative_reward, tolerance_factor) 
        para nivel de dificultad solicitado. Si difficulty no existe, retorna 
        parámetros de 'UNKNOWN' como fallback seguro.
        """