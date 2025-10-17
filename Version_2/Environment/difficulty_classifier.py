from typing import Optional


class ProcessDifficultyClassifier:
    """
    Clasifica la dificultad del proceso según su tiempo de respuesta.
    
    Categorías:
    - EASY: < 60s (procesos rápidos)
    - MEDIUM: 60s - 1800s (procesos normales)
    - DIFFICULT: >= 1800s (procesos lentos, ej: temperatura)
    - UNKNOWN: Sin suficiente información
    
    Args:
        easy_threshold: Umbral para procesos fáciles [s]
        difficult_threshold: Umbral para procesos difíciles [s]
    """
    
    # Umbrales por defecto
    EASY_THRESHOLD = 60.0        # < 1 min
    DIFFICULT_THRESHOLD = 1800.0  # >= 30 min
    
    def __init__(self, 
                 easy_threshold: float = EASY_THRESHOLD,
                 difficult_threshold: float = DIFFICULT_THRESHOLD):
        
        if easy_threshold >= difficult_threshold:
            raise ValueError(
                "easy_threshold debe ser menor que difficult_threshold"
            )
        
        self.easy_threshold = easy_threshold
        self.difficult_threshold = difficult_threshold
        
        self.current_difficulty = "UNKNOWN"
    
    def classify(self, response_time: Optional[float]) -> str:
        """
        Clasificar dificultad según tiempo de respuesta.
        
        Args:
            response_time: Tiempo de respuesta estimado [s]
        
        Returns:
            Categoría: 'EASY', 'MEDIUM', 'DIFFICULT', 'UNKNOWN'
        """
        if response_time is None:
            return "UNKNOWN"
        
        if response_time < self.easy_threshold:
            self.current_difficulty = "EASY"
        elif response_time < self.difficult_threshold:
            self.current_difficulty = "MEDIUM"
        else:
            self.current_difficulty = "DIFFICULT"
        
        return self.current_difficulty
    
    def get_difficulty(self) -> str:
        """Obtener dificultad actual."""
        return self.current_difficulty
    
    def reset(self) -> None:
        """Resetear a estado desconocido."""
        self.current_difficulty = "UNKNOWN"