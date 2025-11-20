"""
Entrenador de controladores PID con validación de estabilidad.

Este módulo encapsula la lógica repetitiva de:
1. Entrenar agente RL durante n episodios
2. Validar PID resultante con criterios de estabilidad
3. Reintentar hasta j veces si no cumple
4. Retornar mejor PID encontrado
"""

import numpy as np
from typing import Tuple, Optional


class PIDTrainer:
    """
    Entrenador de PIDs con validación automática.
    
    Ejecuta el ciclo de entrenamiento-validación para encontrar
    el mejor PID según criterios de estabilidad.
    
    Args:
        stability_criteria: Instancia de StabilityCriteria para validación
    """
    
    def __init__(self, stability_criteria):
        """
        Inicializar entrenador.
        
        Args:
            stability_criteria: Objeto con método validate() que retorna (bool, float)
        """
        self.stability_criteria = stability_criteria
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'total_trainings': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'best_errors': []
        }
    
    def find_best_pid(self,
                      agent,
                      env,
                      var_idx: int,
                      setpoint: float,
                      n_episodes: int,
                      j_max_retries: int,
                      verbose: bool = True) -> Tuple[Tuple[float, float, float], float]:
        """
        Encontrar el mejor PID mediante entrenamiento y validación.
        
        Algoritmo:
        1. Ciclo hasta j_max_retries:
           a. Entrenar agente por n_episodes
           b. Obtener PID resultante
           c. Validar con criterios de estabilidad
           d. Si cumple → retornar PID
           e. Si no cumple → guardar si es mejor que anterior
        2. Después de j intentos → retornar mejor PID encontrado
        
        Args:
            agent: Agente RL con método train(env, n_episodes, var_idx, setpoint)
            env: Ambiente base para entrenamiento
            var_idx: Índice de la variable a controlar
            setpoint: Setpoint objetivo para la variable
            n_episodes: Número de episodios de entrenamiento (n)
            j_max_retries: Número máximo de reintentos de validación (j)
            verbose: Si True, imprime progreso
        
        Returns:
            Tupla con:
            - mejor_pid: (Kp, Ki, Kd) mejor PID encontrado
            - menor_error: Error del mejor PID
        """
        if verbose:
            print(f"\n  {'─'*70}")
            print(f"  PIDTrainer: Buscando mejor PID para variable {var_idx}")
            print(f"  Setpoint: {setpoint:.2f} | Episodios: {n_episodes} | Reintentos: {j_max_retries}")
            print(f"  {'─'*70}")
        
        mejor_pid = None
        menor_error = float('inf')
        encontrado_valido = False
        
        # Ciclo de j reintentos de validación
        for intento in range(j_max_retries):
            if verbose:
                print(f"\n    Intento {intento + 1}/{j_max_retries}:")
            
            # 1. Entrenar agente por n episodios
            if verbose:
                print(f"      Entrenando agente durante {n_episodes} episodios...")
            
            # Entrenar usando ControllerAgent
            pid_final = agent.train(
                env=env,
                n_episodes=n_episodes,
                var_idx=var_idx,
                setpoint=setpoint
            )
            
            if verbose:
                print(f"      Entrenamiento completado")
                print(f"      PID obtenido: Kp={pid_final[0]:.4f}, Ki={pid_final[1]:.4f}, Kd={pid_final[2]:.4f}")
            
            # 2. Validar PID con criterios de estabilidad
            if verbose:
                print(f"      Validando criterios de estabilidad...")
            
            # Validar usando StabilityCriteria
            cumple, error = self.stability_criteria.validate(
                pid=pid_final,
                env=env,
                var_idx=var_idx,
                setpoint=setpoint
            )
            
            # Actualizar estadísticas
            self.training_stats['total_trainings'] += 1
            
            # 3. Verificar resultado de validación
            if cumple:
                # PID válido encontrado
                if verbose:
                    print(f"      PID VÁLIDO encontrado (error={error:.4f})")
                
                self.training_stats['successful_validations'] += 1
                mejor_pid = pid_final
                menor_error = error
                encontrado_valido = True
                break  # Salir del ciclo de reintentos
            
            else:
                # PID no cumple criterios
                if verbose:
                    print(f"      PID NO cumple criterios (error={error:.4f})")
                
                self.training_stats['failed_validations'] += 1
                
                # Comparar con mejor almacenado
                if error < menor_error:
                    if verbose:
                        print(f"       Guardado como mejor hasta ahora (error anterior: {menor_error:.4f})")
                    mejor_pid = pid_final
                    menor_error = error
                else:
                    if verbose:
                        print(f"      ↳ No mejora el mejor error ({menor_error:.4f})")
        
        # 4. Resultado final después de j intentos
        if verbose:
            print(f"\n  {'─'*70}")
            if encontrado_valido:
                print(f"   Resultado: PID VÁLIDO encontrado en intento {intento + 1}")
            else:
                print(f"    Resultado: No se encontró PID válido, usando mejor opción")
            print(f"   PID final: Kp={mejor_pid[0]:.4f}, Ki={mejor_pid[1]:.4f}, Kd={mejor_pid[2]:.4f}")
            print(f"   Error final: {menor_error:.4f}")
            print(f"  {'─'*70}")
        
        # Guardar en historial
        self.training_stats['best_errors'].append(menor_error)
        
        return mejor_pid, menor_error
    
    def get_statistics(self) -> dict:
        """
        Obtener estadísticas de entrenamiento.
        
        Returns:
            Diccionario con estadísticas acumuladas
        """
        stats = self.training_stats.copy()
        
        if stats['total_trainings'] > 0:
            stats['success_rate'] = (
                stats['successful_validations'] / stats['total_trainings']
            )
        else:
            stats['success_rate'] = 0.0
        
        if len(stats['best_errors']) > 0:
            stats['avg_best_error'] = np.mean(stats['best_errors'])
            stats['min_best_error'] = np.min(stats['best_errors'])
        else:
            stats['avg_best_error'] = None
            stats['min_best_error'] = None
        
        return stats
    
    def reset_statistics(self) -> None:
        """Resetear estadísticas de entrenamiento."""
        self.training_stats = {
            'total_trainings': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'best_errors': []
        }
    
    def __repr__(self) -> str:
        """Representación en string."""
        stats = self.get_statistics()
        return (
            f"PIDTrainer(\n"
            f"  total_trainings={stats['total_trainings']},\n"
            f"  success_rate={stats['success_rate']:.2%},\n"
            f"  avg_best_error={stats['avg_best_error']:.4f if stats['avg_best_error'] else 'N/A'}\n"
            ")"
        )


