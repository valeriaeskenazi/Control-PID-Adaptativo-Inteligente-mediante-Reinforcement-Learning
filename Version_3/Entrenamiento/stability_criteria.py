"""
Criterios de estabilidad para validación de controladores PID.

Implementa 4 criterios de validación:
1. Acción razonable (no saturada, no excesiva)
2. Error disminuye o se mantiene pequeño
3. No hay oscilaciones evidentes
4. No hay cambios abruptos/inestabilidad
"""

from typing import Tuple, List, Optional
import numpy as np
from collections import deque


class StabilityCriteria:
    """
    Validador de estabilidad para controladores PID.
    
    Verifica 4 criterios fundamentales:
    - Criterio 1: Acción de control razonable
    - Criterio 2: Error disminuye o se mantiene pequeño
    - Criterio 3: Ausencia de oscilaciones
    - Criterio 4: Ausencia de cambios abruptos
    
    Args:
        action_saturation_threshold: Umbral de saturación (default: 0.95)
        error_increase_tolerance: Tolerancia de aumento de error (default: 1.5x)
        oscillation_window: Ventana para detectar oscilaciones (default: 10)
        max_change_threshold: Umbral de cambio máximo (default: 0.3)
        validation_steps: Pasos de simulación para validar (default: 100)
    """
    
    def __init__(self,
                 action_saturation_threshold: float = 0.95,
                 error_increase_tolerance: float = 1.5,
                 oscillation_window: int = 10,
                 max_change_threshold: float = 0.3,
                 validation_steps: int = 100):
        
        self.action_saturation_threshold = action_saturation_threshold
        self.error_increase_tolerance = error_increase_tolerance
        self.oscillation_window = oscillation_window
        self.max_change_threshold = max_change_threshold
        self.validation_steps = validation_steps
        
        # Estadísticas de validación
        self.validation_stats = {
            'total_validations': 0,
            'criterion_1_fails': 0,
            'criterion_2_fails': 0,
            'criterion_3_fails': 0,
            'criterion_4_fails': 0
        }
    
    def validate(self,
                 pid: Tuple[float, float, float],
                 env,
                 var_idx: int,
                 setpoint: Optional[float] = None,
                 verbose: bool = False) -> Tuple[bool, float]:
        """
        Validar PID con todos los criterios de estabilidad.
        
        Ejecuta una simulación con el PID dado y verifica:
        1. Acción razonable (no saturada)
        2. Error disminuye o se mantiene pequeño
        3. No hay oscilaciones evidentes
        4. No hay cambios abruptos
        
        Args:
            pid: Tupla (Kp, Ki, Kd) a validar
            env: Ambiente base para simular
            var_idx: Índice de la variable a controlar
            setpoint: Setpoint a usar (si None, usa el actual del env)
            verbose: Si True, imprime detalles de cada criterio
        
        Returns:
            Tupla con:
            - cumple: True si pasa todos los criterios
            - error: Error promedio durante la simulación
        """
        self.validation_stats['total_validations'] += 1
        
        if verbose:
            print(f"        Validando PID: Kp={pid[0]:.4f}, Ki={pid[1]:.4f}, Kd={pid[2]:.4f}")
        
        # Configurar ambiente
        if setpoint is not None:
            env.set_setpoint(setpoint, var_idx=var_idx)
        
        # Resetear ambiente
        obs, info = env.reset()
        
        # Historial de simulación
        pv_history = []
        error_history = []
        action_history = []
        
        # Ejecutar simulación con PID
        pv_history, error_history, action_history = self._simulate_pid_execution(
            pid, env, var_idx, self.validation_steps
        )
        
        # Calcular error promedio
        error_promedio = np.mean(np.abs(error_history))
        
        # Verificar cada criterio
        criterios_cumplidos = []
        
        # Criterio 1: Acción razonable
        criterio_1, razon_1 = self._check_criterion_1(action_history, verbose)
        criterios_cumplidos.append(criterio_1)
        if not criterio_1:
            self.validation_stats['criterion_1_fails'] += 1
        
        # Criterio 2: Error disminuye o se mantiene pequeño
        criterio_2, razon_2 = self._check_criterion_2(error_history, verbose)
        criterios_cumplidos.append(criterio_2)
        if not criterio_2:
            self.validation_stats['criterion_2_fails'] += 1
        
        # Criterio 3: No hay oscilaciones evidentes
        criterio_3, razon_3 = self._check_criterion_3(pv_history, verbose)
        criterios_cumplidos.append(criterio_3)
        if not criterio_3:
            self.validation_stats['criterion_3_fails'] += 1
        
        # Criterio 4: No hay cambios abruptos
        criterio_4, razon_4 = self._check_criterion_4(pv_history, verbose)
        criterios_cumplidos.append(criterio_4)
        if not criterio_4:
            self.validation_stats['criterion_4_fails'] += 1
        
        # Resultado final
        cumple_todos = all(criterios_cumplidos)
        
        if verbose:
            if cumple_todos:
                print(f"         Todos los criterios cumplidos")
            else:
                print(f"         Criterios NO cumplidos:")
                if not criterio_1:
                    print(f"           - Criterio 1: {razon_1}")
                if not criterio_2:
                    print(f"           - Criterio 2: {razon_2}")
                if not criterio_3:
                    print(f"           - Criterio 3: {razon_3}")
                if not criterio_4:
                    print(f"           - Criterio 4: {razon_4}")
        
        return cumple_todos, error_promedio
    
    def _check_criterion_1(self, action_history: List[float], verbose: bool) -> Tuple[bool, str]:
        """
        Criterio 1: Acción es razonable (no saturada, no excesiva).
        
        Verifica que las acciones de control no estén cerca de la saturación.
        """
        # Normalizar acciones a [-1, 1] si es necesario
        acciones_abs = np.abs(action_history)
        max_accion = np.max(acciones_abs)
        
        # Porcentaje de acciones cerca de saturación
        saturadas = np.sum(acciones_abs > self.action_saturation_threshold)
        porcentaje_saturacion = saturadas / len(action_history)
        
        cumple = porcentaje_saturacion < 0.1  # Menos del 10% saturado
        
        if verbose:
            print(f"        [Criterio 1] Acción máxima: {max_accion:.3f}, "
                  f"Saturación: {porcentaje_saturacion:.1%} → "
                  f"{'✓' if cumple else '✗'}")
        
        razon = f"Acción saturada en {porcentaje_saturacion:.1%} de los pasos"
        return cumple, razon
    
    def _check_criterion_2(self, error_history: List[float], verbose: bool) -> Tuple[bool, str]:
        """
        Criterio 2: Error disminuyó o se mantiene pequeño.
        
        Verifica que el error no aumente significativamente durante la simulación.
        """
        errores_abs = np.abs(error_history)
        
        # Comparar primera mitad vs segunda mitad
        mitad = len(error_history) // 2
        error_inicial = np.mean(errores_abs[:mitad])
        error_final = np.mean(errores_abs[mitad:])
        
        # El error final no debe ser mucho mayor que el inicial
        cumple = error_final <= error_inicial * self.error_increase_tolerance
        
        if verbose:
            print(f"        [Criterio 2] Error inicial: {error_inicial:.3f}, "
                  f"Error final: {error_final:.3f} → "
                  f"{'✓' if cumple else '✗'}")
        
        razon = f"Error aumentó de {error_inicial:.3f} a {error_final:.3f}"
        return cumple, razon
    
    def _check_criterion_3(self, pv_history: List[float], verbose: bool) -> Tuple[bool, str]:
        """
        Criterio 3: No hay oscilaciones evidentes.
        
        Detecta oscilaciones contando cruces del setpoint.
        """
        if len(pv_history) < self.oscillation_window:
            return True, "Ventana muy pequeña"
        
        # Detectar cruces del valor medio
        pv_array = np.array(pv_history)
        mean_pv = np.mean(pv_array)
        
        # Contar cambios de signo respecto a la media
        diferencias = pv_array - mean_pv
        cruces = np.sum(np.diff(np.sign(diferencias)) != 0)
        
        # Normalizar por longitud
        cruces_por_paso = cruces / len(pv_history)
        
        # No debe haber muchas oscilaciones
        cumple = cruces_por_paso < 0.2  # Menos de 20% de pasos con cruces
        
        if verbose:
            print(f"        [Criterio 3] Cruces detectados: {cruces}, "
                  f"Tasa: {cruces_por_paso:.1%} → "
                  f"{'✓' if cumple else '✗'}")
        
        razon = f"Oscilaciones detectadas ({cruces} cruces, tasa {cruces_por_paso:.1%})"
        return cumple, razon
    
    def _check_criterion_4(self, pv_history: List[float], verbose: bool) -> Tuple[bool, str]:
        """
        Criterio 4: No hay cambios abruptos/inestabilidad.
        
        Verifica que no haya cambios súbitos en el PV.
        """
        if len(pv_history) < 2:
            return True, "Historia muy corta"
        
        # Calcular cambios entre pasos consecutivos
        cambios = np.abs(np.diff(pv_history))
        
        # Normalizar por rango del PV
        rango_pv = np.max(pv_history) - np.min(pv_history)
        if rango_pv > 0:
            cambios_norm = cambios / rango_pv
        else:
            cambios_norm = cambios
        
        # No debe haber cambios muy grandes
        max_cambio = np.max(cambios_norm)
        cambios_grandes = np.sum(cambios_norm > self.max_change_threshold)
        porcentaje_cambios = cambios_grandes / len(cambios_norm)
        
        cumple = porcentaje_cambios < 0.05  # Menos del 5% con cambios grandes
        
        if verbose:
            print(f"        [Criterio 4] Cambio máximo: {max_cambio:.3f}, "
                  f"Cambios grandes: {porcentaje_cambios:.1%} → "
                  f"{'✓' if cumple else '✗'}")
        
        razon = f"Cambios abruptos detectados ({porcentaje_cambios:.1%})"
        return cumple, razon
    
    def _simulate_pid_execution(self,
                                 pid: Tuple[float, float, float],
                                 env,
                                 var_idx: int,
                                 steps: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Simular ejecución del PID en el ambiente.
        
        Configura el PID y ejecuta con acción "mantener" para evaluar estabilidad.
        """
        # 1. Configurar PID en el ambiente
        # Detectar si es multi-variable
        if hasattr(env, 'pid_action_spaces'):
            # Multi-variable: usar el índice correcto
            env.pid_action_spaces[var_idx].set_pid(pid[0], pid[1], pid[2])
        else:
            # Single-variable
            env.pid_action_space.set_pid(pid[0], pid[1], pid[2])
        
        # 2. Reset ambiente
        obs, info = env.reset()
        
        # 3. Ejecutar simulación
        pv_history = []
        error_history = []
        action_history = []
        
        for step in range(steps):
            # Acción 6 = mantener PID actual (no modificar parámetros)
            # Detectar si es multi-variable y crear acción apropiada
            if hasattr(env, 'n_variables') and env.n_variables > 1:
                # Multi-variable: acción "mantener" para todas las variables
                action = [6] * env.n_variables
            else:
                # Single-variable
                action = 6

            obs, reward, terminated, truncated, info = env.step(action)
            
            # Guardar datos
            pv_history.append(obs[0])  # PV
            error_history.append(obs[2])  # Error
            
            # Salida de control de la variable
            if 'control_outputs' in info:
                action_history.append(info['control_outputs'][var_idx])
            else:
                action_history.append(0.0)
            
            if terminated or truncated:
                break
        
        return pv_history, error_history, action_history
    
    def get_statistics(self) -> dict:
        """Obtener estadísticas de validaciones."""
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['criterion_1_fail_rate'] = stats['criterion_1_fails'] / stats['total_validations']
            stats['criterion_2_fail_rate'] = stats['criterion_2_fails'] / stats['total_validations']
            stats['criterion_3_fail_rate'] = stats['criterion_3_fails'] / stats['total_validations']
            stats['criterion_4_fail_rate'] = stats['criterion_4_fails'] / stats['total_validations']
        
        return stats
    
    def reset_statistics(self) -> None:
        """Resetear estadísticas."""
        self.validation_stats = {
            'total_validations': 0,
            'criterion_1_fails': 0,
            'criterion_2_fails': 0,
            'criterion_3_fails': 0,
            'criterion_4_fails': 0
        }


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EJEMPLO: USO DE StabilityCriteria")
    print("="*80)
    
    # Crear validador
    criteria = StabilityCriteria(validation_steps=100)
    
    # Simular validación de un PID bueno
    print("\n>>> Validando PID bueno:")
    pid_bueno = (1.2, 0.15, 0.05)
    cumple, error = criteria.validate(
        pid=pid_bueno,
        env=None,  # Placeholder
        var_idx=0,
        setpoint=50.0,
        verbose=True
    )
    print(f"\nResultado: {'✅ VÁLIDO' if cumple else '❌ INVÁLIDO'} (error={error:.4f})")
    
    # Mostrar estadísticas
    print("\n" + "="*80)
    print("ESTADÍSTICAS")
    print("="*80)
    for key, value in criteria.get_statistics().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
