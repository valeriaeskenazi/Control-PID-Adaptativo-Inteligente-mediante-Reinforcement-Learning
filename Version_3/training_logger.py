"""
Logger simple para visualizar tracking de variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os


class TrainingLogger:
    """Logger para métricas de entrenamiento multi-agente."""
    
    def __init__(self):
        """Inicializar logger."""
        # Datos temporales para cada variable manipulable
        self.manipulable_vars_data = {}  # {var_idx: {'time': [], 'sp': [], 'pv': []}}
        
        # Datos de variable objetivo
        self.target_data = {
            'time': [],
            'pv': [],
            'target': [],
            'deadband_upper': [],
            'deadband_lower': []
        }
        
        # Contador de tiempo global
        self.global_time_step = 0
    
    def log_step(
        self,
        manipulable_pvs: List[float],
        manipulable_sps: List[float],
        target_pv: float,
        target_sp: float,
        deadband: float = 0.0
    ):
        """
        Registrar un paso de ejecución.
        
        Args:
            manipulable_pvs: Valores actuales de variables manipulables [PV_Tc, PV_F]
            manipulable_sps: Setpoints actuales [SP_Tc, SP_F]
            target_pv: Valor actual de la variable objetivo (Cb)
            target_sp: Setpoint objetivo (ej: 0.2)
            deadband: Banda muerta alrededor del target
        """
        self.global_time_step += 1
        
        # Guardar datos de variables manipulables
        for i, (pv, sp) in enumerate(zip(manipulable_pvs, manipulable_sps)):
            if i not in self.manipulable_vars_data:
                self.manipulable_vars_data[i] = {
                    'time': [],
                    'sp': [],
                    'pv': []
                }
            
            self.manipulable_vars_data[i]['time'].append(self.global_time_step)
            self.manipulable_vars_data[i]['sp'].append(sp)
            self.manipulable_vars_data[i]['pv'].append(pv if np.isfinite(pv) else sp)
        
        # Guardar datos de variable objetivo
        self.target_data['time'].append(self.global_time_step)
        self.target_data['pv'].append(target_pv if np.isfinite(target_pv) else 0)
        self.target_data['target'].append(target_sp)
        self.target_data['deadband_upper'].append(target_sp + deadband)
        self.target_data['deadband_lower'].append(target_sp - deadband)
    
    def plot_results(
        self,
        save_dir: str = './results/plots',
        show: bool = False
    ):
        """
        Generar gráficos de tracking.
        
        Args:
            save_dir: Directorio donde guardar gráficos
            show: Si mostrar gráficos interactivamente
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Variables manipulables: SP vs PV
        self._plot_manipulable_tracking(save_dir)
        
        # 2. Variable objetivo: PV vs Target con deadband
        self._plot_target_tracking(save_dir)
        
        print(f"\n📊 Gráficos guardados en: {save_dir}")
        
        if show:
            plt.show()
    
    def _plot_manipulable_tracking(self, save_dir: str):
        """Graficar tracking de variables manipulables."""
        n_vars = len(self.manipulable_vars_data)
        
        if n_vars == 0:
            return
        
        var_names = ['Temperatura Tc (K)', 'Flujo F (m³/s)']
        
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 5*n_vars))
        if n_vars == 1:
            axes = [axes]
        
        for i in range(n_vars):
            data = self.manipulable_vars_data[i]
            
            # Graficar PV
            axes[i].plot(
                data['time'],
                data['pv'],
                color='#2E86AB',
                linewidth=2,
                label='PV (Valor Real)',
                alpha=0.8
            )
            
            # Graficar SP (con escalones visibles)
            axes[i].step(
                data['time'],
                data['sp'],
                where='post',
                color='#D62246',
                linewidth=2.5,
                linestyle='--',
                label='SP (Orquestador)',
                alpha=0.9
            )
            
            axes[i].set_xlabel('Tiempo (steps)', fontsize=12)
            axes[i].set_ylabel(var_names[i], fontsize=12)
            axes[i].set_title(
                f'Variable Manipulable {i}: {var_names[i]} - Tracking SP vs PV',
                fontsize=13,
                fontweight='bold'
            )
            axes[i].legend(fontsize=11, loc='best')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/manipulable_vars_tracking.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_target_tracking(self, save_dir: str):
        """Graficar tracking de variable objetivo con deadband."""
        if len(self.target_data['time']) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar PV
        ax.plot(
            self.target_data['time'],
            self.target_data['pv'],
            color='#06A77D',
            linewidth=2.5,
            label='Cb (Concentración Real)',
            alpha=0.9
        )
        
        # Graficar Target
        ax.plot(
            self.target_data['time'],
            self.target_data['target'],
            color='#D62246',
            linestyle='--',
            linewidth=2,
            label='Target Cb',
            alpha=0.9
        )
        
        # Graficar Deadband (zona sombreada)
        ax.fill_between(
            self.target_data['time'],
            self.target_data['deadband_lower'],
            self.target_data['deadband_upper'],
            color='#F18F01',
            alpha=0.2,
            label='Deadband'
        )
        
        # Líneas de límites del deadband
        ax.plot(
            self.target_data['time'],
            self.target_data['deadband_upper'],
            color='#F18F01',
            linestyle=':',
            linewidth=1.5,
            alpha=0.6
        )
        ax.plot(
            self.target_data['time'],
            self.target_data['deadband_lower'],
            color='#F18F01',
            linestyle=':',
            linewidth=1.5,
            alpha=0.6
        )
        
        ax.set_xlabel('Tiempo (steps)', fontsize=12)
        ax.set_ylabel('Concentración Cb (mol/m³)', fontsize=12)
        ax.set_title(
            'Variable Objetivo: Cb vs Target con Deadband',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/target_tracking.png', dpi=150, bbox_inches='tight')
        plt.close()