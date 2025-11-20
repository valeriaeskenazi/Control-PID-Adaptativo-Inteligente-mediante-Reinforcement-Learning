import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import time

from .base_env import BasePIDControlEnv
from .pid_components import DeltaPIDActionSpace


class RealPLCEnv(BasePIDControlEnv):
    """
    Ambiente para PLC real con controlador PID integrado.
    
    Características:
    - Solo modo 'pid_tuning' (el PLC calcula el control)
    - Envía (Kp, Ki, Kd) al PLC vía protocolo
    - Lee PV del PLC
    - El PLC maneja el bucle de control
    
    Args:
        config: Configuración del ambiente
        plc_config: Configuración específica del PLC
            - 'ip': Dirección IP del PLC
            - 'protocol': 'opcua', 'modbus', 'ethernet_ip', etc.
            - 'tags': Mapeo de tags del PLC
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 plc_config: Optional[Dict[str, Any]] = None):
        
        # Inicializar clase base
        super().__init__(config)
        
        # Configuración PLC
        self.plc_config = plc_config or {}
        self._setup_action_space()
        
        # Conexión PLC
        self.plc_connection = None
        
        # Estadísticas de comunicación
        self.comm_stats = {
            'total_reads': 0,
            'total_writes': 0,
            'read_errors': 0,
            'write_errors': 0,
            'avg_cycle_time': 0.0
        }
    
    def _setup_action_space(self) -> None:
        """Solo modo pid_tuning para PLC real."""
        # Solo tuning PID (el PLC tiene el controlador)
        self.pid_action_space = DeltaPIDActionSpace(
            initial_pid=(1.0, 0.1, 0.05),
            delta_percent=0.2
        )
        self.action_space = spaces.Discrete(self.pid_action_space.n_actions)
        
        # NO crear PIDController (el PLC lo tiene)
        self.pid_controller = None
        
        print("=" * 60)
        print("✅ Modo: PLC Real")
        print(f"   Acciones disponibles: {self.pid_action_space.n_actions}")
        print(f"   PID inicial: {self.pid_action_space.get_current_pid()}")
        print(f"   PIDController: En el PLC (no en Python)")
        print(f"   Protocolo: {self.plc_config.get('protocol', 'No especificado')}")
        print("=" * 60)
    
    def _apply_control(self, action: int) -> Tuple[Optional[float], Tuple]:
        """
        Aplicar acción (solo traducir a parámetros PID).
        
        Args:
            action: Índice de acción (0-6)
        
        Returns:
            Tuple con (None, pid_params)
            - control_output es None porque el PLC lo calcula
        """
        # Solo traducir índice a parámetros
        pid_params = self.pid_action_space.apply_action(action)
        
        # NO calculamos control_output (lo hace el PLC)
        control_output = None
        
        return control_output, pid_params
    
    def _update_process(self, control_output: Optional[float],
                        pid_params: Tuple) -> float:
        """
        Actualizar proceso real vía PLC.
        
        Args:
            control_output: No usado (None)
            pid_params: Parámetros PID a enviar al PLC
        
        Returns:
            Nuevo valor de PV leído del PLC
        """
        if self.plc_connection is None:
            raise RuntimeError(
                "PLC no conectado. Llama a connect_plc() primero."
            )
        
        cycle_start = time.time()
        
        try:
            # 1. Enviar parámetros PID al PLC
            self.plc_connection.write_pid_params(*pid_params)
            self.comm_stats['total_writes'] += 1
            
            # 2. Enviar setpoint al PLC
            self.plc_connection.write_setpoint(self.setpoint)
            self.comm_stats['total_writes'] += 1
            
            # 3. Esperar un ciclo de control del PLC
            time.sleep(self.dt)
            
            # 4. Leer nuevo PV (el PLC ya aplicó el control)
            new_pv = self.plc_connection.read_pv()
            self.comm_stats['total_reads'] += 1
            
            # Actualizar estadísticas
            cycle_time = time.time() - cycle_start
            self.comm_stats['avg_cycle_time'] = (
                0.9 * self.comm_stats['avg_cycle_time'] + 
                0.1 * cycle_time
            )
            
            return new_pv
            
        except Exception as e:
            self.comm_stats['read_errors'] += 1
            if self.logger:
                self.logger.error(f"Error en comunicación PLC: {e}")
            raise RuntimeError(f"Error actualizando proceso PLC: {e}")
    
    def connect_plc(self, plc_connection) -> None:
        """
        Conectar al PLC real.
        
        Args:
            plc_connection: Objeto con interfaz PLC que debe tener:
                - write_pid_params(kp, ki, kd) -> None
                - write_setpoint(sp) -> None
                - read_pv() -> float
                - read_control_output() -> float (opcional)
                - close() -> None (opcional)
        """
        # Validar interfaz
        required_methods = [
            'write_pid_params',
            'write_setpoint',
            'read_pv'
        ]
        
        for method in required_methods:
            if not hasattr(plc_connection, method):
                raise ValueError(
                    f"PLC connection debe tener método: '{method}'"
                )
        
        self.plc_connection = plc_connection
        
        # Test de conexión
        try:
            _ = self.plc_connection.read_pv()
            
            if self.logger:
                self.logger.info("PLC conectado exitosamente")
            else:
                print("✅ PLC conectado exitosamente")
                
        except Exception as e:
            raise RuntimeError(f"Error al conectar con PLC: {e}")
    
    def disconnect_plc(self) -> None:
        """Desconectar del PLC."""
        if self.plc_connection is not None:
            if hasattr(self.plc_connection, 'close'):
                self.plc_connection.close()
            
            self.plc_connection = None
            
            if self.logger:
                self.logger.info("PLC desconectado")
            else:
                print("✅ PLC desconectado")
    
    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resetear ambiente con PLC real.
        
        Args:
            seed: Semilla (no aplicable a proceso real)
            options: Opciones adicionales
        
        Returns:
            Tuple con (observación inicial, info)
        """
        # Reset de la clase base
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset del espacio de acciones PID
        self.pid_action_space.reset()
        
        # Leer PV inicial del PLC
        if self.plc_connection is not None:
            try:
                self.pv = self.plc_connection.read_pv()
                
                # Enviar parámetros PID iniciales al PLC
                initial_pid = self.pid_action_space.get_current_pid()
                self.plc_connection.write_pid_params(*initial_pid)
                self.plc_connection.write_setpoint(self.setpoint)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error en reset PLC: {e}")
                raise RuntimeError(f"Error al resetear PLC: {e}")
        else:
            raise RuntimeError("PLC no conectado. Llama a connect_plc() primero.")
        
        return self._get_observation(), info
    
    def get_comm_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de comunicación con el PLC."""
        return self.comm_stats.copy()
    
    def render(self, mode: str = 'human') -> None:
        """Renderizar estado con información del PLC."""
        if mode == 'human':
            error = self.setpoint - self.pv
            
            # Intentar leer control_output del PLC si es posible
            control_output = None
            if (self.plc_connection is not None and 
                hasattr(self.plc_connection, 'read_control_output')):
                try:
                    control_output = self.plc_connection.read_control_output()
                except:
                    pass
            
            print(
                f"Step: {self.step_count:4d} | "
                f"PV: {self.pv:6.2f} | "
                f"SP: {self.setpoint:6.2f} | "
                f"Error: {error:6.2f} | "
                f"PID: {self.pid_action_space.get_current_pid()} | "
                f"Control: {control_output if control_output else 'N/A'} | "
                f"Comm: R={self.comm_stats['total_reads']} "
                f"W={self.comm_stats['total_writes']}"
            )
    
    def close(self) -> None:
        """Cerrar ambiente y desconectar PLC."""
        self.disconnect_plc()
        if hasattr(super(), 'close'):
            super().close()


# ============================================================
# EJEMPLO: INTERFAZ PLC
# ============================================================

class PLCInterface:
    """
    Interfaz base para comunicación con PLC.
    
    Esta es una clase de ejemplo. En producción, implementarías
    esto con tu biblioteca específica (pycomm3, pyModbusTCP, etc.)
    """
    
    def __init__(self, ip: str, protocol: str = 'opcua'):
        self.ip = ip
        self.protocol = protocol
        self.connected = False
        
        # Tags del PLC (ejemplo)
        self.tags = {
            'pv': 'Process.PV',
            'setpoint': 'PID_Block.Setpoint',
            'kp': 'PID_Block.Kp',
            'ki': 'PID_Block.Ki',
            'kd': 'PID_Block.Kd',
            'control_output': 'PID_Block.Output'
        }
    
    def connect(self) -> None:
        """Conectar al PLC (implementar según protocolo)."""
        # Aquí iría tu código de conexión real
        # Ejemplo con OPC-UA:
        # from opcua import Client
        # self.client = Client(f"opc.tcp://{self.ip}:4840")
        # self.client.connect()
        
        print(f"🔌 Conectando a PLC en {self.ip} via {self.protocol}...")
        self.connected = True
        print("✅ Conectado")
    
    def write_pid_params(self, kp: float, ki: float, kd: float) -> None:
        """Escribir parámetros PID al PLC."""
        if not self.connected:
            raise RuntimeError("PLC no conectado")
        
        # Implementar escritura según protocolo
        # Ejemplo:
        # self.client.get_node(self.tags['kp']).set_value(kp)
        # self.client.get_node(self.tags['ki']).set_value(ki)
        # self.client.get_node(self.tags['kd']).set_value(kd)
        
        print(f"📤 Escribiendo PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    
    def write_setpoint(self, sp: float) -> None:
        """Escribir setpoint al PLC."""
        if not self.connected:
            raise RuntimeError("PLC no conectado")
        
        # Implementar escritura según protocolo
        # self.client.get_node(self.tags['setpoint']).set_value(sp)
        
        print(f"📤 Escribiendo Setpoint: {sp:.2f}")
    
    def read_pv(self) -> float:
        """Leer variable de proceso del PLC."""
        if not self.connected:
            raise RuntimeError("PLC no conectado")
        
        # Implementar lectura según protocolo
        # pv = self.client.get_node(self.tags['pv']).get_value()
        
        # Simulación para ejemplo
        pv = 50.0 + np.random.normal(0, 2)
        print(f"📥 Leyendo PV: {pv:.2f}")
        
        return pv
    
    def read_control_output(self) -> float:
        """Leer salida de control del PLC (opcional)."""
        if not self.connected:
            raise RuntimeError("PLC no conectado")
        
        # Implementar lectura según protocolo
        # output = self.client.get_node(self.tags['control_output']).get_value()
        
        # Simulación
        output = np.random.uniform(-1, 1)
        return output
    
    def close(self) -> None:
        """Cerrar conexión con PLC."""
        if self.connected:
            # self.client.disconnect()
            self.connected = False
            print("🔌 PLC desconectado")