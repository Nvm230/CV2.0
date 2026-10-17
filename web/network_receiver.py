import os
import sys
import time
import socket
import struct
import threading
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

class NetworkVideoReceiver:
    """
    Receptor de video en red para Estación Terrestre (CV2.0).
    Soporta múltiples modos de ingestión de video transmitido por el dron:
    1. 'stream': Ingesta por URL de red (RTSP, HTTP, o UDP vía FFmpeg ej. udp://@:5000)
    2. 'udp_socket': Receptor directo por socket UDP (ideal para scripts ligeros en Raspberry Pi / Jetson)
    3. 'file': Archivo local en bucle infinito para pruebas y simulación táctica
    4. 'device': Cámara local o capturadora HDMI USB (/dev/video0)
    """

    def __init__(self, source: str = "samples/synthetic_drone_mission.mp4", udp_port: int = 5005):
        self.source = source
        self.udp_port = udp_port
        self.mode = self._detect_mode(source)
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.last_frame_time: float = 0.0
        self.connected: bool = False
        
        # Métricas de ingestión
        self.fps_in: float = 0.0
        self.fps_counter: int = 0
        self.fps_timer: float = time.time()
        self.dropped_frames: int = 0
        self.resolution: Tuple[int, int] = (0, 0)
        self.bytes_received: int = 0
        
        # Manejadores internos
        self.cap: Optional[cv2.VideoCapture] = None
        self.sock: Optional[socket.socket] = None

    def _detect_mode(self, src: str) -> str:
        src_lower = str(src).lower()
        if src_lower.startswith("udp_socket:") or src_lower.startswith("socket:"):
            return "udp_socket"
        if src_lower.startswith("rtsp://") or src_lower.startswith("udp://") or src_lower.startswith("http://"):
            return "stream"
        if src_lower.isdigit() or src_lower.startswith("/dev/video"):
            return "device"
        return "file"

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="DroneNetReceiver")
        self.thread.start()
        print(f"[NetReceiver] Iniciado en modo '{self.mode}' con origen: {self.source}")
        return self

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self._cleanup()
        print("[NetReceiver] Receptor detenido.")

    def _cleanup(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.connected = False

    def change_source(self, new_source: str):
        """Permite cambiar la fuente o puerto en caliente desde la interfaz web."""
        with self.lock:
            print(f"[NetReceiver] Cambiando fuente de '{self.source}' a '{new_source}'")
            self.source = new_source
            self.mode = self._detect_mode(new_source)
            self._cleanup()

    def _worker_loop(self):
        while self.running:
            try:
                if self.mode == "udp_socket":
                    self._run_udp_socket_receiver()
                elif self.mode in ("stream", "device", "file"):
                    self._run_opencv_receiver()
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"[NetReceiver] Error en hilo de recepción: {e}")
                time.sleep(1.0)

    def _run_opencv_receiver(self):
        src = self.source
        if self.mode == "device" and str(src).isdigit():
            src_val = int(src)
        elif self.mode == "file" and isinstance(src, str) and not os.path.isabs(src):
            src_val = os.path.join(PROJECT_ROOT, src)
        else:
            src_val = src

        if self.mode == "stream" and str(src).startswith("rtsp://"):
            self.cap = cv2.VideoCapture(src_val, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.mode == "stream" and str(src).startswith("udp://"):
            # Flags de baja latencia para OpenCV con UDP
            self.cap = cv2.VideoCapture(src_val, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.mode == "file":
            self.cap = cv2.VideoCapture(src_val, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src_val)

        if not self.cap.isOpened():
            print(f"[NetReceiver] ERROR: No se pudo abrir la fuente: {src_val}")
            self.connected = False
            time.sleep(1.0)
            return

        print(f"[NetReceiver] Fuente abierta con éxito: {src_val}")
        self.connected = True
        is_file = (self.mode == "file")

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            # Log de diagnóstico en la primera lectura
            if self.frame_id == 0:
                print(f"[NetReceiver] Primera lectura: ret={ret}, frame={'None' if frame is None else frame.shape}", flush=True)
            if not ret or frame is None:
                if is_file:
                    # En archivo de prueba, repetir en bucle continuo
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    self.connected = False
                    time.sleep(0.1)
                    break

            self._update_frame(frame)
            if is_file:
                # Limitar tasa a ~30 FPS para emular video real de cámara
                time.sleep(0.025)

        self._cleanup()

    def _run_udp_socket_receiver(self):
        """
        Escucha paquetes UDP en un puerto de red.
        Soporta tramas JPEG empaquetadas enviadas por el script del dron.
        Estructura de paquete:
        - Bloques de hasta 65000 bytes o paquetes segmentados con cabecera (magic 0xAA55, frame_id, chunk_idx, total_chunks)
        - O datagramas JPEG directos de hasta 65,507 bytes
        """
        port = self.udp_port
        # Si la fuente especifica puerto, e.g. "udp_socket:5005"
        if ":" in self.source:
            parts = self.source.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                port = int(parts[1])

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024) # Buffer de 4MB
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(2.0)
        
        print(f"[NetReceiver] Escuchando paquetes UDP del dron en puerto 0.0.0.0:{port}...")
        self.connected = True

        buffer_chunks: Dict[int, Dict[int, bytes]] = {}
        buffer_totals: Dict[int, int] = {}
        buffer_expire: Dict[int, float] = {}

        while self.running and self.sock:
            try:
                data, addr = self.sock.recvfrom(65536)
                if not data:
                    continue
                
                self.bytes_received += len(data)

                # Comprobación de cabecera de fragmentación (Magic 0xAA55)
                # Formato: 2 bytes magic, 4 bytes frame_id, 2 bytes chunk_id, 2 bytes total_chunks
                if len(data) >= 10 and data[:2] == b'\xAA\x55':
                    magic, f_id, chunk_id, total_chunks = struct.unpack("!H I H H", data[:10])
                    payload = data[10:]
                    
                    now = time.time()
                    if f_id not in buffer_chunks:
                        buffer_chunks[f_id] = {}
                        buffer_totals[f_id] = total_chunks
                        buffer_expire[f_id] = now + 1.0 # Expira en 1s
                    
                    buffer_chunks[f_id][chunk_id] = payload
                    
                    # Limpiar tramas expiradas
                    expired = [fid for fid, exp in buffer_expire.items() if now > exp]
                    for exp_id in expired:
                        buffer_chunks.pop(exp_id, None)
                        buffer_totals.pop(exp_id, None)
                        buffer_expire.pop(exp_id, None)
                    
                    if len(buffer_chunks[f_id]) == buffer_totals[f_id]:
                        # Ensamblar trama completa
                        sorted_chunks = [buffer_chunks[f_id][i] for i in range(buffer_totals[f_id])]
                        full_jpeg = b"".join(sorted_chunks)
                        buffer_chunks.pop(f_id, None)
                        buffer_totals.pop(f_id, None)
                        buffer_expire.pop(f_id, None)
                        
                        np_arr = np.frombuffer(full_jpeg, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self._update_frame(frame)
                else:
                    # Intento directo de decodificación como imagen JPEG completa en un solo datagrama
                    np_arr = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._update_frame(frame)
                        
            except socket.timeout:
                self.connected = False
                continue
            except Exception as e:
                if self.running:
                    print(f"[NetReceiver] Excepción en socket UDP: {e}")
                time.sleep(0.05)

        self._cleanup()

    def _update_frame(self, frame: np.ndarray):
        now = time.time()
        with self.lock:
            self.latest_frame = frame
            self.frame_id += 1
            self.last_frame_time = now
            self.connected = True
            self.resolution = (frame.shape[1], frame.shape[0])

        self.fps_counter += 1
        elapsed = now - self.fps_timer
        if elapsed >= 1.0:
            self.fps_in = round(self.fps_counter / elapsed, 1)
            self.fps_counter = 0
            self.fps_timer = now

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """Obtiene la trama más reciente y estadísticas de recepción."""
        with self.lock:
            if self.latest_frame is None:
                return False, None, self.get_stats()
            # Retorna una vista copiada para evitar condiciones de carrera
            return True, self.latest_frame.copy(), self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "connected": self.connected,
                "mode": self.mode,
                "source": self.source,
                "fps_in": self.fps_in,
                "frame_id": self.frame_id,
                "resolution": f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution[0] > 0 else "N/A",
                "width": self.resolution[0],
                "height": self.resolution[1],
                "last_seen_sec": round(time.time() - self.last_frame_time, 2) if self.last_frame_time > 0 else 999.0,
                "bytes_received": self.bytes_received
            }
