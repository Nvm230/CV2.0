#!/usr/bin/env python3
"""
Servidor Web de Estación Terrestre CV2.0 (FastAPI + WebSockets + MJPEG)
Recibe video transmitido por el dron a través de la red local (UDP / RTSP / Socket)
y lo retransmite en tiempo real con superposiciones de visión por computadora
(YOLO26x + BoT-SORT + Analizador de Incendios) para que cualquier dispositivo en la LAN
(PC, laptop, tablet, smartphone) pueda visualizarlo y operarlo.
"""

import os
import sys
import time
import socket
import asyncio
import threading
from typing import List, Dict, Any, Optional
import yaml
import cv2
import numpy as np

# Añadir directorio raíz al sys.path para importar módulos de src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.detector import DroneVisionDetector
from src.analyzer import SpatialEmergencyAnalyzer, WindowStatus
from src.ground_station import GroundStationHUD
from src.hardware_monitor import HardwareMonitor
from web.network_receiver import NetworkVideoReceiver

def get_lan_ip() -> str:
    """Descubre automáticamente la dirección IP local de la PC en la LAN."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No envía datos reales, solo conecta para determinar la interfaz de salida
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def load_system_config(config_path: str = "config/config.yaml") -> dict:
    abs_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(abs_path):
        print(f"[Web] Configuración no encontrada en {abs_path}, usando valores por defecto")
        return {}
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcast_task = asyncio.create_task(telemetry_broadcaster())
    yield
    broadcast_task.cancel()

# Inicializar FastAPI
app = FastAPI(
    title="CV2.0 Drone Vision Ground Station",
    description="Sistema de Transmisión Web y Telemetría de Visión Artificial para Dron Cautivo",
    version="2.0.0",
    lifespan=lifespan
)

# Permitir CORS desde cualquier dispositivo en la red local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar rutas estáticas y plantillas
templates_dir = os.path.join(CURRENT_DIR, "templates")
static_dir = os.path.join(CURRENT_DIR, "static")
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Variables globales del sistema
config = load_system_config()
hw_monitor = HardwareMonitor()
detector = DroneVisionDetector(config)
analyzer = SpatialEmergencyAnalyzer(config)
hud = GroundStationHUD(config, hardware_monitor=hw_monitor)

# Inicializar receptor de red
initial_source = config.get("stream", {}).get("default_source", "samples/synthetic_drone_mission.mp4")
receiver = NetworkVideoReceiver(source=initial_source, udp_port=5005).start()

# Buffers de fotogramas para transmisión web MJPEG
latest_annotated_jpeg: Optional[bytes] = None
latest_raw_jpeg: Optional[bytes] = None
frame_lock = threading.Lock()
frame_new_event = threading.Event()

# Métricas de ejecución global
pipeline_stats = {
    "fps": 0.0,
    "inference_ms": 0.0,
    "total_frames_processed": 0,
    "last_processed_time": 0.0
}
latest_telemetry: Dict[str, Any] = {}
is_processing = True

# Gestor de conexiones WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            for dead in disconnected:
                if dead in self.active_connections:
                    self.active_connections.remove(dead)

ws_manager = ConnectionManager()

def vision_pipeline_worker():
    """
    Hilo de ejecución en tiempo real del pipeline de visión por computadora:
    1. Lee tramas del receptor de red (dron).
    2. Ejecuta YOLO26x (inferencia + BoT-SORT tracker).
    3. Analiza emergencias (focos de incendio e impacto en ventanas).
    4. Renderiza HUD táctico.
    5. Codifica en JPEG para los clientes web en la LAN.
    """
    global latest_annotated_jpeg, latest_raw_jpeg, pipeline_stats, latest_telemetry
    
    fps_counter = 0
    fps_timer = time.time()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    print("[PipelineWorker] Hilo de visión artificial activo en GPU...")

    while is_processing:
        try:
            has_frame, frame, stream_stats = receiver.get_latest_frame()
            if not has_frame or frame is None:
                time.sleep(0.01)
                continue

            start_inf = time.perf_counter()
            
            # 1. Inferencia YOLO + Tracking
            detections, inf_ms = detector.track(frame)
            
            # 2. Análisis Espacial de Emergencias
            report = analyzer.analyze(detections)
            
            # 3. Renderizado de HUD Táctico
            annotated_frame = hud.render(frame, detections, report, inference_time_ms=inf_ms)
            
            # 4. Codificación en memoria a JPEG para el servidor web
            _, anno_buf = cv2.imencode(".jpg", annotated_frame, encode_params)
            _, raw_buf = cv2.imencode(".jpg", frame, encode_params)
            
            with frame_lock:
                latest_annotated_jpeg = anno_buf.tobytes()
                latest_raw_jpeg = raw_buf.tobytes()
                frame_new_event.set()

            pipeline_stats["total_frames_processed"] += 1
            pipeline_stats["inference_ms"] = round(inf_ms, 2)
            pipeline_stats["last_processed_time"] = time.time()

            # Medición de FPS
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                pipeline_stats["fps"] = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_timer = time.time()

            # Preparar telemetría serializable para WebSockets
            det_list = []
            for d in detections:
                det_list.append({
                    "track_id": d.track_id,
                    "class_name": d.class_name,
                    "confidence": round(float(d.confidence), 2),
                    "bbox": [round(float(v), 1) for v in d.bbox],
                    "area": round(float(d.area), 1)
                })

            window_assessments = []
            for w in report.window_assessments:
                window_assessments.append({
                    "window_id": w.window_id,
                    "status": w.status.value,
                    "severity": round(float(w.severity), 2),
                    "overlap_ratio": round(float(w.overlap_ratio), 2),
                    "distance_to_nearest_fire": round(float(w.distance_to_nearest_fire), 1)
                })

            gpu_stats = hw_monitor.get_gpu_metrics()
            cpu_stats = hw_monitor.get_cpu_metrics()

            latest_telemetry = {
                "timestamp": time.time(),
                "stream_stats": stream_stats,
                "pipeline": {
                    "fps": pipeline_stats["fps"],
                    "inference_ms": pipeline_stats["inference_ms"],
                    "frames_processed": pipeline_stats["total_frames_processed"],
                    "sahi_enabled": detector.sahi_enabled,
                    "recording": hud.is_recording,
                    "trajectories": hud.show_trajectories
                },
                "emergency_report": {
                    "active_fire_count": report.active_fire_count,
                    "total_windows": report.total_windows,
                    "safe_windows": report.safe_windows,
                    "warning_windows": report.warning_windows,
                    "burning_windows": report.burning_windows,
                    "max_severity": round(float(report.max_severity), 2),
                    "total_fire_area_px": round(float(report.total_fire_area_pixels), 1),
                    "fire_spread_velocity_px_s": round(float(report.fire_spread_velocity_px_s), 2),
                    "windows": window_assessments
                },
                "detections": det_list,
                "hardware": {
                    "gpu_name": gpu_stats.get("name", "NVIDIA GPU"),
                    "gpu_utilization": gpu_stats.get("utilization_gpu", 0),
                    "vram_used_mb": gpu_stats.get("vram_used_mb", 0),
                    "vram_total_mb": gpu_stats.get("vram_total_mb", 12288),
                    "vram_percent": gpu_stats.get("vram_percent", 0.0),
                    "temperature_c": gpu_stats.get("temperature_c", 0),
                    "power_watts": gpu_stats.get("power_watts", 0.0),
                    "cpu_percent": cpu_stats.get("cpu_percent", 0.0),
                    "ram_used_gb": cpu_stats.get("memory_used_gb", 0.0),
                    "ram_percent": cpu_stats.get("memory_percent", 0.0)
                }
            }
        except Exception as e:
            print(f"[PipelineWorker] Error procesando trama: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.05)

# Iniciar hilo de procesamiento en segundo plano
processing_thread = threading.Thread(target=vision_pipeline_worker, daemon=True, name="VisionWorker")
processing_thread.start()

# Tarea periódica para emitir WebSockets a los clientes de la red
async def telemetry_broadcaster():
    while True:
        try:
            if latest_telemetry:
                await ws_manager.broadcast(latest_telemetry)
            await asyncio.sleep(0.066) # ~15 Hz actualización de telemetría por WS
        except Exception:
            await asyncio.sleep(0.1)

# --- Rutas Web y Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index_view(request: Request):
    lan_ip = get_lan_ip()
    port = 8000
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "lan_ip": lan_ip,
            "port": port,
            "lan_url": f"http://{lan_ip}:{port}",
            "gpu_name": hw_monitor.gpu_name
        }
    )

def generate_mjpeg_stream(stream_type: str = "annotated"):
    """Generador de flujo continuo multipart/x-mixed-replace (MJPEG)."""
    boundary = b"--frame\r\n"
    while is_processing:
        frame_new_event.wait(timeout=0.1)
        frame_new_event.clear()

        with frame_lock:
            jpeg_bytes = latest_annotated_jpeg if stream_type == "annotated" else latest_raw_jpeg

        if jpeg_bytes is not None:
            yield (
                boundary +
                b"Content-Type: image/jpeg\r\n" +
                f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("utf-8") +
                jpeg_bytes +
                b"\r\n"
            )
        else:
            time.sleep(0.02)

@app.get("/api/stream/annotated")
async def video_stream_annotated():
    """Stream MJPEG con detecciones del sistema de visión y HUD táctico."""
    return StreamingResponse(
        generate_mjpeg_stream("annotated"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/stream/raw")
async def video_stream_raw():
    """Stream MJPEG del video puro transmitido por el dron sin anotaciones."""
    return StreamingResponse(
        generate_mjpeg_stream("raw"),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/snapshot")
async def get_snapshot():
    """Descarga una captura instantánea en alta resolución con anotaciones."""
    with frame_lock:
        if latest_annotated_jpeg:
            return Response(content=latest_annotated_jpeg, media_type="image/jpeg")
    return JSONResponse(status_code=404, content={"error": "No frame available"})

@app.get("/api/status")
async def get_system_status():
    """Devuelve el estado completo del sistema, red y hardware."""
    lan_ip = get_lan_ip()
    return {
        "status": "online",
        "lan_ip": lan_ip,
        "lan_url": f"http://{lan_ip}:8000",
        "active_clients": len(ws_manager.active_connections),
        "telemetry": latest_telemetry
    }

@app.get("/api/incidents")
async def get_incident_log():
    """Devuelve la bitácora histórica de incidentes detectados."""
    return {
        "incidents": analyzer.incident_records[-100:] # Últimos 100 incidentes
    }

class ControlCommand(BaseModel):
    action: str
    value: Optional[Any] = None

@app.post("/api/control")
async def control_pipeline(cmd: ControlCommand):
    """Permite controlar parámetros del sistema de visión de forma remota."""
    action = cmd.action.lower()
    resp = {"status": "ok", "action": action}

    if action == "toggle_sahi":
        new_val = detector.toggle_sahi()
        resp["sahi_enabled"] = new_val
    elif action == "toggle_hud":
        hud.show_hud = not hud.show_hud
        resp["show_hud"] = hud.show_hud
    elif action == "toggle_trajectories":
        hud.show_trajectories = not hud.show_trajectories
        resp["show_trajectories"] = hud.show_trajectories
    elif action == "toggle_recording":
        hud.toggle_recording((1280, 720))
        resp["is_recording"] = hud.is_recording
    elif action == "set_confidence":
        try:
            val = float(cmd.value)
            detector.conf_thresh = max(0.05, min(0.95, val))
            resp["confidence"] = detector.conf_thresh
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
    elif action == "change_source":
        new_source = str(cmd.value)
        receiver.change_source(new_source)
        resp["new_source"] = new_source
    else:
        return JSONResponse(status_code=400, content={"error": f"Unknown action: {action}"})

    return resp

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Canal WebSocket para telemetría vectorial de baja latencia en tiempo real."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Mantener conexión abierta y escuchar comandos de cliente si los hay
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    local_ip = get_lan_ip()
    port = 8000
    print("\n" + "=" * 70)
    print(" 🚀 ESTACIÓN TERRESTRE CV2.0 - SERVIDOR WEB LAN INICIADO")
    print(f"    Acceso Local:             http://127.0.0.1:{port}")
    print(f"    Acceso Red Local (LAN):   http://{local_ip}:{port}")
    print("    Cualquier teléfono, tablet o PC en la misma red Wi-Fi/Ethernet")
    print(f"    puede acceder ingresando en su navegador: http://{local_ip}:{port}")
    print(f"    Puerto UDP Receptor Dron: 0.0.0.0:{receiver.udp_port}")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
