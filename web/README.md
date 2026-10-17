# Servidor Web y Estación Terrestre CV2.0 para Dron Cautivo

Este módulo implementa el servidor web de retransmisión y telemetría táctica para el sistema de visión artificial **CV2.0**. Permite que el dron transmita su flujo de video hacia una PC de procesamiento (Ground Station) mediante un puerto de red (UDP, RTSP o sockets), y que la PC ejecute en tiempo real el pipeline neuronal (YOLO26x, BoT-SORT y análisis espacial de incendios) para difundirlo a **cualquier dispositivo conectado a la red local (LAN)** (laptops, celulares, tablets o monitores del centro de mando) sin necesidad de instalar software adicional.

---

## 1. Topología de Red y Flujo de Datos

```
[ Dron en Vuelo / Cámara ]
           │
           │ Transmisión de video por puerto de red (ej. UDP 5005 o RTSP)
           ▼
[ PC Estación Terrestre (192.168.1.X) ]
   ├── Receptor de Red (web/network_receiver.py)
   ├── Pipeline de Visión: NVIDIA GPU RTX + YOLO26x + BoT-SORT + Analizador Espacial
   └── Servidor Web Asíncrono (FastAPI + WebSockets + MJPEG) (Puerto 8000)
           │
           ├── WiFi / Switch Ethernet LAN
           ▼
[ Dispositivos en la Red Local ]
   ├── Celular / Tablet de Bomberos (http://192.168.1.X:8000)
   ├── Laptop de Mando Táctico      (http://192.168.1.X:8000)
   └── Pantalla en Sala de Crisis   (http://192.168.1.X:8000)
```

---

## 2. Puesta en Marcha Rápida

### A. Iniciar el Servidor Web en la PC (Estación Terrestre)
Desde la raíz del proyecto:
```bash
./web/run_server.sh
```
O directamente con Python:
```bash
source .venv/bin/activate
python web/app.py
```

El servidor detectará automáticamente la dirección IP local de tu PC (por ejemplo, `192.168.1.8`) y abrirá el puerto `8000`:
*   **En la misma PC:** Abre en tu navegador [http://127.0.0.1:8000](http://127.0.0.1:8000)
*   **Desde cualquier otro dispositivo en la red local:** Abre `http://<IP_DE_LA_PC>:8000` (ejemplo: `http://192.168.1.8:8000`).

---

## 3. Modos de Ingesta de Video desde el Dron

El sistema admite varias formas para recibir el video del dron:

### Opción 1: Transmisión directa por Socket UDP (Recomendado para scripts ligeros)
En la computadora a bordo del dron (Raspberry Pi, Jetson Orin Nano, Orange Pi):
```bash
python web/drone_sender_sim.py --dest 192.168.1.8 --port 5005 --source 0 --fps 30
```
*   `--dest`: Dirección IP de la PC que tiene la tarjeta gráfica NVIDIA GPU RTX.
*   `--port`: Puerto UDP de destino (por defecto `5005`).
*   `--source`: Dispositivo de video (ej. `0` para la cámara del dron o ruta de archivo).

### Opción 2: Flujo RTSP / UDP nativo de cámara IP o módulo DJI / SIYI
Si el dron tiene una cámara con emisor RTSP integrado:
1. En la interfaz web, en la sección **"Fuente de Video / Puerto Red"**, escribe la URL RTSP:
   `rtsp://192.168.1.50:8554/live`
2. Presiona **Conectar**.

### Opción 3: Modo Simulación / Prueba sin Dron Físico
Por defecto, si no hay un dron transmitiendo, el sistema arranca con la misión sintética pregrabada en bucle (`samples/synthetic_drone_mission.mp4`) para permitir verificar todas las alertas, métricas de GPU y visualizaciones web de inmediato.

---

## 4. Características de la Interfaz Web Táctica

1.  **Doble Flujo de Video:**
    *   **Anotado (HUD):** Incluye las cajas delimitadoras tácticas de fuego y ventanas, trayectorias de BoT-SORT, máscaras térmicas y telemetría de vuelo.
    *   **Dron Puro (RAW):** Video original sin anotaciones para inspección limpia.
2.  **Matriz de Amenazas y Emergencias:**
    *   Conteo en tiempo real de focos de fuego activos.
    *   Conteo de ventanas seguras, en advertencia y en ignición crítica.
    *   Tasa de expansión y velocidad del fuego ($px/s$).
    *   Indicador visual y auditivo de severidad máxima.
3.  **Alarmas Tácticas en Tiempo Real:**
    *   Sintetizador de audio Web Audio API que emite un pulso sonoro cuando se detecta un incendio crítico o riesgo estructural (conmutador MUTE disponible).
4.  **Telemetría de Hardware en Vivo:**
    *   Porcentaje de uso de la GPU (NVIDIA GPU RTX).
    *   Consumo de VRAM en megabytes y porcentaje.
    *   Temperatura de la GPU y consumo de potencia en Watts.
    *   Carga de CPU y memoria RAM del sistema.
    *   Latencia de inferencia en milisegundos y FPS efectivos.
5.  **Mandos de Control Remoto:**
    *   Activar/desactivar inferencia de alta resolución por teselas (SAHI).
    *   Ocultar/mostrar gráficos HUD.
    *   Ajustar el umbral de confianza de YOLO dinámicamente con un deslizador.
    *   Cambiar de fuente o puerto de recepción en caliente.
    *   Capturar instantáneas en alta resolución (*Snapshot* forense).
    *   Activar grabación de video de la misión.
