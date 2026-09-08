# Arquitectura del Sistema de Visión CV2.0

Este documento detalla el diseño, la topología y el pipeline de inferencia del sistema de visión artificial empleado para la detección temprana de anomalías (incendios) mediante vehículos aéreos no tripulados (UAV).

---

## 1. Topología del Sistema: Edge-to-Ground

El procesamiento de redes neuronales profundas sobre flujos de video en resolución 4K exige una inmensa capacidad de cómputo matricial. Por ello, la arquitectura descarta el enfoque *Edge Computing* tradicional (procesar a bordo del dron) para evitar latencias extremas y sobrecalentamiento.

El diseño se basa en un esquema de **Estación Terrestre Cautiva (Ground Station)**:
1. **Transmisión:** El UAV captura el video en resolución nativa (4K) y lo transmite mediante un enlace de baja latencia (RTSP/UDP) hacia la base.
2. **Procesamiento Centralizado:** Una estación terrestre equipada con hardware de aceleración gráfica procesa el flujo de video y extrae la telemetría.
3. **Telemetría:** Los datos procesados alimentan un HUD (Head-Up Display) táctico en tiempo real.

---

## 2. Pipeline de Inferencia Avanzada

Para lograr la detección de focos incipientes a largas distancias y mantener un procesamiento en tiempo real fluido (60 FPS), el motor de visión utiliza tres capas de optimización:

### A. Modelo de Extracción (YOLO11x)
El núcleo del reconocimiento es el modelo **YOLO11x (Extra Large)**, la arquitectura más profunda de su familia (aprox. 57 millones de parámetros). Su tamaño garantiza la mayor precisión posible y sensibilidad ante características sub-píxel, indispensable para humo lejano.

### B. Segmentación Inteligente (SAHI + Staggering)
Procesar una matriz completa de 4K (3840x2160) en una sola pasada agota la VRAM de la mayoría de tarjetas gráficas y desploma los FPS. Para solucionar esto, el sistema implementa **SAHI (Slicing Aided Hyper Inference)**:
- El fotograma 4K se divide dinámicamente en teselas superpuestas (ej. 640x640).
- La técnica de *Staggering* permite procesar estos parches geométricos en lotes paralelos.
- Esto mantiene la nitidez original del objeto (sin compresión), detectando focos pequeños como si se estuviera haciendo zoom, pero distribuyendo la carga computacional equitativamente.

### C. Aceleración por Hardware (TensorRT)
Para evitar el cuello de botella del intérprete estándar de Python, los pesos originales dinámicos (`.pt`) son exportados y compilados de manera estática a un motor **TensorRT (`.engine`) en precisión FP16**. Esto activa los núcleos tensores (Tensor Cores) dedicados de la GPU, transformando un proceso de varios segundos por frame en latencias de milisegundos.

---

## 3. Requisitos Mínimos del Sistema

Para desplegar y ejecutar esta arquitectura de manera fluida, se requiere el siguiente equipamiento mínimo:

### Estación Terrestre (Procesamiento)
Para mantener la latencia por debajo de los 16 milisegundos (necesario para alcanzar 60 FPS con SAHI):
- **GPU (Tarjeta Gráfica):** Compatible con arquitectura CUDA y equipada con Tensor Cores (Generación Turing o superior).
  - **VRAM Mínima:** 8 GB GDDR6 (necesarios para acomodar el modelo "X" y los lotes de imágenes segmentadas).
- **CPU:** Procesador moderno de 6 núcleos o superior (necesario para el decode del stream 4K sin causar asfixia a la GPU).
- **RAM:** 16 GB de memoria de sistema.

### Nodo Aéreo (UAV)
- **Cámara:** Sensor capaz de captura y transmisión de flujo de video RTSP/UDP en resolución 4K (2160p) a 30/60 cuadros por segundo.
- **Enlace de Datos:** Sistema de transmisión de radiofrecuencia (RF) de banda ancha capaz de enviar video con latencia inferior a los 50ms.
