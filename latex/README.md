# Documentación y Compilación del Informe de Tesis en LaTeX

Este directorio contiene el informe técnico y capítulo formal de tesis correspondiente al **Subsistema de Ingestión de Video en Red, Procesamiento en Estación Terrena y Retransmisión Web Táctica para Dron Cautivo (CV2.0)**.

*   Archivo principal: `latex/informe_sistema_vision_web_dron.tex`

---

## 1. Estructura del Documento

El documento ha sido elaborado siguiendo los estándares académicos de **UTEC** y el formato de publicaciones internacionales (IEEE):

1.  **Portada y Metadatos:** Título formal, autor, afiliación a UTEC y especificaciones de la plataforma de cómputo (NVIDIA GPU RTX / CUDA / PyTorch).
2.  **Resumen y Abstract bilingüe (Español / Inglés).**
3.  **Sección 1: Introducción y Justificación Técnica:** Por qué descartar el cómputo embebido pesado a bordo del dron (*Edge Computing Bottlenecks*) y adoptar una arquitectura *Edge-to-Ground* para drones cautivos.
4.  **Sección 2: Topología y Arquitectura del Sistema:** Diagrama de bloques multinivel en **TikZ** y formulación matemática de la latencia acumulada extremo a extremo ($T_{\text{extremo-a-extremo}}$).
5.  **Sección 3: Protocolos de Ingesta y Recepción de Video:** Comparativa técnica entre UDP Datagrams, RTSP/RTP, WebRTC y TCP. Estructura de cabeceras de datagramas y mitigación de bloqueos reentrantes con `threading.RLock`.
6.  **Sección 4: Pipeline de Procesamiento de Visión Artificial (CV2.0):** YOLOX-26 FP16, inferencia por teselas (SAHI), seguimiento BoT-SORT con GMC, y formulación del cálculo de intersección geométrica IoU y velocidad de expansión del fuego ($v_{\text{fuego}}$).
7.  **Sección 5: Servidor de Retransmisión Web en Red Local (LAN):** Implementación con FastAPI y Uvicorn, streaming continuo HTTP Multipart (MJPEG), telemetría vectorial de alta frecuencia por WebSockets y API RESTful de control bidireccional.
8.  **Sección 6: Diseño de la Interfaz Web Táctica (Dashboard):** Principios de ergonomía cognitiva *Dark Aero Tactical*, diseño responsivo multiplataforma (celulares, tablets, PC) y sintetizador de alertas sonoras Web Audio API.
9.  **Sección 7: Evaluación Experimental y Resultados:** Tablas de latencia medida por etapa (7.90 ms en total en NVIDIA GPU RTX, >100 FPS sostenidos) y rendimiento bajo múltiples clientes concurrentes en LAN.
10. **Sección 8: Conclusiones y Trabajo Futuro.**
11. **Referencias Bibliográficas (Formato IEEE).**

---

## 2. Instrucciones de Compilación

### Opción A: Compilación en Overleaf (Recomendado)
1. Inicia sesión en [Overleaf](https://www.overleaf.com/).
2. Crea un nuevo proyecto en blanco o sube el archivo `informe_sistema_vision_web_dron.tex`.
3. Selecciona el compilador estándar **pdfLaTeX** (o **XeLaTeX**).
4. Haz clic en **Recompile**. Se generará el PDF listo para impresión o inclusión en la memoria de tesis.

### Opción B: Compilación Local en Linux (TeX Live)
Si dispones de `texlive` instalado en el sistema:
```bash
cd latex/
pdflatex informe_sistema_vision_web_dron.tex
pdflatex informe_sistema_vision_web_dron.tex # Segunda pasada para actualizar referencias y tabla de contenidos
```

### Opción C: Inclusión en una Tesis Existente
Si ya cuentas con una plantilla de tesis principal (`main.tex` o `tesis.tex`), puedes incorporar este capítulo utilizando el comando de LaTeX:
```latex
\input{latex/informe_sistema_vision_web_dron.tex}
```
o incorporando las secciones deseadas dentro del cuerpo de tu documento.
