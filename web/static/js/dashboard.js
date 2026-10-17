/**
 * CV2.0 Tactical Ground Station - Client Engine
 * Maneja WebSockets en tiempo real, streaming MJPEG, alarmas sonoras Web Audio
 * y controles de visión por computadora.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elementos DOM principales
  const videoStream = document.getElementById("mainVideoStream");
  const connectionDot = document.getElementById("connectionDot");
  const connectionStatusText = document.getElementById("connectionStatusText");
  const streamSourceBadge = document.getElementById("streamSourceBadge");
  const streamResBadge = document.getElementById("streamResBadge");
  const streamFpsBadge = document.getElementById("streamFpsBadge");
  
  // Métricas rápidas
  const metricActiveFires = document.getElementById("metricActiveFires");
  const metricBurningWindows = document.getElementById("metricBurningWindows");
  const metricSpreadRate = document.getElementById("metricSpreadRate");
  const metricLatency = document.getElementById("metricLatency");
  
  // Threat Matrix
  const threatBanner = document.getElementById("threatBanner");
  const threatStatusTitle = document.getElementById("threatStatusTitle");
  const threatStatusDesc = document.getElementById("threatStatusDesc");
  const severityGaugeFill = document.getElementById("severityGaugeFill");
  const severityGaugeVal = document.getElementById("severityGaugeVal");
  const winSafeCount = document.getElementById("winSafeCount");
  const winWarnCount = document.getElementById("winWarnCount");
  const winCritCount = document.getElementById("winCritCount");
  
  // Detecciones
  const detectionsList = document.getElementById("detectionsList");
  const detCountBadge = document.getElementById("detCountBadge");
  
  // Hardware
  const gpuNameEl = document.getElementById("gpuName");
  const gpuLoadEl = document.getElementById("gpuLoad");
  const gpuTempEl = document.getElementById("gpuTemp");
  const vramTextEl = document.getElementById("vramText");
  const vramBarFill = document.getElementById("vramBarFill");
  const cpuPercentEl = document.getElementById("cpuPercent");
  const ramTextEl = document.getElementById("ramText");
  
  // Botones de control
  const btnModeAnnotated = document.getElementById("btnModeAnnotated");
  const btnModeRaw = document.getElementById("btnModeRaw");
  const btnSnapshot = document.getElementById("btnSnapshot");
  const btnFullscreen = document.getElementById("btnFullscreen");
  const btnAudioAlert = document.getElementById("btnAudioAlert");
  
  const btnToggleSahi = document.getElementById("btnToggleSahi");
  const btnToggleHud = document.getElementById("btnToggleHud");
  const btnToggleTraj = document.getElementById("btnToggleTraj");
  const btnToggleRec = document.getElementById("btnToggleRec");
  const rangeConfidence = document.getElementById("rangeConfidence");
  const valConfidence = document.getElementById("valConfidence");
  const inputSource = document.getElementById("inputSource");
  const btnApplySource = document.getElementById("btnApplySource");
  const lanPill = document.getElementById("lanPill");

  // Estado del cliente
  let isAudioMuted = true;
  let audioCtx = null;
  let lastCriticalState = false;
  let lastBeepTime = 0;

  // 1. Inicialización de Audio Táctico (Web Audio API)
  function initAudio() {
    if (!audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AudioContext();
    }
    if (audioCtx.state === "suspended") {
      audioCtx.resume();
    }
  }

  function playAlertBeep(freq = 880, duration = 0.15) {
    if (isAudioMuted || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("Audio alert error:", e);
    }
  }

  btnAudioAlert.addEventListener("click", () => {
    initAudio();
    isAudioMuted = !isAudioMuted;
    btnAudioAlert.classList.toggle("active", !isAudioMuted);
    btnAudioAlert.innerHTML = !isAudioMuted 
      ? '<span class="icon">🔔</span> Alertas: ON' 
      : '<span class="icon">🔕</span> Alertas: MUTE';
    if (!isAudioMuted) {
      playAlertBeep(660, 0.1);
    }
  });

  // 2. Conexión WebSocket de Telemetría
  let ws = null;
  function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log("[WS] Conectado a la Estación Terrestre CV2.0");
      connectionDot.className = "status-dot";
      connectionStatusText.textContent = "DRON ENLACE ACTIVO";
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateDashboard(data);
      } catch (e) {
        console.error("Error parseando telemetría:", e);
      }
    };

    ws.onclose = () => {
      console.warn("[WS] Conexión perdida. Reintentando en 2s...");
      connectionDot.className = "status-dot danger";
      connectionStatusText.textContent = "DESCONECTADO";
      setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (err) => {
      console.error("[WS] Error:", err);
      ws.close();
    };
  }

  connectWebSocket();

  // 3. Renderizado y actualización reactiva del Dashboard
  function updateDashboard(data) {
    if (!data) return;

    // Stream & Enlace
    const stream = data.stream_stats || {};
    const pipe = data.pipeline || {};
    const report = data.emergency_report || {};
    const hw = data.hardware || {};
    const dets = data.detections || [];

    if (stream.connected) {
      connectionDot.className = "status-dot";
      connectionStatusText.textContent = `RECEPTOR OK (${stream.mode.toUpperCase()})`;
    } else {
      connectionDot.className = "status-dot warn";
      connectionStatusText.textContent = "ESPERANDO DRON...";
    }

    if (streamSourceBadge) streamSourceBadge.textContent = stream.source || "N/A";
    if (streamResBadge) streamResBadge.textContent = stream.resolution || "--x--";
    if (streamFpsBadge) streamFpsBadge.textContent = `${pipe.fps || 0} FPS`;

    // Métricas Rápidas
    if (metricActiveFires) metricActiveFires.textContent = report.active_fire_count || 0;
    if (metricBurningWindows) metricBurningWindows.textContent = report.burning_windows || 0;
    if (metricSpreadRate) metricSpreadRate.textContent = (report.fire_spread_velocity_px_s || 0).toFixed(1);
    if (metricLatency) metricLatency.textContent = (pipe.inference_ms || 0).toFixed(1);

    // Threat Matrix & Estado de Emergencia
    const burning = report.burning_windows || 0;
    const activeFire = report.active_fire_count || 0;
    const maxSev = (report.max_severity || 0) * 100;

    if (severityGaugeFill) severityGaugeFill.style.width = `${Math.min(100, maxSev)}%`;
    if (severityGaugeVal) severityGaugeVal.textContent = `${maxSev.toFixed(0)}%`;

    if (winSafeCount) winSafeCount.textContent = report.safe_windows || 0;
    if (winWarnCount) winWarnCount.textContent = report.warning_windows || 0;
    if (winCritCount) winCritCount.textContent = burning;

    if (burning > 0 || maxSev > 75) {
      threatBanner.className = "threat-badge-card critical";
      threatStatusTitle.textContent = "ALERTA CRÍTICA: ESTRUCTURA EN FUEGO";
      threatStatusDesc.textContent = `${burning} ventanas comprometidas por foco ígneo`;
      
      // Alarma sonora periódica (cada 2 segundos si hay fuego crítico)
      const now = Date.now();
      if (now - lastBeepTime > 2000) {
        playAlertBeep(920, 0.25);
        lastBeepTime = now;
      }
    } else if (activeFire > 0 || (report.warning_windows || 0) > 0) {
      threatBanner.className = "threat-badge-card";
      threatStatusTitle.textContent = "ADVERTENCIA: FOCO TÉRMICO ACTIVO";
      threatStatusDesc.textContent = "Monitoreando proximidad a perímetros";
    } else {
      threatBanner.className = "threat-badge-card";
      threatStatusTitle.style.color = "var(--accent-emerald)";
      threatStatusTitle.textContent = "SECTOR ESTABLE / SEGURO";
      threatStatusDesc.textContent = "Sin anomalías térmicas críticas";
    }

    // Lista de Detecciones
    if (detCountBadge) detCountBadge.textContent = `${dets.length} OBJETOS`;
    if (detectionsList) {
      if (dets.length === 0) {
        detectionsList.innerHTML = '<div style="color:var(--text-muted); font-size:0.75rem; text-align:center; padding:12px;">Sin detecciones activas</div>';
      } else {
        detectionsList.innerHTML = dets.map(d => {
          const isFire = d.class_name.toLowerCase().includes("fire") || d.class_name.toLowerCase().includes("fuego");
          const dotClass = isFire ? "det-dot fire" : "det-dot window";
          return `
            <div class="detection-item">
              <div class="det-tag">
                <span class="${dotClass}"></span>
                <strong>${d.class_name.toUpperCase()} #${d.track_id}</strong>
              </div>
              <div class="det-conf">${(d.confidence * 100).toFixed(0)}% | [${d.bbox[0]}, ${d.bbox[1]}]</div>
            </div>
          `;
        }).join("");
      }
    }

    // Hardware Telemetry
    if (gpuNameEl) gpuNameEl.textContent = hw.gpu_name || "NVIDIA GPU";
    if (gpuLoadEl) gpuLoadEl.textContent = `${hw.gpu_utilization || 0}%`;
    if (gpuTempEl) gpuTempEl.textContent = `${hw.temperature_c || 0} °C`;
    if (vramTextEl) vramTextEl.textContent = `${hw.vram_used_mb || 0} / ${hw.vram_total_mb || 12288} MB`;
    if (vramBarFill) vramBarFill.style.width = `${hw.vram_percent || 0}%`;
    if (cpuPercentEl) cpuPercentEl.textContent = `${(hw.cpu_percent || 0).toFixed(0)}%`;
    if (ramTextEl) ramTextEl.textContent = `${(hw.ram_used_gb || 0).toFixed(1)} GB (${(hw.ram_percent || 0).toFixed(0)}%)`;

    // Botones de Estado Táctico
    if (pipe.sahi_enabled !== undefined) {
      btnToggleSahi.classList.toggle("active", pipe.sahi_enabled);
    }
    if (pipe.recording !== undefined) {
      btnToggleRec.classList.toggle("active", pipe.recording);
    }
    if (pipe.trajectories !== undefined) {
      btnToggleTraj.classList.toggle("active", pipe.trajectories);
    }
  }

  // 4. Conmutación de Modo de Video
  btnModeAnnotated.addEventListener("click", () => {
    btnModeAnnotated.classList.add("active");
    btnModeRaw.classList.remove("active");
    videoStream.src = "/api/stream/annotated";
  });

  btnModeRaw.addEventListener("click", () => {
    btnModeRaw.classList.add("active");
    btnModeAnnotated.classList.remove("active");
    videoStream.src = "/api/stream/raw";
  });

  // 5. Captura de Instantánea (Snapshot)
  btnSnapshot.addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = "/api/snapshot";
    link.download = `drone_cv2_snapshot_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  // 6. Pantalla Completa
  btnFullscreen.addEventListener("click", () => {
    const container = document.getElementById("videoContainer");
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(err => alert(`Error: ${err.message}`));
    } else {
      document.exitFullscreen();
    }
  });

  // 7. Acciones de Control Remoto del Pipeline
  async function sendControl(action, value = null) {
    try {
      const res = await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, value })
      });
      return await res.json();
    } catch (e) {
      console.error("Control API error:", e);
    }
  }

  btnToggleSahi.addEventListener("click", () => sendControl("toggle_sahi"));
  btnToggleHud.addEventListener("click", () => sendControl("toggle_hud"));
  btnToggleTraj.addEventListener("click", () => sendControl("toggle_trajectories"));
  btnToggleRec.addEventListener("click", () => sendControl("toggle_recording"));

  rangeConfidence.addEventListener("input", (e) => {
    valConfidence.textContent = parseFloat(e.target.value).toFixed(2);
  });

  rangeConfidence.addEventListener("change", (e) => {
    sendControl("set_confidence", parseFloat(e.target.value));
  });

  btnApplySource.addEventListener("click", () => {
    const src = inputSource.value.trim();
    if (src) {
      sendControl("change_source", src);
    }
  });

  // Presets de fuente rápida
  document.querySelectorAll(".btn-preset-source").forEach(btn => {
    btn.addEventListener("click", () => {
      const src = btn.getAttribute("data-src");
      inputSource.value = src;
      sendControl("change_source", src);
    });
  });

  // Copiar IP de la LAN al portapapeles
  if (lanPill) {
    lanPill.addEventListener("click", () => {
      const url = lanPill.getAttribute("data-url");
      navigator.clipboard.writeText(url).then(() => {
        const orig = lanPill.innerHTML;
        lanPill.innerHTML = '<span>✅ ¡Copiado!</span>';
        setTimeout(() => { lanPill.innerHTML = orig; }, 1800);
      });
    });
  }
});
