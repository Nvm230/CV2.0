#!/usr/bin/env bash
# ==============================================================================
# Lanzador de la Estación Terrestre Web CV2.0
# Inicia el servidor web FastAPI para transmisión en red local (LAN)
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DIR")"

cd "$ROOT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Detectar IP local
LOCAL_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n 1 || echo "127.0.0.1")

echo "========================================================================"
echo "  🚀 INICIANDO SERVIDOR WEB ESTACIÓN TERRESTRE CV2.0"
echo "  Hardware: NVIDIA GPU RTX + PyTorch CUDA"
echo "  Acceso Local:             http://127.0.0.1:8000"
echo "  Acceso Red Local (LAN):   http://${LOCAL_IP}:8000"
echo "  Puerto de escucha Dron:   0.0.0.0:5005 (UDP)"
echo "========================================================================"
echo ""
echo "  Para ver el video en cualquier celular, tablet o laptop:"
echo "  Asegúrate de estar en el mismo WiFi o red y abre en el navegador:"
echo "  👉 http://${LOCAL_IP}:8000"
echo ""

python3 web/app.py
