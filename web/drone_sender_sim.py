#!/usr/bin/env python3
"""
Simulador / Transmisor de Video desde el Dron hacia la Estación Terrestre (PC).
Este script puede ejecutarse:
1. Directamente en la computadora a bordo del dron (Raspberry Pi, Jetson Nano, etc.) conectada a la cámara.
2. En otra máquina dentro de la red local (LAN) para simular la transmisión inalámbrica RF / WiFi.
3. En la misma PC en otra terminal para pruebas locales de extremo a extremo.

Uso:
    python web/drone_sender_sim.py --dest 192.168.1.8 --port 5005 --source samples/synthetic_drone_mission.mp4
    python web/drone_sender_sim.py --dest 127.0.0.1 --port 5005 --source 0 # Usar cámara web como dron
"""

import sys
import time
import socket
import struct
import argparse
import cv2
import numpy as np

def run_drone_transmitter(dest_ip: str, dest_port: int, source: str, fps: int = 30, quality: int = 80, chunk_size: int = 60000):
    print("=" * 65)
    print(" NODO AEREO (UAV TRANSMITTER) - EMISOR DE VIDEO POR PUERTO UDP")
    print("=" * 65)
    print(f" Destino (PC Estación Terrestre): {dest_ip}:{dest_port}")
    print(f" Fuente de video:                 {source}")
    print(f" Tasa objetivo:                   {fps} FPS")
    print(f" Calidad JPEG:                    {quality}%")
    print(f" Tamaño máx paquete UDP:         {chunk_size} bytes")
    print("=" * 65 + "\n")

    # Abrir fuente de video
    src_val = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_val)
    if not cap.isOpened():
        print(f"[DroneTransmitter] ERROR: No se pudo abrir la fuente: {source}")
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Habilitar broadcast si es necesario o buffer de transmisión
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)

    frame_id = 0
    frame_interval = 1.0 / max(1, fps)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    try:
        while True:
            start_t = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                if not source.isdigit():
                    # Repetir video en bucle
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.01)
                    continue

            frame_id = (frame_id + 1) % 1000000

            # Añadir estampa táctica simulada del dron si se desea
            h, w = frame.shape[:2]
            cv2.putText(frame, f"UAV-TX | ID:{frame_id} | {time.strftime('%H:%M:%S')}", 
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Comprimir trama a JPEG
            success, encoded_img = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            jpeg_bytes = encoded_img.tobytes()
            total_len = len(jpeg_bytes)

            # Si la trama cabe en un solo datagrama UDP (< chunk_size)
            if total_len <= chunk_size:
                # Cabecera de 10 bytes: Magic (0xAA55), frame_id (4B), chunk_id=0 (2B), total_chunks=1 (2B)
                header = struct.unpack("!H I H H", struct.pack("!H I H H", 0xAA55, frame_id, 0, 1))
                packet = struct.pack("!H I H H", 0xAA55, frame_id, 0, 1) + jpeg_bytes
                sock.sendto(packet, (dest_ip, dest_port))
            else:
                # Segmentar trama en fragmentos de tamaño 'chunk_size'
                num_chunks = (total_len + chunk_size - 1) // chunk_size
                for chunk_idx in range(num_chunks):
                    start_offset = chunk_idx * chunk_size
                    end_offset = min(start_offset + chunk_size, total_len)
                    chunk_data = jpeg_bytes[start_offset:end_offset]
                    
                    header = struct.pack("!H I H H", 0xAA55, frame_id, chunk_idx, num_chunks)
                    sock.sendto(header + chunk_data, (dest_ip, dest_port))

            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            if frame_id % 90 == 0:
                actual_fps = 1.0 / max(0.001, elapsed + sleep_time)
                kb_size = total_len / 1024.0
                print(f" [UAV] Transmitiendo Frame #{frame_id} | {w}x{h} | {kb_size:.1f} KB | {actual_fps:.1f} FPS -> {dest_ip}:{dest_port}")

    except KeyboardInterrupt:
        print("\n[DroneTransmitter] Transmisión detenida por el usuario.")
    finally:
        cap.release()
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulador de Transmisión de Video desde Dron hacia PC vía UDP")
    parser.add_argument("--dest", type=str, default="127.0.0.1", help="Dirección IP de la PC (Estación Terrestre)")
    parser.add_argument("--port", type=int, default=5005, help="Puerto UDP de recepción en la PC")
    parser.add_argument("--source", type=str, default="samples/synthetic_drone_mission.mp4", help="Archivo de video o índice de cámara (ej. 0)")
    parser.add_argument("--fps", type=int, default=30, help="Fotogramas por segundo a transmitir")
    parser.add_argument("--quality", type=int, default=80, help="Calidad de compresión JPEG (1-100)")
    args = parser.parse_args()

    run_drone_transmitter(args.dest, args.port, args.source, args.fps, args.quality)
