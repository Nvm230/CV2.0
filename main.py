import os
import sys
import time
import argparse
import yaml
import cv2
import numpy as np
from src.detector import DroneVisionDetector
from src.analyzer import SpatialEmergencyAnalyzer
from src.ground_station import GroundStationHUD
from src.hardware_monitor import HardwareMonitor
from src.stream_handler import VideoStreamHandler
from data.download_samples import download_or_generate_samples

def load_config(config_path: str='config/config.yaml') -> dict:
    if not os.path.exists(config_path):
        print(f'[Main] Error: Config file not found at {config_path}')
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def cmd_run(args, config: dict):
    source = args.source or config.get('stream', {}).get('default_source', '0')
    if isinstance(source, str) and (not source.startswith('rtsp://')) and (not os.path.exists(source)):
        print(f"[Main] Source file '{source}' not found. Generating sample media...")
        download_or_generate_samples()
    print(f'\n' + '=' * 70)
    print(' 🚀 INICIANDO ESTACIÓN TERRESTRE CV2.0 (DRON CAUTIVO)')
    print('    Hardware: NVIDIA GPU RTX (Tensor Cores FP16) + Intel Core Ultra')
    print(f'    Fuente de video: {source}')
    print('=' * 70 + '\n')
    hw_monitor = HardwareMonitor()
    detector = DroneVisionDetector(config)
    analyzer = SpatialEmergencyAnalyzer(config)
    hud = GroundStationHUD(config, hardware_monitor=hw_monitor)
    stream = VideoStreamHandler(source, loop_video=True).start()
    if args.record:
        hud.toggle_recording((stream.width, stream.height))
    window_name = 'CV2.0 - Tactical Ground Station HUD (NVIDIA GPU RTX)'
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    frame_count = 0
    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            frame_count += 1
            detections, inf_ms = detector.track(frame)
            report = analyzer.analyze(detections)
            rendered_frame = hud.render(frame, detections, report, inference_time_ms=inf_ms)
            if not args.headless:
                cv2.imshow(window_name, rendered_frame)
                key = cv2.waitKey(1) & 255
                if key == ord('q') or key == 27:
                    print('\n[Main] Exiting by user command.')
                    break
                elif key == ord('r') or key == ord('R'):
                    hud.toggle_recording((stream.width, stream.height))
                elif key == ord('s') or key == ord('S'):
                    hud.save_snapshot(rendered_frame)
                elif key == ord('t') or key == ord('T'):
                    hud.show_trajectories = not hud.show_trajectories
                    print(f'[Main] Trajectories: {hud.show_trajectories}')
                elif key == ord('p') or key == ord('P'):
                    is_enabled = detector.toggle_sahi()
                    print(f"[Main] SAHI Hyper-Inference: {('ENABLED (GPU Parallel Batches)' if is_enabled else 'DISABLED')}")
                elif key == ord('h') or key == ord('H'):
                    hud.show_hud = not hud.show_hud
                    print(f'[Main] HUD Overlays: {hud.show_hud}')
    except KeyboardInterrupt:
        print('\n[Main] Interrupted by user.')
    finally:
        stream.stop()
        if hud.is_recording:
            hud.toggle_recording((0, 0))
        analyzer.save_incident_log()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f'[Main] Session closed. Total frames processed: {frame_count}')

def cmd_test(args, config: dict):
    source = args.source or config.get('stream', {}).get('default_source', 'samples/youtube_drone_fire.mp4.webm')
    if not os.path.exists(source):
        print(f"[Main] Target sample not found at '{source}'. Generating test samples...")
        download_or_generate_samples()
    output_path = args.output or 'runs/test_output.mp4'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f'\n[Test] Running evaluation on: {source}')
    print(f'[Test] Output annotated video: {output_path}')
    hw_monitor = HardwareMonitor()
    detector = DroneVisionDetector(config)
    analyzer = SpatialEmergencyAnalyzer(config)
    hud = GroundStationHUD(config, hardware_monitor=hw_monitor)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[Test] Failed to open video source: {source}')
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames or total_frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    latencies = []
    burning_events = 0
    frame_idx = 0
    print(f'[Test] Processing up to {max_frames} frames ({width}x{height} @ {fps:.1f} FPS)...')
    start_all = time.perf_counter()
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        detections, inf_ms = detector.track(frame)
        latencies.append(inf_ms)
        report = analyzer.analyze(detections)
        if report.burning_windows > 0:
            burning_events += 1
        rendered = hud.render(frame, detections, report, inference_time_ms=inf_ms)
        writer.write(rendered)
        if frame_idx % 30 == 0 or frame_idx == max_frames:
            avg_lat = np.mean(latencies[-30:]) if latencies else 0.0
            print(f'  -> Frame {frame_idx}/{max_frames} | Latency: {avg_lat:.2f} ms ({1000.0 / avg_lat:.1f} FPS) | Burning: {report.burning_windows}')
    cap.release()
    writer.release()
    total_time = time.perf_counter() - start_all
    analyzer.save_incident_log('runs/test_incidents.json')
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    throughput_fps = frame_idx / total_time if total_time > 0 else 0.0
    print('\n' + '=' * 60)
    print(' 📊 RESULTADOS DE PRUEBA DE VISIÓN ARTIFICIAL (CV2.0)')
    print('=' * 60)
    print(f' Frames Procesados:       {frame_idx}')
    print(f' Tiempo Total:            {total_time:.2f} s')
    print(f' Rendimiento Global:      {throughput_fps:.1f} FPS')
    print(f' Latencia Media Inferencia: {avg_latency:.2f} ms')
    print(f' Cuadros con Fuego Activo: {burning_events} ({burning_events / frame_idx * 100:.1f}%)')
    print(f' Video Anotado Guardado:  {output_path}')
    print(f' Log de Incidentes:       runs/test_incidents.json')
    print('=' * 60 + '\n')

def cmd_benchmark(args, config: dict):
    print('\n' + '=' * 70)
    print(' ⚡ BENCHMARK DE RENDIMIENTO EN NVIDIA GPU RTX')
    print('=' * 70)
    detector = DroneVisionDetector(config)
    hw = HardwareMonitor()
    resolutions = [('720p HD', (1280, 720)), ('1080p Full HD', (1920, 1080)), ('1440p 2K', (2560, 1440)), ('4K Ultra HD', (3840, 2160))]
    warmup_iters = 10
    test_iters = 50
    print(f' Ejecutando {test_iters} iteraciones por resolución con FP16 Tensor Cores...\n')
    print(f"{'Resolución':<16} | {'Dimensiones':<12} | {'Latencia Media':<16} | {'FPS Equivalente':<16} | {'VRAM Utilizada':<14}")
    print('-' * 84)
    for label, (w, h) in resolutions:
        detector.clear_tracks()
        dummy_frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(warmup_iters):
            _ = detector.track(dummy_frame)
        latencies = []
        for _ in range(test_iters):
            _, lat_ms = detector.track(dummy_frame)
            latencies.append(lat_ms)
        avg_lat = float(np.mean(latencies))
        fps_equiv = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        gpu_stats = hw.get_gpu_metrics()
        vram_str = f"{gpu_stats['vram_used_mb']} MB"
        print(f"{label:<16} | {f'{w}x{h}':<12} | {f'{avg_lat:.2f} ms':<16} | {f'{fps_equiv:.1f} FPS':<16} | {vram_str:<14}")
    print('-' * 84)
    print(' Benchmark finalizado con éxito.\n')

def cmd_train(args, config: dict):
    from src.trainer import DroneModelTrainer
    data_yaml = args.data or 'data/dataset/data.yaml'
    model = args.model or config.get('model', {}).get('weights', 'yolo26x.pt')
    epochs = args.epochs or 50
    imgsz = args.imgsz or config.get('model', {}).get('imgsz', 640)
    batch = args.batch or 8
    trainer = DroneModelTrainer(data_yaml, model)
    trainer.train(epochs=epochs, imgsz=imgsz, batch=batch)

def main():
    parser = argparse.ArgumentParser(description='CV2.0 - High-Performance Vision & Ground Station for Tethered Drone', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    subparsers = parser.add_subparsers(dest='command', help='Available subcommands')
    run_parser = subparsers.add_parser('run', help='Launch live tactical ground station')
    run_parser.add_argument('--source', type=str, default=None, help='Video source (0, RTSP URL, or file)')
    run_parser.add_argument('--record', action='store_true', help='Start recording automatically')
    run_parser.add_argument('--headless', action='store_true', help='Run without UI window')
    test_parser = subparsers.add_parser('test', help='Run automated test on sample video')
    test_parser.add_argument('--source', type=str, default=None, help='Video file to test')
    test_parser.add_argument('--output', type=str, default='runs/test_output.mp4', help='Output video path')
    test_parser.add_argument('--max-frames', type=int, default=150, help='Max frames to process')
    subparsers.add_parser('benchmark', help='Benchmark NVIDIA GPU RTX performance across resolutions')
    subparsers.add_parser('download-samples', help='Generate/download realistic drone test media')
    train_parser = subparsers.add_parser('train', help='Train model with drone-tailored augmentations')
    train_parser.add_argument('--data', type=str, default='data/dataset/data.yaml', help='Dataset YAML')
    train_parser.add_argument('--model', type=str, default='yolo26x.pt', help='Base model weights')
    train_parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    train_parser.add_argument('--batch', type=int, default=8, help='Batch size')
    web_parser = subparsers.add_parser('web', help='Launch web dashboard for local network (LAN) streaming')
    web_parser.add_argument('--port', type=int, default=8000, help='Web server port (default 8000)')
    web_parser.add_argument('--source', type=str, default=None, help='Initial video stream or network port')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    config = load_config(args.config)
    if args.command == 'run':
        cmd_run(args, config)
    elif args.command == 'web':
        import uvicorn
        from web.app import app, receiver, get_lan_ip
        if args.source:
            receiver.change_source(args.source)
        local_ip = get_lan_ip()
        print("\n" + "=" * 70)
        print(" 🚀 ESTACIÓN TERRESTRE CV2.0 - SERVIDOR WEB LAN INICIADO")
        print(f"    Acceso Local:             http://127.0.0.1:{args.port}")
        print(f"    Acceso Red Local (LAN):   http://{local_ip}:{args.port}")
        print(f"    Puerto UDP Receptor Dron: 0.0.0.0:{receiver.udp_port}")
        print("=" * 70 + "\n")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    elif args.command == 'test':
        cmd_test(args, config)
    elif args.command == 'benchmark':
        cmd_benchmark(args, config)
    elif args.command == 'download-samples':
        download_or_generate_samples()
    elif args.command == 'train':
        cmd_train(args, config)
if __name__ == '__main__':
    main()
