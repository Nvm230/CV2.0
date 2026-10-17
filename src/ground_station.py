import os
import time
import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.detector import Detection
from src.analyzer import EmergencySceneReport, WindowStatus
from src.hardware_monitor import HardwareMonitor

class GroundStationHUD:
    COLOR_BG_DARK = (15, 18, 22)
    COLOR_PANEL_BG = (22, 28, 35)
    COLOR_BORDER = (45, 55, 72)
    COLOR_TEXT_WHITE = (240, 245, 250)
    COLOR_TEXT_DIM = (140, 150, 165)
    COLOR_SAFE_GREEN = (46, 204, 113)
    COLOR_WARN_AMBER = (0, 165, 255)
    COLOR_CRIT_RED = (40, 40, 235)
    COLOR_FIRE_ORANGE = (20, 100, 255)
    COLOR_CYAN_ACCENT = (230, 200, 0)

    def __init__(self, config: Dict[str, Any], hardware_monitor: Optional[HardwareMonitor]=None):
        self.config = config
        self.hud_cfg = config.get('hud', {})
        self.rec_cfg = config.get('recording', {})
        self.hw_monitor = hardware_monitor or HardwareMonitor()
        self.show_hud = True
        self.show_trajectories = self.hud_cfg.get('show_trajectories', True)
        self.show_masks = self.hud_cfg.get('show_masks', True)
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_dir = self.rec_cfg.get('output_dir', 'runs/recordings')
        self.snapshot_dir = 'runs/snapshots'
        self._ensure_dirs()
        self.flash_counter = 0

    def _ensure_dirs(self):
        os.makedirs(self.record_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def toggle_recording(self, frame_size: Tuple[int, int], fps: float=30.0):
        if not self.is_recording:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.record_dir, f'mission_rec_{timestamp}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
            self.is_recording = True
            print(f'[HUD] Recording started: {filename}')
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print('[HUD] Recording stopped and saved.')

    def save_snapshot(self, frame: np.ndarray):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.snapshot_dir, f'snapshot_{timestamp}.png')
        cv2.imwrite(filename, frame)
        print(f'[HUD] Snapshot saved: {filename}')

    def render(self, frame: np.ndarray, detections: List[Detection], report: EmergencySceneReport, inference_time_ms: float=0.0) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()
        self.flash_counter += 1
        self._draw_detections(canvas, detections, report)
        if self.show_hud:
            self._draw_header_bar(canvas, w)
            self._draw_telemetry_panel(canvas, w, inference_time_ms)
            self._draw_threat_matrix_panel(canvas, report)
            self._draw_footer_controls(canvas, w, h)
            if report.burning_windows > 0 and self.flash_counter % 20 < 10:
                cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), self.COLOR_CRIT_RED, 4)
        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(canvas)
        return canvas

    def _draw_detections(self, canvas: np.ndarray, detections: List[Detection], report: EmergencySceneReport):
        win_status_map = {res.window_id: res for res in report.window_assessments}
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            if self.show_trajectories and len(det.trajectory) > 1:
                pts = np.array(det.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                traj_color = self.COLOR_FIRE_ORANGE if det.class_name == 'fire' else self.COLOR_CYAN_ACCENT
                cv2.polylines(canvas, [pts], isClosed=False, color=traj_color, thickness=2, lineType=cv2.LINE_AA)
            if det.class_name == 'window':
                res = win_status_map.get(det.track_id)
                if res and res.status == WindowStatus.CRITICAL_BURNING:
                    color = self.COLOR_CRIT_RED
                    label = f'WIN #{det.track_id} [CRITICAL BURNING - {int(res.overlap_ratio * 100)}%]'
                    sub_rect = canvas[max(0, y1):min(canvas.shape[0], y2), max(0, x1):min(canvas.shape[1], x2)]
                    if sub_rect.size > 0:
                        red_mask = np.full_like(sub_rect, (0, 0, 180), dtype=np.uint8)
                        cv2.addWeighted(sub_rect, 0.7, red_mask, 0.3, 0, sub_rect)
                elif res and res.status == WindowStatus.WARNING:
                    color = self.COLOR_WARN_AMBER
                    label = f'WIN #{det.track_id} [WARN PROXIMITY: {int(res.distance_to_nearest_fire)}px]'
                else:
                    color = self.COLOR_SAFE_GREEN
                    label = f'WIN #{det.track_id} [SAFE]'
                self._draw_tactical_corners(canvas, x1, y1, x2, y2, color, corner_len=18)
                self._draw_tag(canvas, label, x1, y1 - 8, color)
            elif det.class_name == 'fire':
                color = self.COLOR_FIRE_ORANGE
                label = f'FIRE #{det.track_id} ({det.confidence:.2f})'
                pulse_thick = 3 if self.flash_counter % 12 < 6 else 2
                self._draw_tactical_corners(canvas, x1, y1, x2, y2, color, corner_len=22, thickness=pulse_thick)
                self._draw_tag(canvas, label, x1, y1 - 8, color)
            else:
                if self.config.get('classes', {}).get('filter_only_targets', True):
                    continue
                color = (180, 180, 180)
                label = f'{det.class_name.upper()} #{det.track_id}'
                self._draw_tactical_corners(canvas, x1, y1, x2, y2, color, corner_len=12)
                self._draw_tag(canvas, label, x1, y1 - 6, color)

    def _draw_tactical_corners(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], corner_len: int=15, thickness: int=2):
        w_box = x2 - x1
        h_box = y2 - y1
        cl_x = min(corner_len, w_box // 2)
        cl_y = min(corner_len, h_box // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1 + cl_x, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl_y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2 - cl_x, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl_y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1 + cl_x, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl_y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2 - cl_x, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl_y), color, thickness, cv2.LINE_AA)

    def _draw_tag(self, img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]):
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.45
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        y_bg = max(0, y - th - 4)
        cv2.rectangle(img, (x, y_bg), (x + tw + 6, y + 2), (20, 24, 30), -1)
        cv2.rectangle(img, (x, y_bg), (x + tw + 6, y + 2), color, 1)
        cv2.putText(img, text, (x + 3, y - 2), font, scale, self.COLOR_TEXT_WHITE, thick, cv2.LINE_AA)

    def _draw_header_bar(self, img: np.ndarray, width: int):
        header_h = 36
        overlay = img[:header_h, :width].copy()
        cv2.rectangle(overlay, (0, 0), (width, header_h), self.COLOR_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.85, img[:header_h, :width], 0.15, 0, img[:header_h, :width])
        cv2.line(img, (0, header_h), (width, header_h), self.COLOR_BORDER, 1)
        cv2.putText(img, 'TETHERED DRONE GS [CV2.0] // DUAL DETECTION ENGINE', (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.circle(img, (width // 2 - 80, 18), 5, self.COLOR_SAFE_GREEN, -1)
        cv2.putText(img, 'UMBILICAL: ACTIVE (GMC BOT-SORT)', (width // 2 - 68, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        if self.is_recording:
            rec_color = self.COLOR_CRIT_RED if self.flash_counter % 20 < 10 else (100, 100, 100)
            cv2.circle(img, (width - 150, 18), 6, rec_color, -1)
            cv2.putText(img, 'REC ON', (width - 138, 23), cv2.FONT_HERSHEY_DUPLEX, 0.45, self.COLOR_CRIT_RED, 1, cv2.LINE_AA)

    def _draw_telemetry_panel(self, img: np.ndarray, width: int, inference_time_ms: float):
        metrics = self.hw_monitor.get_all_metrics()
        gpu = metrics['gpu']
        cpu = metrics['cpu']
        fps = self.hw_monitor.tick_fps(inference_time_ms)
        panel_w = 340
        panel_h = 160
        x0 = width - panel_w - 15
        y0 = 48
        sub = img[y0:y0 + panel_h, x0:x0 + panel_w]
        if sub.shape[0] == panel_h and sub.shape[1] == panel_w:
            panel_bg = np.full_like(sub, self.COLOR_PANEL_BG)
            cv2.addWeighted(sub, 0.2, panel_bg, 0.8, 0, sub)
            cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), self.COLOR_BORDER, 1)
        cv2.putText(img, 'HARDWARE TELEMETRY', (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, self.COLOR_CYAN_ACCENT, 1, cv2.LINE_AA)
        fps_color = self.COLOR_SAFE_GREEN if fps >= 30 else self.COLOR_WARN_AMBER
        cv2.putText(img, f'FPS: {fps:.1f}  |  Latency: {inference_time_ms:.1f} ms', (x0 + 10, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)
        gpu_load = gpu.get('utilization_gpu', 0)
        gpu_temp = gpu.get('temperature_c', 0)
        temp_color = self.COLOR_SAFE_GREEN if gpu_temp < 65 else self.COLOR_WARN_AMBER if gpu_temp < 75 else self.COLOR_CRIT_RED
        cv2.putText(img, f'GPU: NVIDIA RTX ({gpu_load}%)  Temp: {gpu_temp} C', (x0 + 10, y0 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, temp_color, 1, cv2.LINE_AA)
        vram_used = gpu.get('vram_used_mb', 0)
        vram_total = gpu.get('vram_total_mb', 12288)
        vram_pct = gpu.get('vram_percent', 0.0)
        cv2.putText(img, f'VRAM: {vram_used}/{vram_total} MB ({vram_pct}%)', (x0 + 10, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        bar_w = panel_w - 20
        bar_h = 6
        cv2.rectangle(img, (x0 + 10, y0 + 100), (x0 + 10 + bar_w, y0 + 100 + bar_h), (50, 60, 75), -1)
        fill_w = int(bar_w * (vram_pct / 100.0))
        cv2.rectangle(img, (x0 + 10, y0 + 100), (x0 + 10 + fill_w, y0 + 100 + bar_h), self.COLOR_CYAN_ACCENT, -1)
        cpu_pct = cpu.get('cpu_percent', 0)
        ram_used = cpu.get('ram_used_gb', 0)
        cv2.putText(img, f'CPU (Core Ultra 7): {cpu_pct}% | RAM: {ram_used} GB', (x0 + 10, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        pwr = gpu.get('power_watts', 0.0)
        if pwr > 0:
            cv2.putText(img, f'Power Draw: {pwr} W', (x0 + 10, y0 + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_threat_matrix_panel(self, img: np.ndarray, report: EmergencySceneReport):
        panel_w = 320
        panel_h = 160
        x0 = 15
        y0 = 48
        sub = img[y0:y0 + panel_h, x0:x0 + panel_w]
        if sub.shape[0] == panel_h and sub.shape[1] == panel_w:
            panel_bg = np.full_like(sub, self.COLOR_PANEL_BG)
            cv2.addWeighted(sub, 0.2, panel_bg, 0.8, 0, sub)
            cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), self.COLOR_BORDER, 1)
        cv2.putText(img, 'EMERGENCY HAZARD MATRIX', (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, self.COLOR_CYAN_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, f'Total Windows Monitored: {report.total_windows}', (x0 + 10, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f'  * SAFE: {report.safe_windows}', (x0 + 10, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_SAFE_GREEN, 1, cv2.LINE_AA)
        cv2.putText(img, f'  * PROXIMITY ALERT: {report.warning_windows}', (x0 + 10, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WARN_AMBER, 1, cv2.LINE_AA)
        crit_color = self.COLOR_CRIT_RED if report.burning_windows > 0 else self.COLOR_TEXT_DIM
        cv2.putText(img, f'  * ACTIVE BURNING: {report.burning_windows}', (x0 + 10, y0 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, crit_color, 1, cv2.LINE_AA)
        cv2.putText(img, f'Fires Detected: {report.active_fire_count}  |  Spread: {report.fire_spread_velocity_px_s:+.0f} px/s', (x0 + 10, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_FIRE_ORANGE, 1, cv2.LINE_AA)
        sev_color = self.COLOR_SAFE_GREEN if report.max_severity < 0.3 else self.COLOR_WARN_AMBER if report.max_severity < 0.7 else self.COLOR_CRIT_RED
        cv2.putText(img, f'Threat Index: {int(report.max_severity * 100)}%', (x0 + 10, y0 + 148), cv2.FONT_HERSHEY_DUPLEX, 0.42, sev_color, 1, cv2.LINE_AA)

    def _draw_footer_controls(self, img: np.ndarray, width: int, height: int):
        footer_h = 26
        y0 = height - footer_h
        cv2.rectangle(img, (0, y0), (width, height), self.COLOR_PANEL_BG, -1)
        cv2.line(img, (0, y0), (width, y0), self.COLOR_BORDER, 1)
        guide = '[Q] Quit  |  [R] Record  |  [S] Snapshot  |  [P] SAHI Hyper-Inference  |  [T] Traj  |  [H] HUD'
        cv2.putText(img, guide, (15, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_DIM, 1, cv2.LINE_AA)
