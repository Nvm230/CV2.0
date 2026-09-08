import os
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2
import torch
from ultralytics import YOLO

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    center: Tuple[float, float]
    area: float
    mask: Optional[np.ndarray] = None
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

class AsyncSAHIRunner:

    def __init__(self, model_path: str, device: str, slice_w: int=640, slice_h: int=640, overlap_x: float=0.2, overlap_y: float=0.2, conf_thresh: float=0.5):
        self.model_path = model_path
        self.device = device
        self.slice_w = slice_w
        self.slice_h = slice_h
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.conf_thresh = conf_thresh
        self.queue = queue.Queue(maxsize=1)
        self.latest_detections: List[Tuple[Tuple[float, float, float, float], float, str]] = []
        self.lock = threading.Lock()
        self.running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def submit(self, frame: np.ndarray):
        if not self.queue.full():
            try:
                self.queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def get_detections(self) -> List[Tuple[Tuple[float, float, float, float], float, str]]:
        with self.lock:
            return list(self.latest_detections)

    def _worker_loop(self):
        try:
            slice_model = YOLO(self.model_path)
        except Exception:
            return
        while self.running:
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            time.sleep(0.05)
            h, w = frame.shape[:2]
            slices = []
            step_x = int(self.slice_w * (1.0 - self.overlap_x))
            step_y = int(self.slice_h * (1.0 - self.overlap_y))
            for y in range(0, h, step_y):
                for x in range(0, w, step_x):
                    x_end = min(x + self.slice_w, w)
                    y_end = min(y + self.slice_h, h)
                    x_start = max(0, x_end - self.slice_w)
                    y_start = max(0, y_end - self.slice_h)
                    slices.append((x_start, y_start, x_end, y_end))
            slice_images = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in slices]
            try:
                all_boxes = []
                all_confs = []
                all_names = []
                batch_size = 4
                for i in range(0, len(slice_images), batch_size):
                    chunk_imgs = slice_images[i:i + batch_size]
                    chunk_slices = slices[i:i + batch_size]
                    results = slice_model.predict(chunk_imgs, conf=self.conf_thresh, imgsz=self.slice_w, device=self.device, verbose=False)
                    for idx, s_res in enumerate(results):
                        x_off, y_off, _, _ = chunk_slices[idx]
                        if s_res.boxes is not None and len(s_res.boxes) > 0:
                            s_xyxy = s_res.boxes.xyxy.cpu().numpy()
                            s_confs = s_res.boxes.conf.cpu().numpy()
                            s_cls = s_res.boxes.cls.int().cpu().tolist()
                            names = s_res.names
                            for b_idx in range(len(s_xyxy)):
                                raw_n = names.get(s_cls[b_idx], '').lower()
                                c_conf = float(s_confs[b_idx])
                                if any((k in raw_n for k in ['fire', 'flame'])):
                                    canonical = 'fire'
                                elif 'smoke' in raw_n and c_conf >= 0.65:
                                    canonical = 'smoke'
                                elif any((k in raw_n for k in ['window', 'ventana', 'balcon'])):
                                    canonical = 'window'
                                else:
                                    continue
                                bx1, by1, bx2, by2 = s_xyxy[b_idx]
                                all_boxes.append([bx1 + x_off, by1 + y_off, bx2 + x_off, by2 + y_off])
                                all_confs.append(c_conf)
                                all_names.append(canonical)
                    time.sleep(0.015)
                if all_boxes:
                    indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, score_threshold=self.conf_thresh, nms_threshold=0.45)
                    merged = []
                    if len(indices) > 0:
                        for i in indices.flatten():
                            merged.append((tuple(all_boxes[i]), all_confs[i], all_names[i]))
                    with self.lock:
                        self.latest_detections = merged
            except Exception:
                pass

    def stop(self):
        self.running = False

class TrackStabilizer:

    def __init__(self, min_hits: int=3, smoothing_alpha: float=0.75, max_lost: int=15):
        self.min_hits = min_hits
        self.alpha = smoothing_alpha
        self.max_lost = max_lost
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, raw_detections: List[Detection]) -> List[Detection]:
        confirmed_detections: List[Detection] = []
        current_seen_ids = set()
        for det in raw_detections:
            t_id = det.track_id
            if t_id == -1:
                if det.confidence >= 0.55:
                    confirmed_detections.append(det)
                continue
            current_seen_ids.add(t_id)
            if t_id not in self.tracks:
                self.tracks[t_id] = {'hits': 1, 'lost': 0, 'bbox': list(det.bbox), 'class_name': det.class_name, 'conf': det.confidence, 'trajectory': list(det.trajectory)}
            else:
                track = self.tracks[t_id]
                track['hits'] += 1
                track['lost'] = 0
                track['conf'] = det.confidence
                track['class_name'] = det.class_name
                prev_b = track['bbox']
                new_b = det.bbox
                track['bbox'] = [self.alpha * new_b[i] + (1.0 - self.alpha) * prev_b[i] for i in range(4)]
                track['trajectory'] = list(det.trajectory)
            track = self.tracks[t_id]
            if track['hits'] >= self.min_hits:
                sb = tuple(track['bbox'])
                scx = (sb[0] + sb[2]) / 2.0
                scy = (sb[1] + sb[3]) / 2.0
                sarea = max(0.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
                confirmed_detections.append(Detection(track_id=t_id, class_id=det.class_id, class_name=track['class_name'], confidence=track['conf'], bbox=sb, center=(scx, scy), area=sarea, trajectory=track['trajectory']))
        for t_id in list(self.tracks.keys()):
            if t_id not in current_seen_ids:
                self.tracks[t_id]['lost'] += 1
                if self.tracks[t_id]['lost'] > self.max_lost:
                    del self.tracks[t_id]
        return confirmed_detections

    def clear(self):
        self.tracks.clear()

class DroneVisionDetector:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get('model', {})
        self.tracker_cfg = config.get('tracker', {})
        self.classes_cfg = config.get('classes', {})
        self.temporal_cfg = config.get('temporal_filter', {})
        self.sahi_cfg = config.get('sahi', {})
        self.device = self._select_device(self.model_cfg.get('device', '0'))
        self.imgsz = self.model_cfg.get('imgsz', 1280)
        self.conf_thresh = self.model_cfg.get('confidence_threshold', 0.4)
        self.win_conf_thresh = self.model_cfg.get('window_confidence_threshold', 0.45)
        self.iou_thresh = self.model_cfg.get('iou_threshold', 0.45)
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.max_history_len = 30
        self.frame_count = 0
        self.cached_windows: List[Detection] = []
        min_hits = self.temporal_cfg.get('min_consecutive_hits', 3)
        alpha = self.temporal_cfg.get('box_smoothing_alpha', 0.75)
        max_lost = self.temporal_cfg.get('max_lost_memory', 15)
        self.stabilizer = TrackStabilizer(min_hits=min_hits, smoothing_alpha=alpha, max_lost=max_lost)
        self.fire_weights_path = self._resolve_fire_weights()
        print(f'[Detector] Loading primary fire model: {self.fire_weights_path}')
        self.model = YOLO(self.fire_weights_path)
        self.window_model_path = self._resolve_window_weights()
        self.window_model = None
        if os.path.isfile(self.window_model_path):
            try:
                print(f'[Detector] Loading window detector: {self.window_model_path}')
                self.window_model = YOLO(self.window_model_path)
            except Exception as e:
                print(f'[Detector] Window model note: {e}')
        self.sahi_enabled = self.sahi_cfg.get('enabled', False)
        self.async_sahi: Optional[AsyncSAHIRunner] = None
        if self.sahi_enabled:
            self._start_sahi()
        self._warmup()
        self.filter_only_targets = self.classes_cfg.get('filter_only_targets', True)
        self.fire_labels = set((c.lower() for c in self.classes_cfg.get('fire_classes', ['fire', 'flame', 'smoke', 'fuego', 'humo'])))
        self.window_labels = set((c.lower() for c in self.classes_cfg.get('window_classes', ['window', 'window_frame', 'balcony', 'ventana', 'balcon'])))

    def _select_device(self, dev: str) -> str:
        if torch.cuda.is_available():
            dev_str = f'cuda:{dev}' if dev.isdigit() else 'cuda'
            print(f'[Detector] Using GPU: {torch.cuda.get_device_name(0)} ({dev_str})')
            return dev_str
        print('[Detector] Warning: CUDA not available, using CPU.')
        return 'cpu'

    def _resolve_fire_weights(self) -> str:
        custom = self.model_cfg.get('custom_weights')
        if custom and os.path.isfile(custom):
            return custom
        engine_path = 'weights/firedetect-11x.engine'
        if os.path.isfile(engine_path):
            print(f'[Detector] Using compiled TensorRT Engine on RTX 5070: {engine_path}')
            return engine_path
        primary = self.model_cfg.get('weights', 'weights/firedetect-11x.pt')
        if os.path.isfile(primary):
            return primary
        return 'yolo11x.pt'

    def _resolve_window_weights(self) -> str:
        engine_path = 'weights/window_detector.engine'
        if os.path.isfile(engine_path):
            print(f'[Detector] Using compiled Window TensorRT Engine: {engine_path}')
            return engine_path
        return self.model_cfg.get('window_weights', 'weights/window_detector.pt')

    def _start_sahi(self):
        py_weights = 'weights/firedetect-11x.pt' if os.path.isfile('weights/firedetect-11x.pt') else self.fire_weights_path
        self.async_sahi = AsyncSAHIRunner(model_path=py_weights, device=self.device, slice_w=self.sahi_cfg.get('slice_width', 640), slice_h=self.sahi_cfg.get('slice_height', 640), overlap_x=self.sahi_cfg.get('overlap_width_ratio', 0.2), overlap_y=self.sahi_cfg.get('overlap_height_ratio', 0.2), conf_thresh=0.5)

    def toggle_sahi(self):
        self.sahi_enabled = not self.sahi_enabled
        if self.sahi_enabled and self.async_sahi is None:
            self._start_sahi()
        elif not self.sahi_enabled and self.async_sahi is not None:
            self.async_sahi.stop()
            self.async_sahi = None
        return self.sahi_enabled

    def _warmup(self):
        if 'cuda' in self.device:
            dummy_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            try:
                _ = self.model.predict(dummy_img, device=self.device, imgsz=self.imgsz, verbose=False)
                if self.window_model is not None:
                    _ = self.window_model.predict(dummy_img, device=self.device, imgsz=640, verbose=False)
                torch.cuda.synchronize()
                print('[Detector] GPU TensorRT warmup completed successfully.')
            except Exception as e:
                print(f'[Detector] Warmup note: {e}')

    def track(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        start_t = time.perf_counter()
        self.frame_count += 1
        if self.sahi_enabled and self.async_sahi is not None:
            self.async_sahi.submit(frame)
        predict_kwargs = {'source': frame, 'conf': self.conf_thresh, 'iou': self.iou_thresh, 'imgsz': self.imgsz, 'device': self.device, 'verbose': False}
        results = self.model.predict(**predict_kwargs)
        if 'cuda' in self.device:
            torch.cuda.synchronize()
        inference_time_ms = (time.perf_counter() - start_t) * 1000.0
        raw_detections: List[Detection] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.int().cpu().tolist()
                names = res.names
                for idx in range(len(boxes)):
                    t_id = track_ids[idx]
                    c_id = cls_ids[idx]
                    raw_name = names.get(c_id, str(c_id)).lower()
                    conf = float(confs[idx])
                    x1, y1, x2, y2 = map(float, xyxy[idx])
                    cx, cy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    area = max(0.0, (x2 - x1) * (y2 - y1))
                    canonical_name = self._categorize_class(raw_name)
                    if self.filter_only_targets and canonical_name not in ('fire', 'window'):
                        continue
                    if t_id != -1:
                        if t_id not in self.track_history:
                            self.track_history[t_id] = []
                        self.track_history[t_id].append((cx, cy))
                        if len(self.track_history[t_id]) > self.max_history_len:
                            self.track_history[t_id].pop(0)
                        trajectory = list(self.track_history[t_id])
                    else:
                        trajectory = [(cx, cy)]
                    raw_detections.append(Detection(track_id=t_id, class_id=c_id, class_name=canonical_name, confidence=conf, bbox=(x1, y1, x2, y2), center=(cx, cy), area=area, trajectory=trajectory))
        if self.window_model is not None:
            if self.frame_count % 4 == 0:
                new_windows = []
                try:
                    win_res = self.window_model.predict(frame, conf=self.win_conf_thresh, device=self.device, imgsz=640, verbose=False)
                    if win_res and len(win_res) > 0 and (win_res[0].boxes is not None):
                        w_boxes = win_res[0].boxes
                        w_xyxy = w_boxes.xyxy.cpu().numpy()
                        w_confs = w_boxes.conf.cpu().numpy()
                        w_cls = w_boxes.cls.int().cpu().tolist()
                        w_names = win_res[0].names
                        for w_idx in range(len(w_boxes)):
                            raw_w_name = w_names.get(w_cls[w_idx], '').lower()
                            if 'window' in raw_w_name or 'ventana' in raw_w_name or 'balcon' in raw_w_name:
                                wx1, wy1, wx2, wy2 = map(float, w_xyxy[w_idx])
                                wcx, wcy = ((wx1 + wx2) / 2.0, (wy1 + wy2) / 2.0)
                                w_area = (wx2 - wx1) * (wy2 - wy1)
                                new_windows.append(Detection(track_id=500 + w_idx, class_id=1, class_name='window', confidence=float(w_confs[w_idx]), bbox=(wx1, wy1, wx2, wy2), center=(wcx, wcy), area=w_area, trajectory=[(wcx, wcy)]))
                except Exception:
                    pass
                self.cached_windows = new_windows
            raw_detections.extend(self.cached_windows)
        if self.sahi_enabled and self.async_sahi is not None:
            sahi_dets = self.async_sahi.get_detections()
            for s_bbox, s_conf, s_name in sahi_dets:
                scx = (s_bbox[0] + s_bbox[2]) / 2.0
                scy = (s_bbox[1] + s_bbox[3]) / 2.0
                already_tracked = any((abs(d.center[0] - scx) < 45 and abs(d.center[1] - scy) < 45 for d in raw_detections))
                if not already_tracked and s_conf >= 0.55:
                    s_area = (s_bbox[2] - s_bbox[0]) * (s_bbox[3] - s_bbox[1])
                    raw_detections.append(Detection(track_id=-1, class_id=0 if s_name == 'fire' else 1, class_name=s_name, confidence=s_conf, bbox=s_bbox, center=(scx, scy), area=s_area, trajectory=[(scx, scy)]))
        stable_detections = self.stabilizer.update(raw_detections)
        return (stable_detections, inference_time_ms)

    def _categorize_class(self, raw_name: str) -> str:
        if any((fire_word in raw_name for fire_word in self.fire_labels)):
            return 'fire'
        if any((win_word in raw_name for win_word in self.window_labels)):
            return 'window'
        return raw_name

    def clear_tracks(self):
        self.track_history.clear()
        self.stabilizer.clear()
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            self.model.predictor = None
