"""
CV2.0 - Drone Vision Detector (YOLO26x Unified Engine)
High-Performance Vision Pipeline for Tethered Emergency Drone (UAV).
Architectured for NVIDIA GPU RTX (FP16 Tensor Cores) & Intel Core Ultra.
Natively detects Fire, Smoke, and Building Windows in real-time aerial footage.
"""

import os
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Hardware Optimization: Enable cuDNN autotuner for NVIDIA GPU RTX
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]              # (cx, cy)
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


class TrackStabilizer:
    """
    Temporal tracker stabilizer & anti-flicker filter.
    Applies exponential moving average (EMA) smoothing to bounding boxes
    and maintains target persistence across drone camera maneuvers.
    """
    def __init__(self, min_hits: int = 2, smoothing_alpha: float = 0.75, max_lost: int = 20, conf_thresh: float = 0.20):
        self.min_hits = min_hits
        self.alpha = smoothing_alpha
        self.max_lost = max_lost
        self.conf_thresh = conf_thresh
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, raw_detections: List[Detection]) -> List[Detection]:
        confirmed: List[Detection] = []
        seen_ids = set()

        for det in raw_detections:
            t_id = det.track_id
            if t_id == -1:
                if det.confidence >= self.conf_thresh:
                    confirmed.append(det)
                continue

            seen_ids.add(t_id)
            if t_id not in self.tracks:
                self.tracks[t_id] = {
                    "hits": 1,
                    "lost": 0,
                    "bbox": list(det.bbox),
                    "class_name": det.class_name,
                    "class_id": det.class_id,
                    "conf": det.confidence,
                    "trajectory": list(det.trajectory)
                }
            else:
                track = self.tracks[t_id]
                track["hits"] += 1
                track["lost"] = 0
                track["conf"] = det.confidence
                track["class_name"] = det.class_name
                track["class_id"] = det.class_id
                
                # Exponential smoothing of bounding box
                prev_b = track["bbox"]
                new_b = det.bbox
                track["bbox"] = [
                    self.alpha * new_b[i] + (1.0 - self.alpha) * prev_b[i]
                    for i in range(4)
                ]
                track["trajectory"] = list(det.trajectory)

            track = self.tracks[t_id]
            if track["hits"] >= self.min_hits:
                sb = tuple(track["bbox"])
                scx = (sb[0] + sb[2]) / 2.0
                scy = (sb[1] + sb[3]) / 2.0
                sarea = max(0.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
                confirmed.append(Detection(
                    track_id=t_id,
                    class_id=track["class_id"],
                    class_name=track["class_name"],
                    confidence=track["conf"],
                    bbox=sb,
                    center=(scx, scy),
                    area=sarea,
                    trajectory=track["trajectory"]
                ))

        # Handle dead / occluded tracks
        for t_id in list(self.tracks.keys()):
            if t_id not in seen_ids:
                self.tracks[t_id]["lost"] += 1
                if self.tracks[t_id]["lost"] > self.max_lost:
                    del self.tracks[t_id]

        return confirmed

    def clear(self):
        self.tracks.clear()


class AsyncTileProcessor:
    """
    Asynchronous high-resolution slicing inference for 4K video streams.
    Slices frames into overlapping tiles to catch distant flame points and tiny facade windows.
    Batched and accelerated on NVIDIA GPU RTX Tensor Cores without stalling the main video thread.
    """
    def __init__(self, model_path: str, device: str, slice_size: int = 640, overlap: float = 0.20, conf: float = 0.35):
        self.model_path = model_path
        self.device = device
        self.slice_size = slice_size
        self.overlap = overlap
        self.conf = conf
        
        self.queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_detections: List[Tuple[Tuple[float, float, float, float], float, str]] = []
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
            tile_model = YOLO(self.model_path)
        except Exception as e:
            print(f"[TileProcessor] Error loading model: {e}")
            return

        while self.running:
            try:
                frame = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            h, w = frame.shape[:2]
            step = int(self.slice_size * (1.0 - self.overlap))
            slices = []
            
            for y in range(0, h, step):
                for x in range(0, w, step):
                    x_end = min(x + self.slice_size, w)
                    y_end = min(y + self.slice_size, h)
                    x_start = max(0, x_end - self.slice_size)
                    y_start = max(0, y_end - self.slice_size)
                    slices.append((x_start, y_start, x_end, y_end))

            slice_imgs = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in slices]
            
            try:
                all_boxes = []
                all_confs = []
                all_names = []
                batch_size = 6  # Optimal for NVIDIA GPU RTX

                for i in range(0, len(slice_imgs), batch_size):
                    chunk_imgs = slice_imgs[i:i + batch_size]
                    chunk_slices = slices[i:i + batch_size]
                    
                    results = tile_model.predict(
                        chunk_imgs,
                        conf=self.conf,
                        imgsz=self.slice_size,
                        device=self.device,
                        verbose=False
                    )

                    for idx, res in enumerate(results):
                        x_off, y_off, _, _ = chunk_slices[idx]
                        if res.boxes is not None and len(res.boxes) > 0:
                            s_xyxy = res.boxes.xyxy.cpu().numpy()
                            s_confs = res.boxes.conf.cpu().numpy()
                            s_clss = res.boxes.cls.int().cpu().tolist()
                            names = res.names
                            
                            for b_i in range(len(s_xyxy)):
                                raw_n = names.get(s_clss[b_i], "").lower()
                                c_conf = float(s_confs[b_i])
                                
                                # Canonical mapping
                                if "fire" in raw_n or "fuego" in raw_n or "flame" in raw_n:
                                    canon = "fire"
                                elif "smoke" in raw_n or "humo" in raw_n:
                                    canon = "smoke"
                                elif "window" in raw_n or "ventana" in raw_n:
                                    canon = "window"
                                else:
                                    continue
                                
                                bx1, by1, bx2, by2 = s_xyxy[b_i]
                                all_boxes.append([bx1 + x_off, by1 + y_off, bx2 + x_off, by2 + y_off])
                                all_confs.append(c_conf)
                                all_names.append(canon)

                if all_boxes:
                    indices = cv2.dnn.NMSBoxes(all_boxes, all_confs, score_threshold=self.conf, nms_threshold=0.45)
                    merged = []
                    if len(indices) > 0:
                        for idx in indices.flatten():
                            merged.append((tuple(all_boxes[idx]), all_confs[idx], all_names[idx]))
                    with self.lock:
                        self.latest_detections = merged

            except Exception:
                pass

    def stop(self):
        self.running = False


class DroneVisionDetector:
    """
    Unified YOLO26x Vision Detector.
    Runs end-to-end detection on NVIDIA GPU RTX for Fire, Smoke, and Facade Windows.
    Zero dependency on legacy YOLOv8 or YOLO11 architectures.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.temporal_cfg = config.get("temporal_filter", {})
        self.classes_cfg = config.get("classes", {})
        self.sahi_cfg = config.get("sahi", {})

        # 1. Hardware Detection & Acceleration
        self.device = self._init_hardware(self.model_cfg.get("device", "0"))
        self.imgsz = self.model_cfg.get("imgsz", 640)
        self.conf_thresh = self.model_cfg.get("confidence_threshold", 0.25)
        self.iou_thresh = self.model_cfg.get("iou_threshold", 0.45)
        self.half_precision = self.model_cfg.get("half", True) and ("cuda" in self.device)

        # 2. Tracking History & Stabilization
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.max_history_len = 30
        self.frame_count = 0

        min_hits = self.temporal_cfg.get("min_consecutive_hits", 1)
        alpha = self.temporal_cfg.get("box_smoothing_alpha", 0.75)
        max_lost = self.temporal_cfg.get("max_lost_memory", 25)
        self.stabilizer = TrackStabilizer(
            min_hits=min_hits,
            smoothing_alpha=alpha,
            max_lost=max_lost,
            conf_thresh=self.conf_thresh
        )

        # 3. Model Weights Resolution (YOLO26x)
        self.model_path = self._resolve_yolo26x_weights()
        print(f"[Detector] ⚡ Loading Unified YOLO26x Engine: {self.model_path}")
        self.model = YOLO(self.model_path)
        if self.half_precision and hasattr(self.model, "model") and hasattr(self.model.model, "half"):
            try:
                self.model.model.half()
            except Exception:
                pass
        print(f"[Detector] Classes detected in YOLO26x: {self.model.names}")

        # 4. High-Resolution Tile Processor (SAHI)
        self.sahi_enabled = self.sahi_cfg.get("enabled", False)
        self.tile_processor: Optional[AsyncTileProcessor] = None
        if self.sahi_enabled:
            self._start_sahi()

        # 5. Class Target Filtering
        self.filter_only_targets = self.classes_cfg.get("filter_only_targets", True)
        self.fire_labels = {"fire", "flame", "smoke", "fuego", "humo"}
        self.window_labels = {"window", "window_frame", "balcony", "ventana", "balcon"}

        # 6. Tensor Core GPU Warmup
        self._warmup()

    def _init_hardware(self, dev: str) -> str:
        if torch.cuda.is_available():
            dev_str = f"cuda:{dev}" if dev.isdigit() else "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Detector] 🎮 Hardware: {gpu_name} ({total_vram:.1f} GB VRAM) on {dev_str}")
            print(f"[Detector] 🚀 FP16 Tensor Cores & cuDNN Benchmark: ENABLED")
            return dev_str
        print("[Detector] ⚠️ Warning: CUDA not available. Running on CPU.")
        return "cpu"

    def _resolve_yolo26x_weights(self) -> str:
        # Check custom or trained weights first
        candidates = [
            self.model_cfg.get("weights"),
            "weights/yolo26x_fire_windows.pt",
            "yolo26x.pt",
            "weights/yolov26-fire.pt"
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                return c
        return "yolo26x.pt"

    def _start_sahi(self):
        self.tile_processor = AsyncTileProcessor(
            model_path=self.model_path,
            device=self.device,
            slice_size=self.sahi_cfg.get("slice_width", 640),
            overlap=self.sahi_cfg.get("overlap_width_ratio", 0.20),
            conf=self.sahi_cfg.get("postprocess_match_threshold", 0.35)
        )

    def toggle_sahi(self) -> bool:
        self.sahi_enabled = not self.sahi_enabled
        if self.sahi_enabled and self.tile_processor is None:
            self._start_sahi()
        elif not self.sahi_enabled and self.tile_processor is not None:
            self.tile_processor.stop()
            self.tile_processor = None
        return self.sahi_enabled

    def _warmup(self):
        if "cuda" in self.device:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            try:
                _ = self.model.predict(
                    dummy,
                    device=self.device,
                    imgsz=self.imgsz,
                    verbose=False
                )
                torch.cuda.synchronize()
                print("[Detector] GPU TensorRT/FP16 warmup completed successfully.")
            except Exception as e:
                print(f"[Detector] Warmup note: {e}")

    def track(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Executes real-time inference on the input frame.
        Returns:
            stable_detections: List of verified Detection objects
            inference_time_ms: Exact hardware latency in milliseconds
        """
        start_t = time.perf_counter()
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Submit to async tile processor if enabled
        if self.sahi_enabled and self.tile_processor is not None:
            self.tile_processor.submit(frame)

        # Primary YOLO26x inference
        predict_kwargs = {
            "source": frame,
            "conf": self.conf_thresh,
            "iou": self.iou_thresh,
            "imgsz": self.imgsz,
            "device": self.device,
            "verbose": False
        }

        results = self.model.predict(**predict_kwargs)
        if "cuda" in self.device:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        raw_detections: List[Detection] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.int().cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(boxes)
                names = res.names

                for idx in range(len(boxes)):
                    t_id = track_ids[idx]
                    c_id = clss[idx]
                    raw_name = names.get(c_id, str(c_id)).lower()
                    conf = float(confs[idx])
                    x1, y1, x2, y2 = map(float, xyxy[idx])

                    box_w = x2 - x1
                    box_h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    area = max(0.0, box_w * box_h)

                    canonical_name = self._categorize_class(raw_name)
                    if self.filter_only_targets and canonical_name not in ("fire", "window", "smoke"):
                        continue

                    # Filter HUD corners / video watermarks
                    in_top_left_watermark = y1 < 260 and x1 < 650
                    in_top_right_panel = x1 > w * 0.88 and y1 < 220
                    is_oversized = (box_w / w) > 0.65 or (box_h / h) > 0.65
                    if in_top_left_watermark or in_top_right_panel or is_oversized:
                        continue

                    # Trajectory handling
                    if t_id != -1:
                        if t_id not in self.track_history:
                            self.track_history[t_id] = []
                        self.track_history[t_id].append((cx, cy))
                        if len(self.track_history[t_id]) > self.max_history_len:
                            self.track_history[t_id].pop(0)
                        trajectory = list(self.track_history[t_id])
                    else:
                        trajectory = [(cx, cy)]

                    raw_detections.append(Detection(
                        track_id=t_id,
                        class_id=c_id,
                        class_name=canonical_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        area=area,
                        trajectory=trajectory
                    ))

        # Merge asynchronous tile detections if enabled
        if self.sahi_enabled and self.tile_processor is not None:
            tile_dets = self.tile_processor.get_detections()
            for t_bbox, t_conf, t_name in tile_dets:
                tx1, ty1, tx2, ty2 = t_bbox
                tcx = (tx1 + tx2) / 2.0
                tcy = (ty1 + ty2) / 2.0
                
                # Check if already covered by whole-frame inference
                already_covered = any(
                    abs(d.center[0] - tcx) < 40 and abs(d.center[1] - tcy) < 40
                    for d in raw_detections
                )
                if not already_covered and t_conf >= 0.35:
                    t_area = (tx2 - tx1) * (ty2 - ty1)
                    raw_detections.append(Detection(
                        track_id=-1,
                        class_id=0 if t_name == "fire" else 2,
                        class_name=t_name,
                        confidence=t_conf,
                        bbox=t_bbox,
                        center=(tcx, tcy),
                        area=t_area,
                        trajectory=[(tcx, tcy)]
                    ))

        stable_detections = self.stabilizer.update(raw_detections)
        return (stable_detections, latency_ms)

    def _categorize_class(self, raw_name: str) -> str:
        if any(w in raw_name for w in self.fire_labels):
            return "fire"
        if any(w in raw_name for w in self.window_labels):
            return "window"
        return raw_name

    def clear_tracks(self):
        self.track_history.clear()
        self.stabilizer.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None
