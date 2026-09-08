import os
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import numpy as np
from src.detector import Detection

class WindowStatus(str, Enum):
    SAFE = 'SAFE'
    WARNING = 'WARNING'
    CRITICAL_BURNING = 'CRITICAL_BURNING'

@dataclass
class WindowAnalysisResult:
    window_id: int
    bbox: Tuple[float, float, float, float]
    status: WindowStatus
    severity: float
    intersecting_fire_ids: List[int]
    distance_to_nearest_fire: float
    overlap_area_pixels: float
    overlap_ratio: float

@dataclass
class EmergencySceneReport:
    timestamp: float
    total_windows: int
    safe_windows: int
    warning_windows: int
    burning_windows: int
    active_fire_count: int
    max_severity: float
    total_fire_area_pixels: float
    fire_spread_velocity_px_s: float
    window_assessments: List[WindowAnalysisResult]

class SpatialEmergencyAnalyzer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer_cfg = config.get('analyzer', {})
        self.iou_thresh = self.analyzer_cfg.get('iou_collision_thresh', 0.02)
        self.proximity_radius = self.analyzer_cfg.get('proximity_pixel_radius', 90.0)
        self.containment_weight = self.analyzer_cfg.get('containment_weight', 0.7)
        self.log_incidents = self.analyzer_cfg.get('log_incidents', True)
        self.log_file = self.analyzer_cfg.get('incident_log_file', 'runs/incidents.json')
        self.fire_area_history: List[Tuple[float, float]] = []
        self.history_len = self.analyzer_cfg.get('spread_rate_history_len', 30)
        self.incident_records: List[Dict[str, Any]] = []
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    @staticmethod
    def _box_iou_and_intersection(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h
        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        union_area = areaA + areaB - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        ratio_A = inter_area / areaA if areaA > 0 else 0.0
        return (iou, inter_area, ratio_A)

    @staticmethod
    def _box_distance(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        cxA, cyA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
        cxB, cyB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
        return math.hypot(cxA - cxB, cyA - cyB)

    def analyze(self, detections: List[Detection]) -> EmergencySceneReport:
        now = time.time()
        windows = [d for d in detections if d.class_name == 'window']
        fires = [d for d in detections if d.class_name == 'fire']
        total_fire_area = sum((f.area for f in fires))
        self.fire_area_history.append((now, total_fire_area))
        if len(self.fire_area_history) > self.history_len:
            self.fire_area_history.pop(0)
        fire_spread_velocity = 0.0
        if len(self.fire_area_history) >= 2:
            dt = self.fire_area_history[-1][0] - self.fire_area_history[0][0]
            if dt > 0.1:
                d_area = self.fire_area_history[-1][1] - self.fire_area_history[0][1]
                fire_spread_velocity = round(d_area / dt, 1)
        window_assessments: List[WindowAnalysisResult] = []
        safe_count = 0
        warning_count = 0
        burning_count = 0
        max_severity = 0.0
        for win in windows:
            intersecting_fire_ids = []
            min_dist = float('inf')
            total_overlap_area = 0.0
            max_overlap_ratio = 0.0
            for fire in fires:
                iou, inter_area, ratio_win = self._box_iou_and_intersection(win.bbox, fire.bbox)
                dist = self._box_distance(win.bbox, fire.bbox)
                if dist < min_dist:
                    min_dist = dist
                if inter_area > 0 or iou >= self.iou_thresh:
                    intersecting_fire_ids.append(fire.track_id)
                    total_overlap_area += inter_area
                    if ratio_win > max_overlap_ratio:
                        max_overlap_ratio = ratio_win
            if len(intersecting_fire_ids) > 0 or max_overlap_ratio > self.iou_thresh:
                status = WindowStatus.CRITICAL_BURNING
                severity = min(1.0, 0.5 + 0.5 * max_overlap_ratio)
                burning_count += 1
            elif min_dist <= self.proximity_radius and len(fires) > 0:
                status = WindowStatus.WARNING
                proximity_factor = max(0.0, 1.0 - min_dist / self.proximity_radius)
                severity = min(0.49, 0.1 + 0.35 * proximity_factor)
                warning_count += 1
            else:
                status = WindowStatus.SAFE
                severity = 0.0
                safe_count += 1
            if severity > max_severity:
                max_severity = severity
            assessment = WindowAnalysisResult(window_id=win.track_id, bbox=win.bbox, status=status, severity=round(severity, 3), intersecting_fire_ids=intersecting_fire_ids, distance_to_nearest_fire=round(min_dist, 1) if min_dist != float('inf') else -1.0, overlap_area_pixels=round(total_overlap_area, 1), overlap_ratio=round(max_overlap_ratio, 3))
            window_assessments.append(assessment)
            if status == WindowStatus.CRITICAL_BURNING and self.log_incidents:
                self._record_incident(now, assessment)
        report = EmergencySceneReport(timestamp=now, total_windows=len(windows), safe_windows=safe_count, warning_windows=warning_count, burning_windows=burning_count, active_fire_count=len(fires), max_severity=round(max_severity, 3), total_fire_area_pixels=round(total_fire_area, 1), fire_spread_velocity_px_s=fire_spread_velocity, window_assessments=window_assessments)
        return report

    def _record_incident(self, timestamp: float, assessment: WindowAnalysisResult):
        record = {'timestamp': float(timestamp), 'iso_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)), 'window_id': int(assessment.window_id), 'status': str(assessment.status.value), 'severity': float(assessment.severity), 'bbox': [float(x) for x in assessment.bbox], 'fire_ids': [int(x) for x in assessment.intersecting_fire_ids], 'overlap_ratio': float(assessment.overlap_ratio)}
        self.incident_records.append(record)

    def save_incident_log(self, filepath: Optional[str]=None):
        target = filepath or self.log_file
        self._ensure_log_dir()
        try:
            with open(target, 'w', encoding='utf-8') as f:
                json.dump(self.incident_records, f, indent=2)
            print(f'[Analyzer] Saved {len(self.incident_records)} incident records to {target}')
        except Exception as e:
            print(f'[Analyzer] Failed to save incident log: {e}')
