import time
import threading
import cv2
import numpy as np
from typing import Optional, Tuple, Union

class VideoStreamHandler:

    def __init__(self, source: Union[str, int], loop_video: bool=True, max_fps: int=60):
        self.source = source
        self.loop_video = loop_video
        self.max_fps = max_fps
        self.is_file = isinstance(source, str) and (not source.startswith('rtsp://')) and (not source.startswith('http://')) and (not source.startswith('/dev/video')) and (not (source.isdigit() if isinstance(source, str) else False))
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.ret: bool = False
        self.running: bool = False
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.width: int = 0
        self.height: int = 0
        self.source_fps: float = 60.0
        self.frame_count: int = 0
        self._open_stream()

    def _open_stream(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        is_device = isinstance(self.source, int) or (isinstance(self.source, str) and (self.source.isdigit() or self.source.startswith('/dev/video')))
        src = int(self.source) if isinstance(self.source, str) and self.source.isdigit() else self.source
        if isinstance(src, str) and src.startswith('rtsp://'):
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif is_device:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 60.0)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f'[StreamHandler] Warning: Could not open video source: {self.source}')
            return False
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        if self.source_fps <= 0 or np.isnan(self.source_fps):
            self.source_fps = 60.0
        print(f'[StreamHandler] Stream Connected: {self.source} | {self.width}x{self.height} @ {self.source_fps:.1f} FPS (YUYV 60Hz)')
        return True

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        timeout = 5.0
        start_t = time.time()
        while self.frame is None and time.time() - start_t < timeout:
            time.sleep(0.01)
        return self

    def _update_loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.5)
                self._open_stream()
                continue
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file and self.loop_video:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.05)
                    continue
            with self.lock:
                self.frame = frame
                self.ret = ret
                self.frame_count += 1
            self.new_frame_event.set()

    def read(self, timeout: float=0.05) -> Tuple[bool, Optional[np.ndarray]]:
        self.new_frame_event.wait(timeout=timeout)
        self.new_frame_event.clear()
        with self.lock:
            if self.frame is None:
                return (False, None)
            return (True, self.frame.copy())

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print('[StreamHandler] Stream stopped.')

    def __del__(self):
        self.stop()
