import time
import psutil
from typing import Dict, Any, Optional
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_smi as pynvml
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False

class HardwareMonitor:

    def __init__(self, gpu_index: int=0):
        self.gpu_index = gpu_index
        self.nvml_initialized = False
        self.handle = None
        self.gpu_name = 'NVIDIA GeForce GPU'
        self._fps_history = []
        self._last_time = time.perf_counter()
        self._fps_ema = 0.0
        self._inference_time_ms = 0.0
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                name_bytes = pynvml.nvmlDeviceGetName(self.handle)
                self.gpu_name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else str(name_bytes)
                self.nvml_initialized = True
            except Exception as e:
                self.nvml_initialized = False

    def get_gpu_metrics(self) -> Dict[str, Any]:
        metrics = {'name': self.gpu_name, 'utilization_gpu': 0, 'utilization_mem': 0, 'vram_used_mb': 0, 'vram_total_mb': 12288, 'vram_percent': 0.0, 'temperature_c': 0, 'power_watts': 0.0, 'power_limit_watts': 250.0}
        if not self.nvml_initialized or self.handle is None:
            return metrics
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            metrics['utilization_gpu'] = util.gpu
            metrics['utilization_mem'] = util.memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            metrics['vram_used_mb'] = int(mem.used / (1024 * 1024))
            metrics['vram_total_mb'] = int(mem.total / (1024 * 1024))
            metrics['vram_percent'] = round(mem.used / mem.total * 100, 1)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics['temperature_c'] = temp
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                metrics['power_watts'] = round(power / 1000.0, 1)
            except Exception:
                pass
        except Exception:
            pass
        return metrics

    def get_cpu_metrics(self) -> Dict[str, Any]:
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            return {'cpu_percent': round(cpu_percent, 1), 'ram_used_gb': round(mem.used / 1024 ** 3, 2), 'ram_total_gb': round(mem.total / 1024 ** 3, 2), 'ram_percent': round(mem.percent, 1)}
        except Exception:
            return {'cpu_percent': 0.0, 'ram_used_gb': 0.0, 'ram_total_gb': 16.0, 'ram_percent': 0.0}

    def tick_fps(self, inference_ms: Optional[float]=None) -> float:
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            if self._fps_ema == 0.0:
                self._fps_ema = instant_fps
            else:
                self._fps_ema = 0.9 * self._fps_ema + 0.1 * instant_fps
        if inference_ms is not None:
            self._inference_time_ms = inference_ms
        return round(self._fps_ema, 1)

    def get_fps(self) -> float:
        return round(self._fps_ema, 1)

    def get_inference_ms(self) -> float:
        return round(self._inference_time_ms, 2)

    def get_all_metrics(self) -> Dict[str, Any]:
        return {'fps': self.get_fps(), 'inference_ms': self.get_inference_ms(), 'gpu': self.get_gpu_metrics(), 'cpu': self.get_cpu_metrics()}

    def __del__(self):
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
