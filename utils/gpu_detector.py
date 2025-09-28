import pycuda.driver as cuda
import pycuda.autoinit
from arch import *


class GPUDetector:
    """GPU Model Detector"""

    # GPU model mapping table
    GPU_MODEL_MAP = {
        "NVIDIA H100 80GB HBM3": H100,
        "NVIDIA GeForce RTX 4090": RTX4090,
        "Quadro GV100": V100,
    }

    @classmethod
    def detect_gpu(cls):
        """Detect current GPU model"""
        try:
            device = cuda.Context.get_device()
            device_name = device.name()
            print(f"Detected GPU: {device_name}")

            # Search for matching GPU in the mapping table
            if device_name in cls.GPU_MODEL_MAP.keys():
                return cls.GPU_MODEL_MAP[device_name]
            else:
                raise NotImplementedError(
                    f"Unsupported GPU model detection: {device_name}, please report to us."
                    f"Or you can insert your own GPU model in the mapping table. But we can "
                    f"only ensure the correct configuration of {cls.GPU_MODEL_MAP.keys()}."
                )

        except Exception as e:
            print(f"GPU detection failed: {e}")
            exit(-1)

    @classmethod
    def create_gpu_instance(cls):
        """Create corresponding GPU class instance"""
        gpu_class = cls.detect_gpu()
        return gpu_class()
