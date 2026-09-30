import os
import platform
import torch

def get_cpu_name():
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", "Unknown CPU")
    except ImportError:
        return platform.processor() or "Unknown CPU"

def get_logical_threads():
    return os.cpu_count() or 1

def get_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 0.0

def get_gpu_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "None (CPU only)"

def get_cuda_available():
    return torch.cuda.is_available()