"""Unit tests for compute-device detection and labelling (CPU / CUDA / ROCm).

These tests are hermetic: they inject a *fake* torch module into ``sys.modules``
so they run without a real torch install or GPU. AMD GPUs run through
ROCm/HIP and are surfaced by torch under the same ``cuda`` device string, so we
distinguish them via ``torch.version.hip`` (set only on ROCm builds).
"""

import importlib
import sys
import types


def _load_models(hip, device_name, *, cuda_available=True):
    """Builds a fake torch and (re)loads :mod:`src.models` against it."""
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip=hip)
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: cuda_available
    cuda.get_device_name = lambda i: device_name
    xpu = types.SimpleNamespace()
    xpu.is_available = lambda: False
    torch.cuda = cuda
    torch.xpu = xpu
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.get_num_threads = lambda: 8
    sys.modules["torch"] = torch
    return importlib.reload(__import__("src.models", fromlist=["*"])), torch


def test_device_details_nvidia_cuda():
    m, _ = _load_models(hip=None, device_name="NVIDIA GeForce RTX 3060")
    assert m.is_amd_rocm() is False
    assert m.device_details("cuda") == "NVIDIA GPU (CUDA): NVIDIA GeForce RTX 3060"


def test_device_details_amd_rocm():
    m, _ = _load_models(hip="5.6", device_name="AMD Radeon RX 580")
    assert m.is_amd_rocm() is True
    assert m.device_details("cuda") == "AMD GPU (ROCm/HIP): AMD Radeon RX 580"


def test_device_details_cpu_and_unknown():
    m, _ = _load_models(hip=None, device_name="irrelevant")
    assert m.device_details("cpu").startswith("CPU (")
    assert m.device_details("does-not-exist") == "does-not-exist"


def test_available_devices_lists_cuda_when_gpu_present():
    m, _ = _load_models(hip="6.2", device_name="AMD Radeon RX 7900 XTX")
    devices = m.available_devices()
    assert devices[0] == "cpu"
    assert "cuda" in devices
