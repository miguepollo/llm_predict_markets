"""Kronos model registry and predictor loading.

Kronos has no official PyPI package, so the repo is vendored under
``vendor/Kronos`` and imported from there. Models available on
HuggingFace (https://huggingface.co/NeoQuasar):

- mini : 4.1M params, context 2048, 2k tokenizer   -> ideal for CPU
- small: 24.7M params, context 512                 -> balanced (default)
- base : 102.3M params, context 512                -> best quality, slow on CPU
- large: 499.2M params, NOT published
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

VENDOR_KRONOS_PATH = Path(__file__).resolve().parent.parent / "vendor" / "Kronos"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_model_id: str
    hf_tokenizer_id: str
    max_context: int
    params: str
    description: str


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "mini": ModelConfig(
        name="mini",
        hf_model_id="NeoQuasar/Kronos-mini",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        params="4.1M",
        description="Lightweight and fast, ideal for CPU. Long context (2048).",
    ),
    "small": ModelConfig(
        name="small",
        hf_model_id="NeoQuasar/Kronos-small",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params="24.7M",
        description="Quality/speed balance. Recommended default.",
    ),
    "base": ModelConfig(
        name="base",
        hf_model_id="NeoQuasar/Kronos-base",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params="102.3M",
        description="Best available quality, but slow on CPU.",
    ),
}

DEFAULT_MODEL = "small"


def device_details(device: str) -> str:
    """Human-readable description of a compute device (hardware, driver)."""
    try:
        import torch

        if device == "cpu":
            return f"CPU ({torch.get_num_threads()} torch threads)"
        if device == "cuda":
            return f"NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0)}"
        if device == "xpu":
            try:
                props = torch.xpu.get_device_properties(0)
                driver = getattr(props, "driver_version", None) or "unknown"
                return (
                    f"Intel GPU (XPU): {torch.xpu.get_device_name(0)}, "
                    f"driver {driver}"
                )
            except Exception:
                return "Intel GPU (XPU)"
        if device == "mps":
            return "Apple Silicon GPU (MPS)"
    except Exception:
        pass
    return device


def available_devices() -> list[str]:
    """Compute devices available in this environment.

    - ``cpu``: always available.
    - ``cuda``: NVIDIA GPU (torch with CUDA).
    - ``xpu``: Intel GPU (Arc / Iris Xe) via torch XPU backend. On Linux
      requires a torch XPU build (download.pytorch.org/whl/xpu) and,
      depending on the GPU, intel-extension-for-pytorch.
    - ``mps``: Apple Silicon.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    logger.debug(
        "Available compute devices: %s",
        {d: device_details(d) for d in devices},
    )
    return devices


def _ensure_kronos_importable() -> None:
    """Adds vendor/Kronos to sys.path so `from model import ...` works."""
    path = str(VENDOR_KRONOS_PATH)
    if not VENDOR_KRONOS_PATH.is_dir():
        raise FileNotFoundError(
            f"Vendored Kronos repo not found at {VENDOR_KRONOS_PATH}. "
            "Clone it with: git clone https://github.com/shiyu-coder/Kronos vendor/Kronos"
        )
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=3)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Loads (cached) tokenizer + model and returns a KronosPredictor.

    The first call downloads the weights from HuggingFace (~10-400 MB
    depending on the model). Subsequent calls reuse the instance.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    cfg = MODEL_REGISTRY[model_name]

    import torch

    logger.info(
        "Initializing Kronos-%s (%s params) | tokenizer=%s model=%s",
        model_name, cfg.params, cfg.hf_tokenizer_id, cfg.hf_model_id,
    )
    logger.info("torch %s | inference device: %s -> %s",
                torch.__version__, device, device_details(device))

    t0 = time.time()
    _ensure_kronos_importable()
    from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: PLC0415

    tokenizer = KronosTokenizer.from_pretrained(cfg.hf_tokenizer_id)
    model = Kronos.from_pretrained(cfg.hf_model_id)
    predictor = KronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=cfg.max_context,
    )
    predictor.eval = getattr(model, "eval", lambda: None)  # ensure eval mode
    logger.info(
        "Kronos-%s ready on %s in %.1fs (max_context=%d)",
        model_name, device, time.time() - t0, cfg.max_context,
    )
    return predictor
