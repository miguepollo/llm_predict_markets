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

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
    return predictor
