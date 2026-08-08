"""Registro de modelos Kronos y carga del predictor.

Kronos no tiene paquete oficial en PyPI, así que se vendoriza el repo en
``vendor/Kronos`` y se importa desde ahí. Modelos disponibles en
HuggingFace (https://huggingface.co/NeoQuasar):

- mini : 4.1M params, contexto 2048, tokenizer 2k   -> ideal para CPU
- small: 24.7M params, contexto 512                 -> equilibrio (default)
- base : 102.3M params, contexto 512                -> mejor calidad, lento en CPU
- large: 499.2M params, NO publicado
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
        description="Ligero y rápido, ideal para CPU. Contexto largo (2048).",
    ),
    "small": ModelConfig(
        name="small",
        hf_model_id="NeoQuasar/Kronos-small",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params="24.7M",
        description="Equilibrio calidad/velocidad. Recomendado por defecto.",
    ),
    "base": ModelConfig(
        name="base",
        hf_model_id="NeoQuasar/Kronos-base",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        params="102.3M",
        description="Mayor calidad disponible, pero lento en CPU.",
    ),
}

DEFAULT_MODEL = "small"


def _ensure_kronos_importable() -> None:
    """Añade vendor/Kronos a sys.path para poder hacer `from model import ...`."""
    path = str(VENDOR_KRONOS_PATH)
    if not VENDOR_KRONOS_PATH.is_dir():
        raise FileNotFoundError(
            f"No se encuentra el repo vendorizado de Kronos en {VENDOR_KRONOS_PATH}. "
            "Clónalo con: git clone https://github.com/shiyu-coder/Kronos vendor/Kronos"
        )
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=3)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Carga (cacheada) tokenizer + modelo y devuelve un KronosPredictor.

    La primera llamada descarga los pesos desde HuggingFace (~10-400 MB
    según el modelo). Las llamadas posteriores reutilizan la instancia.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo desconocido: {model_name!r}. "
            f"Disponibles: {sorted(MODEL_REGISTRY)}"
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
    predictor.eval = getattr(model, "eval", lambda: None)  # asegurar modo eval
    return predictor
