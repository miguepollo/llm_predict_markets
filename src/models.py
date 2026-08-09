"""TimesFM model registry and predictor loading.

Google's TimesFM (Time Series Foundation Model) is a univariate point-forecast
foundation model. We use TimesFM 2.5 (200M params), which dropped the
frequency-indicator input and supports long contexts (up to 16k) plus an
optional continuous quantile head. Weights are downloaded from HuggingFace
(https://huggingface.co/google/timesfm-2.5-200m-pytorch).

TimesFM predicts a single target series per call. This app feeds it the
**close** price series; the other candle fields (open/high/low/volume/amount)
are reconstructed around the forecast closes so candles keep rendering (see
:class:`TimesFMPredictor`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TimesFM 2.5 patches the context into windows of `input_patch_len` (32).
# Context/inputs must be a multiple of this.
PATCH_LEN = 32


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_model_id: str
    max_context: int
    max_horizon: int
    params: str
    description: str


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "2.5": ModelConfig(
        name="2.5",
        hf_model_id="google/timesfm-2.5-200m-pytorch",
        max_context=1024,
        max_horizon=256,
        params="200M",
        description="TimesFM 2.5 (200M). Default, good CPU/quality balance.",
    ),
}

DEFAULT_MODEL = "2.5"


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

    Note: TimesFM 2.5's torch inference only accelerates on CUDA and falls back
    to CPU otherwise, so XPU/MPS are intentionally not offered here (unlike the
    old Kronos backend).
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    logger.debug(
        "Available compute devices: %s",
        {d: device_details(d) for d in devices},
    )
    return devices


class TimesFMPredictor:
    """Wraps TimesFM and exposes a predictor-compatible ``predict()`` interface.

    TimesFM is a univariate point-forecast model, but its ``forecast()`` method
    accepts several series in a single batched call. This app forecasts the
    five OHLCV series (**open, high, low, close, volume**) independently in one
    forward pass and then reconciles each candle so the geometry is consistent:

    - ``open`` / ``close``: TimesFM point forecasts.
    - ``high`` = max(hi_forecast, open, close).
    - ``low``  = min(lo_forecast, open, close).
    - ``volume``: TimesFM forecast, clamped at >= 0.
    - ``amount`` = volume * mean price.

    The returned DataFrame is indexed by the future timestamps, with columns
    ``open, high, low, close, volume, amount``.
    """

    # Column name -> position in the batched ``inputs`` list.
    _SERIES = ["open", "high", "low", "close", "volume"]

    def __init__(self, model, max_context: int):
        self.model = model
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 120,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Predicts ``pred_len`` candles from the OHLCV context ``df``."""
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        inputs = [df[c].to_numpy(dtype=np.float64) for c in self._SERIES]
        point, _ = self.model.forecast(horizon=int(pred_len), inputs=inputs)
        pred = np.asarray(point, dtype=np.float64)  # shape (5, pred_len)

        f_open, f_high, f_low, f_close, f_volume = pred

        # Reconcile candle geometry: high/low must enclose the open/close body.
        high = np.maximum(f_high, np.maximum(f_open, f_close))
        low = np.minimum(f_low, np.minimum(f_open, f_close))
        volume = np.maximum(f_volume, 0.0)
        amount = volume * (f_open + high + low + f_close) / 4.0

        index = pd.DatetimeIndex(y_timestamp)
        index.name = "timestamps"
        return pd.DataFrame(
            {
                "open": f_open,
                "high": high,
                "low": low,
                "close": f_close,
                "volume": volume,
                "amount": amount,
            },
            index=index,
        )


@lru_cache(maxsize=3)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Loads (cached) the TimesFM model and returns a TimesFMPredictor.

    The first call downloads the weights from HuggingFace (~800 MB) and
    compiles the forecast loop. Subsequent calls reuse the instance.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    cfg = MODEL_REGISTRY[model_name]

    import torch

    logger.info(
        "Initializing TimesFM-%s (%s params) | model=%s device=%s",
        model_name, cfg.params, cfg.hf_model_id, device,
    )
    logger.info("torch %s | inference device: %s -> %s",
                torch.__version__, device, device_details(device))
    torch.set_float32_matmul_precision("high")

    t0 = time.time()

    import timesfm  # noqa: PLC0415

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cfg.hf_model_id)
    model.compile(
        timesfm.ForecastConfig(
            max_context=cfg.max_context,
            max_horizon=cfg.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    model.model.eval()

    predictor = TimesFMPredictor(model, max_context=cfg.max_context)
    logger.info(
        "TimesFM-%s ready on %s in %.1fs (max_context=%d, max_horizon=%d)",
        model_name, device, time.time() - t0, cfg.max_context, cfg.max_horizon,
    )
    return predictor