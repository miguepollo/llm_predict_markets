"""Model registry and predictor loading for TimesFM and Moirai.

Two time-series foundation models are supported, selected from the UI:

- **TimesFM 2.5** (``google/timesfm-2.5-200m-pytorch``, 200M, Apache-2.0): a
  univariate point-forecast model. Its ``forecast()`` accepts several series in
  one batched call, so this app forecasts open/high/low/close/volume
  independently in a single forward pass.
- **Moirai** (``Salesforce/moirai-1.1-R-*``, 14M/91M/311M, CC-BY-NC-4.0): a
  truly multivariate probabilistic transformer that forecasts all variates
  jointly in one forward pass, capturing cross-series correlation. Loaded via
  the ``uni2ts`` package.

Both backends expose the same ``predict(df, x_timestamp, y_timestamp,
pred_len, verbose)`` interface and reconcile candle geometry so
``high >= max(open, close)``, ``low <= min(open, close)`` and ``volume >= 0``.
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

# OHLCV variates shared by both backends (column order).
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str  # "timesfm" | "moirai"
    hf_model_id: str
    max_context: int
    max_horizon: int
    params: str
    description: str
    num_samples: int | None = None  # only used by Moirai (probabilistic)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "2.5": ModelConfig(
        name="2.5",
        backend="timesfm",
        hf_model_id="google/timesfm-2.5-200m-pytorch",
        max_context=1024,
        max_horizon=256,
        params="200M",
        description="TimesFM 2.5 (200M, Apache-2.0). Point forecast; each OHLCV "
                    "series forecast independently in one batched call.",
    ),
    "moirai-small": ModelConfig(
        name="moirai-small",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-small",
        max_context=512,
        max_horizon=256,
        params="14M",
        num_samples=20,
        description="Moirai 1.1 small (14M, CC-BY-NC). True multivariate "
                    "probabilistic; all OHLCV variates together. Fast on CPU.",
    ),
    "moirai-base": ModelConfig(
        name="moirai-base",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-base",
        max_context=512,
        max_horizon=256,
        params="91M",
        num_samples=10,
        description="Moirai 1.1 base (91M, CC-BY-NC). Multivariate probabilistic; "
                    "quality/speed balance.",
    ),
    "moirai-large": ModelConfig(
        name="moirai-large",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-large",
        max_context=512,
        max_horizon=256,
        params="311M",
        num_samples=10,
        description="Moirai 1.1 large (311M, CC-BY-NC). Best quality, slow on CPU.",
    ),
}

DEFAULT_MODEL = "moirai-base"


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


class MoiraiPredictor:
    """Wraps a Moirai module and exposes the app ``predict()`` interface.

    Moirai is a truly multivariate probabilistic transformer: all OHLCV
    variates are fed as one series (``target_dim = 5``) and forecast jointly in
    a single forward pass, so the cross-series relationship is learned by the
    model. We use the median over ``num_samples`` sampled trajectories as the
    point forecast and reconcile candle geometry (``high >= max(open, close)``,
    ``low <= min(open, close)``, ``volume >= 0``) as a safety net.
    """

    def __init__(self, module, max_context: int, num_samples: int = 20,
                 patch_size: str = "auto"):
        self.module = module
        self.max_context = max_context
        self.num_samples = num_samples
        self.patch_size = patch_size

    def predict(self, df, x_timestamp, y_timestamp, pred_len=120, verbose=False):
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        from gluonts.dataset.pandas import PandasDataset
        from uni2ts.model.moirai import MoiraiForecast

        wide = df[OHLCV].copy()
        wide.index = pd.DatetimeIndex(x_timestamp)
        ds = PandasDataset(wide, target=OHLCV)

        model = MoiraiForecast(
            module=self.module,
            prediction_length=int(pred_len),
            context_length=int(len(wide)),
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=len(OHLCV),
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = model.create_predictor(batch_size=1)
        forecast = next(iter(predictor.predict(ds)))
        median = np.asarray(forecast.median, dtype=np.float64)  # (pred_len, 5)

        f_open = median[:, 0]
        f_high = median[:, 1]
        f_low = median[:, 2]
        f_close = median[:, 3]
        f_volume = median[:, 4]

        high = np.maximum(f_high, np.maximum(f_open, f_close))
        low = np.minimum(f_low, np.minimum(f_open, f_close))
        volume = np.maximum(f_volume, 0.0)
        amount = volume * (f_open + high + low + f_close) / 4.0

        index = pd.DatetimeIndex(y_timestamp)
        index.name = "timestamps"
        return pd.DataFrame(
            {"open": f_open, "high": high, "low": low, "close": f_close,
             "volume": volume, "amount": amount},
            index=index,
        )


def _load_timesfm(cfg: ModelConfig, device: str):
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
    return TimesFMPredictor(model, max_context=cfg.max_context)


def _load_moirai(cfg: ModelConfig, device: str):
    from uni2ts.model.moirai import MoiraiModule

    module = MoiraiModule.from_pretrained(cfg.hf_model_id)
    module.eval()
    return MoiraiPredictor(
        module,
        max_context=cfg.max_context,
        num_samples=cfg.num_samples or 20,
    )


@lru_cache(maxsize=4)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Loads (cached) the selected model's predictor.

    Dispatches to TimesFM or Moirai based on the registry backend. The first
    call downloads the weights from HuggingFace; subsequent calls reuse it.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    cfg = MODEL_REGISTRY[model_name]

    import torch

    logger.info(
        "Initializing %s-%s (%s params) | model=%s device=%s",
        cfg.backend, model_name, cfg.params, cfg.hf_model_id, device,
    )
    logger.info("torch %s | inference device: %s -> %s",
                torch.__version__, device, device_details(device))
    torch.set_float32_matmul_precision("high")

    t0 = time.time()
    if cfg.backend == "timesfm":
        predictor = _load_timesfm(cfg, device)
    elif cfg.backend == "moirai":
        predictor = _load_moirai(cfg, device)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r} for {model_name!r}.")
    logger.info(
        "%s-%s ready on %s in %.1fs (max_context=%d)",
        cfg.backend, model_name, device, time.time() - t0, cfg.max_context,
    )
    return predictor