"""Model registry and predictor loading for TimesFM, Moirai and Kronos.

Three time-series foundation models are supported, selected from the UI:

- **TimesFM 2.5** (``google/timesfm-2.5-200m-pytorch``, 200M, Apache-2.0): a
  univariate point-forecast model. Its ``forecast()`` accepts several series in
  one batched call, so this app forecasts open/high/low/close/volume
  independently in a single forward pass.
- **Moirai** (``Salesforce/moirai-1.1-R-*``, 14M/91M/311M, CC-BY-NC-4.0): a
  truly multivariate probabilistic transformer that forecasts all variates
  jointly in one forward pass, capturing cross-series correlation. Loaded via
  the ``uni2ts`` package.
- **Kronos** (``NeoQuasar/Kronos-{mini,small,base}``, 4.1M/24.7M/102.3M,
  Apache-2.0): a generative time-series foundation model vendored under
  ``vendor/Kronos`` (no PyPI package). It produces candles by token sampling
  controlled by ``temperature``/``top_p``/``sample_count`` and averages the
  sampled paths into a point forecast.

All backends expose the same ``predict(df, x_timestamp, y_timestamp, pred_len,
verbose, temperature, top_p, sample_count)`` interface and reconcile candle
geometry so ``high >= max(open, close)``, ``low <= min(open, close)`` and
``volume >= 0``.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TimesFM 2.5 patches the context into windows of `input_patch_len` (32).
# Context/inputs must be a multiple of this.
PATCH_LEN = 32

# OHLCV variates shared by all backends (column order).
OHLCV = ["open", "high", "low", "close", "volume"]

# Kronos has no PyPI package; the repo is vendored under vendor/Kronos.
VENDOR_KRONOS_PATH = Path(__file__).resolve().parent.parent / "vendor" / "Kronos"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str  # "timesfm" | "moirai" | "kronos"
    hf_model_id: str
    max_context: int
    max_horizon: int
    params: str
    description: str
    num_samples: int | None = None  # only used by Moirai (probabilistic)
    hf_tokenizer_id: str | None = None  # only used by Kronos (vendored models)


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
    "kronos-mini": ModelConfig(
        name="kronos-mini",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-mini",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        max_horizon=512,
        params="4.1M",
        description="Kronos mini (4.1M, Apache-2.0). Generative; token sampling. "
                    "Lightest and ideal for CPU, with the longest context (2048).",
    ),
    "kronos-small": ModelConfig(
        name="kronos-small",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-small",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        max_horizon=256,
        params="24.7M",
        description="Kronos small (24.7M, Apache-2.0). Generative; sampled point "
                    "forecast (temperature/top-p/sample count). Balanced.",
    ),
    "kronos-base": ModelConfig(
        name="kronos-base",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-base",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        max_horizon=256,
        params="102.3M",
        description="Kronos base (102.3M, Apache-2.0). Generative; best quality, "
                    "slower on CPU.",
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
    - ``xpu``: Intel GPU (Arc / Iris Xe) via torch XPU backend. On Linux
      requires a torch XPU build (download.pytorch.org/whl/xpu) and,
      depending on the GPU, intel-extension-for-pytorch.
    - ``mps``: Apple Silicon.

    Note: TimesFM 2.5 / Moirai torch inference only accelerates on CUDA and
    falls back to CPU otherwise, so the UI only offers ``cpu``/``cuda``. Kronos
    can also use XPU/MPS when present.
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
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Predicts ``pred_len`` candles from the OHLCV context ``df``.

        ``temperature``/``top_p``/``sample_count`` are accepted for a uniform
        interface across backends but are ignored (TimesFM is deterministic).
        """
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

    def predict(self, df, x_timestamp, y_timestamp, pred_len=120, verbose=False,
                temperature=1.0, top_p=0.9, sample_count=1):
        """Forecast all OHLCV variates jointly via a Moirai predictive head.

        ``temperature``/``top_p``/``sample_count`` are accepted for a uniform
        interface across backends but unused here (sampling is controlled by the
        registry's ``num_samples``).
        """
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


def _ensure_kronos_importable() -> None:
    """Adds vendor/Kronos to sys.path so `from model import ...` works."""
    if not VENDOR_KRONOS_PATH.is_dir():
        raise FileNotFoundError(
            f"Vendored Kronos repo not found at {VENDOR_KRONOS_PATH}. "
            "Clone it with: git clone "
            "https://github.com/shiyu-coder/Kronos vendor/Kronos"
        )
    path = str(VENDOR_KRONOS_PATH)
    if path not in sys.path:
        sys.path.insert(0, path)


class KronosPredictor:
    """Wraps the vendored NeoQuasar Kronos predictor for the app ``predict()``.

    Kronos is a generative, token-based foundation model. It produces candle
    sequences by sampling from the next-token distribution, controlled by
    ``temperature``, ``top_p`` and ``sample_count`` (the sampled paths are
    averaged). It is not published on PyPI, so it is loaded from the vendored
    ``vendor/Kronos`` repo (see ``_ensure_kronos_importable``).

    The wrapped predictor returns a DataFrame indexed by the future timestamps
    with OHLCV columns; this wrapper still reconciles candle geometry
    (``high >= max(open, close)``, ``low <= min(open, close)``, ``volume >= 0``)
    as a safety net.
    """

    def __init__(self, predictor, max_context: int):
        self.predictor = predictor
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 120,
        verbose: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        pred = self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        return _reconcile_kronos_candles(pred, y_timestamp)


def _reconcile_kronos_candles(pred: pd.DataFrame, y_timestamp) -> pd.DataFrame:
    """Reconciles the vendored Kronos output into the app candle schema."""
    out = pred.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    open_ = out["open"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    volume = out["volume"].to_numpy(dtype=np.float64)

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.maximum(volume, 0.0)
    amount = volume * (open_ + high + low + close) / 4.0

    index = pd.DatetimeIndex(y_timestamp)
    index.name = "timestamps"
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        },
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


def _load_kronos(cfg: ModelConfig, device: str):
    _ensure_kronos_importable()
    from model import (  # noqa: PLC0415
        Kronos,
        KronosPredictor as _VendoredKronosPredictor,
        KronosTokenizer,
    )

    tokenizer = KronosTokenizer.from_pretrained(cfg.hf_tokenizer_id)
    model = Kronos.from_pretrained(cfg.hf_model_id)
    predictor = _VendoredKronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=cfg.max_context,
    )
    predictor.eval = getattr(model, "eval", lambda: None)  # ensure eval mode
    return KronosPredictor(predictor, max_context=cfg.max_context)


@lru_cache(maxsize=6)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Loads (cached) the selected model's predictor.

    Dispatches to TimesFM, Moirai or Kronos based on the registry backend.
    The first call downloads the weights from HuggingFace; subsequent calls
    reuse it.
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
    elif cfg.backend == "kronos":
        predictor = _load_kronos(cfg, device)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r} for {model_name!r}.")
    logger.info(
        "%s-%s ready on %s in %.1fs (max_context=%d)",
        cfg.backend, model_name, device, time.time() - t0, cfg.max_context,
    )
    return predictor