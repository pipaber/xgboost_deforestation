"""
FastAPI-based prediction API for the trained XGBoost bundle.

Goals:
- Load the trained bundle once at startup (joblib bundle.joblib).
- Enforce strict feature ordering (as trained) using `feature_columns.json`.
- Validate payloads and provide clear errors for missing/unknown features.
- Support single and batch predictions.
- Be friendly to web apps: JSON in/out, stable response schema.

Run (from repo root):
  uv run uvicorn deforestation.api:app --host 0.0.0.0 --port 8000

Example request:
  POST /predict
  {
    "features": {
      "YEAR": 2020,
      "Coca_ha": 0.0,
      "...": 123
    }
  }

Notes:
- This assumes the model inside the bundle is an xgboost sklearn estimator
  (e.g., xgboost.XGBRegressor) supporting `.predict(X)`.
- If your web app may omit some features, decide on a policy (reject vs default).
  This implementation rejects missing required features by default.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_MODEL_DIR = Path("models/xgb_timecv_v1_gpu")
DEFAULT_BUNDLE_PATH = DEFAULT_MODEL_DIR / "bundle.joblib"
DEFAULT_FEATURES_PATH = DEFAULT_MODEL_DIR / "feature_columns.json"


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key, "").strip()
    return Path(raw) if raw else default


BUNDLE_PATH = _env_path("DEFORESTATION_BUNDLE_PATH", DEFAULT_BUNDLE_PATH)
FEATURES_PATH = _env_path("DEFORESTATION_FEATURES_PATH", DEFAULT_FEATURES_PATH)

# Policy knobs (env-configurable)
ALLOW_EXTRA_FEATURES = os.environ.get("DEFORESTATION_ALLOW_EXTRA_FEATURES", "0").strip() not in ("", "0", "false", "False")
ALLOW_MISSING_FEATURES = os.environ.get("DEFORESTATION_ALLOW_MISSING_FEATURES", "0").strip() not in ("", "0", "false", "False")
MISSING_FEATURE_DEFAULT = os.environ.get("DEFORESTATION_MISSING_FEATURE_DEFAULT", "0").strip()
# Convert missing default to float if possible, else keep as string; we’ll only use it if missing is allowed.
try:
    MISSING_FEATURE_DEFAULT_VALUE: Any = float(MISSING_FEATURE_DEFAULT)
except Exception:
    MISSING_FEATURE_DEFAULT_VALUE = MISSING_FEATURE_DEFAULT


# -----------------------------
# Models (Pydantic)
# -----------------------------

Number = Union[int, float]


class PredictRequest(BaseModel):
    """
    Single prediction request.

    You MUST provide all required features (unless ALLOW_MISSING_FEATURES=1).
    """
    model_config = ConfigDict(extra="forbid")
    features: Dict[str, Any] = Field(..., description="Feature name -> value mapping")


class PredictBatchRequest(BaseModel):
    """
    Batch prediction request: list of rows. Each row is a dict of feature name -> value.
    """
    model_config = ConfigDict(extra="forbid")
    rows: List[Dict[str, Any]] = Field(..., min_length=1, description="List of feature dicts")


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: Optional[str] = None
    n_features: int
    feature_order: List[str]
    predictions: List[float]
    # Optional additional metadata (timings, etc.)
    meta: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]
    model_loaded: bool
    bundle_path: str
    features_path: str
    n_features: Optional[int] = None
    model_type: Optional[str] = None


# -----------------------------
# Internal predictor
# -----------------------------

@dataclass(frozen=True)
class LoadedArtifacts:
    model: Any
    feature_order: List[str]
    bundle_meta: Dict[str, Any]


class Predictor:
    def __init__(self, bundle_path: Path, features_path: Path) -> None:
        self._bundle_path = bundle_path
        self._features_path = features_path
        self._loaded: Optional[LoadedArtifacts] = None

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    def load(self) -> LoadedArtifacts:
        if self._loaded is not None:
            return self._loaded

        if not self._bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {self._bundle_path}")
        if not self._features_path.exists():
            raise FileNotFoundError(f"Feature columns file not found: {self._features_path}")

        bundle = joblib.load(self._bundle_path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise KeyError(
                "Bundle does not look like the expected dict with key 'model'. "
                f"Got type={type(bundle)!r}."
            )

        model = bundle["model"]
        if not hasattr(model, "predict"):
            raise TypeError(f"bundle['model'] does not have predict(). Got type={type(model)!r}")

        try:
            features_obj = json.loads(self._features_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Could not parse feature columns JSON at {self._features_path}: {e}") from e

        if not isinstance(features_obj, dict) or "columns" not in features_obj or not isinstance(features_obj["columns"], list):
            raise ValueError(f"Invalid feature_columns.json schema at {self._features_path}. Expected {{'columns': [..]}}")

        feature_order = [str(c) for c in features_obj["columns"]]
        if len(feature_order) == 0:
            raise ValueError("feature_columns.json contains empty 'columns' list")

        # Best-effort metadata
        meta: Dict[str, Any] = {}
        for k in ("version", "created_at", "training_data", "notes"):
            if isinstance(bundle, dict) and k in bundle:
                meta[k] = bundle.get(k)

        self._loaded = LoadedArtifacts(model=model, feature_order=feature_order, bundle_meta=meta)
        return self._loaded

    def _validate_and_vectorize_one(self, features: Dict[str, Any], feature_order: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate a single row dict and return (row_vector, meta).
        """
        # Identify missing/extra
        provided_keys = set(features.keys())
        expected_keys = set(feature_order)

        missing = [k for k in feature_order if k not in provided_keys]
        extra = sorted(list(provided_keys - expected_keys))

        if missing and not ALLOW_MISSING_FEATURES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "missing_features",
                    "message": "Request is missing required features",
                    "missing": missing,
                    "allow_missing_features": ALLOW_MISSING_FEATURES,
                },
            )

        if extra and not ALLOW_EXTRA_FEATURES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "extra_features",
                    "message": "Request contains unknown features",
                    "extra": extra,
                    "allow_extra_features": ALLOW_EXTRA_FEATURES,
                },
            )

        # Build ordered vector; apply defaults if allowed
        ordered_values: List[Any] = []
        used_defaults: List[str] = []

        for col in feature_order:
            if col in features:
                ordered_values.append(features[col])
            else:
                ordered_values.append(MISSING_FEATURE_DEFAULT_VALUE)
                used_defaults.append(col)

        # Convert to numeric where possible; keep NaNs if user passes null
        # Pandas is good at coercion; we’ll do 1-row DataFrame -> numeric conversion
        df = pd.DataFrame([ordered_values], columns=feature_order)
        # Coerce all columns to numeric; invalid parsing becomes NaN
        df = df.apply(pd.to_numeric, errors="coerce")

        # xgboost sklearn wrapper generally accepts numpy arrays
        x = df.to_numpy(dtype=np.float32, copy=False)

        meta = {
            "missing_filled": used_defaults,
            "extra_ignored": extra if ALLOW_EXTRA_FEATURES else [],
            "nan_count": int(np.isnan(x).sum()),
        }
        return x, meta

    def predict_one(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        art = self.load()
        x, meta = self._validate_and_vectorize_one(features, art.feature_order)
        y = art.model.predict(x)
        # Handle various return shapes
        try:
            pred = float(np.asarray(y).reshape(-1)[0])
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "prediction_output_invalid", "message": f"Could not parse prediction output: {e}"},
            )
        return pred, meta

    def predict_batch(self, rows: List[Dict[str, Any]]) -> Tuple[List[float], Dict[str, Any]]:
        art = self.load()

        xs: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                raise HTTPException(status_code=422, detail={"error": "row_not_object", "index": i})
            x, meta = self._validate_and_vectorize_one(row, art.feature_order)
            xs.append(x)
            metas.append(meta)

        x_all = np.vstack(xs) if len(xs) > 1 else xs[0]
        y = art.model.predict(x_all)
        arr = np.asarray(y).reshape(-1)
        preds = [float(v) for v in arr.tolist()]

        batch_meta = {
            "rows": len(rows),
            "row_meta": metas,
            "nan_total": int(np.isnan(x_all).sum()),
        }
        return preds, batch_meta


predictor = Predictor(bundle_path=BUNDLE_PATH, features_path=FEATURES_PATH)


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="Deforestation XGBoost Prediction API",
    version="1.0.0",
    description="Predict deforestation using a trained XGBoost model bundle (joblib).",
)


@app.exception_handler(FileNotFoundError)
async def _file_not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "model_files_not_found",
            "message": str(exc),
            "bundle_path": str(BUNDLE_PATH),
            "features_path": str(FEATURES_PATH),
        },
    )


@app.exception_handler(KeyError)
async def _key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "bundle_invalid", "message": str(exc)},
    )


@app.on_event("startup")
def _startup() -> None:
    # Preload so the first request is fast and we fail early if files are missing.
    predictor.load()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_type = None
    n_features = None
    if predictor.loaded:
        art = predictor.load()
        model_type = f"{type(art.model).__module__}.{type(art.model).__name__}"
        n_features = len(art.feature_order)

    return HealthResponse(
        status="ok",
        model_loaded=predictor.loaded,
        bundle_path=str(BUNDLE_PATH),
        features_path=str(FEATURES_PATH),
        n_features=n_features,
        model_type=model_type,
    )


@app.get("/schema/features")
def feature_schema() -> Dict[str, Any]:
    """
    Returns the feature order the model expects.
    Useful for building forms/clients.
    """
    art = predictor.load()
    return {
        "n_features": len(art.feature_order),
        "feature_order": art.feature_order,
        "allow_missing_features": ALLOW_MISSING_FEATURES,
        "missing_feature_default": MISSING_FEATURE_DEFAULT_VALUE if ALLOW_MISSING_FEATURES else None,
        "allow_extra_features": ALLOW_EXTRA_FEATURES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    art = predictor.load()

    t0 = time.perf_counter()
    pred, meta = predictor.predict_one(req.features)
    ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        model_name="xgb_timecv_v1_gpu",
        model_version=str(art.bundle_meta.get("version")) if art.bundle_meta.get("version") is not None else None,
        n_features=len(art.feature_order),
        feature_order=art.feature_order,
        predictions=[pred],
        meta={"latency_ms": ms, **meta},
    )


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(req: PredictBatchRequest) -> PredictResponse:
    art = predictor.load()

    t0 = time.perf_counter()
    preds, meta = predictor.predict_batch(req.rows)
    ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        model_name="xgb_timecv_v1_gpu",
        model_version=str(art.bundle_meta.get("version")) if art.bundle_meta.get("version") is not None else None,
        n_features=len(art.feature_order),
        feature_order=art.feature_order,
        predictions=preds,
        meta={"latency_ms": ms, **meta},
    )
