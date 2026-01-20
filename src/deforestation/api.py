"""
Prediction API for Peru deforestation model (XGBoost) with grouped SHAP contributions.

What this API is for
--------------------
You want an API where the user does NOT need to provide every feature the model expects.
Instead, the user provides:
- stable categorical context: Región, NOMBDEP
- year: YEAR
- only the numeric variables they want to change (Minería, Pobreza, etc.)

The API will:
1) Build a *baseline* feature vector from a server-side dataset row (UBIGEO + baseline_year).
2) Apply any numeric overrides provided by the caller (and YEAR if provided).
3) One-hot encode Región and NOMBDEP (no Cluster assumed), sanitize feature names, align to training schema.
4) Predict deforestation in hectares (model predicts log1p(Def_ha), API returns expm1(pred)).
5) Compute SHAP values for that request and return:
   - driver-group percentage contributions (Mining, Infrastructure, Agriculture, Climate, Socioeconomic, Geography/Admin, Other)

Important interpretability note
-------------------------------
"Percent contribution" is computed as:

  pct(group) = sum(|SHAP features in group|) / sum(|SHAP all features|)

This is a "share of model reasoning" for the prediction, NOT a causal attribution.

Run
---
uv run uvicorn deforestation.api:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
Model artifacts:
- DEFORESTATION_BUNDLE_PATH      (default: models/xgb_timecv_v1/bundle.joblib)
- DEFORESTATION_FEATURES_PATH    (default: models/xgb_timecv_v1/feature_columns.json)

Server dataset for baseline defaults:
- DEFORESTATION_DATASET_PATH     (default: deforestation_dataset_PERU_imputed_coca.csv)
- DEFORESTATION_DATASET_SEP      (default: ;)

Baseline policy:
- DEFORESTATION_BASELINE_YEAR    (default: 2020)

SHAP:
- Requires 'shap' installed in the API environment.

"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_MODEL_DIR = Path("models/xgb_timecv_v1")
DEFAULT_BUNDLE_PATH = DEFAULT_MODEL_DIR / "bundle.joblib"
DEFAULT_FEATURES_PATH = DEFAULT_MODEL_DIR / "feature_columns.json"

DEFAULT_DATASET_PATH = Path("deforestation_dataset_PERU_imputed_coca.csv")
DEFAULT_DATASET_SEP = os.environ.get("DEFORESTATION_DATASET_SEP", ";").strip() or ";"

DEFAULT_BASELINE_YEAR = int(
    os.environ.get("DEFORESTATION_BASELINE_YEAR", "2020").strip() or "2020"
)

BUNDLE_PATH = Path(
    os.environ.get("DEFORESTATION_BUNDLE_PATH", str(DEFAULT_BUNDLE_PATH)).strip()
    or str(DEFAULT_BUNDLE_PATH)
)
FEATURES_PATH = Path(
    os.environ.get("DEFORESTATION_FEATURES_PATH", str(DEFAULT_FEATURES_PATH)).strip()
    or str(DEFAULT_FEATURES_PATH)
)
DATASET_PATH = Path(
    os.environ.get("DEFORESTATION_DATASET_PATH", str(DEFAULT_DATASET_PATH)).strip()
    or str(DEFAULT_DATASET_PATH)
)
DATASET_SEP = DEFAULT_DATASET_SEP


# Driver-group labels for output
DRIVER_GROUPS = [
    "Mining",
    "Infrastructure",
    "Agriculture",
    "Climate",
    "Socioeconomic",
    "Geography/Admin",
    "Other",
]

# Heuristic substring matching over sanitized feature names
# (these should align with your data dictionary and feature engineering choices)
_DRIVER_GROUP_PATTERNS: Dict[str, List[str]] = {
    "Mining": ["Minería", "Mineria", "Dist_cat_Min"],
    "Infrastructure": [
        "Infraestructura",
        "Dist_vías",
        "Dist_vias",
        "Dist_vias",
        "vías",
        "vias",
    ],
    "Agriculture": ["area_agropec", "form_boscosa", "Yuca_ha", "Coca_ha"],
    "Climate": ["pp", "tmean", "hum_suelo"],
    "Socioeconomic": [
        "Población",
        "Poblacion",
        "dens_pob",
        "Pobreza",
        "IDH",
        "Pbi_dist",
        "Efic_gasto",
    ],
    "Geography/Admin": ["Región_", "Region_", "NOMBDEP_"],
}


# -----------------------------
# Utilities (consistent with training)
# -----------------------------


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """
    XGBoost forbids '[', ']' and '<' in feature names, and requires strings.
    We also normalize whitespace and replace spaces with underscores.
    """
    sanitized: List[str] = []
    seen: Dict[str, int] = {}
    for c in columns:
        s = str(c)
        s = s.replace("[", "(").replace("]", ")").replace("<", "lt")
        s = s.replace("\n", " ").replace("\r", " ").strip()
        s = s.replace(" ", "_")
        if s == "":
            s = "feature"
        if s in seen:
            seen[s] += 1
            s = f"{s}__{seen[s]}"
        else:
            seen[s] = 0
        sanitized.append(s)
    return sanitized


def safe_expm1(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


def normalize_ubigeo(ubigeo: str) -> str:
    """
    Normalize UBIGEO to 6-digit code when numeric-like.
    """
    u = str(ubigeo).strip()
    if u == "":
        return u
    # Avoid pandas scalar typing issues by using a plain try/except conversion.
    try:
        return str(int(float(u))).zfill(6)
    except Exception:
        return u


def group_driver(feature_name: str) -> str:
    for g, pats in _DRIVER_GROUP_PATTERNS.items():
        for p in pats:
            if p in feature_name:
                return g
    return "Other"


def driver_contributions_from_shap(
    feature_names: List[str], shap_values: np.ndarray
) -> Dict[str, Any]:
    """
    Convert SHAP values to driver-group percentage contributions.
    """
    sv = np.asarray(shap_values).reshape(-1)
    abs_sv = np.abs(sv)
    denom = float(np.sum(abs_sv))
    if denom <= 0.0 or not np.isfinite(denom):
        return {
            "method": "shap",
            "denominator": "sum_abs_shap",
            "groups": {g: 0.0 for g in DRIVER_GROUPS},
        }

    group_abs: Dict[str, float] = {g: 0.0 for g in DRIVER_GROUPS}
    for name, aval in zip(feature_names, abs_sv.tolist()):
        g = group_driver(str(name))
        if g not in group_abs:
            g = "Other"
        group_abs[g] += float(aval)

    out = {g: float(group_abs[g] / denom) for g in DRIVER_GROUPS}
    return {"method": "shap", "denominator": "sum_abs_shap", "groups": out}


# -----------------------------
# Data + model loading
# -----------------------------


@dataclass(frozen=True)
class LoadedArtifacts:
    model: Any
    feature_columns: List[str]
    train_config: Dict[str, Any]
    bundle_meta: Dict[str, Any]


_dataset_cache: Optional[pd.DataFrame] = None


def load_dataset() -> pd.DataFrame:
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, sep=DATASET_SEP)

    if "UBIGEO" not in df.columns or "YEAR" not in df.columns:
        raise ValueError("Dataset must contain UBIGEO and YEAR columns.")

    df["UBIGEO"] = df["UBIGEO"].astype("string")
    # Keep YEAR numeric-like but avoid strict dtype conversions that confuse type-checkers.
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

    _dataset_cache = df
    return df


def load_feature_columns() -> List[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found: {FEATURES_PATH}")
    obj = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(obj, dict)
        or "columns" not in obj
        or not isinstance(obj["columns"], list)
    ):
        raise ValueError('feature_columns.json must be {"columns": [...]}')

    cols = [str(c) for c in obj["columns"]]
    if not cols:
        raise ValueError("feature_columns.json has empty columns list")
    return cols


def load_bundle() -> LoadedArtifacts:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Bundle not found: {BUNDLE_PATH}")

    bundle = joblib.load(BUNDLE_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise KeyError("bundle.joblib must be a dict with key 'model'")

    model = bundle["model"]
    if not hasattr(model, "predict"):
        raise TypeError("bundle['model'] must implement predict()")

    # Train config helps us mirror preprocessing. If missing, we can still run by using
    # our known schema from feature_columns.json, but we prefer the bundle metadata.
    train_cfg = bundle.get("train_config")
    if not isinstance(train_cfg, dict):
        train_cfg = {}

    feature_cols = bundle.get("feature_columns")
    if isinstance(feature_cols, list) and feature_cols:
        feature_columns = [str(c) for c in feature_cols]
    else:
        feature_columns = load_feature_columns()

    meta: Dict[str, Any] = {}
    for k in ("version", "created_at", "training_data", "notes"):
        if k in bundle:
            meta[k] = bundle.get(k)

    return LoadedArtifacts(
        model=model,
        feature_columns=feature_columns,
        train_config=train_cfg,
        bundle_meta=meta,
    )


_artifacts: Optional[LoadedArtifacts] = None
_shap_explainer: Any = None


def artifacts() -> LoadedArtifacts:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_bundle()
    return _artifacts


def ensure_shap_explainer() -> Any:
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer

    if shap is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "shap_not_available",
                "message": "SHAP is not installed in this environment; cannot compute contributions.",
            },
        )

    try:
        _shap_explainer = shap.TreeExplainer(artifacts().model)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "shap_explainer_init_failed", "message": str(e)},
        )
    return _shap_explainer


# -----------------------------
# Preprocessing for raw input
# -----------------------------


def build_X_from_raw(raw: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Convert a raw feature dict into the trained feature space:
    - ensure Región and NOMBDEP exist (categoricals)
    - one-hot encode them with dummy_na=True
    - sanitize feature names
    - align to feature_columns (add missing with 0, drop extras)
    """
    df1 = pd.DataFrame([raw])

    # Ensure categoricals are strings if present
    for c in ("Región", "NOMBDEP"):
        if c in df1.columns:
            df1[c] = df1[c].astype("string")

    X = pd.get_dummies(
        df1,
        columns=[c for c in ("Región", "NOMBDEP") if c in df1.columns],
        dummy_na=True,
    )

    X.columns = sanitize_feature_names(list(X.columns))

    # Coerce all to numeric where possible; invalid parsing becomes NaN.
    # Explicitly wrap as DataFrame to keep the return type stable for static type checkers.
    X = pd.DataFrame(X.applymap(lambda v: pd.to_numeric(v, errors="coerce")))

    # Align to schema
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)

    # Return as a concrete DataFrame (not a potentially-typed slice) to avoid type-checker confusion.
    return pd.DataFrame(X.reindex(columns=feature_columns))


def baseline_row_for_ubigeo(
    df: pd.DataFrame, ubigeo: str, baseline_year: int
) -> pd.Series:
    u = normalize_ubigeo(ubigeo)
    sub = df[(df["UBIGEO"] == u) & (df["YEAR"] == baseline_year)]
    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "baseline_row_not_found",
                "message": "No baseline row found for UBIGEO at baseline_year.",
                "ubigeo": u,
                "baseline_year": baseline_year,
                "dataset_path": str(DATASET_PATH),
            },
        )
    return sub.iloc[0]


def apply_overrides(
    base: pd.Series, overrides: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Start from the baseline row and apply user overrides.
    Returns (raw_feature_dict, meta).
    """
    raw = base.to_dict()

    # Never allow UBIGEO / names to become model features; they are dropped/ignored later anyway.
    # Keep them in meta only.
    baseline_year: Optional[int] = None
    if "YEAR" in base.index:
        yv = base.get("YEAR")
        try:
            if yv is not None and pd.notna(yv):
                baseline_year = int(float(yv))
        except Exception:
            baseline_year = None

    meta: Dict[str, Any] = {
        "overrides_applied": [],
        "overrides_ignored": [],
        "baseline_year": baseline_year,
    }

    for k, v in (overrides or {}).items():
        if k is None:
            continue
        key = str(k).strip()
        if key == "":
            continue

        # UBIGEO is not a model feature (it is dropped in training); ignore if provided as override.
        if key.upper() == "UBIGEO":
            meta["overrides_ignored"].append(key)
            continue

        raw[key] = v
        meta["overrides_applied"].append(key)

    # Ensure YEAR exists as int if provided
    if "YEAR" in raw:
        try:
            raw["YEAR"] = int(float(raw["YEAR"]))
        except Exception:
            # Leave as-is; preprocessing will coerce/NaN it if invalid.
            pass
    return raw, meta


# -----------------------------
# API models
# -----------------------------


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"]
    model_loaded: bool
    bundle_path: str
    features_path: str
    dataset_path: str
    dataset_sep: str
    baseline_year: int
    n_features: Optional[int] = None
    model_type: Optional[str] = None


class PredictByUbigeoBaselineRequest(BaseModel):
    """
    The recommended endpoint.

    Client provides only:
    - UBIGEO (to find baseline row)
    - YEAR (optional override; allows forecasting for 2021+ if you provide YEAR and other variables)
    - Región / NOMBDEP overrides are allowed but typically stable.
    - numeric overrides for variables that "change" (Minería, Pobreza, etc.)

    Any omitted variables are defaulted from the baseline row.
    """

    model_config = ConfigDict(extra="forbid")

    ubigeo: str = Field(..., description="District UBIGEO code")
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw feature overrides (e.g., YEAR, Minería, Pobreza, pp, tmean, etc.)",
    )
    include_contributions: bool = Field(
        default=True,
        description="If true, compute grouped SHAP contribution percentages.",
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: Optional[str] = None
    predictions_ha: List[float]
    driver_contributions: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="Deforestation XGBoost Prediction API",
    version="2.0.0",
    description="Predict deforestation (ha) from baseline defaults + overrides, with grouped SHAP contributions.",
)


@app.exception_handler(FileNotFoundError)
async def _file_not_found_handler(_: Request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "file_not_found",
            "message": str(exc),
            "bundle_path": str(BUNDLE_PATH),
            "features_path": str(FEATURES_PATH),
            "dataset_path": str(DATASET_PATH),
        },
    )


@app.exception_handler(KeyError)
async def _key_error_handler(_: Request, exc: KeyError) -> JSONResponse:
    return JSONResponse(
        status_code=500, content={"error": "bundle_invalid", "message": str(exc)}
    )


@app.on_event("startup")
def _startup() -> None:
    # Load model and dataset early (fail fast)
    _ = artifacts()
    _ = load_dataset()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _artifacts is not None
    model_type = None
    n_features = None
    if loaded:
        a = artifacts()
        model_type = f"{type(a.model).__module__}.{type(a.model).__name__}"
        n_features = len(a.feature_columns)

    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        bundle_path=str(BUNDLE_PATH),
        features_path=str(FEATURES_PATH),
        dataset_path=str(DATASET_PATH),
        dataset_sep=DATASET_SEP,
        baseline_year=DEFAULT_BASELINE_YEAR,
        n_features=n_features,
        model_type=model_type,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictByUbigeoBaselineRequest) -> PredictResponse:
    t0 = time.perf_counter()
    a = artifacts()

    df = load_dataset()
    ub = normalize_ubigeo(req.ubigeo)
    base = baseline_row_for_ubigeo(df, ub, DEFAULT_BASELINE_YEAR)

    # Apply overrides (YEAR + variables that change)
    raw, override_meta = apply_overrides(base, req.overrides)

    # Build trained feature matrix
    X = build_X_from_raw(raw, feature_columns=a.feature_columns)

    # Predict (bundle model predicts log1p target)
    pred_log = a.model.predict(X)
    pred_ha = float(safe_expm1(pred_log).reshape(-1)[0])

    driver_contrib = None
    if req.include_contributions:
        explainer = ensure_shap_explainer()
        shap_vals = explainer.shap_values(X)
        driver_contrib = driver_contributions_from_shap(
            list(X.columns), np.asarray(shap_vals).reshape(-1)
        )

    ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        model_name="xgb_timecv_v1",
        model_version=str(a.bundle_meta.get("version"))
        if a.bundle_meta.get("version") is not None
        else None,
        predictions_ha=[pred_ha],
        driver_contributions=driver_contrib,
        meta={
            "latency_ms": ms,
            "ubigeo": ub,
            "baseline_year": DEFAULT_BASELINE_YEAR,
            "override_meta": override_meta,
            "effective_year": raw.get("YEAR"),
        },
    )
