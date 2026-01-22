"""
Prediction API for Peru deforestation model (XGBoost) with marginal-effects endpoints.

What this API is for
--------------------
You want an API where the user does NOT need to provide every feature the model expects.
Instead, the user provides:
- UBIGEO (district id used only to lookup a baseline row in the server dataset)
- YEAR (typically > baseline year, e.g. 2024)
- only the numeric variables they want to change (Minería, Pobreza, etc.)

The API will:
1) Build a *baseline* feature vector from a server-side dataset row (UBIGEO + baseline_year).
2) Apply any overrides provided by the caller (YEAR and/or covariates).
3) Optionally apply a "hindcast" macro projection policy for YEAR > baseline:
   - population growth (Población/dens_pob)
   - NOAA temperature anomaly delta (tmean)
   - precipitation multipliers (pp) with region scaling
   - simple multipliers for Minería/Infraestructura/area_agropec/Coca_ha
4) One-hot encode Región and NOMBDEP, sanitize feature names, align to the training schema.
5) Predict deforestation in hectares (model predicts log1p(Def_ha), API returns expm1(pred)).
6) Optionally compute marginal effects (finite differences) for feature deltas.

Endpoints
---------
POST /predict
- Predict for a single district (UBIGEO baseline row) with optional overrides.

POST /predict/aggregate
- Predict for ALL districts (baseline-year slice), apply optional overrides globally,
  and return totals aggregated by department or province.

POST /marginal/department
POST /marginal/province
POST /marginal/district
- Compute marginal effects (delta in predicted ha) by administrative level.

Important interpretability note
-------------------------------
Marginal effects are finite-difference "what-if" deltas in predicted hectares
for a specified feature change. They are not causal effects.

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

CORS:
- DEFORESTATION_CORS_ORIGINS     (default: allow localhost/127.0.0.1/[::1] on any port; set to * to allow all)

Hindcast tuning (optional overrides; defaults used if not set):
- DEFORESTATION_HINDCAST_MINERIA_FACTOR
- DEFORESTATION_HINDCAST_INFRA_FACTOR
- DEFORESTATION_HINDCAST_AGROPEC_FACTOR
- DEFORESTATION_HINDCAST_COCA_FACTOR

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


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

# -----------------------------
# CORS
# -----------------------------

# Default: allow local dev frontends (localhost/127.0.0.1/[::1]) on any port.
# Override with a comma-separated list via `DEFORESTATION_CORS_ORIGINS`, or set to
# `*` to allow all origins.
DEFAULT_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
CORS_ORIGINS_ENV = os.environ.get("DEFORESTATION_CORS_ORIGINS", "").strip()
if CORS_ORIGINS_ENV:
    if CORS_ORIGINS_ENV == "*":
        CORS_ALLOW_ORIGINS = ["*"]
    else:
        CORS_ALLOW_ORIGINS = [
            origin.strip()
            for origin in CORS_ORIGINS_ENV.split(",")
            if origin.strip()
        ]
    CORS_ALLOW_ORIGIN_REGEX: str | None = None
else:
    CORS_ALLOW_ORIGINS = []
    CORS_ALLOW_ORIGIN_REGEX = DEFAULT_CORS_ORIGIN_REGEX

# -----------------------------
# Hindcast defaults (macro assumptions)
# -----------------------------

HINDCAST_POP_GROWTH = {
    2021: 1.00,
    2022: 1.0096,
    2023: 1.0111,
    2024: 1.0248,
}

HINDCAST_NOAA_ANOMALY = {2020: 1.02, 2021: 0.87, 2022: 0.90, 2023: 1.19, 2024: 1.28}

HINDCAST_PP_FACTORS = {2021: 1.00, 2022: 0.99, 2023: 0.98, 2024: 0.97}

HINDCAST_MINERIA_FACTOR = float(
    os.environ.get("DEFORESTATION_HINDCAST_MINERIA_FACTOR", "1.02").strip() or "1.02"
)
HINDCAST_INFRA_FACTOR = float(
    os.environ.get("DEFORESTATION_HINDCAST_INFRA_FACTOR", "1.01").strip() or "1.01"
)
HINDCAST_AGROPEC_FACTOR = float(
    os.environ.get("DEFORESTATION_HINDCAST_AGROPEC_FACTOR", "1.01").strip() or "1.01"
)
HINDCAST_COCA_FACTOR = float(
    os.environ.get("DEFORESTATION_HINDCAST_COCA_FACTOR", "1.00").strip() or "1.00"
)

HINDCAST_PP_REGION_MULT = {
    "SELVA": 1.0,
    "SIERRA": 0.6,
    "COSTA": 0.0,
}


# Marginal-effects defaults
DEFAULT_MARGINAL_DELTA = 1.0
MARGINAL_EXCLUDE_COLS = {
    "UBIGEO",
    "YEAR",
    "NOMBDEP",
    "NOMBPROB",
    "NOMBDIST",
    "Def_ha",
}
MARGINAL_LEVEL_COLUMNS = {
    "department": "NOMBDEP",
    "province": "NOMBPROB",
    "district": "NOMBDIST",
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


def artifacts() -> LoadedArtifacts:
    global _artifacts
    if _artifacts is None:
        _artifacts = load_bundle()
    return _artifacts


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


def build_X_from_df(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Convert a raw DataFrame into the trained feature space:
    - one-hot encode Región and NOMBDEP with dummy_na=True
    - sanitize feature names
    - align to feature_columns (add missing with 0, drop extras)
    """
    X = df.copy()

    for c in ("Región", "NOMBDEP"):
        if c in X.columns:
            X[c] = X[c].astype("string")

    X = pd.get_dummies(
        X,
        columns=[c for c in ("Región", "NOMBDEP") if c in X.columns],
        dummy_na=True,
    )

    X.columns = sanitize_feature_names(list(X.columns))

    X = pd.DataFrame(X.applymap(lambda v: pd.to_numeric(v, errors="coerce")))

    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)

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


def apply_hindcast(
    raw: Dict[str, Any],
    year: int,
    baseline_year: int,
    locked: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply hindcast-style macro adjustments to a raw feature dict.
    Skips any keys present in `locked` (user overrides win).
    """
    locked_set = {str(k) for k in (locked or []) if k is not None}

    meta: Dict[str, Any] = {
        "mode": "hindcast",
        "active": False,
        "baseline_year": baseline_year,
        "year": year,
        "applied": [],
        "skipped": [],
        "factors": {},
    }

    if year <= baseline_year:
        return raw, meta

    def _set_if_unlocked(key: str, value: float) -> None:
        if key in locked_set:
            meta["skipped"].append(key)
            return
        raw[key] = value
        meta["applied"].append(key)

    def _mul_if_unlocked(key: str, factor: float) -> None:
        if key in locked_set:
            meta["skipped"].append(key)
            return
        if key not in raw:
            return
        val = pd.to_numeric(raw.get(key), errors="coerce")
        if pd.isna(val):
            return
        raw[key] = float(val) * float(factor)
        meta["applied"].append(key)

    def _add_if_unlocked(key: str, delta: float) -> None:
        if key in locked_set:
            meta["skipped"].append(key)
            return
        if key not in raw:
            return
        val = pd.to_numeric(raw.get(key), errors="coerce")
        if pd.isna(val):
            return
        raw[key] = float(val) + float(delta)
        meta["applied"].append(key)

    # Population growth (compound from 2021..year)
    factor = 1.0
    for y in range(2021, year + 1):
        factor *= HINDCAST_POP_GROWTH.get(y, 1.0)
    meta["factors"]["population_factor"] = float(factor)
    for c in ["Población", "Poblacion", "dens_pob"]:
        _mul_if_unlocked(c, factor)

    # Temperature anomaly (NOAA delta vs baseline year)
    base_anom = HINDCAST_NOAA_ANOMALY.get(baseline_year, 1.02)
    year_anom = HINDCAST_NOAA_ANOMALY.get(year)
    if year_anom is not None:
        delta = float(year_anom - base_anom)
        meta["factors"]["tmean_delta"] = float(delta)
        _add_if_unlocked("tmean", delta)

    # Precipitation factor (year + region scaling)
    pp_factor = float(HINDCAST_PP_FACTORS.get(year, 1.0))
    region = str(raw.get("Región") or "").strip().upper()
    region_mult = float(HINDCAST_PP_REGION_MULT.get(region, 1.0))
    meta["factors"]["pp_factor"] = float(pp_factor)
    meta["factors"]["pp_region_mult"] = float(region_mult)
    _mul_if_unlocked("pp", pp_factor * region_mult)

    # Simple multipliers (vs baseline)
    meta["factors"]["mineria_factor"] = float(HINDCAST_MINERIA_FACTOR)
    meta["factors"]["infra_factor"] = float(HINDCAST_INFRA_FACTOR)
    meta["factors"]["agropec_factor"] = float(HINDCAST_AGROPEC_FACTOR)
    meta["factors"]["coca_factor"] = float(HINDCAST_COCA_FACTOR)

    _mul_if_unlocked("Minería", HINDCAST_MINERIA_FACTOR)
    _mul_if_unlocked("Infraestructura", HINDCAST_INFRA_FACTOR)
    _mul_if_unlocked("area_agropec", HINDCAST_AGROPEC_FACTOR)
    _mul_if_unlocked("Coca_ha", HINDCAST_COCA_FACTOR)

    meta["active"] = True
    return raw, meta


def apply_hindcast_df(
    df: pd.DataFrame, year: int, baseline_year: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply hindcast-style macro adjustments to a DataFrame (vectorized).
    """
    meta: Dict[str, Any] = {
        "mode": "hindcast",
        "active": False,
        "baseline_year": baseline_year,
        "year": year,
        "factors": {},
        "applied_columns": [],
    }

    if year <= baseline_year:
        return df, meta

    out = df.copy()

    # Population growth (compound from 2021..year)
    factor = 1.0
    for y in range(2021, year + 1):
        factor *= HINDCAST_POP_GROWTH.get(y, 1.0)
    meta["factors"]["population_factor"] = float(factor)
    for c in ["Población", "Poblacion", "dens_pob"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * factor
            meta["applied_columns"].append(c)

    # Temperature anomaly (NOAA delta vs baseline year)
    base_anom = HINDCAST_NOAA_ANOMALY.get(baseline_year, 1.02)
    year_anom = HINDCAST_NOAA_ANOMALY.get(year)
    if year_anom is not None and "tmean" in out.columns:
        delta = float(year_anom - base_anom)
        meta["factors"]["tmean_delta"] = float(delta)
        out["tmean"] = pd.to_numeric(out["tmean"], errors="coerce") + delta
        meta["applied_columns"].append("tmean")

    # Precipitation factor (year + region scaling)
    if "pp" in out.columns:
        pp_factor = float(HINDCAST_PP_FACTORS.get(year, 1.0))
        region = out.get("Región")
        if region is None:
            out["pp"] = pd.to_numeric(out["pp"], errors="coerce") * pp_factor
            meta["factors"]["pp_factor"] = float(pp_factor)
            meta["factors"]["pp_region_mult"] = 1.0
        else:
            r = region.astype("string").str.upper()
            mult = r.map(HINDCAST_PP_REGION_MULT).fillna(1.0).astype(float)
            out["pp"] = pd.to_numeric(out["pp"], errors="coerce") * pp_factor * mult
            meta["factors"]["pp_factor"] = float(pp_factor)
            meta["factors"]["pp_region_mult"] = "by_region"
        meta["applied_columns"].append("pp")

    # Simple multipliers
    meta["factors"]["mineria_factor"] = float(HINDCAST_MINERIA_FACTOR)
    meta["factors"]["infra_factor"] = float(HINDCAST_INFRA_FACTOR)
    meta["factors"]["agropec_factor"] = float(HINDCAST_AGROPEC_FACTOR)
    meta["factors"]["coca_factor"] = float(HINDCAST_COCA_FACTOR)

    for col, factor in [
        ("Minería", HINDCAST_MINERIA_FACTOR),
        ("Infraestructura", HINDCAST_INFRA_FACTOR),
        ("area_agropec", HINDCAST_AGROPEC_FACTOR),
        ("Coca_ha", HINDCAST_COCA_FACTOR),
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(factor)
            meta["applied_columns"].append(col)

    meta["active"] = True
    return out, meta


# -----------------------------
# Marginal effects helpers
# -----------------------------


def _is_excluded_feature(name: str) -> bool:
    if name in MARGINAL_EXCLUDE_COLS:
        return True
    return str(name).strip().lower().startswith("regi")


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    return pd.to_numeric(series, errors="coerce").notna().any()


def _coerce_delta(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def resolve_feature_deltas(
    df: pd.DataFrame,
    features: Optional[List[str]],
    deltas: Dict[str, Any],
    default_delta: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    resolved: Dict[str, float] = {}
    meta: Dict[str, Any] = {
        "missing_features": [],
        "excluded_features": [],
        "non_numeric_features": [],
        "invalid_deltas": [],
    }

    if features:
        candidates = [
            str(f).strip() for f in features if f is not None and str(f).strip()
        ]
    elif deltas:
        candidates = [
            str(f).strip() for f in deltas.keys() if f is not None and str(f).strip()
        ]
    else:
        candidates = [str(c) for c in df.columns]

    seen = set()
    for feat in candidates:
        if feat in seen:
            continue
        seen.add(feat)

        if feat not in df.columns:
            meta["missing_features"].append(feat)
            continue
        if _is_excluded_feature(feat):
            meta["excluded_features"].append(feat)
            continue
        if not _is_numeric_series(df[feat]):
            meta["non_numeric_features"].append(feat)
            continue

        delta = _coerce_delta(deltas.get(feat, default_delta))
        if delta is None:
            meta["invalid_deltas"].append(feat)
            continue
        resolved[feat] = delta

    return resolved, meta


def predict_ha_from_df(a: LoadedArtifacts, df: pd.DataFrame) -> np.ndarray:
    X = build_X_from_df(df, feature_columns=a.feature_columns)
    pred_log = a.model.predict(X)
    return safe_expm1(pred_log).reshape(-1)


def _group_key(val: Any) -> str:
    if val is None or pd.isna(val):
        return "NA"
    return str(val)


def compute_marginal_effects_by_group(
    base: pd.DataFrame,
    group_col: str,
    feature_deltas: Dict[str, float],
    artifacts_obj: LoadedArtifacts,
) -> Dict[str, Any]:
    pred_base = predict_ha_from_df(artifacts_obj, base)
    base_with_pred = base.copy()
    base_with_pred["_pred_ha"] = pred_base
    base_group = base_with_pred.groupby(group_col, dropna=False)["_pred_ha"].sum()

    results: Dict[str, Any] = {}
    for key, base_val in base_group.items():
        results[_group_key(key)] = {
            "baseline_ha": float(base_val),
            "effects": {},
        }

    for feat, delta in feature_deltas.items():
        modified = base.copy()
        modified[feat] = pd.to_numeric(modified[feat], errors="coerce") + float(delta)
        pred_mod = predict_ha_from_df(artifacts_obj, modified)
        modified["_pred_ha"] = pred_mod
        mod_group = modified.groupby(group_col, dropna=False)["_pred_ha"].sum()

        for key, base_val in base_group.items():
            key_str = _group_key(key)
            new_val = float(mod_group.get(key, 0.0))
            delta_ha = float(new_val - float(base_val))
            results[key_str]["effects"][feat] = {
                "delta": float(delta),
                "delta_ha": delta_ha,
                "delta_per_unit": float(delta_ha / delta) if delta != 0 else None,
                "new_ha": new_val,
            }

    return results


def prepare_base_slice(
    overrides: Dict[str, Any],
    mode: Literal["hindcast", "baseline"],
) -> Tuple[pd.DataFrame, int, Optional[Dict[str, Any]], List[str]]:
    df = load_dataset()
    base = df[df["YEAR"] == DEFAULT_BASELINE_YEAR].copy()
    if base.empty:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "baseline_rows_not_found",
                "message": "No baseline rows found for baseline year.",
                "baseline_year": DEFAULT_BASELINE_YEAR,
            },
        )

    override_keys = [
        str(k).strip()
        for k in (overrides or {}).keys()
        if k is not None and str(k).strip() != ""
    ]

    for k, v in (overrides or {}).items():
        key = str(k).strip()
        if key == "" or key.upper() == "UBIGEO":
            continue
        base[key] = v

    effective_year = DEFAULT_BASELINE_YEAR
    if "YEAR" in base.columns:
        try:
            effective_year = int(float(base["YEAR"].iloc[0]))
        except Exception:
            effective_year = DEFAULT_BASELINE_YEAR

    hindcast_meta = None
    if mode == "hindcast":
        base, hindcast_meta = apply_hindcast_df(
            base, year=effective_year, baseline_year=DEFAULT_BASELINE_YEAR
        )
        for k, v in (overrides or {}).items():
            key = str(k).strip()
            if key == "" or key.upper() == "UBIGEO":
                continue
            base[key] = v

    return base, effective_year, hindcast_meta, override_keys


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
    mode: Literal["hindcast", "baseline"] = Field(
        default="hindcast",
        description="hindcast applies 2021+ macro adjustments by default; baseline uses raw 2020 values + overrides only.",
    )
    # Kept minimal: marginal effects are available via separate endpoints.


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: Optional[str] = None
    predictions_ha: List[float]
    features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw features used for prediction (before encoding).",
    )
    meta: Dict[str, Any] = Field(default_factory=dict)


class PredictAggregateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_by: Literal["department", "province"] = Field(
        ..., description="Aggregation level: department or province"
    )
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw feature overrides applied to all districts (e.g., YEAR, Minería, pp, tmean)",
    )
    mode: Literal["hindcast", "baseline"] = Field(
        default="hindcast",
        description="hindcast applies macro adjustments for YEAR > baseline; baseline uses raw baseline values + overrides only.",
    )
    # Kept minimal: marginal effects are available via separate endpoints.


class PredictAggregateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: Optional[str] = None
    group_by: Literal["department", "province"]
    year: int
    results: Dict[str, Any]
    total_pred_ha: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class MarginalEffectsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw feature overrides applied to all districts (e.g., YEAR, Miner¡a, pp, tmean).",
    )
    mode: Literal["hindcast", "baseline"] = Field(
        default="hindcast",
        description="hindcast applies macro adjustments for YEAR > baseline; baseline uses raw baseline values + overrides only.",
    )
    features: Optional[List[str]] = Field(
        default=None,
        description="Optional list of feature names to compute marginal effects for.",
    )
    deltas: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-feature additive deltas in native units (overrides default_delta).",
    )
    default_delta: float = Field(
        default=DEFAULT_MARGINAL_DELTA,
        description="Default additive delta for features not in deltas.",
    )


class MarginalEffectsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    model_version: Optional[str] = None
    level: Literal["department", "province", "district"]
    year: int
    results: Dict[str, Any]
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="Deforestation XGBoost Prediction API",
    version="2.0.0",
    description="Predict deforestation (ha) from baseline defaults + overrides, with marginal-effects endpoints.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Hindcast mode (default): auto-adjust covariates for YEAR > baseline
    effective_year = DEFAULT_BASELINE_YEAR
    if "YEAR" in raw:
        try:
            effective_year = int(float(raw["YEAR"]))
        except Exception:
            effective_year = DEFAULT_BASELINE_YEAR

    hindcast_meta = None
    if req.mode == "hindcast":
        raw, hindcast_meta = apply_hindcast(
            raw,
            year=effective_year,
            baseline_year=DEFAULT_BASELINE_YEAR,
            locked=override_meta.get("overrides_applied") or [],
        )

    # Build trained feature matrix
    X = build_X_from_raw(raw, feature_columns=a.feature_columns)

    # Predict (bundle model predicts log1p target)
    pred_log = a.model.predict(X)
    pred_ha = float(safe_expm1(pred_log).reshape(-1)[0])

    ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        model_name="xgb_timecv_v1",
        model_version=str(a.bundle_meta.get("version"))
        if a.bundle_meta.get("version") is not None
        else None,
        predictions_ha=[pred_ha],
        features=raw,
        meta={
            "latency_ms": ms,
            "ubigeo": ub,
            "baseline_year": DEFAULT_BASELINE_YEAR,
            "override_meta": override_meta,
            "hindcast_meta": hindcast_meta,
            "mode": req.mode,
            "effective_year": raw.get("YEAR"),
        },
    )


@app.post("/predict/aggregate", response_model=PredictAggregateResponse)
def predict_aggregate(req: PredictAggregateRequest) -> PredictAggregateResponse:
    t0 = time.perf_counter()
    a = artifacts()

    overrides = dict(req.overrides or {})
    base, effective_year, hindcast_meta, override_keys = prepare_base_slice(
        overrides, req.mode
    )

    # Build trained feature matrix
    X = build_X_from_df(base, feature_columns=a.feature_columns)

    # Predict
    pred_log = a.model.predict(X)
    pred_ha = safe_expm1(pred_log).reshape(-1)
    base["pred_def_ha"] = pred_ha

    # Choose group column
    group_col = "NOMBDEP"
    if req.group_by == "province":
        if "NOMBPROV" in base.columns:
            group_col = "NOMBPROV"
        elif "NOMBPROB" in base.columns:
            group_col = "NOMBPROB"
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "province_column_missing",
                    "message": "Dataset does not contain NOMBPROV or NOMBPROB.",
                },
            )

    if req.group_by == "province":
        dep_col = "NOMBDEP"
        if dep_col not in base.columns:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "department_column_missing",
                    "message": "Dataset does not contain NOMBDEP for nested province output.",
                },
            )

        grouped = (
            base.groupby([dep_col, group_col], dropna=False)["pred_def_ha"]
            .sum()
            .sort_values(ascending=False)
        )

        results: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for (dep, prov), val in grouped.items():
            dep_key = str(dep) if dep is not None else "NA"
            prov_key = str(prov) if prov is not None else "NA"
            results.setdefault(dep_key, {})[prov_key] = {"pred_ha": float(val)}
    else:
        grouped = (
            base.groupby(group_col, dropna=False)["pred_def_ha"]
            .sum()
            .sort_values(ascending=False)
        )
        results = {
            str(idx) if idx is not None else "NA": {"pred_ha": float(val)}
            for idx, val in grouped.items()
        }

    total_pred = float(np.sum(pred_ha))
    ms = (time.perf_counter() - t0) * 1000.0

    return PredictAggregateResponse(
        model_name="xgb_timecv_v1",
        model_version=str(a.bundle_meta.get("version"))
        if a.bundle_meta.get("version") is not None
        else None,
        group_by=req.group_by,
        year=effective_year,
        results=results,
        total_pred_ha=total_pred,
        meta={
            "latency_ms": ms,
            "baseline_year": DEFAULT_BASELINE_YEAR,
            "mode": req.mode,
            "effective_year": effective_year,
            "group_column": group_col,
            "overrides_applied": override_keys,
            "hindcast_meta": hindcast_meta,
        },
    )


def _marginal_effects_for_level(
    req: MarginalEffectsRequest, level: Literal["department", "province", "district"]
) -> MarginalEffectsResponse:
    t0 = time.perf_counter()
    a = artifacts()

    overrides = dict(req.overrides or {})
    base, effective_year, hindcast_meta, override_keys = prepare_base_slice(
        overrides, req.mode
    )

    group_col = MARGINAL_LEVEL_COLUMNS[level]
    if (
        level == "province"
        and group_col not in base.columns
        and "NOMBPROV" in base.columns
    ):
        group_col = "NOMBPROV"
    if group_col not in base.columns:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "group_column_missing",
                "message": f"Dataset does not contain {group_col}.",
            },
        )

    feature_deltas, feature_meta = resolve_feature_deltas(
        base, req.features, req.deltas, req.default_delta
    )
    if not feature_deltas:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_valid_features",
                "message": "No valid numeric features found for marginal effects.",
                "feature_meta": feature_meta,
            },
        )

    results = compute_marginal_effects_by_group(base, group_col, feature_deltas, a)
    ms = (time.perf_counter() - t0) * 1000.0

    return MarginalEffectsResponse(
        model_name="xgb_timecv_v1",
        model_version=str(a.bundle_meta.get("version"))
        if a.bundle_meta.get("version") is not None
        else None,
        level=level,
        year=effective_year,
        results=results,
        meta={
            "latency_ms": ms,
            "baseline_year": DEFAULT_BASELINE_YEAR,
            "mode": req.mode,
            "effective_year": effective_year,
            "group_column": group_col,
            "overrides_applied": override_keys,
            "hindcast_meta": hindcast_meta,
            "feature_deltas": feature_deltas,
            "feature_meta": feature_meta,
        },
    )


@app.post("/marginal/department", response_model=MarginalEffectsResponse)
def marginal_department(req: MarginalEffectsRequest) -> MarginalEffectsResponse:
    return _marginal_effects_for_level(req, "department")


@app.post("/marginal/province", response_model=MarginalEffectsResponse)
def marginal_province(req: MarginalEffectsRequest) -> MarginalEffectsResponse:
    return _marginal_effects_for_level(req, "province")


@app.post("/marginal/district", response_model=MarginalEffectsResponse)
def marginal_district(req: MarginalEffectsRequest) -> MarginalEffectsResponse:
    return _marginal_effects_for_level(req, "district")
