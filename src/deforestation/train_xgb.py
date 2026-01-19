"""
XGBoost training for Peru deforestation (district-year) regression.

What this script does
---------------------
- Loads the dataset (supports ';' or '\t' separated depending on file)
- Builds a leakage-aware time split:
    Train: 2001–2016
    Val:   2017–2018
    Test:  2019–2020
- Trains an XGBoost regressor on log1p(Def_ha)
- Runs hyperparameter search (random search) using the validation set + early stopping
- Logs EVERY hyperparameter trial to CSV (so you never lose progress)
- Evaluates metrics on train/val/test:
    MAE, RMSE, R2
  and also metrics on the subset where true Def_ha > 0
- Exports:
    - model JSON
    - feature schema (columns)
    - metrics report JSON
    - predictions CSV for test set
    - trials_log.csv (all trials)
    - bundle.joblib (model + feature columns + config)

GPU mode
--------
This script supports optional GPU training via:
- --device cuda

For XGBoost 3.x, prefer the decoupled configuration:
- tree_method='hist'
- device='cuda'

If you do not have a compatible GPU / CUDA-enabled XGBoost build, keep the default:
- --device cpu  (tree_method='hist', device='cpu')

Important: pickling / bundle portability
----------------------------------------
The SHAP script loads `bundle.joblib`. To avoid pickle errors like:
    AttributeError: Can't get attribute 'SplitConfig' on <module '__main__' ...>
we store `SplitConfig` and `TrainConfig` in the bundle as plain dicts (not dataclass instances),
so the bundle is portable across entrypoints.

Important: Feature name sanitization
------------------------------------
XGBoost requires feature names to be strings and forbids certain characters like
'[' , ']' and '<'. One-hot encoding (or raw column names like 'P1171$01') can
introduce such characters, causing:
    ValueError: feature_names must be string, and may not contain [, ] or <

This script sanitizes feature names after one-hot encoding to ensure training works.

Notes / Assumptions
-------------------
- XGBoost handles numeric NaN natively. We do NOT impute numeric features here.
- Categorical columns are one-hot encoded via pandas.get_dummies.
- We intentionally drop:
    - identifiers and free-text names that can lead to memorization
    - extreme-missingness blocks (~90–95% missing) that are likely to encode "is missing" artifacts
- If you want to forecast future years, add lag features carefully with time-safe construction.

Run
---
CPU:
  uv run python src/deforestation/train_xgb.py --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --trials 200 --run-name xgb_tune_v1 --device cpu

GPU (if available):
  uv run python src/deforestation/train_xgb.py --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --trials 200 --run-name xgb_tune_v1 --device cuda

Artifacts
---------
Saved under:
- models/xgb_defha_<timestamp>/

New behavior:
- You can optionally choose an explicit run directory name (so re-runs append trials instead of creating a new folder).
- If the run directory already exists, this script will REUSE it and append to trials_log.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class SplitConfig:
    train_end_year: int = 2016
    val_start_year: int = 2017
    val_end_year: int = 2018
    test_start_year: int = 2019
    test_end_year: int = 2020


@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "Def_ha"
    year_col: str = "YEAR"
    ubigeo_col: str = "UBIGEO"

    # Feature handling
    categorical_cols: Tuple[str, ...] = ("Región", "NOMBDEP", "Cluster")
    drop_cols: Tuple[str, ...] = (
        # IDs / names (high leakage risk / not stable)
        "UBIGEO",
        "NOMBPROB",
        "NOMBDIST",
        # very sparse distances
        "Dist_ríos",
        "Dist_vías",
        "Dist_comunid",
        "Dist_conc_mad",
        # very sparse employment block
        "Emp_estado",
        "Emp_cientificos",
        "Emp_tecnicos",
        "Emp_oficinistas",
        "Emp_servicios",
        "Emp_agricultura",
        "Emp_obreros_minasyenerg",
        "Emp_obreros",
        "Emp_nocalif_agric_indust",
        "Emp_otros",
        # very sparse migration totals
        "tot_salieron",
        "tot_ingresaron",
    )

    # Optional: explicitly keep only this set (leave empty to "keep all except drop")
    allowlist_cols: Tuple[str, ...] = ()

    # XGBoost training
    random_seed: int = 42
    n_trials: int = 40
    early_stopping_rounds: int = 200
    max_estimators: int = 5000  # early stopping controls effective trees

    # Objective on log1p(target)
    objective: str = "reg:squarederror"


# -----------------------------
# Utilities
# -----------------------------


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """
    Sanitize feature names to satisfy XGBoost constraints.

    XGBoost requires:
    - feature names are strings
    - feature names do NOT contain: '[' , ']' , '<'

    We also normalize whitespace and replace other problematic characters to keep
    the model portable across environments.
    """
    sanitized: List[str] = []
    seen: Dict[str, int] = {}

    for c in columns:
        s = str(c)

        # Replace forbidden characters
        s = s.replace("[", "(").replace("]", ")").replace("<", "lt")

        # Additional normalization for portability/readability
        s = s.replace("\n", " ").replace("\r", " ").strip()
        s = s.replace(" ", "_")

        # Avoid empty names
        if s == "":
            s = "feature"

        # Ensure uniqueness after sanitization
        if s in seen:
            seen[s] += 1
            s = f"{s}__{seen[s]}"
        else:
            seen[s] = 0

        sanitized.append(s)

    return sanitized


def apply_sanitized_feature_names(*frames: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Apply the same sanitized feature names to all provided DataFrames.
    Assumes they share the same column order.
    """
    if not frames:
        return tuple()

    cols = list(frames[0].columns)
    new_cols = sanitize_feature_names(cols)

    out: List[pd.DataFrame] = []
    for f in frames:
        f2 = f.copy()
        f2.columns = new_cols
        out.append(f2)

    return tuple(out)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_log1p(y: np.ndarray) -> np.ndarray:
    # Def_ha is non-negative; still guard.
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Target contains negative values; cannot apply log1p safely.")
    return np.log1p(y)


def _safe_expm1(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def subset_metrics_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(y_true, dtype=float) > 0
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0.0}
    m = regression_metrics(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
    m["n"] = float(mask.sum())
    return m


# -----------------------------
# Data loading / preprocessing
# -----------------------------


def load_dataset(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # Normalize YEAR to int
    if "YEAR" not in df.columns:
        raise ValueError(f"Expected YEAR column in dataset. Columns={df.columns.tolist()}")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    if df["YEAR"].isna().any():
        bad = df[df["YEAR"].isna()].head(10)
        raise ValueError(f"Found non-numeric YEAR values. Example rows:\n{bad.to_string(index=False)}")
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def time_split(df: pd.DataFrame, split: SplitConfig, year_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[year_col] <= split.train_end_year].copy()
    val = df[(df[year_col] >= split.val_start_year) & (df[year_col] <= split.val_end_year)].copy()
    test = df[(df[year_col] >= split.test_start_year) & (df[year_col] <= split.test_end_year)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split detected. train={train.shape}, val={val.shape}, test={test.shape}. "
            f"Check YEAR range or SplitConfig."
        )
    return train, val, test


def select_features(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")
    y = df[cfg.target_col].astype(float)

    # Start with all columns except target
    X = df.drop(columns=[cfg.target_col])

    # Optional allowlist handling
    if cfg.allowlist_cols:
        keep = set(cfg.allowlist_cols) | {cfg.year_col} | set(cfg.categorical_cols)
        present = [c for c in X.columns if c in keep]
        X = X[present].copy()

    # Drop specified columns if present
    drop = [c for c in cfg.drop_cols if c in X.columns]
    X = X.drop(columns=drop)

    # Ensure YEAR kept (it's a useful temporal feature even without lags)
    if cfg.year_col not in X.columns and cfg.year_col in df.columns:
        X[cfg.year_col] = df[cfg.year_col]

    return X, y


def one_hot_encode(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: Tuple[str, ...],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Convert categoricals to string to avoid mixed dtypes
    for c in categorical_cols:
        for X in (X_train, X_val, X_test):
            if c in X.columns:
                X[c] = X[c].astype("string")

    # get_dummies on concatenated frame to ensure aligned columns
    all_X = pd.concat(
        [
            X_train.assign(_split="train"),
            X_val.assign(_split="val"),
            X_test.assign(_split="test"),
        ],
        axis=0,
        ignore_index=True,
    )

    cat_present = [c for c in categorical_cols if c in all_X.columns]
    all_enc = pd.get_dummies(all_X, columns=cat_present, dummy_na=True)

    # Split back
    train_enc = all_enc[all_enc["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_enc = all_enc[all_enc["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    test_enc = all_enc[all_enc["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

    # XGBoost requires sanitized feature names (no '[' ']' '<', etc.)
    train_enc, val_enc, test_enc = apply_sanitized_feature_names(train_enc, val_enc, test_enc)

    return train_enc, val_enc, test_enc


# -----------------------------
# Hyperparameter search
# -----------------------------


def sample_params(rng: random.Random, cfg: TrainConfig, device: str) -> Dict[str, Any]:
    """
    Randomly sample hyperparameters from sensible ranges.

    device:
      - 'cpu' (default): tree_method='hist'
      - 'cuda'         : tree_method='gpu_hist', device='cuda'
    """
    # log-uniform helpers
    def log_uniform(a: float, b: float) -> float:
        return float(math.exp(rng.uniform(math.log(a), math.log(b))))

    params = {
        "n_estimators": cfg.max_estimators,
        "learning_rate": log_uniform(0.01, 0.2),
        "max_depth": rng.randint(3, 10),
        "min_child_weight": log_uniform(0.5, 30.0),
        "subsample": rng.uniform(0.6, 1.0),
        "colsample_bytree": rng.uniform(0.6, 1.0),
        "reg_alpha": log_uniform(1e-8, 10.0),
        "reg_lambda": log_uniform(0.5, 50.0),
        "gamma": log_uniform(1e-8, 10.0),
        "objective": cfg.objective,
        "random_state": cfg.random_seed,
        "n_jobs": max(1, os.cpu_count() or 1),
    }

    # XGBoost 3.x: decouple algorithm from device selection.
    # Use hist everywhere, and select hardware via `device`.
    params["tree_method"] = "hist"
    if device == "cuda":
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

    return params


def train_one(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
    params: Dict[str, Any],
    cfg: TrainConfig,
) -> Tuple[XGBRegressor, Dict[str, Any]]:
    # XGBoost v3.1.3 sklearn API does not accept `early_stopping_rounds=` nor `callbacks=`
    # in .fit(). Instead, configure early stopping in the constructor and provide eval_set.
    params = dict(params)
    params["early_stopping_rounds"] = cfg.early_stopping_rounds
    params["eval_metric"] = "rmse"

    model = XGBRegressor(**params)

    model.fit(
        X_train,
        y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False,
    )

    # best_iteration is available after early stopping (when supported by the backend)
    best_iter = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)

    info = {
        "best_iteration": int(best_iter) if best_iter is not None else None,
        "best_score": float(best_score) if best_score is not None else None,
    }
    return model, info


def evaluate_model_on_splits(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    # Predict in log-space and invert
    pred_train = _safe_expm1(model.predict(X_train))
    pred_val = _safe_expm1(model.predict(X_val))
    pred_test = _safe_expm1(model.predict(X_test))

    out: Dict[str, Any] = {
        "train": regression_metrics(y_train, pred_train),
        "val": regression_metrics(y_val, pred_val),
        "test": regression_metrics(y_test, pred_test),
        "train_nonzero": subset_metrics_nonzero(y_train, pred_train),
        "val_nonzero": subset_metrics_nonzero(y_val, pred_val),
        "test_nonzero": subset_metrics_nonzero(y_test, pred_test),
    }
    return out


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost to predict Def_ha with time-based split + tuning.")
    parser.add_argument("--data", required=True, help="Path to dataset CSV.")
    parser.add_argument("--sep", required=True, help="CSV separator, e.g. ';' or '\\t'.")
    parser.add_argument("--outdir", default="models", help="Base output directory.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name under --outdir. If provided and exists, trials will append to trials_log.csv.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. Use 'cuda' to enable GPU training (requires compatible NVIDIA GPU + CUDA-enabled XGBoost).",
    )
    parser.add_argument("--trials", type=int, default=None, help="Override number of hyperparameter trials.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    split_cfg = SplitConfig()
    train_cfg = TrainConfig(
        n_trials=args.trials if args.trials is not None else TrainConfig.n_trials,
        random_seed=args.seed if args.seed is not None else TrainConfig.random_seed,
    )

    rng = random.Random(train_cfg.random_seed)
    np.random.seed(train_cfg.random_seed)

    df = load_dataset(args.data, sep=args.sep)

    # Split
    df_train, df_val, df_test = time_split(df, split_cfg, year_col=train_cfg.year_col)

    # Feature selection
    X_train_raw, y_train = select_features(df_train, train_cfg)
    X_val_raw, y_val = select_features(df_val, train_cfg)
    X_test_raw, y_test = select_features(df_test, train_cfg)

    # One-hot
    X_train, X_val, X_test = one_hot_encode(X_train_raw, X_val_raw, X_test_raw, train_cfg.categorical_cols)

    # Align target transforms
    y_train_log = _safe_log1p(y_train.to_numpy())
    y_val_log = _safe_log1p(y_val.to_numpy())
    y_test_log = _safe_log1p(y_test.to_numpy())

    # Export artifacts directory (REUSE if run-name is provided)
    if args.run_name:
        out_base = Path(args.outdir) / args.run_name
    else:
        out_base = Path(args.outdir) / f"xgb_defha_{_now_stamp()}"
    _ensure_dir(out_base)

    # Trials log: append-only CSV so we never lose progress
    trials_log_path = out_base / "trials_log.csv"
    if trials_log_path.exists():
        try:
            trials_log = pd.read_csv(trials_log_path)
            prev_trials = int(len(trials_log))
        except Exception:
            trials_log = pd.DataFrame()
            prev_trials = 0
    else:
        trials_log = pd.DataFrame()
        prev_trials = 0

    # Hyperparameter search using validation RMSE in original space (ha)
    best = {
        "trial": None,
        "params": None,
        "val_rmse": float("inf"),
        "metrics": None,
        "fit_info": None,
        "model": None,
    }

    print(f"[INFO] Dataset: {args.data} sep={repr(args.sep)} rows={df.shape[0]} cols={df.shape[1]}")
    print(f"[INFO] Split sizes: train={df_train.shape}, val={df_val.shape}, test={df_test.shape}")
    print(f"[INFO] Feature matrix shapes after encoding: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"[INFO] Trials (this run): {train_cfg.n_trials} seed={train_cfg.random_seed}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Output dir: {out_base}")
    print(f"[INFO] Trials log: {trials_log_path} (existing rows: {prev_trials})")

    # Helper to persist log incrementally
    def _append_trial_row(row: Dict[str, Any]) -> None:
        nonlocal trials_log
        new_row = pd.DataFrame([row])
        if trials_log.empty:
            trials_log = new_row
        else:
            trials_log = pd.concat([trials_log, new_row], ignore_index=True)
        trials_log.to_csv(trials_log_path, index=False)

    for t in range(train_cfg.n_trials):
        params = sample_params(rng, train_cfg, device=args.device)
        model, fit_info = train_one(X_train, y_train_log, X_val, y_val_log, params, train_cfg)

        # Evaluate in original space
        preds_val = _safe_expm1(model.predict(X_val))
        val_rmse = _rmse(y_val.to_numpy(), preds_val)

        # Persist every trial
        _append_trial_row(
            {
                "trial_in_run": t,
                "trial_global": prev_trials + t,
                "val_rmse": float(val_rmse),
                "best_iteration": fit_info.get("best_iteration"),
                "best_score": fit_info.get("best_score"),
                **{f"param__{k}": v for k, v in params.items()},
            }
        )

        if val_rmse < best["val_rmse"]:
            metrics = evaluate_model_on_splits(
                model=model,
                X_train=X_train,
                y_train=y_train.to_numpy(),
                X_val=X_val,
                y_val=y_val.to_numpy(),
                X_test=X_test,
                y_test=y_test.to_numpy(),
            )
            best.update(
                {
                    "trial": prev_trials + t,
                    "params": params,
                    "val_rmse": val_rmse,
                    "metrics": metrics,
                    "fit_info": fit_info,
                    "model": model,
                }
            )

        if (t + 1) % 5 == 0 or t == 0:
            print(
                f"[TUNE] trial={t+1:03d}/{train_cfg.n_trials} "
                f"current_val_rmse={val_rmse:.4f} best_val_rmse={best['val_rmse']:.4f}"
            )

    if best["model"] is None:
        raise RuntimeError("Hyperparameter search did not produce a model (unexpected).")

    # Save model
    model_path = out_base / "model.json"
    best["model"].save_model(str(model_path))

    # Save feature schema
    schema_path = out_base / "feature_columns.json"
    schema_path.write_text(json.dumps({"columns": X_train.columns.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save best params / metrics
    report = {
        "data": {"path": args.data, "sep": args.sep},
        "split": asdict(split_cfg),
        "train_config": {
            **asdict(train_cfg),
            "categorical_cols": list(train_cfg.categorical_cols),
            "drop_cols": list(train_cfg.drop_cols),
            "allowlist_cols": list(train_cfg.allowlist_cols),
        },
        "best": {
            "trial": best["trial"],
            "val_rmse": float(best["val_rmse"]),
            "fit_info": best["fit_info"],
            "params": best["params"],
            "metrics": best["metrics"],
        },
    }
    report_path = out_base / "metrics_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save test predictions for auditing
    preds_test = _safe_expm1(best["model"].predict(X_test))
    pred_df = df_test[[train_cfg.year_col]].copy()
    # UBIGEO is dropped from features; keep for output if present in raw df
    if train_cfg.ubigeo_col in df_test.columns:
        pred_df[train_cfg.ubigeo_col] = df_test[train_cfg.ubigeo_col]
    pred_df["y_true_def_ha"] = y_test.to_numpy()
    pred_df["y_pred_def_ha"] = preds_test
    pred_df.to_csv(out_base / "test_predictions.csv", index=False)

    # Also save a joblib bundle that includes the model + columns (handy for Python inference).
    # IMPORTANT: store configs as plain dicts to avoid pickle issues when loading from a different entrypoint.
    bundle = {
        "model": best["model"],
        "feature_columns": X_train.columns.tolist(),
        "split": asdict(split_cfg),
        "train_config": {
            **asdict(train_cfg),
            "categorical_cols": list(train_cfg.categorical_cols),
            "drop_cols": list(train_cfg.drop_cols),
            "allowlist_cols": list(train_cfg.allowlist_cols),
        },
    }
    joblib.dump(bundle, out_base / "bundle.joblib")

    # Print final summary
    m_test = best["metrics"]["test"]
    print()
    print("[RESULT] Best global trial:", best["trial"])
    print("[RESULT] Val RMSE (ha):", f"{best['val_rmse']:.4f}")
    print("[RESULT] Test metrics:", json.dumps(m_test, indent=2))
    print("[RESULT] Artifacts saved to:", str(out_base))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
