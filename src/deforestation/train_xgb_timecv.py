"""
Rolling-window time-series cross-validation tuning script for XGBoost.

Purpose
-------
Tune XGBoost hyperparameters using a rolling/expanding time CV scheme to avoid
over-optimizing to a single validation window (e.g., only 2017–2018).

This script:
- Loads the dataset (supports ';' or '\t' separators)
- Builds rolling CV folds across years
- For each trial (random search), trains with early stopping per fold
- Scores each fold in ORIGINAL space (ha) after training on log1p(Def_ha)
- Selects the best params by average CV RMSE (optionally also MAE)
- Retrains a final model on the full training window and evaluates on a final holdout test window
- Saves artifacts:
  - model.json
  - bundle.joblib (portable dict-based config)
  - feature_columns.json
  - metrics_report.json (CV + final test)
  - trials_log.csv (append-only)

GPU mode (optional)
-------------------
You can run training on GPU if you have a compatible NVIDIA GPU and a CUDA-enabled XGBoost build.

Enable GPU:
  --device cuda

For XGBoost 3.x, prefer the decoupled configuration:
- tree_method='hist'
- device='cuda'

Default is CPU:
  --device cpu  (tree_method='hist', device='cpu')

Recommended usage
-----------------
1) Choose a final test window you will NOT use for tuning (default 2019–2020).
2) Tune on earlier years via rolling CV (default folds validate on 2011, 2012, 2015, 2017).
3) Train final model on <= 2018 and test on 2019–2020 (default).

Example (CPU):
  uv run python src/deforestation/train_xgb_timecv.py \
    --data deforestation_dataset_PERU_imputed_coca.csv \
    --sep ';' \
    --trials 200 \
    --seed 42 \
    --outdir models \
    --run-name xgb_timecv_v1 \
    --device cpu

Example (GPU):
  uv run python src/deforestation/train_xgb_timecv.py \
    --data deforestation_dataset_PERU_imputed_coca.csv \
    --sep ';' \
    --trials 200 \
    --seed 42 \
    --outdir models \
    --run-name xgb_timecv_v1 \
    --device cuda

Notes
-----
- XGBoost handles numeric NaN natively. We do NOT impute numeric features here.
- Categorical columns are one-hot encoded with dummy_na=True.
- We drop extreme-missingness blocks (~90–95%) and ID/name fields, matching prior approach.
- Feature names are sanitized for XGBoost constraints (no '[', ']', '<').
- Early stopping: configured in the constructor (XGBoost sklearn API v3.x).

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "Def_ha"
    year_col: str = "YEAR"
    ubigeo_col: str = "UBIGEO"

    categorical_cols: Tuple[str, ...] = ("Región", "NOMBDEP", "Cluster")

    # Conservative drop list (IDs + extreme sparse blocks)
    drop_cols: Tuple[str, ...] = (
        "UBIGEO",
        "NOMBPROB",
        "NOMBDIST",
        "Dist_ríos",
        "Dist_vías",
        "Dist_comunid",
        "Dist_conc_mad",
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
        "tot_salieron",
        "tot_ingresaron",
    )

    # Optional: keep only specific columns (rarely used; empty means "all except drops")
    allowlist_cols: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CVConfig:
    """
    Rolling/expanding time CV definition.

    For each fold:
      - train years <= train_end_year
      - validate years in [val_start_year, val_end_year]
    """

    # Default folds (chosen because the dataset contains key socio vars for some years,
    # but we only rely on core features anyway). You can edit via CLI.
    folds: Tuple[Tuple[int, int, int], ...] = (
        # (train_end_year, val_start_year, val_end_year)
        (2010, 2011, 2011),
        (2011, 2012, 2012),
        (2014, 2015, 2015),
        (2016, 2017, 2017),
        (2016, 2018, 2018),
    )

    # Final train window for the "production" model (after tuning) and final test window
    final_train_end_year: int = 2018
    test_start_year: int = 2019
    test_end_year: int = 2020


@dataclass(frozen=True)
class TuneConfig:
    random_seed: int = 42
    n_trials: int = 200

    # Early stopping
    early_stopping_rounds: int = 200
    max_estimators: int = 5000

    # Objective
    objective: str = "reg:squarederror"

    # CV selection metric
    selection_metric: str = "rmse"  # rmse or mae


# -----------------------------
# Utilities
# -----------------------------


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_log1p(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.any(y < 0):
        raise ValueError("Target contains negative values; cannot apply log1p safely.")
    return np.log1p(y)


def _safe_expm1(y_log: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y_log, dtype=float))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "n": float(len(y_true)),
    }


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """
    XGBoost feature name constraints:
      - must be strings
      - must not contain '[', ']' or '<'

    We also normalize whitespace.
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


# -----------------------------
# Data / features
# -----------------------------


def load_dataset(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    if "YEAR" not in df.columns:
        raise ValueError(f"Expected YEAR column. Columns={df.columns.tolist()}")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    if df["YEAR"].isna().any():
        bad = df[df["YEAR"].isna()].head(10)
        raise ValueError(f"Found non-numeric YEAR values. Example rows:\n{bad.to_string(index=False)}")
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def select_features(df: pd.DataFrame, feat_cfg: FeatureConfig) -> Tuple[pd.DataFrame, pd.Series]:
    if feat_cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{feat_cfg.target_col}' not found.")
    y = df[feat_cfg.target_col].astype(float)
    X = df.drop(columns=[feat_cfg.target_col])

    if feat_cfg.allowlist_cols:
        keep = set(feat_cfg.allowlist_cols) | {feat_cfg.year_col} | set(feat_cfg.categorical_cols)
        X = X[[c for c in X.columns if c in keep]].copy()

    drops = [c for c in feat_cfg.drop_cols if c in X.columns]
    X = X.drop(columns=drops)

    # Ensure YEAR present
    if feat_cfg.year_col not in X.columns and feat_cfg.year_col in df.columns:
        X[feat_cfg.year_col] = df[feat_cfg.year_col]

    return X, y


def one_hot_encode_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode train/val together to ensure aligned columns, then sanitize feature names.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    for c in categorical_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("string")
        if c in X_val.columns:
            X_val[c] = X_val[c].astype("string")

    all_X = pd.concat([X_train.assign(_split="train"), X_val.assign(_split="val")], ignore_index=True)
    cats = [c for c in categorical_cols if c in all_X.columns]
    all_enc = pd.get_dummies(all_X, columns=cats, dummy_na=True)

    tr = all_enc[all_enc["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    va = all_enc[all_enc["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)

    new_cols = sanitize_feature_names(list(tr.columns))
    tr.columns = new_cols
    va.columns = new_cols

    return tr, va


# -----------------------------
# Hyperparameter sampling / training
# -----------------------------


def sample_params(rng: random.Random, tune_cfg: TuneConfig, device: str) -> Dict[str, Any]:
    def log_uniform(a: float, b: float) -> float:
        return float(math.exp(rng.uniform(math.log(a), math.log(b))))

    params = {
        "n_estimators": tune_cfg.max_estimators,
        "learning_rate": log_uniform(0.01, 0.2),
        "max_depth": rng.randint(3, 10),
        "min_child_weight": log_uniform(0.5, 30.0),
        "subsample": rng.uniform(0.6, 1.0),
        "colsample_bytree": rng.uniform(0.6, 1.0),
        "reg_alpha": log_uniform(1e-8, 10.0),
        "reg_lambda": log_uniform(0.5, 50.0),
        "gamma": log_uniform(1e-8, 10.0),
        "objective": tune_cfg.objective,
        "random_state": tune_cfg.random_seed,
        "n_jobs": max(1, os.cpu_count() or 1),
        "early_stopping_rounds": tune_cfg.early_stopping_rounds,
        "eval_metric": "rmse",
    }

    # XGBoost 3.x: decouple algorithm from device selection.
    # Use hist everywhere, and select hardware via `device`.
    params["tree_method"] = "hist"
    if device == "cuda":
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"

    return params


def fit_one_fold(
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=False)
    return model


def score_fold_original_space(model: XGBRegressor, X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, float]:
    pred_val = _safe_expm1(model.predict(X_val))
    return {
        "rmse": _rmse(y_val, pred_val),
        "mae": _mae(y_val, pred_val),
        "r2": float(r2_score(y_val, pred_val)),
        "n": float(len(y_val)),
    }


# -----------------------------
# CV logic
# -----------------------------


def iter_folds(df: pd.DataFrame, cv_cfg: CVConfig, year_col: str) -> List[Dict[str, Any]]:
    folds_out: List[Dict[str, Any]] = []
    for (train_end, val_start, val_end) in cv_cfg.folds:
        train_df = df[df[year_col] <= train_end].copy()
        val_df = df[(df[year_col] >= val_start) & (df[year_col] <= val_end)].copy()
        if train_df.empty or val_df.empty:
            # Skip empty folds, but keep note
            folds_out.append(
                {
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "skipped": True,
                }
            )
            continue
        folds_out.append(
            {
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "train_df": train_df,
                "val_df": val_df,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "skipped": False,
            }
        )
    return folds_out


def aggregate_cv_scores(per_fold: List[Dict[str, float]]) -> Dict[str, float]:
    # Average only across non-empty folds
    if not per_fold:
        return {"rmse_mean": float("inf"), "mae_mean": float("inf"), "r2_mean": float("nan"), "folds": 0.0}

    rmse_vals = [f["rmse"] for f in per_fold]
    mae_vals = [f["mae"] for f in per_fold]
    r2_vals = [f["r2"] for f in per_fold if not math.isnan(f["r2"])]

    return {
        "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else float("inf"),
        "rmse_std": float(np.std(rmse_vals)) if rmse_vals else float("nan"),
        "mae_mean": float(np.mean(mae_vals)) if mae_vals else float("inf"),
        "mae_std": float(np.std(mae_vals)) if mae_vals else float("nan"),
        "r2_mean": float(np.mean(r2_vals)) if r2_vals else float("nan"),
        "folds": float(len(per_fold)),
    }


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune XGBoost using rolling time-series CV, then train final model and evaluate on holdout test.")
    ap.add_argument("--data", required=True, help="Path to dataset CSV.")
    ap.add_argument("--sep", required=True, help="Separator, e.g. ';' or '\\t'.")
    ap.add_argument("--outdir", default="models", help="Base output directory.")
    ap.add_argument("--run-name", default=None, help="Optional run directory name. If exists, append to trials_log.csv.")
    ap.add_argument("--trials", type=int, default=200, help="Number of random-search trials.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--selection-metric", choices=["rmse", "mae"], default="rmse", help="Metric to minimize for selecting best params.")
    ap.add_argument("--final-train-end", type=int, default=2018, help="Final train end year for the production model.")
    ap.add_argument("--test-start", type=int, default=2019, help="Test start year.")
    ap.add_argument("--test-end", type=int, default=2020, help="Test end year.")
    ap.add_argument("--sample-folds", default=None, help="Optional override folds: 'trainEnd:valStart-valEnd,...' e.g. '2010:2011-2011,2014:2015-2015'")
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device. Use 'cuda' for GPU training (requires compatible NVIDIA GPU + CUDA-enabled XGBoost).",
    )
    args = ap.parse_args()

    feat_cfg = FeatureConfig()
    tune_cfg = TuneConfig(random_seed=args.seed, n_trials=args.trials, selection_metric=args.selection_metric)

    # CV config
    cv_cfg = CVConfig(final_train_end_year=args.final_train_end, test_start_year=args.test_start, test_end_year=args.test_end)
    if args.sample_folds:
        folds: List[Tuple[int, int, int]] = []
        for part in args.sample_folds.split(","):
            part = part.strip()
            if not part:
                continue
            train_end_s, val_range = part.split(":")
            val_start_s, val_end_s = val_range.split("-")
            folds.append((int(train_end_s), int(val_start_s), int(val_end_s)))
        cv_cfg = CVConfig(folds=tuple(folds), final_train_end_year=args.final_train_end, test_start_year=args.test_start, test_end_year=args.test_end)

    rng = random.Random(tune_cfg.random_seed)
    np.random.seed(tune_cfg.random_seed)

    df = load_dataset(args.data, sep=args.sep)

    # Output dir
    if args.run_name:
        out_base = Path(args.outdir) / args.run_name
    else:
        out_base = Path(args.outdir) / f"xgb_timecv_{_now_stamp()}"
    _ensure_dir(out_base)

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

    def append_trial_row(row: Dict[str, Any]) -> None:
        nonlocal trials_log
        new_row = pd.DataFrame([row])
        trials_log = pd.concat([trials_log, new_row], ignore_index=True) if not trials_log.empty else new_row
        trials_log.to_csv(trials_log_path, index=False)

    # Prepare folds
    folds = iter_folds(df, cv_cfg, year_col=feat_cfg.year_col)
    usable_folds = [f for f in folds if not f.get("skipped", False)]
    if not usable_folds:
        raise ValueError("No usable folds. Check folds definition and YEAR range.")

    print(f"[INFO] Data: {args.data} sep={repr(args.sep)} shape={df.shape}")
    print(f"[INFO] Output: {out_base}")
    print(f"[INFO] Trials: {tune_cfg.n_trials} seed={tune_cfg.random_seed} selection_metric={tune_cfg.selection_metric}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Folds: {len(usable_folds)} usable / {len(folds)} total")
    for f in folds:
        if f.get("skipped"):
            print(f"  - SKIP fold train<= {f['train_end']} val={f['val_start']}-{f['val_end']} (train_rows={f['train_rows']} val_rows={f['val_rows']})")
        else:
            print(f"  - fold train<= {f['train_end']} val={f['val_start']}-{f['val_end']} (train_rows={f['train_rows']} val_rows={f['val_rows']})")

    best = {
        "trial": None,
        "params": None,
        "cv_score": float("inf"),
        "cv_agg": None,
        "per_fold": None,
        "model": None,  # final trained model (after selection)
    }

    # Tuning loop
    for t in range(tune_cfg.n_trials):
        params = sample_params(rng, tune_cfg, device=args.device)

        fold_scores: List[Dict[str, float]] = []
        # For each fold: build features & fit
        for f in usable_folds:
            train_df = f["train_df"]
            val_df = f["val_df"]

            X_tr_raw, y_tr = select_features(train_df, feat_cfg)
            X_va_raw, y_va = select_features(val_df, feat_cfg)

            X_tr, X_va = one_hot_encode_splits(X_tr_raw, X_va_raw, feat_cfg.categorical_cols)

            y_tr_log = _safe_log1p(y_tr.to_numpy())
            y_va_log = _safe_log1p(y_va.to_numpy())

            model = fit_one_fold(params, X_tr, y_tr_log, X_va, y_va_log)
            fold_scores.append(score_fold_original_space(model, X_va, y_va.to_numpy()))

        agg = aggregate_cv_scores(fold_scores)
        metric_to_min = agg["rmse_mean"] if tune_cfg.selection_metric == "rmse" else agg["mae_mean"]

        # Log every trial
        append_trial_row(
            {
                "trial_in_run": t,
                "trial_global": prev_trials + t,
                "cv_rmse_mean": agg["rmse_mean"],
                "cv_rmse_std": agg["rmse_std"],
                "cv_mae_mean": agg["mae_mean"],
                "cv_mae_std": agg["mae_std"],
                "cv_r2_mean": agg["r2_mean"],
                "cv_folds": agg["folds"],
                "selection_metric": tune_cfg.selection_metric,
                "selection_score": float(metric_to_min),
                **{f"param__{k}": v for k, v in params.items()},
            }
        )

        if metric_to_min < best["cv_score"]:
            best.update(
                {
                    "trial": prev_trials + t,
                    "params": params,
                    "cv_score": float(metric_to_min),
                    "cv_agg": agg,
                    "per_fold": fold_scores,
                }
            )

        if (t + 1) % 10 == 0 or t == 0:
            print(f"[TUNE] {t+1:03d}/{tune_cfg.n_trials} current={metric_to_min:.4f} best={best['cv_score']:.4f}")

    if best["params"] is None:
        raise RuntimeError("No best params selected (unexpected).")

    # Train final model on <= final_train_end_year and evaluate on test window
    final_train_df = df[df[feat_cfg.year_col] <= cv_cfg.final_train_end_year].copy()
    test_df = df[(df[feat_cfg.year_col] >= cv_cfg.test_start_year) & (df[feat_cfg.year_col] <= cv_cfg.test_end_year)].copy()
    if final_train_df.empty or test_df.empty:
        raise ValueError(f"Empty final train/test: train={final_train_df.shape} test={test_df.shape}")

    X_tr_raw, y_tr = select_features(final_train_df, feat_cfg)
    X_te_raw, y_te = select_features(test_df, feat_cfg)

    # One-hot together for alignment
    X_tr, X_te = one_hot_encode_splits(X_tr_raw, X_te_raw, feat_cfg.categorical_cols)

    y_tr_log = _safe_log1p(y_tr.to_numpy())
    y_te_log = _safe_log1p(y_te.to_numpy())

    # Fit with early stopping on a small tail split of training? For simplicity, we use test_df as eval_set ONLY for early stopping? NO (leakage).
    # Instead: split off the last 2 years of training as internal early stop (2017–2018) if available.
    internal_val_df = final_train_df[(final_train_df[feat_cfg.year_col] >= 2017) & (final_train_df[feat_cfg.year_col] <= 2018)].copy()
    if internal_val_df.empty:
        # fallback: 10% random slice of final_train_df (still time-safe-ish but not ideal)
        internal_val_df = final_train_df.sample(frac=0.1, random_state=tune_cfg.random_seed)

    internal_train_df = final_train_df.drop(index=internal_val_df.index)

    X_itr_raw, y_itr = select_features(internal_train_df, feat_cfg)
    X_iva_raw, y_iva = select_features(internal_val_df, feat_cfg)
    X_itr, X_iva = one_hot_encode_splits(X_itr_raw, X_iva_raw, feat_cfg.categorical_cols)
    y_itr_log = _safe_log1p(y_itr.to_numpy())
    y_iva_log = _safe_log1p(y_iva.to_numpy())

    final_params = dict(best["params"])
    # Ensure early stopping config present
    final_params["early_stopping_rounds"] = tune_cfg.early_stopping_rounds
    final_params["eval_metric"] = "rmse"

    final_model = XGBRegressor(**final_params)
    final_model.fit(X_itr, y_itr_log, eval_set=[(X_iva, y_iva_log)], verbose=False)

    # Align test features to training columns (they should match because one_hot_encode_splits was done separately)
    # To be safe, ensure exact column alignment
    for col in X_itr.columns:
        if col not in X_te.columns:
            X_te[col] = 0.0
    extra_cols = [c for c in X_te.columns if c not in X_itr.columns]
    if extra_cols:
        X_te = X_te.drop(columns=extra_cols)
    X_te = X_te[X_itr.columns]

    y_pred_test = _safe_expm1(final_model.predict(X_te))
    test_metrics = regression_metrics(y_te.to_numpy(), y_pred_test)

    # Save artifacts
    model_path = out_base / "model.json"
    final_model.save_model(str(model_path))

    feature_columns = list(X_itr.columns)
    (out_base / "feature_columns.json").write_text(json.dumps({"columns": feature_columns}, ensure_ascii=False, indent=2), encoding="utf-8")

    bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "split": {
            "cv_folds": [dict(train_end=f[0], val_start=f[1], val_end=f[2]) for f in cv_cfg.folds],
            "final_train_end_year": cv_cfg.final_train_end_year,
            "test_start_year": cv_cfg.test_start_year,
            "test_end_year": cv_cfg.test_end_year,
        },
        "train_config": {
            **asdict(feat_cfg),
            "categorical_cols": list(feat_cfg.categorical_cols),
            "drop_cols": list(feat_cfg.drop_cols),
            "allowlist_cols": list(feat_cfg.allowlist_cols),
        },
        "tune_config": asdict(tune_cfg),
    }
    joblib.dump(bundle, out_base / "bundle.joblib")

    # Metrics report
    report = {
        "data": {"path": args.data, "sep": args.sep},
        "cv": {
            "folds": [dict(train_end=f[0], val_start=f[1], val_end=f[2]) for f in cv_cfg.folds],
            "usable_folds": int(len(usable_folds)),
"selection_metric": tune_cfg.selection_metric,
            "best_trial": best["trial"],
            "best_cv_score": best["cv_score"],
            "best_cv_aggregate": best["cv_agg"],
            "best_cv_per_fold": best["per_fold"],
            "best_params": best["params"],
        },
        "final_train": {"end_year": cv_cfg.final_train_end_year, "rows": int(len(final_train_df))},
        "test": {"years": f"{cv_cfg.test_start_year}-{cv_cfg.test_end_year}", "rows": int(len(test_df)), "metrics": test_metrics},
    }
    (out_base / "metrics_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Test predictions for audit
    pred_out = test_df[[feat_cfg.year_col]].copy()
    if feat_cfg.ubigeo_col in test_df.columns:
        pred_out[feat_cfg.ubigeo_col] = test_df[feat_cfg.ubigeo_col]
    pred_out["y_true_def_ha"] = y_te.to_numpy()
    pred_out["y_pred_def_ha"] = y_pred_test
    pred_out.to_csv(out_base / "test_predictions.csv", index=False)

    print()
    print("[RESULT] Best CV trial:", best["trial"])
    print("[RESULT] Best CV aggregate:", json.dumps(best["cv_agg"], indent=2))
    print("[RESULT] Final test metrics:", json.dumps(test_metrics, indent=2))
    print("[RESULT] Artifacts saved to:", str(out_base))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
