"""
Spatial holdout evaluation for the Peru deforestation XGBoost model.

Goal
----
Hold out 20% of UBIGEO entirely (all years) as a spatial test set, train on the
remaining UBIGEO with a time-based split and evaluate on future years.

This answers: "Does the model generalize to unseen districts (spatial correlation)?"

Split definition
----------------
- Spatial split:
  - Train UBIGEO set: 80%
  - Holdout UBIGEO set: 20%
- Time split inside Train UBIGEO set:
  - Train years: <= 2016
  - Val years:   2017–2018  (early stopping)
- Evaluation (spatial holdout + future):
  - Test years:  2019–2020 on holdout UBIGEO set

Run
---
uv run python src/deforestation/eval_spatial_holdout.py --data deforestation_dataset_PERU_imputed_coca.csv --sep ';' --holdout-frac 0.2 --seed 42 --out reports/spatial_holdout_ubigeo20_seed42.json

Notes
-----
- XGBoost handles numeric NaN natively, so we do not impute numeric features.
- We mimic the same feature policy as train_xgb.py:
  - drop ID/name columns and the extreme-missingness blocks (~90–95% missing)
  - one-hot encode Región / NOMBDEP / Cluster (dummy_na=True)
  - sanitize feature names for XGBoost constraints (no '[', ']', '<')
- This script uses a reasonable fixed set of XGBoost params. If you want to
  evaluate using your tuned params, update the `MODEL_PARAMS` dict accordingly
  (or extend the script to load models/xgb_tune_v1/metrics_report.json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DROP_COLS_DEFAULT = [
    # IDs / names (leakage risk)
    "UBIGEO",
    "NOMBPROB",
    "NOMBDIST",
    # very sparse distances (~95% missing)
    "Dist_ríos",
    "Dist_vías",
    "Dist_comunid",
    "Dist_conc_mad",
    # very sparse employment block (~90% missing)
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
]

CATEGORICAL_COLS_DEFAULT = ["Región", "NOMBDEP", "Cluster"]

MODEL_PARAMS = {
    # Use early stopping to determine effective tree count
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "eval_metric": "rmse",
    "early_stopping_rounds": 200,
}


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "n": float(len(y_true)),
    }


def log1p_safe(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if (y < 0).any():
        raise ValueError("Negative target encountered; cannot log1p.")
    return np.log1p(y)


def expm1(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


def sanitize_feature_names(cols: List[str]) -> List[str]:
    """
    XGBoost requires feature names to be strings and forbids '[', ']' and '<'.
    """
    sanitized: List[str] = []
    seen: Dict[str, int] = {}
    for c in cols:
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


def load_df(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)

    if "YEAR" not in df.columns or "UBIGEO" not in df.columns or "Def_ha" not in df.columns:
        raise ValueError(
            "Dataset must contain YEAR, UBIGEO, Def_ha. "
            f"Columns={df.columns.tolist()}"
        )

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype(int)
    df["UBIGEO"] = df["UBIGEO"].astype(str).str.zfill(6)
    df["Def_ha"] = pd.to_numeric(df["Def_ha"], errors="coerce")

    # Drop rows where target missing (can't train/evaluate)
    df = df.dropna(subset=["Def_ha"]).copy()

    return df


def make_holdout_ubigeos(df: pd.DataFrame, holdout_frac: float, seed: int) -> Tuple[List[str], List[str]]:
    ubigeos = sorted(df["UBIGEO"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(ubigeos)
    k = max(1, int(round(len(ubigeos) * holdout_frac)))
    holdout = sorted(ubigeos[:k])
    train = sorted(ubigeos[k:])
    return train, holdout


def prep_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["Def_ha"].astype(float)
    X = df.drop(columns=["Def_ha"])

    drops = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=drops)

    # Ensure YEAR present
    if "YEAR" not in X.columns:
        X["YEAR"] = df["YEAR"]

    # Cast categoricals
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")

    return X, y


def one_hot_align(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_X = pd.concat(
        [
            train_X.assign(_split="train"),
            val_X.assign(_split="val"),
            test_X.assign(_split="test"),
        ],
        ignore_index=True,
    )

    cats = [c for c in categorical_cols if c in all_X.columns]
    all_enc = pd.get_dummies(all_X, columns=cats, dummy_na=True)

    tr = all_enc[all_enc["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    va = all_enc[all_enc["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    te = all_enc[all_enc["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)

    # Sanitize feature names for XGBoost
    new_cols = sanitize_feature_names(list(tr.columns))
    tr.columns = new_cols
    va.columns = new_cols
    te.columns = new_cols

    return tr, va, te


def main() -> int:
    ap = argparse.ArgumentParser(description="Spatial holdout evaluation (20% UBIGEO) for XGBoost deforestation model.")
    ap.add_argument("--data", required=True, help="Dataset CSV path.")
    ap.add_argument("--sep", required=True, help="CSV separator, e.g. ';' or '\\t'.")
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="Fraction of UBIGEO to hold out (default 0.2).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for selecting holdout UBIGEO.")
    ap.add_argument("--out", required=True, help="Output JSON report path.")
    args = ap.parse_args()

    df = load_df(args.data, args.sep)
    train_ubs, holdout_ubs = make_holdout_ubigeos(df, args.holdout_frac, args.seed)

    # Train/val from TRAIN UBIGEO only
    df_train_ub = df[df["UBIGEO"].isin(train_ubs)].copy()
    train_df = df_train_ub[df_train_ub["YEAR"] <= 2016].copy()
    val_df = df_train_ub[(df_train_ub["YEAR"] >= 2017) & (df_train_ub["YEAR"] <= 2018)].copy()

    # Test = HOLDOUT UBIGEO, future years
    test_df = df[(df["UBIGEO"].isin(holdout_ubs)) & (df["YEAR"] >= 2019) & (df["YEAR"] <= 2020)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(f"Empty split: train={train_df.shape} val={val_df.shape} test={test_df.shape}")

    X_tr_raw, y_tr = prep_features(train_df, CATEGORICAL_COLS_DEFAULT, DROP_COLS_DEFAULT)
    X_va_raw, y_va = prep_features(val_df, CATEGORICAL_COLS_DEFAULT, DROP_COLS_DEFAULT)
    X_te_raw, y_te = prep_features(test_df, CATEGORICAL_COLS_DEFAULT, DROP_COLS_DEFAULT)

    X_tr, X_va, X_te = one_hot_align(X_tr_raw, X_va_raw, X_te_raw, CATEGORICAL_COLS_DEFAULT)

    params = dict(MODEL_PARAMS)
    params["n_jobs"] = max(1, os.cpu_count() or 1)
    params["random_state"] = args.seed

    model = XGBRegressor(**params)
    model.fit(X_tr, log1p_safe(y_tr.to_numpy()), eval_set=[(X_va, log1p_safe(y_va.to_numpy()))], verbose=False)

    pred_test = expm1(model.predict(X_te))

    report = {
        "data": {"path": args.data, "sep": args.sep},
        "holdout": {
            "holdout_frac": args.holdout_frac,
            "seed": args.seed,
            "train_ubigeo_n": len(train_ubs),
            "holdout_ubigeo_n": len(holdout_ubs),
        },
        "splits": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_years": "2001-2016",
            "val_years": "2017-2018",
            "test_years": "2019-2020 (holdout UBIGEO)",
        },
        "metrics_test_holdout_2019_2020": metrics(y_te.to_numpy(), pred_test),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out_path}")
    print(json.dumps(report["metrics_test_holdout_2019_2020"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
