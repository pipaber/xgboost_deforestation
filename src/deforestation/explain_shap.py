"""
Generate SHAP explainability plots for a trained XGBoost deforestation model.

This script is designed to work with artifacts produced by:
- src/deforestation/train_xgb.py

It loads:
- models/<run>/bundle.joblib
- models/<run>/feature_columns.json
and rebuilds the same feature matrix for a chosen dataset + split, then generates SHAP plots.

Key outputs (written under reports/shap/<run>/):
- shap_summary_beeswarm.png
- shap_summary_bar.png
- shap_dependence_<feature>.png (top-k features)
- shap_waterfall_example.png (one example row)

Requirements (in the same environment as `uv run`):
- shap
- numba
- numpy (numba requires numpy <= 2.3.x)
- matplotlib
- pandas
- joblib
- xgboost

Recommended commands:
  uv run python src/deforestation/explain_shap.py \
    --bundle models/xgb_tune_v1/bundle.joblib \
    --data deforestation_dataset_PERU_imputed_coca.csv \
    --sep ';' \
    --split test \
    --out reports/shap/xgb_tune_v1 \
    --topk 15

Notes:
- This script intentionally mirrors the feature prep in train_xgb.py:
  - drops columns configured in the saved TrainConfig
  - one-hot encodes categoricals with dummy_na=True
  - sanitizes feature names to satisfy XGBoost feature naming constraints
  - aligns columns to the training schema (feature_columns.json or bundle feature list)
- For speed/memory, SHAP is computed on a sample of rows (configurable).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import joblib

# Use a non-interactive backend (safe for headless runs)
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

# ----------------------------
# Utilities mirrored from train_xgb.py (kept minimal + consistent)
# ----------------------------


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """
    Sanitize feature names to satisfy XGBoost constraints.

    XGBoost forbids '[', ']', '<' in feature names, and requires strings.
    """
    sanitized: List[str] = []
    seen: dict[str, int] = {}

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


def apply_sanitized_feature_names(*frames: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    if not frames:
        return tuple()
    new_cols = sanitize_feature_names(list(frames[0].columns))
    out: list[pd.DataFrame] = []
    for f in frames:
        f2 = f.copy()
        f2.columns = new_cols
        out.append(f2)
    return tuple(out)


def load_dataset(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    if "YEAR" not in df.columns:
        raise ValueError(f"Expected YEAR column. Columns={df.columns.tolist()}")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    if df["YEAR"].isna().any():
        bad = df[df["YEAR"].isna()].head(10)
        raise ValueError(
            f"Found non-numeric YEAR values. Example rows:\n{bad.to_string(index=False)}"
        )
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["YEAR"] <= 2016].copy()
    val = df[(df["YEAR"] >= 2017) & (df["YEAR"] <= 2018)].copy()
    test = df[(df["YEAR"] >= 2019) & (df["YEAR"] <= 2020)].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Empty split. train={train.shape}, val={val.shape}, test={test.shape}"
        )
    return train, val, test


def select_features(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: List[str],
    allowlist_cols: List[str],
    categorical_cols: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found.")
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])

    if allowlist_cols:
        keep = set(allowlist_cols) | {"YEAR"} | set(categorical_cols)
        X = X[[c for c in X.columns if c in keep]].copy()

    drops = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=drops)

    if "YEAR" not in X.columns and "YEAR" in df.columns:
        X["YEAR"] = df["YEAR"]
    return X, y


def one_hot_encode(X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")
    cat_present = [c for c in categorical_cols if c in X.columns]
    X_enc = pd.get_dummies(X, columns=cat_present, dummy_na=True)
    (X_enc,) = apply_sanitized_feature_names(X_enc)
    return X_enc


def align_to_training_schema(
    X: pd.DataFrame, feature_columns: List[str]
) -> pd.DataFrame:
    """
    Align X to the exact feature columns used during training:
    - add missing columns with 0
    - drop extra columns
    - enforce ordering
    """
    X = X.copy()
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)
    X = X[feature_columns]
    return X


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# SHAP plotting
# ----------------------------


def save_summary_plots(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    out_dir: Path,
    sample_n: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Computes SHAP values on a sampled subset and saves summary plots.
    Returns (shap_values, X_sample) used for plots.
    """
    rng = np.random.default_rng(seed)
    if len(X) > sample_n:
        idx = rng.choice(len(X), size=sample_n, replace=False)
        Xs = X.iloc[idx].copy()
    else:
        Xs = X.copy()

    shap_values = explainer.shap_values(Xs)

    # Beeswarm
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, Xs, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=200)
    plt.close()

    # Bar
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, Xs, plot_type="bar", show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=200)
    plt.close()

    return shap_values, Xs


def topk_features_by_mean_abs_shap(
    shap_values: np.ndarray, feature_names: List[str], k: int
) -> List[str]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    top_idx = order[: min(k, len(order))]
    return [feature_names[i] for i in top_idx]


def save_dependence_plots(
    shap_values: np.ndarray, Xs: pd.DataFrame, features: List[str], out_dir: Path
) -> None:
    for feat in features:
        # Dependence plot can fail if feature not present (should not happen after alignment)
        try:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feat, shap_values, Xs, show=False, interaction_index="auto"
            )
            plt.tight_layout()
            safe = feat.replace("/", "_")
            plt.savefig(out_dir / f"shap_dependence_{safe}.png", dpi=200)
            plt.close()
        except Exception as e:
            # Keep going; write a small marker file to show failure for this feature
            (out_dir / f"shap_dependence_{feat}_FAILED.txt").write_text(
                str(e), encoding="utf-8"
            )


def save_waterfall_plot(
    explainer: shap.TreeExplainer, X: pd.DataFrame, out_dir: Path, row_index: int
) -> None:
    row_index = int(row_index)
    if row_index < 0 or row_index >= len(X):
        row_index = 0
    x_row = X.iloc[[row_index]].copy()
    shap_exp = explainer(x_row)

    # shap.plots.waterfall expects an Explanation object element
    try:
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_exp[0], show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_waterfall_example.png", dpi=200)
        plt.close()
    except Exception as e:
        (out_dir / "shap_waterfall_example_FAILED.txt").write_text(
            str(e), encoding="utf-8"
        )


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SHAP plots for the trained XGBoost model bundle."
    )
    parser.add_argument(
        "--bundle", required=True, help="Path to bundle.joblib produced by training."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Dataset CSV used for training/eval (same schema).",
    )
    parser.add_argument(
        "--sep", required=True, help="CSV separator for --data, e.g. ';' or '\\t'."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to explain.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for plots, e.g. reports/shap/xgb_tune_v1",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=1500,
        help="Number of rows to sample for SHAP summary plots.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Number of top features to create dependence plots for.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling."
    )
    parser.add_argument(
        "--waterfall-row",
        type=int,
        default=0,
        help="Row index (within chosen split) for waterfall plot.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    bundle = joblib.load(args.bundle)
    model = bundle["model"]

    # train_config is stored as a plain dict (for pickle portability across entrypoints)
    train_cfg = bundle.get("train_config")
    if not isinstance(train_cfg, dict):
        raise ValueError(
            "Expected bundle['train_config'] to be a dict. "
            "Re-run training to regenerate bundle.joblib with dict-based config."
        )

    target_col = train_cfg.get("target_col", "Def_ha")
    categorical_cols = list(
        train_cfg.get("categorical_cols", ["Región", "NOMBDEP", "Cluster"])
    )
    drop_cols = list(train_cfg.get("drop_cols", []))
    allowlist_cols = list(train_cfg.get("allowlist_cols", []))

    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        raise ValueError("bundle.joblib does not contain 'feature_columns'.")

    df = load_dataset(args.data, sep=args.sep)

    # Build split
    if args.split == "all":
        df_use = df.copy()
    else:
        df_train, df_val, df_test = time_split(df)
        if args.split == "train":
            df_use = df_train
        elif args.split == "val":
            df_use = df_val
        else:
            df_use = df_test

    # Build features same as training
    X_raw, y = select_features(
        df_use,
        target_col=target_col,
        drop_cols=drop_cols,
        allowlist_cols=allowlist_cols,
        categorical_cols=categorical_cols,
    )
    X_enc = one_hot_encode(X_raw, categorical_cols=categorical_cols)
    X_enc = align_to_training_schema(X_enc, feature_columns=feature_columns)

    # Write a small metadata file for reproducibility
    meta = {
        "bundle": str(args.bundle),
        "data": {"path": args.data, "sep": args.sep},
        "split": args.split,
        "rows_explained": int(len(X_enc)),
        "sample_n": int(args.sample_n),
        "topk": int(args.topk),
        "seed": int(args.seed),
        "train_config": {
            "target_col": target_col,
            "categorical_cols": categorical_cols,
            "drop_cols": drop_cols,
            "allowlist_cols": allowlist_cols,
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "shap": getattr(shap, "__version__", "unknown"),
        },
    }
    (out_dir / "shap_run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Build explainer
    explainer = shap.TreeExplainer(model)

    shap_values, Xs = save_summary_plots(
        explainer, X_enc, out_dir, sample_n=args.sample_n, seed=args.seed
    )

    top_feats = topk_features_by_mean_abs_shap(
        shap_values, list(Xs.columns), k=args.topk
    )
    (out_dir / "top_features.txt").write_text("\n".join(top_feats), encoding="utf-8")

    save_dependence_plots(shap_values, Xs, top_feats, out_dir)
    save_waterfall_plot(explainer, X_enc, out_dir, row_index=args.waterfall_row)

    # Also export a compact CSV of mean(|SHAP|) for audit
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    imp = (
        pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(out_dir / "shap_mean_abs.csv", index=False)

    print(f"[OK] Wrote SHAP outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
