"""
Hindcast / projection evaluation for 2021–2024 using:
- Observed forest loss by district (curated CSV with 2001_ha..2024_ha columns)
- A trained XGBoost model bundle (trained on log1p(Def_ha))
- Simple macro assumptions to extrapolate covariates beyond 2020

Goal
----
Provide a reproducible way to compare:
  observed loss (ha) vs model-predicted deforestation (ha)
for years 2021–2024, aggregated by Department and Province.

Important caveats
-----------------
1) This is NOT a strict forecast unless you provide true covariates for 2021–2024.
   Here, we create plausible covariate projections from the 2020 baseline using macro multipliers
   and/or scenario-like assumptions. Treat results as a "scenario-based hindcast".
2) Some variables in the training dataset may be partially missing. XGBoost can handle NaN.
3) The projections below are intentionally transparent and easy to change.

Inputs
------
- --bundle: models/<run>/bundle.joblib (must include "model" and ideally "feature_columns")
- --data:  deforestation_dataset_PERU_imputed_coca.csv (semicolon-separated by default)
- --loss:  Bosque_y_perdida_de_bosques_por_Distrito_al_2024_curated.csv (district observed loss)
- --out:   output directory

Macro assumptions included
-------------------------
Temperature anomaly (NOAA global anomaly, user-provided):
  2020: 1.02
  2021: 0.87
  2022: 0.90
  2023: 1.19
  2024: 1.28

We translate this into relative warming increments vs baseline year 2020:
  delta_t = anomaly_year - anomaly_2020

Population (macrotrends; user-provided growth rates):
  2022 vs 2021: +0.96%
  2023 vs 2022: +1.11%
  2024 vs 2023: +2.48%
We apply these growth factors to district Población and dens_pob.

Precipitation:
  We apply a scenario-like multiplier by region (SELVA/SIERRA/COSTA) and year.
  Defaults are mild anomalies; edit in code or pass --pp-factor-*.

Infrastructure / mining / agriculture:
  Default: small changes (political uncertainty) - mild growth.
  You can override multipliers.

Coca:
  We support DEPARTMENT-level coca scaling from a markdown report (or any extracted table),
  preserving within-department district shares:
    1) Parse department-level coca totals for years 2020..2024 from coca_report_2021_2024.md
    2) Compute per-department multipliers for each year relative to 2020
    3) Apply multiplier to district Coca_ha for districts in that department

Illegal mining:
  Not explicitly modeled here unless you adjust Minería multipliers or provide better covariates.

Outputs
-------
- CSVs:
  - observed_vs_pred_district_2021_2024.csv
  - observed_vs_pred_by_department_2021_2024.csv
  - observed_vs_pred_by_province_2021_2024.csv
  - metrics_by_year.csv
  - metrics_by_department.csv

- Plots (PNG):
  - by_department_trends_observed_vs_pred.png
  - by_department_trends_observed_vs_pred_log1p.png
  - by_province_topN_trends_observed_vs_pred.png
  - by_province_topN_trends_observed_vs_pred_log1p.png
  - scatter_observed_vs_pred_2021_2024.png
  - residuals_hist.png

Run
---
uv run python src/deforestation/analysis/hindcast_2021_2024.py \
  --bundle models/xgb_timecv_v1/bundle.joblib \
  --data deforestation_dataset_PERU_imputed_coca.csv \
  --sep ';' \
    --loss Bosque_y_perdida_de_bosques_por_Distrito_al_2024_curated.csv \
    --out reports/hindcast_2021_2024 \
    --coca-dept-csv data_external/coca_department_2020_2024.csv

"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Helpers: metrics
# -----------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def safe_expm1(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


# -----------------------------
# Helpers: feature preprocessing (mirrors training convention)
# -----------------------------


def sanitize_feature_names(columns: List[str]) -> List[str]:
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


def one_hot_encode(df_X: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    df_X = df_X.copy()
    for c in categorical_cols:
        if c in df_X.columns:
            df_X[c] = df_X[c].astype("string")
    cats = [c for c in categorical_cols if c in df_X.columns]
    out = pd.get_dummies(df_X, columns=cats, dummy_na=True)
    out.columns = sanitize_feature_names(list(out.columns))
    return out


def align_to_schema(X: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    X = X.copy()
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    extra = [c for c in X.columns if c not in feature_columns]
    if extra:
        X = X.drop(columns=extra)
    return X[feature_columns]


def build_feature_matrix(
    df: pd.DataFrame, train_config: Dict[str, Any], feature_columns: List[str]
) -> pd.DataFrame:
    target_col = str(train_config.get("target_col", "Def_ha"))
    drop_cols = list(train_config.get("drop_cols", []))
    categorical_cols = list(train_config.get("categorical_cols", ["Región", "NOMBDEP"]))

    X = df.drop(columns=[c for c in [target_col] if c in df.columns]).copy()
    drops = [c for c in drop_cols if c in X.columns]
    if drops:
        X = X.drop(columns=drops)

    if "YEAR" in df.columns and "YEAR" not in X.columns:
        X["YEAR"] = df["YEAR"]

    X_enc = one_hot_encode(X, categorical_cols=categorical_cols)
    X_enc = align_to_schema(X_enc, feature_columns=feature_columns)
    return X_enc


# -----------------------------
# Data loading / cleaning
# -----------------------------


def _to_num_cell(v):
    if pd.isna(v):
        return v
    s = str(v).strip().replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def load_loss_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize UBIGEO
    df["UBIGEO"] = df["UBIGEO"].astype(str).str.strip().str.zfill(6)

    # Convert year columns + forest remaining to numeric
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}_ha", str(c))]
    for c in year_cols:
        df[c] = df[c].map(_to_num_cell)
    if "BOSQUE AL 2024_ha" in df.columns:
        df["BOSQUE AL 2024_ha"] = df["BOSQUE AL 2024_ha"].map(_to_num_cell)

    return df


def load_model_dataset(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    if "UBIGEO" in df.columns:
        df["UBIGEO"] = df["UBIGEO"].astype(str).str.strip().str.zfill(6)
    if "YEAR" not in df.columns:
        raise ValueError("Training dataset must contain YEAR.")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype(int)
    return df


# -----------------------------
# Projection logic (macro assumptions)
# -----------------------------


def apply_population_growth(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Applies macro population growth to district-level Población and dens_pob.

    Growth rates are applied relative to the previous year:
      2022: +0.96% vs 2021
      2023: +1.11% vs 2022
      2024: +2.48% vs 2023

    We approximate 2021 vs 2020 as 0% by default (you can adjust if needed).
    """
    growth = {
        2021: 1.00,
        2022: 1.0096,
        2023: 1.0111,
        2024: 1.0248,
    }
    # Convert to factor relative to 2020 baseline by chaining from 2021..year.
    factor = 1.0
    for y in range(2021, year + 1):
        factor *= growth.get(y, 1.0)

    out = df.copy()
    for c in ["Población", "Poblacion", "dens_pob"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") * factor
    return out


def apply_temperature_anomaly(
    df: pd.DataFrame,
    year: int,
    temp_col: str = "tmean",
    baseline_anomaly_2020: float = 1.02,
) -> pd.DataFrame:
    """
    Apply additive temperature anomaly based on NOAA global anomaly values.

    We use:
      delta = anomaly_year - anomaly_2020
    and add delta (°C) to tmean.

    Baseline anomaly for 2020 is 1.02°C (user-provided).
    """
    noaa = {2020: 1.02, 2021: 0.87, 2022: 0.90, 2023: 1.19, 2024: 1.28}
    if year not in noaa:
        return df.copy()

    delta = float(noaa[year] - baseline_anomaly_2020)

    out = df.copy()
    if temp_col in out.columns:
        out[temp_col] = pd.to_numeric(out[temp_col], errors="coerce") + delta
    return out


def apply_precip_multiplier_by_region(
    df: pd.DataFrame,
    year: int,
    pp_col: str = "pp",
    base_factor: float = 1.0,
    selva_mult: float = 1.0,
    sierra_mult: float = 0.6,
    costa_mult: float = 0.0,
) -> pd.DataFrame:
    """
    Mimics the climate_anomaly_mask logic in scenarios.yaml:
    - apply a global base_factor per year
    - then scale by region multipliers

    Default region multipliers follow scenarios.yaml (pp: SELVA=1.0, SIERRA=0.6, COSTA=0.0).

    You can set base_factor per year from outside (we keep it flat by default).
    """
    out = df.copy()
    if pp_col not in out.columns:
        return out
    if "Región" not in out.columns:
        out[pp_col] = pd.to_numeric(out[pp_col], errors="coerce") * float(base_factor)
        return out

    region = out["Región"].astype("string")
    scale = np.ones(len(out), dtype=float) * float(base_factor)
    scale = np.where(region == "SELVA", scale * float(selva_mult), scale)
    scale = np.where(region == "SIERRA", scale * float(sierra_mult), scale)
    scale = np.where(region == "COSTA", scale * float(costa_mult), scale)

    out[pp_col] = pd.to_numeric(out[pp_col], errors="coerce") * scale
    return out


def apply_simple_multipliers(
    df: pd.DataFrame,
    mineria_factor: float,
    infra_factor: float,
    agropec_factor: float,
    coca_factor: float,
) -> pd.DataFrame:
    """
    Apply simple multiplicative adjustments to human pressure variables.

    Notes:
    - These factors are applied uniformly (per district).
    - Coca is handled separately via department-level scaling when a coca report is provided.
      We keep coca_factor here as a fallback (default 1.0 recommended when using dept scaling).
    """
    out = df.copy()

    def mul(col: str, factor: float):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(factor)

    mul("Minería", mineria_factor)
    mul("Infraestructura", infra_factor)
    mul("area_agropec", agropec_factor)
    mul("Coca_ha", coca_factor)

    return out


# -----------------------------
# Main evaluation
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Hindcast 2021–2024 vs observed loss CSV.")
    ap.add_argument("--bundle", required=True, help="Path to trained bundle.joblib.")
    ap.add_argument(
        "--data", required=True, help="Training dataset CSV (for baseline rows)."
    )
    ap.add_argument("--sep", default=";", help="Separator for --data (default ';').")
    ap.add_argument(
        "--loss", required=True, help="Curated observed loss CSV (district-year loss)."
    )
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument(
        "--baseline-year", type=int, default=2020, help="Baseline year for covariates."
    )
    ap.add_argument(
        "--temp-baseline-anomaly",
        type=float,
        default=1.02,
        help="NOAA global temperature anomaly for 2020 (°C). Used for delta = anomaly_year - anomaly_2020.",
    )
    ap.add_argument(
        "--top-provinces", type=int, default=20, help="Top N provinces for plots."
    )
    ap.add_argument("--pp-factor-2021", type=float, default=1.00)
    ap.add_argument("--pp-factor-2022", type=float, default=0.99)
    ap.add_argument("--pp-factor-2023", type=float, default=0.98)
    ap.add_argument("--pp-factor-2024", type=float, default=0.97)
    ap.add_argument(
        "--mineria-factor",
        type=float,
        default=1.02,
        help="Annual mining multiplier (applied vs baseline; used for all 2021–2024 years).",
    )
    ap.add_argument(
        "--infra-factor",
        type=float,
        default=1.01,
        help="Annual infrastructure multiplier.",
    )
    ap.add_argument(
        "--agropec-factor",
        type=float,
        default=1.01,
        help="Annual agriculture area multiplier.",
    )
    ap.add_argument(
        "--coca-factor",
        type=float,
        default=1.00,
        help="Fallback annual coca multiplier. Prefer department-level scaling below.",
    )
    ap.add_argument(
        "--coca-dept-csv",
        default="",
        help=(
            "Optional path to a CSV with department coca totals for 2020–2024. "
            "Expected columns: Departamento,2020,2021,2022,2023,2024. "
            "If provided, department-level multipliers vs 2020 are applied to district Coca_ha."
        ),
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bundle / schema
    bundle = joblib.load(args.bundle)
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns") or []
    if not feature_columns:
        raise ValueError("bundle.joblib missing 'feature_columns'.")
    feature_columns = [str(c) for c in feature_columns]

    train_cfg = bundle.get("train_config")
    if not isinstance(train_cfg, dict):
        raise ValueError("bundle['train_config'] must be a dict.")

    # Load data
    df = load_model_dataset(Path(args.data), sep=args.sep)
    loss = load_loss_csv(Path(args.loss))

    # Baseline covariates from baseline year
    base = df[df["YEAR"] == int(args.baseline_year)].copy()
    if base.empty:
        raise ValueError(
            f"No baseline rows found for YEAR={args.baseline_year} in {args.data}"
        )

    # Ensure we have stable keys
    for c in ["UBIGEO", "Región", "NOMBDEP"]:
        if c not in base.columns:
            raise ValueError(f"Baseline data missing required column: {c}")

    # Build projected covariates for each year 2021–2024 from baseline
    years = [2021, 2022, 2023, 2024]
    preds_rows = []

    pp_factors = {
        2021: float(args.pp_factor_2021),
        2022: float(args.pp_factor_2022),
        2023: float(args.pp_factor_2023),
        2024: float(args.pp_factor_2024),
    }

    def _load_coca_dept_totals_csv(csv_path: Path) -> Dict[str, Dict[int, float]]:
        """
        Load department-level coca totals from a CSV.
        Expected columns: Departamento,2020,2021,2022,2023,2024

        Returns:
          {DEPARTAMENTO_UPPER: {year: total_ha}}
        """
        dfc = pd.read_csv(csv_path)

        required = ["Departamento", "2020", "2021", "2022", "2023", "2024"]
        missing = [c for c in required if c not in dfc.columns]
        if missing:
            raise ValueError(
                f"Coca dept CSV missing required columns: {missing}. "
                f"Expected at least: {required}"
            )

        # Normalize department name
        dfc["Departamento"] = dfc["Departamento"].astype(str).str.strip().str.upper()

        # Coerce numeric (allow thousands separators just in case)
        for y in [2020, 2021, 2022, 2023, 2024]:
            col = str(y)
            dfc[col] = (
                dfc[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("nan", np.nan)
            )
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

        out: Dict[str, Dict[int, float]] = {}
        for _, row in dfc.iterrows():
            dep = str(row["Departamento"]).strip().upper()
            if dep in ("TOTAL", "TOTAL*") or dep == "":
                continue
            out[dep] = {}
            for y in [2020, 2021, 2022, 2023, 2024]:
                v = row.get(str(y))
                if v is not None and pd.notna(v):
                    out[dep][y] = float(v)

        return out

    coca_dept_totals: Dict[str, Dict[int, float]] = {}
    coca_dept_multipliers: Dict[str, Dict[int, float]] = {}
    if args.coca_dept_csv:
        csvp = Path(args.coca_dept_csv)
        if csvp.exists():
            coca_dept_totals = _load_coca_dept_totals_csv(csvp)
            # Convert totals to multipliers relative to 2020
            for dep, m in coca_dept_totals.items():
                base2020 = m.get(2020)
                if base2020 is None or base2020 <= 0:
                    continue
                coca_dept_multipliers[dep] = {}
                for y in years:
                    if y in m and m[y] is not None:
                        coca_dept_multipliers[dep][y] = float(m[y] / base2020)

    for y in years:
        d = base.copy()
        d["YEAR"] = y

        # Apply macro assumptions
        d = apply_population_growth(d, year=y)
        d = apply_temperature_anomaly(
            d, year=y, baseline_anomaly_2020=float(args.temp_baseline_anomaly)
        )
        d = apply_precip_multiplier_by_region(d, year=y, base_factor=pp_factors[y])

        # Small “political uncertainty” changes (uniform multipliers)
        # Interpreted as 1-step vs baseline; if you prefer compounded year-to-year, adjust here.
        d = apply_simple_multipliers(
            d,
            mineria_factor=float(args.mineria_factor),
            infra_factor=float(args.infra_factor),
            agropec_factor=float(args.agropec_factor),
            coca_factor=float(args.coca_factor),
        )

        # Department-level coca scaling (preferred)
        if coca_dept_multipliers and "Coca_ha" in d.columns and "NOMBDEP" in d.columns:
            dep_upper = d["NOMBDEP"].astype(str).str.upper()
            mult = np.ones(len(d), dtype=float)
            for dep, per_year in coca_dept_multipliers.items():
                if y in per_year:
                    mult = np.where(dep_upper == dep, mult * float(per_year[y]), mult)
            d["Coca_ha"] = pd.to_numeric(d["Coca_ha"], errors="coerce") * mult

        # Predict
        X = build_feature_matrix(d, train_cfg, feature_columns)
        pred_log = model.predict(X)
        pred_ha = safe_expm1(pred_log)

        out = pd.DataFrame(
            {
                "UBIGEO": d["UBIGEO"].astype(str),
                "YEAR": y,
                "Región": d["Región"].astype(str),
                "NOMBDEP": d["NOMBDEP"].astype(str),
                "pred_def_ha": pred_ha.astype(float),
            }
        )
        preds_rows.append(out)

    pred_all = pd.concat(preds_rows, ignore_index=True)

    # Observed loss: use loss CSV columns 2021_ha..2024_ha
    obs_cols = [f"{y}_ha" for y in years]
    for c in obs_cols:
        if c not in loss.columns:
            raise ValueError(f"Loss CSV missing column {c}")

    obs_long = loss[
        ["DEPARTAMENTO", "PROVINCIA", "DISTRITO", "UBIGEO"] + obs_cols
    ].copy()
    obs_long = obs_long.melt(
        id_vars=["DEPARTAMENTO", "PROVINCIA", "DISTRITO", "UBIGEO"],
        var_name="YEAR",
        value_name="observed_loss_ha",
    )
    obs_long["YEAR"] = obs_long["YEAR"].str.replace("_ha", "", regex=False).astype(int)
    obs_long["UBIGEO"] = obs_long["UBIGEO"].astype(str).str.strip().str.zfill(6)

    # Join observed and predicted at district-year
    joined = obs_long.merge(pred_all, on=["UBIGEO", "YEAR"], how="left")

    # If some UBIGEOs were not in the model dataset baseline, pred_def_ha will be NaN.
    joined["pred_def_ha"] = pd.to_numeric(joined["pred_def_ha"], errors="coerce")

    joined["residual_ha"] = joined["pred_def_ha"] - joined["observed_loss_ha"]
    joined.to_csv(out_dir / "observed_vs_pred_district_2021_2024.csv", index=False)

    # Aggregate by department/year
    dep = (
        joined.groupby(["DEPARTAMENTO", "YEAR"], dropna=False)[
            ["observed_loss_ha", "pred_def_ha"]
        ]
        .sum()
        .reset_index()
    )
    dep["residual_ha"] = dep["pred_def_ha"] - dep["observed_loss_ha"]
    dep.to_csv(out_dir / "observed_vs_pred_by_department_2021_2024.csv", index=False)

    # Aggregate by province/year
    prov = (
        joined.groupby(["DEPARTAMENTO", "PROVINCIA", "YEAR"], dropna=False)[
            ["observed_loss_ha", "pred_def_ha"]
        ]
        .sum()
        .reset_index()
    )
    prov["residual_ha"] = prov["pred_def_ha"] - prov["observed_loss_ha"]
    prov.to_csv(out_dir / "observed_vs_pred_by_province_2021_2024.csv", index=False)

    # Metrics by year (district-level)
    metrics_rows = []
    for y in years:
        sub = joined[joined["YEAR"] == y].copy()
        sub = sub[
            np.isfinite(sub["observed_loss_ha"]) & np.isfinite(sub["pred_def_ha"])
        ]
        if sub.empty:
            continue
        yt = sub["observed_loss_ha"].to_numpy(dtype=float)
        yp = sub["pred_def_ha"].to_numpy(dtype=float)
        metrics_rows.append(
            {
                "year": y,
                "n": int(len(sub)),
                "mae": mae(yt, yp),
                "rmse": rmse(yt, yp),
                "r2": r2(yt, yp),
                "sum_observed": float(np.sum(yt)),
                "sum_pred": float(np.sum(yp)),
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_by_year.csv", index=False)

    # Metrics by department (2021–2024 combined)
    dep_metrics = []
    for dep_name, g in joined.groupby("DEPARTAMENTO", dropna=False):
        g2 = g[np.isfinite(g["observed_loss_ha"]) & np.isfinite(g["pred_def_ha"])]
        if g2.empty:
            continue
        yt = g2["observed_loss_ha"].to_numpy(dtype=float)
        yp = g2["pred_def_ha"].to_numpy(dtype=float)
        dep_metrics.append(
            {
                "DEPARTAMENTO": dep_name,
                "n": int(len(g2)),
                "mae": mae(yt, yp),
                "rmse": rmse(yt, yp),
                "r2": r2(yt, yp),
                "sum_observed": float(np.sum(yt)),
                "sum_pred": float(np.sum(yp)),
            }
        )
    dep_metrics_df = pd.DataFrame(dep_metrics).sort_values("rmse", ascending=True)
    dep_metrics_df.to_csv(out_dir / "metrics_by_department.csv", index=False)

    # -----------------------------
    # Plots
    # -----------------------------

    # Department trends (observed vs predicted)
    plt.figure(figsize=(12, 6))
    for dep_name, g in dep.groupby("DEPARTAMENTO", dropna=False):
        g = g.sort_values("YEAR")
        plt.plot(g["YEAR"], g["observed_loss_ha"], linewidth=1.2, alpha=0.6)
    # overlay totals (all departments combined)
    dep_tot = (
        dep.groupby("YEAR")[["observed_loss_ha", "pred_def_ha"]].sum().reset_index()
    )
    plt.plot(
        dep_tot["YEAR"],
        dep_tot["observed_loss_ha"],
        marker="o",
        linewidth=3,
        label="Observed total",
        color="black",
    )
    plt.plot(
        dep_tot["YEAR"],
        dep_tot["pred_def_ha"],
        marker="o",
        linewidth=3,
        label="Predicted total",
        color="red",
    )
    plt.title(
        "Observed vs predicted forest loss (department lines faint; totals highlighted)"
    )
    plt.xlabel("Year")
    plt.ylabel("Loss / predicted deforestation (ha)")
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "by_department_trends_observed_vs_pred.png", dpi=220)
    plt.close()

    # Department trends (log1p scale for readability across magnitudes)
    plt.figure(figsize=(12, 6))
    for dep_name, g in dep.groupby("DEPARTAMENTO", dropna=False):
        g = g.sort_values("YEAR")
        plt.plot(
            g["YEAR"],
            np.log1p(g["observed_loss_ha"]),
            linewidth=1.2,
            alpha=0.6,
        )
    plt.plot(
        dep_tot["YEAR"],
        np.log1p(dep_tot["observed_loss_ha"]),
        marker="o",
        linewidth=3,
        label="Observed total (log1p)",
        color="black",
    )
    plt.plot(
        dep_tot["YEAR"],
        np.log1p(dep_tot["pred_def_ha"]),
        marker="o",
        linewidth=3,
        label="Predicted total (log1p)",
        color="red",
    )
    plt.title(
        "Observed vs predicted forest loss (log1p scale; department lines faint; totals highlighted)"
    )
    plt.xlabel("Year")
    plt.ylabel("log1p(ha)")
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "by_department_trends_observed_vs_pred_log1p.png", dpi=220)
    plt.close()

    # Province top-N by observed 2021–2024 total
    prov_tot = (
        prov.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False)["observed_loss_ha"]
        .sum()
        .sort_values(ascending=False)
        .head(int(args.top_provinces))
        .reset_index()
    )
    top_keys = set(
        tuple(r)
        for r in prov_tot[["DEPARTAMENTO", "PROVINCIA"]].to_records(index=False)
    )
    prov2 = prov.copy()
    prov2["_key"] = list(zip(prov2["DEPARTAMENTO"], prov2["PROVINCIA"]))
    prov2 = prov2[prov2["_key"].isin(top_keys)]

    plt.figure(figsize=(13, 7))
    for (dname, pname), g in prov2.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False):
        g = g.sort_values("YEAR")
        plt.plot(
            g["YEAR"],
            g["observed_loss_ha"],
            linewidth=1.2,
            alpha=0.75,
            label=f"Obs {dname}-{pname}",
        )
        plt.plot(
            g["YEAR"],
            g["pred_def_ha"],
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
            label=f"Pred {dname}-{pname}",
        )
    plt.title(
        f"Top {int(args.top_provinces)} provinces: observed (solid) vs predicted (dashed)"
    )
    plt.xlabel("Year")
    plt.ylabel("ha")
    plt.xticks(years)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "by_province_topN_trends_observed_vs_pred.png", dpi=220)
    plt.close()

    # Province trends (log1p scale)
    plt.figure(figsize=(13, 7))
    for (dname, pname), g in prov2.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False):
        g = g.sort_values("YEAR")
        plt.plot(
            g["YEAR"],
            np.log1p(g["observed_loss_ha"]),
            linewidth=1.2,
            alpha=0.75,
            label=f"Obs {dname}-{pname} (log1p)",
        )
        plt.plot(
            g["YEAR"],
            np.log1p(g["pred_def_ha"]),
            linewidth=1.2,
            alpha=0.75,
            linestyle="--",
            label=f"Pred {dname}-{pname} (log1p)",
        )
    plt.title(
        f"Top {int(args.top_provinces)} provinces: observed (solid) vs predicted (dashed), log1p scale"
    )
    plt.xlabel("Year")
    plt.ylabel("log1p(ha)")
    plt.xticks(years)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "by_province_topN_trends_observed_vs_pred_log1p.png", dpi=220)
    plt.close()

    # Scatter observed vs predicted (all district-years)
    scat = joined[
        np.isfinite(joined["observed_loss_ha"]) & np.isfinite(joined["pred_def_ha"])
    ].copy()
    plt.figure(figsize=(7, 7))
    plt.scatter(scat["observed_loss_ha"], scat["pred_def_ha"], s=10, alpha=0.35)
    maxv = float(max(scat["observed_loss_ha"].max(), scat["pred_def_ha"].max()))
    plt.plot([0, maxv], [0, maxv], color="black", linewidth=1)
    plt.title("Observed vs predicted (district-year, 2021–2024)")
    plt.xlabel("Observed loss (ha)")
    plt.ylabel("Predicted deforestation (ha)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_observed_vs_pred_2021_2024.png", dpi=220)
    plt.close()

    # Scatter observed vs predicted (log1p scale; safe with zeros)
    plt.figure(figsize=(7, 7))
    x_log = np.log1p(scat["observed_loss_ha"].to_numpy(dtype=float))
    y_log = np.log1p(scat["pred_def_ha"].to_numpy(dtype=float))
    plt.scatter(x_log, y_log, s=10, alpha=0.35)
    maxv_log = float(max(np.nanmax(x_log), np.nanmax(y_log)))
    plt.plot([0, maxv_log], [0, maxv_log], color="black", linewidth=1)
    plt.title("Observed vs predicted (district-year, 2021–2024) — log1p scale")
    plt.xlabel("log1p(observed loss ha)")
    plt.ylabel("log1p(predicted deforestation ha)")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_observed_vs_pred_2021_2024_log1p.png", dpi=220)
    plt.close()

    # Residual histogram
    plt.figure(figsize=(10, 5))
    plt.hist(scat["residual_ha"].to_numpy(dtype=float), bins=60)
    plt.title("Residuals (pred - observed), district-year 2021–2024")
    plt.xlabel("Residual (ha)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "residuals_hist.png", dpi=220)
    plt.close()

    # Save assumptions used
    assumptions = {
        "baseline_year": int(args.baseline_year),
        "temp_baseline_anomaly_2020": float(args.temp_baseline_anomaly),
        "noaa_anomaly": {2020: 1.02, 2021: 0.87, 2022: 0.90, 2023: 1.19, 2024: 1.28},
        "pp_factors": pp_factors,
        "multipliers": {
            "mineria_factor": float(args.mineria_factor),
            "infra_factor": float(args.infra_factor),
            "agropec_factor": float(args.agropec_factor),
            "coca_factor_fallback": float(args.coca_factor),
            "coca_department_scaling": bool(args.coca_dept_csv),
        },
        "population_growth": {
            "2022_vs_2021": 0.0096,
            "2023_vs_2022": 0.0111,
            "2024_vs_2023": 0.0248,
        },
        "coca_dept_csv_path": str(args.coca_dept_csv) if args.coca_dept_csv else "",
    }
    (out_dir / "assumptions_used.json").write_text(
        json.dumps(assumptions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Wrote hindcast evaluation to: {out_dir}")
    if not metrics_df.empty:
        print("[INFO] Metrics by year:")
        print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
