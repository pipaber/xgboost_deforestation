"""
Scenario runner for Peru deforestation model.

What this does
--------------
- Loads a trained XGBoost model bundle (joblib) produced by train_xgb.py or train_xgb_timecv.py
- Loads the dataset and selects baseline rows for YEAR=2020 (configurable)
- Applies scenario transformations (good/mild/bad) defined in scenarios/scenarios.yaml
  with optional masks defined in the scenario configuration
- Scores baseline and scenarios with the trained model
- Writes:
  - per-district predictions + deltas (CSV)
  - aggregated summaries by Región and NOMBDEP (CSV)
  - plots (PNG): delta distributions, region bars, top-20 delta bar

Run (Windows / Linux)
---------------------
uv run python src/deforestation/run_scenarios.py \
  --bundle models/xgb_timecv_v1/bundle.joblib \
  --data deforestation_dataset_PERU_imputed_coca.csv \
  --sep ';' \
  --scenarios scenarios/scenarios.yaml \
  --out reports/scenarios/xgb_timecv_v1_gpu_baseline2020

Notes
-----
- Uses the same one-hot encoding approach as training:
  - get_dummies(dummy_na=True) for Región / NOMBDEP / Cluster
  - aligns columns to bundle's feature_columns
- XGBoost handles numeric NaN natively. We do not impute.
- Applies constraints after transforms: non-negative on specified columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib

# plotting
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # pyyaml
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

# Better diverging palette for deltas (centered at 0)
try:
    import cmocean  # type: ignore
except Exception:  # pragma: no cover
    cmocean = None  # type: ignore

# mapping (optional at runtime; required if you want map outputs)
try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore


# ----------------------------
# Utilities: feature naming/encoding compatible with training
# ----------------------------


def sanitize_feature_names(columns: List[str]) -> List[str]:
    """
    XGBoost requires feature names to be strings and forbids '[', ']' and '<'.
    We also normalize whitespace for portability.
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


def safe_expm1(y: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(y, dtype=float))


# ----------------------------
# Scenario config parsing
# ----------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_region_mask(df: pd.DataFrame, include_regions: List[str]) -> pd.Series:
    if "Región" not in df.columns:
        raise ValueError("Dataset must contain 'Región' for region masking.")
    return df["Región"].isin(include_regions)


def build_mask(df: pd.DataFrame, mask_def: Dict[str, Any]) -> pd.Series:
    """
    Supported mask condition:
      - column, op, value
    ops: >, >=, <, <=, ==, !=
    """
    cond = mask_def.get("condition", {})
    col = cond.get("column")
    op = cond.get("op")
    val = cond.get("value")
    if col is None or op is None:
        raise ValueError(f"Invalid mask definition: {mask_def}")

    if col not in df.columns:
        # if the column is missing, the mask cannot be applied; default all False
        return pd.Series(False, index=df.index)

    s = pd.to_numeric(df[col], errors="coerce")
    if op == ">":
        return s > val
    if op == ">=":
        return s >= val
    if op == "<":
        return s < val
    if op == "<=":
        return s <= val
    if op == "==":
        return s == val
    if op == "!=":
        return s != val
    raise ValueError(f"Unsupported op in mask: {op}")


def climate_region_scalars(
    cfg: Dict[str, Any], df: pd.DataFrame, var: str
) -> pd.Series:
    """
    Returns per-row scalar for climate anomaly scaling by Región for given var in {"pp","tmean"}.
    If no mapping present, returns 1.0 for all rows.
    """
    clim = cfg.get("climate_anomaly_mask", {})
    region_mult = (clim.get("region_multipliers") or {}).get(var, {})
    if "Región" not in df.columns:
        return pd.Series(1.0, index=df.index)
    # default 1.0 if region not present in map
    return df["Región"].map(lambda r: float(region_mult.get(str(r), 1.0))).fillna(1.0)


# ----------------------------
# Transform application
# ----------------------------


def apply_constraints(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    cons = cfg.get("constraints", {})
    nonneg = cons.get("non_negative_columns", []) or []
    for c in nonneg:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].clip(lower=0)

    caps = cons.get("caps", {}) or {}
    pp_min = caps.get("pp_min", None)
    if pp_min is not None and "pp" in out.columns:
        out["pp"] = pd.to_numeric(out["pp"], errors="coerce").clip(lower=float(pp_min))

    tmin = caps.get("tmean_min", None)
    tmax = caps.get("tmean_max", None)
    if "tmean" in out.columns:
        if tmin is not None:
            out["tmean"] = pd.to_numeric(out["tmean"], errors="coerce").clip(
                lower=float(tmin)
            )
        if tmax is not None:
            out["tmean"] = pd.to_numeric(out["tmean"], errors="coerce").clip(
                upper=float(tmax)
            )

    return out


def apply_transforms(
    baseline_df: pd.DataFrame,
    scenario_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
    masks_cfg: Dict[str, Any],
    region_mask: pd.Series,
) -> pd.DataFrame:
    """
    Apply transforms in order to a copy of baseline_df.

    region_mask: Región in {SELVA, SIERRA}
    """
    df = baseline_df.copy()

    for t in scenario_cfg.get("transforms", []) or []:
        ttype = t.get("type")
        col = t.get("column")
        if not ttype or not col:
            raise ValueError(f"Invalid transform entry: {t}")

        if col not in df.columns:
            # Skip silently if column doesn't exist
            continue

        # Build apply mask
        m = pd.Series(True, index=df.index)

        if t.get("apply_region_mask", False):
            m = m & region_mask

        # optional named mask from config
        mask_name = t.get("apply_mask", None)
        if mask_name:
            mdef = masks_cfg.get(mask_name)
            if mdef:
                m = m & build_mask(baseline_df, mdef)  # based on baseline
            else:
                raise ValueError(f"Unknown mask referenced: {mask_name}")

        # climate mask scaling (per-region multipliers)
        apply_climate_mask = bool(t.get("apply_climate_mask", False))
        if apply_climate_mask:
            if col == "pp":
                scale = climate_region_scalars(global_cfg, df, "pp")
                # If a row is outside region_mask, scale may be 0.0 per yaml mapping; still ok.
            elif col == "tmean":
                scale = climate_region_scalars(global_cfg, df, "tmean")
            else:
                scale = pd.Series(1.0, index=df.index)
        else:
            scale = pd.Series(1.0, index=df.index)

        # Apply transform
        if ttype == "multiply":
            factor = float(t.get("factor"))
            df.loc[m, col] = pd.to_numeric(df.loc[m, col], errors="coerce") * (
                factor * scale.loc[m]
            )
        elif ttype == "add":
            value = float(t.get("value"))
            df.loc[m, col] = pd.to_numeric(df.loc[m, col], errors="coerce") + (
                value * scale.loc[m]
            )
        else:
            raise ValueError(f"Unsupported transform type: {ttype}")

    # Enforce constraints after all transforms
    df = apply_constraints(df, global_cfg)
    return df


# ----------------------------
# Scoring
# ----------------------------


def build_feature_matrix(
    df: pd.DataFrame,
    train_config: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Rebuild the feature matrix similarly to training:
    - drop target col
    - drop configured drop_cols if present
    - one-hot encode configured categorical_cols
    - sanitize feature names
    - align to training feature_columns
    """
    target_col = train_config.get("target_col", "Def_ha")
    drop_cols = list(train_config.get("drop_cols", []))
    categorical_cols = list(
        train_config.get("categorical_cols", ["Región", "NOMBDEP", "Cluster"])
    )

    X = df.drop(columns=[c for c in [target_col] if c in df.columns]).copy()
    drops = [c for c in drop_cols if c in X.columns]
    if drops:
        X = X.drop(columns=drops)

    # ensure YEAR present
    if "YEAR" in df.columns and "YEAR" not in X.columns:
        X["YEAR"] = df["YEAR"]

    X_enc = one_hot_encode(X, categorical_cols=categorical_cols)
    X_enc = align_to_schema(X_enc, feature_columns=feature_columns)
    return X_enc


def score_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Model was trained on log1p(Def_ha), so predictions are log-space; invert with expm1.
    """
    pred_log = model.predict(X)
    return safe_expm1(pred_log)


# ----------------------------
# Reporting / plotting
# ----------------------------


def _load_peru_districts_gdf(shapefile_zip: Path, target_crs: str = "EPSG:4326"):
    """
    Load INEI district (admin 3) polygons from the provided ZIP shapefile.

    Expects the ZIP to contain:
      - DISTRITOS_inei_geogpsperu_suyopomalia.shp (+ sidecars .dbf/.shx/.prj/.cpg)

    Returns a GeoDataFrame with a normalized 'UBIGEO' column (string, zero-padded to 6 when possible)
    and geometry in target_crs.
    """
    if gpd is None:
        raise RuntimeError(
            "geopandas is required for map plotting. Install it (and deps) and re-run."
        )

    if not shapefile_zip.exists():
        raise FileNotFoundError(f"District shapefile ZIP not found: {shapefile_zip}")

    # GeoPandas can read directly from a zip:// URI
    uri = f"zip://{shapefile_zip}"
    gdf = gpd.read_file(uri)

    # Normalize UBIGEO join key
    # Try common field names first, then fall back to any field containing 'ubigeo'
    ubigeo_col = None
    for c in ["UBIGEO", "ubigeo", "COD_UBIGEO", "CODUBIGEO", "CCDDCCPPCCDI", "CODIGO"]:
        if c in gdf.columns:
            ubigeo_col = c
            break
    if ubigeo_col is None:
        for c in gdf.columns:
            if "ubigeo" in str(c).lower():
                ubigeo_col = c
                break

    if ubigeo_col is None:
        raise ValueError(
            "Could not find an UBIGEO-like column in the district shapefile. "
            f"Available columns: {list(gdf.columns)}"
        )

    gdf = gdf.rename(columns={ubigeo_col: "UBIGEO"})
    gdf["UBIGEO"] = gdf["UBIGEO"].astype("string")

    # If UBIGEO is numeric-like, zero-pad to 6 (Peru UBIGEO district code length)
    as_num = pd.to_numeric(gdf["UBIGEO"], errors="coerce")
    if as_num.notna().any():
        # keep original strings when not numeric-like
        gdf.loc[as_num.notna(), "UBIGEO"] = (
            as_num.dropna().astype(int).astype(str).str.zfill(6)
        )

    # Reproject for consistent plotting
    try:
        if target_crs:
            gdf = gdf.to_crs(target_crs)
    except Exception:
        # If CRS is missing/broken in source, keep as-is; plotting still works.
        pass

    return gdf


def plot_bubble_map_peru_districts(
    df_results: pd.DataFrame,
    districts_zip_path: Path,
    out_path: Path,
    title: str,
    size_col: str = "scenario_pred_ha",
    color_col: str = "delta_ha",
    cmap: Optional[str] = None,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    boundary_level: str = "admin2",
) -> None:
    """
    Proportional-symbol ("bubble") map for Peru districts (admin-3).

    Visual encoding (redundant by design, to improve public interpretability):
      - Boundary context: outlines of admin-2 polygons (derived/dissolved from admin-3),
        to reduce visual clutter vs drawing all district boundaries.
      - Bubble size: model prediction (ha) from `size_col` (default scenario_pred_ha),
        using sqrt scaling to avoid domination by extreme values.
      - Bubble color + colorbar label: delta vs baseline (ha) from `color_col` (default delta_ha),
        using a diverging colormap centered at 0.
      - Size legend: 3 reference bubble sizes labeled with values in hectares.

    Notes:
      - We plot bubbles at polygon centroids, computed in a metric CRS for accuracy.
      - Rows missing `color_col` or `size_col` are omitted from bubble plotting.
      - `boundary_level` currently supports: "admin2" (recommended) or "admin3".
    """
    if gpd is None:
        raise RuntimeError(
            "geopandas is required for map plotting. Install it (and deps) and re-run."
        )

    if "UBIGEO" not in df_results.columns:
        raise ValueError(
            "df_results must include UBIGEO for joining to district polygons."
        )
    if size_col not in df_results.columns:
        raise ValueError(
            f"df_results missing column '{size_col}' required for bubble size."
        )
    if color_col not in df_results.columns:
        raise ValueError(
            f"df_results missing column '{color_col}' required for bubble color."
        )

    # Prep results key
    res = df_results.copy()
    res["UBIGEO"] = res["UBIGEO"].astype("string")
    res_num = pd.to_numeric(res["UBIGEO"], errors="coerce")
    if res_num.notna().any():
        res.loc[res_num.notna(), "UBIGEO"] = (
            res_num.dropna().astype(int).astype(str).str.zfill(6)
        )

    gdf_dist = _load_peru_districts_gdf(districts_zip_path)

    # Join predictions to polygons
    gdf = gdf_dist.merge(res[["UBIGEO", size_col, color_col]], on="UBIGEO", how="left")

    # Compute centroids in a metric CRS (for correct centroid placement)
    gdf_metric = gdf
    try:
        gdf_metric = gdf.to_crs("EPSG:3857")
    except Exception:
        # If CRS is missing/broken, centroid will be computed in whatever CRS exists.
        pass

    cent = gdf_metric.geometry.centroid
    gdf_metric = gdf_metric.assign(_x=cent.x, _y=cent.y)

    # Prepare numeric series
    gdf_metric[size_col] = pd.to_numeric(gdf_metric[size_col], errors="coerce")
    gdf_metric[color_col] = pd.to_numeric(gdf_metric[color_col], errors="coerce")

    # Keep only rows with both size and color
    pts = gdf_metric[
        np.isfinite(gdf_metric[size_col]) & np.isfinite(gdf_metric[color_col])
    ].copy()
    if pts.empty:
        raise ValueError(
            "No districts have finite values for both size and color columns after join."
        )

    # Color range: symmetric by default (delta centered at 0)
    vals = pts[color_col].to_numpy()
    finite = vals[np.isfinite(vals)]
    if color_vmin is None or color_vmax is None:
        if finite.size == 0:
            vmin0, vmax0 = -1.0, 1.0
        else:
            m = float(np.nanmax(np.abs(finite)))
            if m == 0:
                m = 1.0
            vmin0, vmax0 = -m, m
        if color_vmin is None:
            color_vmin = vmin0
        if color_vmax is None:
            color_vmax = vmax0

    # Use TwoSlopeNorm centered at zero for diverging deltas
    norm = TwoSlopeNorm(vmin=float(color_vmin), vcenter=0.0, vmax=float(color_vmax))

    # Default colormap: cmocean.balance (if available), else fall back
    if cmap is None:
        if cmocean is not None:
            cmap = cmocean.cm.balance
        else:
            cmap = "coolwarm"

    # Bubble size scaling (sqrt): matplotlib 's' is area in points^2.
    sizes = pts[size_col].to_numpy()
    sizes = np.clip(sizes, 0.0, np.inf)

    s_sqrt = np.sqrt(sizes)
    q95 = float(np.nanpercentile(s_sqrt, 95))
    if not np.isfinite(q95) or q95 <= 0:
        q95 = float(np.nanmax(s_sqrt)) if np.isfinite(np.nanmax(s_sqrt)) else 1.0
        if q95 <= 0:
            q95 = 1.0

    s_max_pts2 = 900.0  # largest bubble area (points^2)
    s_min_pts2 = 6.0  # smallest bubble area to keep small values visible
    s_pts2 = (s_sqrt / q95) * s_max_pts2
    s_pts2 = np.clip(s_pts2, s_min_pts2, s_max_pts2)

    # Plot (work in metric CRS for centroids + boundaries)
    fig, ax = plt.subplots(figsize=(9.5, 10.5))
    ax.set_axis_off()

    # Context layer: boundaries (admin-2 preferred to reduce clutter)
    if boundary_level not in {"admin2", "admin3"}:
        raise ValueError("boundary_level must be 'admin2' or 'admin3'.")

    if boundary_level == "admin3":
        # Optional: draw district outlines (can be visually busy)
        try:
            gdf_metric.boundary.plot(ax=ax, linewidth=0.15, color="#333333", alpha=0.6)
        except Exception:
            try:
                gdf_dist.boundary.plot(
                    ax=ax, linewidth=0.15, color="#333333", alpha=0.6
                )
            except Exception:
                pass
    else:
        # admin-2 boundaries derived from admin-3 polygons.
        # Peru UBIGEO structure: first 2 digits = department, next 2 = province, last 2 = district.
        # admin-2 (province) can be approximated by dissolving districts by UBIGEO[:4].
        tmp = gdf_metric.copy()
        tmp["UBIGEO"] = tmp["UBIGEO"].astype("string")
        tmp["_UBI4"] = tmp["UBIGEO"].str.slice(0, 4)

        # Dissolve to admin-2 (province) boundaries; fallback to department (admin-1) if needed.
        try:
            admin2 = tmp.dissolve(by="_UBI4", as_index=False)
        except Exception:
            tmp["_UBI2"] = tmp["UBIGEO"].str.slice(0, 2)
            admin2 = tmp.dissolve(by="_UBI2", as_index=False)

        try:
            admin2.boundary.plot(ax=ax, linewidth=0.6, color="#222222", alpha=0.8)
        except Exception:
            pass

    # Bubbles: size = prediction, color = delta
    sc = ax.scatter(
        pts["_x"],
        pts["_y"],
        s=s_pts2,
        c=pts[color_col],
        cmap=cmap,
        norm=norm,
        alpha=0.75,
        linewidths=0.25,
        edgecolors="black",
    )

    # Title with redundancy (explicitly naming both encodings)
    ax.set_title(
        f"{title}\nBubble size = {size_col} (ha). Bubble color = {color_col} (ha, diverging around 0)."
    )

    # Color legend (colorbar) with explicit label (redundancy gain)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"{color_col} (ha): scenario − baseline")

    # Size legend with labeled reference circles (redundancy gain)
    # Choose 3 reference sizes from the (positive) distribution.
    pos = sizes[np.isfinite(sizes) & (sizes > 0)]
    if pos.size == 0:
        ref_vals = [0.0, 1.0, 10.0]
    else:
        ref_vals = [
            float(np.nanpercentile(pos, 25)),
            float(np.nanpercentile(pos, 50)),
            float(np.nanpercentile(pos, 90)),
        ]
        # Ensure strictly increasing and > 0
        ref_vals = [max(0.0, v) for v in ref_vals]
        ref_vals = sorted(set(ref_vals))
        while len(ref_vals) < 3:
            ref_vals.append(ref_vals[-1] * 2 if ref_vals[-1] > 0 else 10.0)
        ref_vals = ref_vals[:3]

    ref_s = np.sqrt(np.asarray(ref_vals))
    ref_s_pts2 = (ref_s / q95) * s_max_pts2
    ref_s_pts2 = np.clip(ref_s_pts2, s_min_pts2, s_max_pts2)

    handles = []
    labels = []
    for rv, rs in zip(ref_vals, ref_s_pts2):
        h = ax.scatter(
            [], [], s=rs, facecolors="none", edgecolors="black", linewidths=0.8
        )
        handles.append(h)
        labels.append(f"{rv:,.0f} ha")

    leg = ax.legend(
        handles,
        labels,
        title=f"Bubble size legend\n({size_col})",
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=9,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.8,
    )
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_delta_hist(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["delta_ha"].to_numpy(), bins=50)
    plt.title(title)
    plt.xlabel("Delta predicted deforestation (ha) vs baseline")
    plt.ylabel("Count of districts")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_region_delta_bar(df: pd.DataFrame, out_path: Path, title: str) -> None:
    # sum delta by region
    g = (
        df.groupby("Región", dropna=False)["delta_ha"]
        .sum()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(10, 6))
    g.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Sum of delta (ha)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_topk_delta(df: pd.DataFrame, out_path: Path, title: str, k: int = 20) -> None:
    top = df.sort_values("delta_ha", ascending=False).head(k).copy()
    # label by UBIGEO (and NOMBDEP if present)
    if "NOMBDEP" in top.columns:
        top["label"] = top["UBIGEO"].astype(str) + " - " + top["NOMBDEP"].astype(str)
    else:
        top["label"] = top["UBIGEO"].astype(str)
    plt.figure(figsize=(12, 8))
    plt.barh(top["label"][::-1], top["delta_ha"][::-1])
    plt.title(title)
    plt.xlabel("Delta (ha)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_outputs(
    out_dir: Path,
    scenario_name: str,
    df_out: pd.DataFrame,
    districts_zip: Optional[Path] = None,
) -> None:
    ensure_dir(out_dir)

    # Per-district output
    df_out.to_csv(out_dir / f"{scenario_name}_district_results.csv", index=False)

    # Aggregates
    if "Región" in df_out.columns:
        reg = (
            df_out.groupby("Región", dropna=False)[
                ["baseline_pred_ha", "scenario_pred_ha", "delta_ha"]
            ]
            .sum()
            .reset_index()
        )
        reg.to_csv(out_dir / f"{scenario_name}_by_region.csv", index=False)
    if "NOMBDEP" in df_out.columns:
        dep = (
            df_out.groupby("NOMBDEP", dropna=False)[
                ["baseline_pred_ha", "scenario_pred_ha", "delta_ha"]
            ]
            .sum()
            .reset_index()
        )
        dep.to_csv(out_dir / f"{scenario_name}_by_department.csv", index=False)

    # Plots
    plot_delta_hist(
        df_out,
        out_dir / f"{scenario_name}_delta_hist.png",
        f"{scenario_name}: distribution of delta vs baseline",
    )
    if "Región" in df_out.columns:
        plot_region_delta_bar(
            df_out,
            out_dir / f"{scenario_name}_delta_by_region.png",
            f"{scenario_name}: total delta by Región",
        )
    plot_topk_delta(
        df_out,
        out_dir / f"{scenario_name}_top20_delta.png",
        f"{scenario_name}: top 20 districts by delta",
        k=20,
    )

    # Map (optional): proportional symbols
    # - bubble size: scenario_pred_ha
    # - bubble color: delta_ha (scenario - baseline)
    if districts_zip is not None:
        try:
            plot_bubble_map_peru_districts(
                df_results=df_out,
                districts_zip_path=districts_zip,
                out_path=out_dir
                / f"{scenario_name}_bubblemap_pred_size_delta_color.png",
                title=f"{scenario_name}: district bubbles (admin 3)",
                size_col="scenario_pred_ha",
                color_col="delta_ha",
            )
        except Exception as e:
            # Don't fail the whole run due to mapping issues; leave a breadcrumb.
            (
                out_dir / f"{scenario_name}_bubblemap_pred_size_delta_color.ERROR.txt"
            ).write_text(str(e), encoding="utf-8")


def plot_scenarios_comparison(
    all_results: Dict[str, pd.DataFrame], out_dir: Path
) -> None:
    """
    Create a single comparison plot: total predicted deforestation per scenario.
    """
    ensure_dir(out_dir)
    rows = []
    for name, df in all_results.items():
        rows.append(
            {
                "scenario": name,
                "total_pred_ha": float(df["scenario_pred_ha"].sum()),
                "total_delta_ha": float(df["delta_ha"].sum()),
            }
        )
    summ = pd.DataFrame(rows).sort_values("scenario")

    plt.figure(figsize=(10, 6))
    plt.bar(summ["scenario"], summ["total_pred_ha"])
    plt.title("Total predicted deforestation (ha) by scenario (baseline year rows)")
    plt.ylabel("Total predicted Def_ha (ha)")
    plt.tight_layout()
    plt.savefig(out_dir / "scenarios_total_pred_ha.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(summ["scenario"], summ["total_delta_ha"])
    plt.title("Total delta vs baseline (ha) by scenario")
    plt.ylabel("Total delta (ha)")
    plt.tight_layout()
    plt.savefig(out_dir / "scenarios_total_delta_ha.png", dpi=200)
    plt.close()

    summ.to_csv(out_dir / "scenarios_totals.csv", index=False)


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run baseline-2020 scenarios (good/mild/bad) for trained XGBoost deforestation model."
    )
    ap.add_argument(
        "--bundle",
        required=True,
        help="Path to bundle.joblib (trained model + schema).",
    )
    ap.add_argument("--data", required=True, help="Dataset CSV path.")
    ap.add_argument(
        "--sep", required=True, help="Dataset separator, e.g. ';' or '\\t'."
    )
    ap.add_argument(
        "--scenarios",
        required=True,
        help="Path to scenarios YAML, e.g. scenarios/scenarios.yaml",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory, e.g. reports/scenarios/xgb_timecv_v1_gpu_baseline2020",
    )
    ap.add_argument(
        "--districts-zip",
        default="DISTRITOS_inei_geogpsperu_suyopomalia.zip",
        help="Path to INEI districts (admin 3) shapefile ZIP for map plots.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Load bundle
    bundle = joblib.load(args.bundle)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    train_cfg = bundle.get("train_config")
    if not isinstance(train_cfg, dict):
        raise ValueError(
            "bundle['train_config'] must be a dict (portable). Recreate bundle.joblib if needed."
        )

    # Load scenario config
    cfg = load_yaml(args.scenarios)
    baseline_year = int((cfg.get("baseline") or {}).get("year", 2020))
    include_regions = ((cfg.get("baseline") or {}).get("region_mask") or {}).get(
        "include", ["SELVA", "SIERRA"]
    )
    masks_cfg = cfg.get("masks") or {}

    # Resolve districts shapefile zip (optional map plotting)
    districts_zip = Path(args.districts_zip) if args.districts_zip else None
    if districts_zip is not None and not districts_zip.is_absolute():
        # allow relative to project root / current working directory
        districts_zip = Path.cwd() / districts_zip

    # Load data
    df = pd.read_csv(args.data, sep=args.sep)
    if "YEAR" not in df.columns:
        raise ValueError("Dataset missing YEAR column.")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype(int)

    # Baseline rows
    base = df[df["YEAR"] == baseline_year].copy()
    if base.empty:
        raise ValueError(f"No rows found for baseline year {baseline_year}.")

    # Required columns
    for c in [
        "Región",
        "UBIGEO",
        "NOMBDEP",
        "Cluster",
        "pp",
        "tmean",
        "Minería",
        "Infraestructura",
        "area_agropec",
        "Coca_ha",
    ]:
        if c not in base.columns:
            # Not all are strictly required, but user expects these; fail loudly for clarity
            # If you later want more flexibility, relax this check.
            raise ValueError(f"Baseline data is missing required column: {c}")

    # Build masks on baseline rows
    region_mask = build_region_mask(base, include_regions=include_regions)

    # Score baseline
    X_base = build_feature_matrix(base, train_cfg, feature_columns)
    baseline_pred = score_model(model, X_base)

    # Keep identifying columns for outputs
    id_cols = [
        c for c in ["UBIGEO", "NOMBDEP", "Región", "Cluster"] if c in base.columns
    ]
    base_out = base[id_cols].copy()
    base_out["baseline_pred_ha"] = baseline_pred

    # Run each scenario
    results: Dict[str, pd.DataFrame] = {}
    scenarios = cfg.get("scenarios") or {}
    if not scenarios:
        raise ValueError("No scenarios found in scenarios YAML under key 'scenarios'.")

    # Save config snapshot
    (out_dir / "scenario_config_used.yaml").write_text(
        Path(args.scenarios).read_text(encoding="utf-8"), encoding="utf-8"
    )

    for name, sc in scenarios.items():
        df_sc = apply_transforms(
            baseline_df=base,
            scenario_cfg=sc,
            global_cfg=cfg,
            masks_cfg=masks_cfg,
            region_mask=region_mask,
        )

        X_sc = build_feature_matrix(df_sc, train_cfg, feature_columns)
        scenario_pred = score_model(model, X_sc)

        out = base_out.copy()
        out["scenario"] = name
        out["scenario_pred_ha"] = scenario_pred
        out["delta_ha"] = out["scenario_pred_ha"] - out["baseline_pred_ha"]
        out["pct_delta"] = np.where(
            out["baseline_pred_ha"] > 0,
            out["delta_ha"] / out["baseline_pred_ha"],
            np.nan,
        )

        scenario_dir = out_dir / name
        write_outputs(scenario_dir, name, out, districts_zip=districts_zip)

        results[name] = out

    # Combined comparison plot + totals
    plot_scenarios_comparison(results, out_dir)

    # Write a combined CSV of all scenarios stacked
    combined = pd.concat(
        [df.assign(scenario=name) for name, df in results.items()], ignore_index=True
    )
    combined.to_csv(out_dir / "all_scenarios_district_results.csv", index=False)

    # Also write a short JSON summary for quick inspection
    summary = {}
    for name, df_res in results.items():
        summary[name] = {
            "total_pred_ha": float(df_res["scenario_pred_ha"].sum()),
            "total_delta_ha": float(df_res["delta_ha"].sum()),
            "mean_delta_ha": float(df_res["delta_ha"].mean()),
        }
    (out_dir / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[OK] Scenario outputs written to: {out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
