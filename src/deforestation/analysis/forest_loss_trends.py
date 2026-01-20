"""
Forest loss trends (2001–2024) from curated district CSV.

Inputs
------
- Bosque_y_perdida_de_bosques_por_Distrito_al_2024_curated.csv
  Columns:
    - DEPARTAMENTO, PROVINCIA, DISTRITO, UBIGEO
    - 2001_ha ... 2024_ha (annual loss in hectares)
    - HIDROGRAFÍA_ha
    - BOSQUE AL 2024_ha

Outputs
-------
Writes plots and tables under:
  reports/forest_loss_trends/

Generated artifacts:
- department_trends_2001_2024.png
- department_trends_2021_2024.png
- province_trends_topN_2021_2024.png
- province_trends_topN_2001_2024.png
- remaining_forest_2024_by_department.png
- remaining_forest_2024_topN_provinces.png

- department_loss_long.csv
- province_loss_long.csv
- department_summary_2021_2024.csv
- province_summary_2021_2024.csv

Run
---
From repo root (deforestacion/):
  uv run python src/deforestation/analysis/forest_loss_trends.py \
    --csv Bosque_y_perdida_de_bosques_por_Distrito_al_2024_curated.csv \
    --out reports/forest_loss_trends \
    --top-provinces 20

Notes
-----
- The CSV uses commas as thousand separators in some numeric cells (e.g., "2,403.72").
  This script strips commas before numeric coercion.
- UBIGEO is normalized to 6-digit string (zero-padded) for joins later.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _year_columns(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}_ha", str(c))]
    years = sorted(int(c.split("_")[0]) for c in year_cols)
    year_cols = [f"{y}_ha" for y in years]
    return year_cols, years


def _to_num_cell(v):
    if pd.isna(v):
        return v
    s = str(v).strip().replace(",", "")
    return pd.to_numeric(s, errors="coerce")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize UBIGEO
    if "UBIGEO" in df.columns:
        df["UBIGEO"] = df["UBIGEO"].astype(str).str.strip().str.zfill(6)

    year_cols, _ = _year_columns(df)

    numeric_cols = list(year_cols)
    for c in ["HIDROGRAFÍA_ha", "BOSQUE AL 2024_ha"]:
        if c in df.columns:
            numeric_cols.append(c)

    for c in numeric_cols:
        df[c] = df[c].map(_to_num_cell)

    return df


def melt_loss_long(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    year_cols, years = _year_columns(df)
    long = (
        df.groupby(group_cols, dropna=False)[year_cols]
        .sum()
        .reset_index()
        .melt(id_vars=group_cols, var_name="year", value_name="loss_ha")
    )
    long["year"] = long["year"].str.replace("_ha", "", regex=False).astype(int)
    long = long.sort_values(group_cols + ["year"]).reset_index(drop=True)
    return long


def summarize_period(
    long_df: pd.DataFrame, group_cols: List[str], start_year: int, end_year: int
) -> pd.DataFrame:
    sub = long_df[
        (long_df["year"] >= start_year) & (long_df["year"] <= end_year)
    ].copy()
    g = (
        sub.groupby(group_cols, dropna=False)["loss_ha"]
        .sum()
        .reset_index()
        .rename(columns={"loss_ha": f"loss_{start_year}_{end_year}_ha"})
        .sort_values(f"loss_{start_year}_{end_year}_ha", ascending=False)
        .reset_index(drop=True)
    )
    return g


def plot_lines(
    long_df: pd.DataFrame,
    group_col: str,
    title: str,
    out_path: Path,
    years: Tuple[int, int] | None = None,
    max_series: int | None = None,
) -> None:
    data = long_df.copy()
    if years is not None:
        data = data[(data["year"] >= years[0]) & (data["year"] <= years[1])].copy()

    # Choose series to plot (avoid unreadable “spaghetti”)
    series_names = list(data[group_col].dropna().unique())
    if max_series is not None and len(series_names) > max_series:
        # Keep the top max_series by total loss in the time window
        totals = (
            data.groupby(group_col, dropna=False)["loss_ha"]
            .sum()
            .sort_values(ascending=False)
        )
        keep = totals.head(max_series).index.tolist()
        data = data[data[group_col].isin(keep)]
        series_names = keep

    plt.figure(figsize=(12, 6))
    for name, g in data.groupby(group_col, dropna=False):
        g = g.sort_values("year")
        plt.plot(g["year"], g["loss_ha"], marker="o", linewidth=1.6, label=str(name))

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Forest loss (ha)")
    plt.xticks(sorted(data["year"].unique()))
    if len(series_names) <= 12:
        plt.legend(loc="best", fontsize=8, ncol=2)
    else:
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_top_provinces_2021_2024(
    prov_long: pd.DataFrame, out_path: Path, top_n: int = 20
) -> None:
    # Identify top provinces by total 2021–2024 loss
    sub = prov_long[(prov_long["year"] >= 2021) & (prov_long["year"] <= 2024)].copy()
    totals = (
        sub.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False)["loss_ha"]
        .sum()
        .sort_values(ascending=False)
    )
    top = totals.head(top_n).reset_index()[["DEPARTAMENTO", "PROVINCIA"]]
    top = [tuple(r) for r in top.to_records(index=False)]

    data = prov_long.copy()
    data["_key"] = list(zip(data["DEPARTAMENTO"], data["PROVINCIA"]))
    data = data[data["_key"].isin(top)]
    data = data[(data["year"] >= 2021) & (data["year"] <= 2024)]

    plt.figure(figsize=(13, 7))
    for (dep, prov), g in data.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False):
        g = g.sort_values("year")
        plt.plot(
            g["year"], g["loss_ha"], marker="o", linewidth=1.4, label=f"{dep} - {prov}"
        )

    plt.title(f"Top {top_n} provinces by total forest loss (2021–2024)")
    plt.xlabel("Year")
    plt.ylabel("Forest loss (ha)")
    plt.xticks([2021, 2022, 2023, 2024])
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_remaining_forest_2024_by_department(df: pd.DataFrame, out_path: Path) -> None:
    if "BOSQUE AL 2024_ha" not in df.columns:
        return
    g = (
        df.groupby("DEPARTAMENTO", dropna=False)["BOSQUE AL 2024_ha"]
        .sum()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(12, 6))
    g.plot(kind="bar")
    plt.title("Remaining forest area by department (BOSQUE AL 2024)")
    plt.ylabel("Forest area (ha)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_remaining_forest_top_provinces(
    df: pd.DataFrame, out_path: Path, top_n: int = 20
) -> None:
    if "BOSQUE AL 2024_ha" not in df.columns:
        return
    g = (
        df.groupby(["DEPARTAMENTO", "PROVINCIA"], dropna=False)["BOSQUE AL 2024_ha"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    labels = [f"{d} - {p}" for (d, p) in g.index.tolist()]
    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], g.to_numpy()[::-1])
    plt.title(f"Top {top_n} provinces by remaining forest area (BOSQUE AL 2024)")
    plt.xlabel("Forest area (ha)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot forest loss trends 2001–2024 from curated district CSV."
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to Bosque_y_perdida_de_bosques_por_Distrito_al_2024_curated.csv",
    )
    ap.add_argument(
        "--out",
        default="reports/forest_loss_trends",
        help="Output directory for plots/CSVs.",
    )
    ap.add_argument(
        "--top-provinces",
        type=int,
        default=20,
        help="Top N provinces to include in province trend plots.",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(csv_path)

    # Long tables
    dep_long = melt_loss_long(df, group_cols=["DEPARTAMENTO"])
    prov_long = melt_loss_long(df, group_cols=["DEPARTAMENTO", "PROVINCIA"])

    dep_long.to_csv(out_dir / "department_loss_long.csv", index=False)
    prov_long.to_csv(out_dir / "province_loss_long.csv", index=False)

    # Summaries for 2021–2024
    dep_2124 = summarize_period(dep_long, ["DEPARTAMENTO"], 2021, 2024)
    prov_2124 = summarize_period(prov_long, ["DEPARTAMENTO", "PROVINCIA"], 2021, 2024)
    dep_2124.to_csv(out_dir / "department_summary_2021_2024.csv", index=False)
    prov_2124.to_csv(out_dir / "province_summary_2021_2024.csv", index=False)

    # Department trends
    plot_lines(
        dep_long,
        group_col="DEPARTAMENTO",
        title="Forest loss trends by department (2001–2024, sum over districts)",
        out_path=out_dir / "department_trends_2001_2024.png",
        years=(2001, 2024),
        max_series=30,  # departments are few; safe
    )
    plot_lines(
        dep_long,
        group_col="DEPARTAMENTO",
        title="Forest loss trends by department (2021–2024, sum over districts)",
        out_path=out_dir / "department_trends_2021_2024.png",
        years=(2021, 2024),
        max_series=30,
    )

    # Province trends (top N provinces)
    plot_top_provinces_2021_2024(
        prov_long,
        out_path=out_dir / "province_trends_topN_2021_2024.png",
        top_n=int(args.top_provinces),
    )

    # Also a longer horizon plot for the same top-N (optional, useful context)
    # Identify top provinces by 2021–2024 totals, then plot 2001–2024 for those provinces.
    top_keys = set(
        summarize_period(prov_long, ["DEPARTAMENTO", "PROVINCIA"], 2021, 2024)
        .head(int(args.top_provinces))[["DEPARTAMENTO", "PROVINCIA"]]
        .itertuples(index=False, name=None)
    )
    prov_long2 = prov_long.copy()
    prov_long2["_key"] = list(zip(prov_long2["DEPARTAMENTO"], prov_long2["PROVINCIA"]))
    prov_long2 = prov_long2[prov_long2["_key"].isin(top_keys)]
    plt.figure(figsize=(13, 7))
    for (dep, prov), g in prov_long2.groupby(
        ["DEPARTAMENTO", "PROVINCIA"], dropna=False
    ):
        g = g.sort_values("year")
        plt.plot(g["year"], g["loss_ha"], linewidth=1.2, label=f"{dep} - {prov}")
    plt.title(
        f"Top {int(args.top_provinces)} provinces (by 2021–2024 loss): trends 2001–2024"
    )
    plt.xlabel("Year")
    plt.ylabel("Forest loss (ha)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "province_trends_topN_2001_2024.png", dpi=220)
    plt.close()

    # Remaining forest 2024 plots
    plot_remaining_forest_2024_by_department(
        df, out_dir / "remaining_forest_2024_by_department.png"
    )
    plot_remaining_forest_top_provinces(
        df,
        out_dir / "remaining_forest_2024_topN_provinces.png",
        top_n=int(args.top_provinces),
    )

    # Small console summary
    total_2124 = float(
        df[[f"{y}_ha" for y in range(2021, 2025)]].sum(numeric_only=True).sum()
    )
    print(f"[OK] Wrote outputs to: {out_dir}")
    print(f"[INFO] Total loss 2021–2024 (all districts): {total_2124:,.2f} ha")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
