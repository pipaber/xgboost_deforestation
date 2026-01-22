from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd


def normalize_ubigeo(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.zfill(6)
    return s


def load_districts(zip_path: Path) -> gpd.GeoDataFrame:
    uri = f"zip://{zip_path}"
    gdf = gpd.read_file(uri)

    ubigeo_col: Optional[str] = None
    for col in ["UBIGEO", "ubigeo", "COD_UBIGEO", "CODUBIGEO", "CCDDCCPPCCDI", "CODIGO"]:
        if col in gdf.columns:
            ubigeo_col = col
            break
    if ubigeo_col is None:
        for col in gdf.columns:
            if "ubigeo" in str(col).lower():
                ubigeo_col = col
                break

    if ubigeo_col is None:
        raise ValueError("Could not find UBIGEO column in the shapefile.")

    gdf = gdf.rename(columns={ubigeo_col: "UBIGEO"})
    gdf["UBIGEO"] = normalize_ubigeo(gdf["UBIGEO"])
    return gdf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate department/province GeoJSON from districts shapefile."
    )
    parser.add_argument(
        "--districts-zip",
        default="DISTRITOS_inei_geogpsperu_suyopomalia.zip",
        help="Path to INEI district shapefile ZIP.",
    )
    parser.add_argument(
        "--dataset",
        default="deforestation_dataset_PERU_imputed_coca.csv",
        help="Dataset CSV path for NOMBDEP/NOMBPROB mapping.",
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="Dataset separator (default ';').",
    )
    parser.add_argument(
        "--out-dir",
        default="frontend/data",
        help="Output directory for GeoJSON files.",
    )
    args = parser.parse_args()

    zip_path = Path(args.districts_zip)
    if not zip_path.exists():
        print(f"[ERROR] Shapefile ZIP not found: {zip_path}")
        return 1

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return 1

    df = pd.read_csv(dataset_path, sep=args.sep)
    if "UBIGEO" not in df.columns:
        print("[ERROR] Dataset is missing UBIGEO column.")
        return 1
    if "NOMBDEP" not in df.columns or "NOMBPROB" not in df.columns:
        print("[ERROR] Dataset is missing NOMBDEP/NOMBPROB columns.")
        return 1

    df["UBIGEO"] = normalize_ubigeo(df["UBIGEO"])
    df["UBI4"] = df["UBIGEO"].str.slice(0, 4)
    df["UBI2"] = df["UBIGEO"].str.slice(0, 2)

    prov_names = df[["UBI4", "NOMBPROB"]].dropna().drop_duplicates()
    dep_names = df[["UBI2", "NOMBDEP"]].dropna().drop_duplicates()

    gdf = load_districts(zip_path)
    gdf["UBI4"] = gdf["UBIGEO"].str.slice(0, 4)
    gdf["UBI2"] = gdf["UBIGEO"].str.slice(0, 2)

    provinces = gdf.dissolve(by="UBI4", as_index=False).merge(
        prov_names, on="UBI4", how="left"
    )
    departments = gdf.dissolve(by="UBI2", as_index=False).merge(
        dep_names, on="UBI2", how="left"
    )

    if "NOMBDEP" not in departments.columns and "NOMBDEP_x" in departments.columns:
        departments = departments.rename(columns={"NOMBDEP_x": "NOMBDEP"})
    if "NOMBPROB" not in provinces.columns and "NOMBPROV" in provinces.columns:
        provinces = provinces.rename(columns={"NOMBPROV": "NOMBPROB"})

    if "NOMBPROB" not in provinces.columns:
        raise ValueError("Could not find NOMBPROB in provinces output.")
    if "NOMBDEP" not in departments.columns:
        raise ValueError("Could not find NOMBDEP in departments output.")

    provinces = provinces.to_crs("EPSG:4326")[["NOMBPROB", "geometry"]]
    departments = departments.to_crs("EPSG:4326")[["NOMBDEP", "geometry"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prov_path = out_dir / "peru_provinces.geojson"
    dep_path = out_dir / "peru_departments.geojson"

    provinces.to_file(prov_path, driver="GeoJSON")
    departments.to_file(dep_path, driver="GeoJSON")

    print(f"[OK] Wrote: {prov_path}")
    print(f"[OK] Wrote: {dep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
