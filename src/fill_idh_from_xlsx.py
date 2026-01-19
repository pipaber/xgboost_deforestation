"""
Fill missing IDH values in the main deforestation dataset using the official IDH Excel file.

Inputs (expected in repo root `deforestation/`):
- deforestation_dataset_PERU_imputed_coca.csv  (semicolon-separated)
- IDH-y-Componentes-2003-2019.xlsx

Outputs:
- deforestation_dataset_PERU_imputed_coca_idh.csv (semicolon-separated)
  Adds/updates:
    - IDH (filled where missing, when excel provides a value for UBIGEO+YEAR)
    - IDH_filled_from_xlsx (boolean indicator)
    - IDH_source (string: 'original' or 'xlsx')

Why this script exists:
- The IDH Excel workbook has non-standard headers (multi-row headers, "Unnamed:" columns),
  and separator rows (Provincia/Distrito) and region separators.
- This script extracts a clean long-format table: (UBIGEO, YEAR, IDH) and merges it.

Run:
  uv run python src/fill_idh_from_xlsx.py

Notes:
- UBIGEO is normalized to 6-digit strings (zfill(6)) to preserve leading zeros.
- IDH availability in the workbook is only for years: 2003, 2007, 2010, 2011, 2012, 2015, 2017, 2018, 2019.
  Therefore, this will not fill IDH for 2001–2002 or 2004–2006, etc., unless you choose to
  later interpolate/model-impute those remaining gaps.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


RE_UBIGEO_6 = re.compile(r"^\d{6}$")


@dataclass(frozen=True)
class Paths:
    base_csv: Path
    idh_xlsx: Path
    out_csv: Path


def _repo_root() -> Path:
    # Script is at deforestation/src/fill_idh_from_xlsx.py, so root is parent of src
    return Path(__file__).resolve().parents[1]


def _coerce_ubigeo_6(x) -> Optional[str]:
    if pd.isna(x):
        return None
    m = re.search(r"(\d+)", str(x))
    if not m:
        return None
    s = m.group(1).zfill(6)
    if RE_UBIGEO_6.fullmatch(s) is None:
        return None
    return s


def _clean_sheet_keep_geo_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the separator rows like:
      - Provincia / Distrito
      - empty rows
    and keep only rows with a valid UBIGEO.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "UBIGEO" not in df.columns:
        raise ValueError("Expected a column named 'UBIGEO' in the IDH sheet.")

    df["UBIGEO"] = df["UBIGEO"].apply(_coerce_ubigeo_6)
    df = df[df["UBIGEO"].notna()].copy()
    df = df[df["UBIGEO"].apply(lambda s: RE_UBIGEO_6.fullmatch(str(s)) is not None)].copy()
    return df


def _find_idh_block_start_col(columns: Iterable[str]) -> int:
    """
    In 'Variables del IDH 2003-2017' the IDH columns are presented as a block with a header
    like 'Índice de Desarrollo Humano' followed by a run of year columns.

    The actual file often has:
      - a first header row with section titles
      - a second header row with units
      - the real year values as row content, but columns are still Unnamed.

    Pandas reads it into columns where the IDH block is anchored by
    'Índice de Desarrollo Humano' (exact string in the workbook).
    """
    cols = list(columns)
    for i, c in enumerate(cols):
        c_norm = str(c).strip()
        if (
            "Índice de Desarrollo Humano" in c_norm
            or "Indice de Desarrollo Humano" in c_norm
            or c_norm.upper() == "IDH"
        ):
            return i
    raise ValueError(
        "Could not locate the IDH block start column in sheet 'Variables del IDH 2003-2017'. "
        "Expected a column containing 'Índice de Desarrollo Humano'."
    )


def extract_idh_2003_2017(idh_xlsx: Path) -> pd.DataFrame:
    """
    Extract IDH values for years 2003, 2007, 2010, 2011, 2012, 2015, 2017
    from sheet 'Variables del IDH 2003-2017'.

    Returns a long-format DataFrame with columns:
      - UBIGEO (6-digit string)
      - YEAR (int)
      - IDH (float)
      - IDH_source_sheet (str)
    """
    sheet = "Variables del IDH 2003-2017"
    df = pd.read_excel(idh_xlsx, sheet_name=sheet)
    df = _clean_sheet_keep_geo_rows(df)

    # Determine where the IDH block starts
    start_idx = _find_idh_block_start_col(df.columns)

    # Known year columns in this sheet (as observed in the workbook)
    years = [2003, 2007, 2010, 2011, 2012, 2015, 2017]

    # The IDH block spans start_idx .. start_idx+len(years)-1
    idh_cols = list(df.columns[start_idx : start_idx + len(years)])
    if len(idh_cols) != len(years):
        raise ValueError(
            f"Unexpected IDH block width in '{sheet}'. "
            f"Expected {len(years)} columns, got {len(idh_cols)}."
        )

    wide = df[["UBIGEO"]].copy()
    for col, yr in zip(idh_cols, years):
        wide[str(yr)] = pd.to_numeric(df[col], errors="coerce")

    long = wide.melt(id_vars=["UBIGEO"], var_name="YEAR", value_name="IDH")
    long["YEAR"] = long["YEAR"].astype(int)
    long["IDH_source_sheet"] = sheet
    long = long.dropna(subset=["IDH"])

    return long


def _pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "None of the candidate columns were found. "
        f"Candidates={candidates}. Available={df.columns.tolist()}"
    )


def extract_idh_single_year(idh_xlsx: Path, sheet: str, year: int) -> pd.DataFrame:
    """
    Extract IDH values from the single-year sheets (IDH 2018, IDH 2019).

    Returns a DataFrame with:
      - UBIGEO (6-digit string)
      - YEAR (int)
      - IDH (float)
      - IDH_source_sheet (str)
    """
    df = pd.read_excel(idh_xlsx, sheet_name=sheet)
    df = _clean_sheet_keep_geo_rows(df)

    df.columns = [str(c).strip() for c in df.columns]

    # The IDH column name differs between sheets.
    idh_col = _pick_first_existing_column(
        df,
        candidates=[
            "Índice de Desarrollo Humano (IDH)",
            "Indice de Desarrollo Humano (IDH)",
            "Índice de desarrollo Humano (IDH)",
            "Indice de desarrollo Humano (IDH)",
            "Índice de desarrollo Humano (IDH)",
            "Índice de Desarrollo Humano (IDH)",
            "IDH",
        ],
    )

    out = df[["UBIGEO"]].copy()
    out["YEAR"] = int(year)
    out["IDH"] = pd.to_numeric(df[idh_col], errors="coerce")
    out["IDH_source_sheet"] = sheet
    out = out.dropna(subset=["IDH"])
    return out


def build_idh_long_table(idh_xlsx: Path) -> pd.DataFrame:
    """
    Build a consolidated long-format IDH table from the workbook:
      - 2003–2017 (selected years)
      - 2018
      - 2019

    Output columns:
      - UBIGEO, YEAR, IDH, IDH_source_sheet
    """
    idh_0317 = extract_idh_2003_2017(idh_xlsx)
    idh_2018 = extract_idh_single_year(idh_xlsx, sheet="IDH 2018", year=2018)
    idh_2019 = extract_idh_single_year(idh_xlsx, sheet="IDH 2019", year=2019)

    idh_all = pd.concat([idh_0317, idh_2018, idh_2019], ignore_index=True)

    # De-duplicate if necessary: keep the first occurrence per key
    idh_all = (
        idh_all.sort_values(["UBIGEO", "YEAR", "IDH_source_sheet"])
        .drop_duplicates(subset=["UBIGEO", "YEAR"], keep="first")
        .reset_index(drop=True)
    )

    return idh_all


def fill_idh(base_df: pd.DataFrame, idh_long: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IDH values into base_df and fill missing base_df['IDH'] using xlsx values.

    Adds:
      - IDH_filled_from_xlsx: bool
      - IDH_source: 'original' or 'xlsx'
    """
    if "IDH" not in base_df.columns:
        raise ValueError("Base dataset does not contain column 'IDH'.")

    out = base_df.copy()

    # Normalize keys
    out["UBIGEO"] = out["UBIGEO"].apply(_coerce_ubigeo_6)
    if out["UBIGEO"].isna().any():
        # If this ever happens, the join will break; fail fast with a helpful message.
        bad = out[out["UBIGEO"].isna()].head(10)
        raise ValueError(
            "Found rows in base dataset with invalid UBIGEO after normalization. "
            f"Example rows:\n{bad.to_string(index=False)}"
        )

    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce").astype("Int64")
    if out["YEAR"].isna().any():
        bad = out[out["YEAR"].isna()].head(10)
        raise ValueError(
            "Found rows in base dataset with invalid YEAR values. "
            f"Example rows:\n{bad.to_string(index=False)}"
        )
    out["YEAR"] = out["YEAR"].astype(int)

    idh = idh_long[["UBIGEO", "YEAR", "IDH"]].copy()
    idh["UBIGEO"] = idh["UBIGEO"].apply(_coerce_ubigeo_6)
    idh["YEAR"] = pd.to_numeric(idh["YEAR"], errors="coerce").astype(int)

    merged = out.merge(idh, on=["UBIGEO", "YEAR"], how="left", suffixes=("", "_xlsx"))

    merged["IDH_filled_from_xlsx"] = merged["IDH"].isna() & merged["IDH_xlsx"].notna()
    merged.loc[merged["IDH_filled_from_xlsx"], "IDH"] = merged.loc[
        merged["IDH_filled_from_xlsx"], "IDH_xlsx"
    ]

    merged["IDH_source"] = "original"
    merged.loc[merged["IDH_filled_from_xlsx"], "IDH_source"] = "xlsx"

    merged = merged.drop(columns=["IDH_xlsx"])
    return merged


def main() -> int:
    root = _repo_root()
    paths = Paths(
        base_csv=root / "deforestation_dataset_PERU_imputed_coca.csv",
        idh_xlsx=root / "IDH-y-Componentes-2003-2019.xlsx",
        out_csv=root / "deforestation_dataset_PERU_imputed_coca_idh.csv",
    )

    if not paths.base_csv.exists():
        print(f"[ERROR] Base CSV not found: {paths.base_csv}", file=sys.stderr)
        return 2
    if not paths.idh_xlsx.exists():
        print(f"[ERROR] IDH XLSX not found: {paths.idh_xlsx}", file=sys.stderr)
        return 2

    # Important: the coca-imputed CSV you generated is semicolon-separated
    base = pd.read_csv(paths.base_csv, sep=";")

    before_missing = int(base["IDH"].isna().sum()) if "IDH" in base.columns else None
    print(f"[INFO] Loaded base: shape={base.shape}, IDH_missing={before_missing}")

    idh_long = build_idh_long_table(paths.idh_xlsx)
    print(
        "[INFO] Extracted IDH from xlsx: "
        f"rows={idh_long.shape[0]}, ubigeo_n={idh_long['UBIGEO'].nunique()}, years={sorted(idh_long['YEAR'].unique().tolist())}"
    )

    filled = fill_idh(base, idh_long)

    after_missing = int(filled["IDH"].isna().sum())
    filled_count = int(filled["IDH_filled_from_xlsx"].sum())
    print(f"[INFO] Filled from xlsx: {filled_count}")
    print(f"[INFO] IDH missing after: {after_missing}")

    filled.to_csv(paths.out_csv, sep=";", index=False)
    print(f"[INFO] Wrote: {paths.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
