import pandas as pd
from pathlib import Path
import re
import numpy as np

# Configuration
DATA_PATH = Path("deforestation_dataset_PERU.csv")
COCA_DIR = Path("coca")
OUTPUT_PATH = Path("deforestation_dataset_PERU_imputed_coca.csv")

# Mapping: Zone/Region Name -> List of identifiable substrings in NOMBPROB/NOMBDEP/NOMBDIST
# We will match these against the dataset to find valid UBIGEOs.
ZONE_TO_LOCATIONS = {
    # Zones in reports
    "Aguaytía": ["Padre Abad"],
    "Alto Chicama": ["Otuzco", "Santiago de Chuco"],
    "Amazonas": ["Amazonas"],
    "Bajo Amazonas": ["Loreto", "Mariscal Ramón Castilla"],
    "Bajo Huallaga": ["San Martín"],
    "Callería": ["Callería"],
    "Contamana": ["Contamana"],
    "Huallaga": ["Huallaga"],
    "Kosñipata": ["Kosñipata"],
    "La Convención-Lares": ["La Convención", "Lares", "Yanatile"],
    "Madre de Dios": ["Madre de Dios"],
    "Marañón": ["Marañón"],
    "Pichis-Palcazu-Pachitea": ["Oxapampa", "Puerto Inca"],
    "Putumayo": ["Putumayo"],
    "San Gabán": ["San Gabán"],
    "VRAEM": ["Ayacucho", "Cusco", "Junín"],
    "Inambari-Tambopata": ["Tambopata", "Inambari"],
    "Ucayali": ["Ucayali"],
    "Huánuco": ["Huánuco"],
    "Pasco": ["Pasco"],
    "Puno": ["Puno"],
    "Cusco": ["Cusco"],
    "Ayacucho": ["Ayacucho"],
    "Junín": ["Junín"],
    "Loreto": ["Loreto"],
    "San Martín": ["San Martín"],
}


def load_dataset():
    """Loads main dataset with correct separator."""
    try:
        # Based on file inspection, the separator is semicolon
        df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
    except Exception:
        # Fallback for absolute path if running from subdir
        df = pd.read_csv(
            Path(r"C:\Users\LENOVO\Documents\MIT_UTEC\deforestation") / DATA_PATH.name,
            sep=";",
            encoding="latin1",
        )

    return df


def get_ubigeos_for_zone(zone_name, df):
    """
    Finds UBIGEOs corresponding to a zone name.
    """
    keys = ZONE_TO_LOCATIONS.get(zone_name, [zone_name])
    ubigeos = set()

    for key in keys:
        pattern = f"(?i){re.escape(key)}"

        # Search Districts, Provinces, Regions
        for col in ["NOMBDIST", "NOMBPROB", "NOMBDEP"]:
            if col in df.columns:
                matches = df[df[col].str.contains(pattern, na=False)]["UBIGEO"].tolist()
                ubigeos.update(matches)

    return list(ubigeos)


def parse_tabular_summary(path):
    """Parses coca_2018-2022_resumen.txt"""
    print(f"Parsing tabular summary: {path}")
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_years = []

    for line in lines:
        parts = re.split(r"\s{2,}|\t", line.strip())

        if "Zona" in line and not header_years:
            header_years = [int(p) for p in line.split() if p.isdigit() and len(p) == 4]
            continue

        if header_years and len(parts) >= len(header_years) + 1:
            zone = parts[0].strip()
            vals = parts[1:]

            for yr, val in zip(header_years, vals):
                val = val.replace(",", "").replace("-", "0").strip()
                try:
                    ha = float(val)
                    data.append({"Zone": zone, "Year": yr, "Coca_ha": ha})
                except ValueError:
                    continue
    return pd.DataFrame(data)


def parse_narrative_report(path, year):
    """Parses unstructured text."""
    print(f"Parsing narrative: {path} for year {year}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\n", " ")

    extracted = []

    for zone in ZONE_TO_LOCATIONS.keys():
        # Pattern: ZoneName ... 12,345 ha
        pattern = rf"(?i){re.escape(zone)}.{0, 50}?(\d{{1,3}}(?:,\d{{3}})*) ?ha"
        matches = re.finditer(pattern, text)
        for m in matches:
            num_str = m.group(1).replace(",", "")
            try:
                ha = float(num_str)
                extracted.append({"Zone": zone, "Year": year, "Coca_ha": ha})
            except:
                pass

    return pd.DataFrame(extracted)


def main():
    df_main = load_dataset()
    print(f"Dataset loaded: {len(df_main)} rows.")

    if "Coca_ha" not in df_main.columns:
        # Create it if missing, though it should be there with NaNs
        df_main["Coca_ha"] = np.nan

    print(f"Initial Missing Coca_ha: {df_main['Coca_ha'].isna().sum()}")

    files = list(COCA_DIR.glob("*.txt"))
    all_extracted = []

    for f in files:
        match = re.search(r"(\d{4})", f.name)
        if "2018-2022" in f.name:
            all_extracted.append(parse_tabular_summary(f))
        elif match:
            year = int(match.group(1))
            all_extracted.append(parse_narrative_report(f, year))

    if not all_extracted:
        print("No data extracted from text files.")
        return

    df_coca = pd.concat(all_extracted, ignore_index=True)
    # Deduplicate: take max for (Zone, Year)
    df_coca = df_coca.groupby(["Zone", "Year"], as_index=False)["Coca_ha"].max()

    print(f"Extracted {len(df_coca)} Zone-Year records.")

    imputation_map = {}

    for _, row in df_coca.iterrows():
        zone = row["Zone"]
        year = row["Year"]
        total_ha = row["Coca_ha"]

        ubigeos = get_ubigeos_for_zone(zone, df_main)

        relevant_rows = df_main[
            (df_main["YEAR"] == year) & (df_main["UBIGEO"].isin(ubigeos))
        ]
        valid_ubigeos = relevant_rows["UBIGEO"].tolist()

        if valid_ubigeos:
            ha_per_dist = total_ha / len(valid_ubigeos)
            for u in valid_ubigeos:
                key = (u, year)
                curr = imputation_map.get(key, 0)
                imputation_map[key] = max(curr, ha_per_dist)

    imputed_count = 0
    for idx, row in df_main.iterrows():
        if pd.isna(row["Coca_ha"]):
            key = (row["UBIGEO"], row["YEAR"])
            if key in imputation_map:
                df_main.at[idx, "Coca_ha"] = imputation_map[key]
                imputed_count += 1

    print(f"Imputed {imputed_count} missing values.")
    print(f"Final Missing Coca_ha: {df_main['Coca_ha'].isna().sum()}")

    df_main.to_csv(OUTPUT_PATH, sep=";", index=False, encoding="latin1")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
