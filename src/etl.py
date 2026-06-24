import glob
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import geo_matching as gm
import config

def del_qtn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values in the "QTN" column
    :param df:
    :return: df
    """
    df.dropna(subset=["QTN"], inplace=True)
    return df

def date_min_max(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return the DataFrame with the minimum date is 2014-01-01 and the maximum date is 2023-12-31
    :param df:
    :return: df
    """
    df = df[(df["AAAAMMJJ"] >= "20140101") & (df["AAAAMMJJ"] <= "20231231")]
    return df

def missing_rate_per_station(df: pd.DataFrame, seuil: float = 35.0) -> pd.DataFrame:
    """
    Supprime les stations ayant un taux de valeurs manquantes supérieur au seuil.
    :param df: DataFrame avec colonnes NUM_POSTE, AAAAMMJJ, TN
    :param seuil: pourcentage max de valeurs manquantes autorisé (défaut 35%)
    :return: df filtré
    """
    rows = []
    for num_poste, g in df.groupby("NUM_POSTE"):
        nb_present = len(g)
        nb_missing_tn = g["TN"].isna().sum()
        date_min, date_max = g["AAAAMMJJ"].min(), g["AAAAMMJJ"].max()
        nb_expected = (date_max - date_min).days + 1
        nb_absent_rows = max(0, nb_expected - nb_present)
        nb_missing_total = nb_missing_tn + nb_absent_rows
        taux = 100 * nb_missing_total / nb_expected if nb_expected > 0 else 100.0
        if taux <= seuil:
            rows.append(num_poste)
    return df[df["NUM_POSTE"].isin(rows)]

def build_valid_station_ids(seuil: float = config.MAX_MISSING_PERCENT) -> set:
    """
    Lit tous les fichiers météo, filtre sur 2014-2023,
    et retourne l'ensemble des NUM_POSTE passant le critère de qualité.
    """
    pattern = os.path.join(config.METEO_RAW_DIR, "*RR-T-Vent*.csv.gz")
    meteo_files = sorted(glob.glob(pattern))

    valid_ids = set()
    for filepath in meteo_files:
        try:
            df = pd.read_csv(
                filepath, sep=";", compression="gzip",
                usecols=["NUM_POSTE", "AAAAMMJJ", "TN"],
                dtype={"NUM_POSTE": str},
            )
            df = date_min_max(df)
            if df.empty:
                continue
            df["AAAAMMJJ"] = pd.to_datetime(
                df["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce"
            )
            df = df.dropna(subset=["AAAAMMJJ"])
            valid_df = missing_rate_per_station(df, seuil)
            valid_ids.update(valid_df["NUM_POSTE"].unique())
        except Exception:
            continue
    return valid_ids

def clean_communes(df: pd.DataFrame) -> None:
    """
    Sélectionne, renomme les colonnes et associe la station la plus proche à chaque commune.
    :param df: DataFrame brut des communes
    :return: df avec colonnes insee_code, name, dep_code, dep_name, lat, lon,
             closest_station_name, closest_station_num_poste, station_dept
    """

    cols = ["code_insee", "nom_standard", "dep_code", "dep_nom", "latitude_centre", "longitude_centre"]
    df = df[cols].copy()

    for nom, (lat, lon) in config.MISSING_CITIES_LAT_LON.items():
        mask = df["nom_standard"].str.contains(nom, case=False, na=False)
        df.loc[mask & df["latitude_centre"].isna(), "latitude_centre"] = lat
        df.loc[mask & df["longitude_centre"].isna(), "longitude_centre"] = lon

    df = df.dropna(subset=["latitude_centre", "longitude_centre"])
    df = df.rename(columns={
        "code_insee": "insee_code",
        "nom_standard": "name",
        "dep_nom": "dep_name",
        "latitude_centre": "lat",
        "longitude_centre": "lon",
    })
    df = df.sort_values("insee_code").reset_index(drop=True)

    valid_ids = build_valid_station_ids()
    stations = gm.build_stations_cache(force=True)
    stations = stations[stations["NUM_POSTE"].isin(valid_ids)].reset_index(drop=True)
    lats_s = stations["LAT"].values.astype(float)
    lons_s = stations["LON"].values.astype(float)

    closest_names, closest_nums, closest_depts = [], [], []
    for _, row in df.iterrows():
        dists = gm.haversine_km_vec(row["lat"], row["lon"], lats_s, lons_s)
        idx = int(np.argmin(dists))
        station = stations.iloc[idx]
        closest_names.append(station["NOM_USUEL"])
        closest_nums.append(station["NUM_POSTE"])
        closest_depts.append(station["dept"])

    df["closest_station_name"] = closest_names
    df["closest_station_num_poste"] = closest_nums
    df["station_dept"] = closest_depts

    save_dataset(df, os.path.join(config.PROCESSED_DIR, "city_df.csv"))

def load_communes() -> None:
    """
    Charge le fichier communes-france-2025.csv.gz et appelle clean_communes.
    """
    path = os.path.join(config.COMMUNES_RAW_DIR, "communes-france-2025.csv.gz")
    df = pd.read_csv(path, dtype={"code_insee": str}, low_memory=False)
    clean_communes(df)

def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Save the dataset to a CSV file.

    :param df: DataFrame containing the dataset.
    :param path: Path to the CSV file.
    :return: None
    """
    df.to_csv(path, index=False)

if __name__ == "__main__":
    load_communes()