import glob
import os
import sys
import unicodedata
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

def missing_rate_per_station(df: pd.DataFrame, seuil: float = config.MAX_MISSING_PERCENT) -> pd.DataFrame:
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
                dtype={"NUM_POSTE": str, "AAAAMMJJ": str},
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

    save_dataset(df, os.path.join(config.VALIDATION_DIR, "city_df_complete.csv"))

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

def process_city_weather(city_name: str, dept: str, output_dir: str = config.VALIDATION_DIR) -> None:
    """
    Pour une commune donnée, extrait les données météo de sa station la plus proche valide
    et sauvegarde un CSV au format :
    station_id, station_name, latitude, longitude, alti, date, tmin, frost_day, year, month, day
    :param city_name: nom de la commune (recherche partielle, insensible à la casse)
    :param dept: code département (ex: "01", "04")
    :param output_dir: dossier de sortie
    """
    city_df = pd.read_csv(
        os.path.join(config.VALIDATION_DIR, "city_df_complete.csv"),
        dtype={"insee_code": str, "dep_code": str, "closest_station_num_poste": str},
    )

    def strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        ).lower()

    dept_norm = str(dept).zfill(2)
    search = strip_accents(city_name)
    mask = (
        city_df["name"].apply(lambda x: strip_accents(str(x))).str.contains(search, na=False)
        & (city_df["dep_code"] == dept_norm)
    )
    if not mask.any():
        raise ValueError(f"Commune '{city_name}' (dept {dept}) introuvable dans city_df_complete.csv")

    row = city_df[mask].iloc[0]
    num_poste = str(row["closest_station_num_poste"])
    station_dept = str(int(row["station_dept"])).zfill(2)

    meteo_files = glob.glob(os.path.join(config.METEO_RAW_DIR, f"Q_{station_dept}_*RR-T-Vent*.csv.gz"))
    if not meteo_files:
        raise FileNotFoundError(f"Aucun fichier météo pour le dept {station_dept}")

    df = pd.read_csv(
        meteo_files[0], sep=";", compression="gzip",
        usecols=["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMMJJ", "TN"],
        dtype={"NUM_POSTE": str},
    )
    df = df[df["NUM_POSTE"] == num_poste].copy()
    if df.empty:
        raise ValueError(f"Station {num_poste} introuvable dans le fichier météo dept {station_dept}")

    stations = gm.build_stations_cache(force=False)
    match = stations[stations["NUM_POSTE"] == num_poste]
    tn_scale = float(match.iloc[0]["tn_scale"]) if not match.empty else 1.0

    df["date"] = pd.to_datetime(df["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= "2014-01-01") & (df["date"] <= "2023-12-31")]
    df["tmin"] = pd.to_numeric(df["TN"], errors="coerce") * tn_scale
    df["frost_day"] = df["tmin"] <= config.FROST_THRESHOLD_C
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    out = df.rename(columns={
        "NUM_POSTE": "station_id",
        "NOM_USUEL": "station_name",
        "LAT": "latitude",
        "LON": "longitude",
        "ALTI": "alti",
    })[["station_id", "station_name", "latitude", "longitude", "alti",
        "date", "tmin", "frost_day", "year", "month", "day"]]

    city_safe = city_name.replace(" ", "_")
    filename = f"{city_safe}_{dept}_completed.csv"
    save_dataset(out, os.path.join(config.VALIDATION_DIR, filename))
    print(f"[etl] {city_name} ({dept}) → {filename} ({len(out)} lignes)")

def load_stations() -> pd.DataFrame:
    stations = gm.get_all_station()
    return clean_stations(stations)

def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    valid_ids = build_valid_station_ids()
    df = df[df["NUM_POSTE"].isin(valid_ids)].reset_index(drop=True)
    df = df.rename(columns={"NUM_POSTE": "station_id", "NOM_USUEL": "station_name"})
    save_dataset(df, os.path.join(config.VALIDATION_DIR, "stations_df_complete.csv"))
    return df

if __name__ == "__main__":
    load_stations()
    load_communes()

    cities = [
        ("Asnières-sur-Saône", "01"),
        ("Digne-les-Bains", "04"),
        ("Espinchal", "63"),
        ("Marseille", "13"),
        ("Montfalcon", "38"),
        ("Paris", "75"),
    ]

    for city, dept in cities:
        process_city_weather(city, dept)