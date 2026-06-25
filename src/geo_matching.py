"""
geo_matching.py
---------------
Matching commune → stations météo les plus proches.

Stratégie :
  1. Charger le cache des stations (stations.parquet) qui recense TOUTES les
     stations de TOUS les départements téléchargés, avec leur (LAT, LON).
  2. Pour une commune donnée, calculer la distance Haversine vectorisée vers
     toutes les stations connues, sans restriction de département.
  3. Retourner les NUM_NEAREST_STATIONS candidats les plus proches (triés par
     distance croissante) ; le filtrage qualité (35%) est ensuite délégué à
     frost_calculator.py qui sait lire les fichiers méteo département par
     département.

Construction du cache stations
--------------------------------
Le cache est construit à la demande (lazy) la première fois, en parcourant
TOUS les fichiers Q_<dept>_*_RR-T-Vent.csv.gz présents dans METEO_RAW_DIR et
en extrayant les colonnes (NUM_POSTE, NOM_USUEL, LAT, LON, département).
Il est sauvegardé dans STATIONS_CACHE_PATH pour les appels suivants.

Détection de l'unité de TN
---------------------------
Certains exports Météo-France exposent TN en °C (ex : -1.5),
d'autres gardent l'unité de stockage en 1/10 °C (ex : -15).
La fonction `detect_tn_scale` lit un échantillon du fichier et retourne
le facteur 0.1 si les valeurs semblent être des dixièmes, sinon 1.0.
"""

from __future__ import annotations

import glob
import os
import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Haversine vectorisé
# ---------------------------------------------------------------------------

def haversine_km_vec(
    lat1: float,
    lon1: float,
    lats2: np.ndarray,
    lons2: np.ndarray,
) -> np.ndarray:
    """Distance Haversine (km) d'un point vers un tableau de points."""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lats2_r = np.radians(lats2)
    lons2_r = np.radians(lons2)
    dlat = lats2_r - lat1_r
    dlon = lons2_r - lon1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lats2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Détection de l'unité de TN
# ---------------------------------------------------------------------------

def detect_tn_scale(filepath: str, n_sample: int = 500) -> float:
    """
    Retourne le facteur d'échelle à appliquer à TN pour obtenir des °C.

    Règle empirique :
      - Si la médiane des valeurs non-nulles |TN| > 50  → dixièmes → facteur 0.1
      - Sinon → déjà en °C → facteur 1.0

    Cette heuristique fonctionne car des températures minimales en °C ne
    dépassent quasiment jamais ±50 °C en France métropolitaine/DOM.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep=";",
            compression="gzip",
            usecols=["TN"],
            nrows=n_sample,
        )
        tn_valid = df["TN"].dropna()
        if tn_valid.empty:
            return 1.0
        median_abs = tn_valid.abs().median()
        if median_abs > 50:
            return 0.1
        return 1.0
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Construction et chargement du cache des stations
# ---------------------------------------------------------------------------

def _extract_dept_from_path(path: str) -> str:
    """Extrait le code département depuis le nom de fichier.

    Format attendu : Q_{DEPT}_previous-..._RR-T-Vent.csv.gz
    """
    basename = os.path.basename(path)
    m = re.match(r"Q_([^_]+)_", basename)
    return m.group(1) if m else "XX"

def get_all_station():
    pattern = os.path.join(config.METEO_RAW_DIR, "*RR-T-Vent*.csv.gz")
    meteo_files = sorted(glob.glob(pattern))

    if not meteo_files:
        raise FileNotFoundError(
            f"Aucun fichier météo trouvé dans {config.METEO_RAW_DIR}. "
            "Lancez d'abord src/download_data.py."
        )

    records = []
    for filepath in meteo_files:
        try:
            df = pd.read_csv(
                filepath,
                sep=";",
                compression="gzip",
                usecols=["NUM_POSTE", "NOM_USUEL", "LAT", "LON"],
                dtype={"NUM_POSTE": str},
            )
        except Exception:
            continue

        stations = (
            df[["NUM_POSTE", "NOM_USUEL", "LAT", "LON"]]
            .drop_duplicates("NUM_POSTE")
            .dropna(subset=["LAT", "LON"])
        )
        records.append(stations)
    if not records:
        raise ValueError("Aucune station trouvée dans les fichiers météo.")

    all_stations = pd.concat(records, ignore_index=True)
    # En cas de doublon (même station présente dans plusieurs fichiers), on
    # garde la ligne avec le tn_scale le plus fréquent pour ce NUM_POSTE.
    all_stations = (
        all_stations
        .sort_values("NUM_POSTE")
        .drop_duplicates("NUM_POSTE")
        .reset_index(drop=True)
    )
    return all_stations

def build_stations_cache(force: bool = False) -> pd.DataFrame:
    """
    Construit (et met en cache parquet) le référentiel de toutes les stations
    disponibles localement, avec leur département d'appartenance.

    Colonnes du DataFrame résultant :
        NUM_POSTE, NOM_USUEL, LAT, LON, dept, tn_scale
    """
    if not force and os.path.exists(config.STATIONS_CACHE_PATH):
        return pd.read_parquet(config.STATIONS_CACHE_PATH)

    pattern = os.path.join(config.METEO_RAW_DIR, "*RR-T-Vent*.csv.gz")
    meteo_files = sorted(glob.glob(pattern))

    if not meteo_files:
        raise FileNotFoundError(
            f"Aucun fichier météo trouvé dans {config.METEO_RAW_DIR}. "
            "Lancez d'abord src/download_data.py."
        )

    records = []
    for filepath in meteo_files:
        dept = _extract_dept_from_path(filepath)
        scale = detect_tn_scale(filepath)
        try:
            df = pd.read_csv(
                filepath,
                sep=";",
                compression="gzip",
                usecols=["NUM_POSTE", "NOM_USUEL", "LAT", "LON"],
                dtype={"NUM_POSTE": str},
            )
        except Exception:
            continue

        stations = (
            df[["NUM_POSTE", "NOM_USUEL", "LAT", "LON"]]
            .drop_duplicates("NUM_POSTE")
            .dropna(subset=["LAT", "LON"])
        )
        stations = stations.copy()
        stations["dept"] = dept
        stations["tn_scale"] = scale
        records.append(stations)

    if not records:
        raise ValueError("Aucune station trouvée dans les fichiers météo.")

    all_stations = pd.concat(records, ignore_index=True)
    # En cas de doublon (même station présente dans plusieurs fichiers), on
    # garde la ligne avec le tn_scale le plus fréquent pour ce NUM_POSTE.
    all_stations = (
        all_stations
        .sort_values("NUM_POSTE")
        .drop_duplicates("NUM_POSTE")
        .reset_index(drop=True)
    )

    # On remet quand même le dept de la station originale (premier fichier rencontré)
    all_stations.to_parquet(config.STATIONS_CACHE_PATH, index=False)
    print(f"[geo_matching] Cache stations construit : {len(all_stations)} stations.")
    return all_stations


@lru_cache(maxsize=1)
def _load_stations_cached() -> pd.DataFrame:
    """Wrapper mis en cache process-wide (LRU)."""
    return build_stations_cache(force=False)


def get_stations() -> pd.DataFrame:
    """Retourne le DataFrame des stations (depuis cache ou fichier parquet)."""
    return _load_stations_cached()


# ---------------------------------------------------------------------------
# Chargement du référentiel des communes
# ---------------------------------------------------------------------------

def _load_communes_raw() -> pd.DataFrame:
    pattern = os.path.join(config.COMMUNES_RAW_DIR, "*.csv.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier communes trouvé dans {config.COMMUNES_RAW_DIR}."
        )
    return pd.read_csv(
        files[0],
        compression="gzip",
        dtype={"code_insee": str, "dep_code": str},
        low_memory=False,
    )


def build_communes_cache(force: bool = False) -> pd.DataFrame:
    """
    Construit (et met en cache parquet) le référentiel des communes avec
    coordonnées GPS complétées pour les cas connus manquants.

    Colonnes clés conservées :
        nom_standard, dep_code, lat, lon
    """
    if not force and os.path.exists(config.COMMUNES_CACHE_PATH):
        return pd.read_parquet(config.COMMUNES_CACHE_PATH)

    communes = _load_communes_raw()

    # Détection automatique des colonnes de coordonnées
    communes.columns.str.lower()
    lat_col = next(
        (c for c in communes.columns if "latitude_centre" in c.lower()), None
    ) or next(
        (c for c in communes.columns if "latitude" in c.lower()), None
    )
    lon_col = next(
        (c for c in communes.columns if "longitude_centre" in c.lower()), None
    ) or next(
        (c for c in communes.columns if "longitude" in c.lower()), None
    )

    if lat_col is None or lon_col is None:
        raise ValueError(
            f"Colonnes latitude/longitude introuvables. Colonnes dispo : {list(communes.columns)}"
        )

    communes = communes.rename(columns={lat_col: "lat", lon_col: "lon"})

    # Complétion des coordonnées manquantes
    for nom, (lat, lon) in config.MISSING_CITIES_LAT_LON.items():
        mask = communes["nom_standard"].str.contains(nom, case=False, na=False)
        communes.loc[mask & communes["lat"].isna(), "lat"] = lat
        communes.loc[mask & communes["lon"].isna(), "lon"] = lon

    # Nettoyage minimal
    keep_cols = [c for c in ["nom_standard", "dep_code", "lat", "lon"] if c in communes.columns]
    communes = communes[keep_cols].dropna(subset=["lat", "lon"])
    communes["dep_code"] = communes["dep_code"].astype(str).str.zfill(2).str.upper()

    communes.to_parquet(config.COMMUNES_CACHE_PATH, index=False)
    print(f"[geo_matching] Cache communes construit : {len(communes)} communes avec coordonnées.")
    return communes


@lru_cache(maxsize=1)
def _load_communes_cached() -> pd.DataFrame:
    return build_communes_cache(force=False)


def get_communes() -> pd.DataFrame:
    """Retourne le DataFrame des communes (depuis cache ou fichier parquet)."""
    return _load_communes_cached()


# ---------------------------------------------------------------------------
# Recherche de la commune
# ---------------------------------------------------------------------------

def find_commune(nom: str, dept: Optional[str] = None) -> pd.Series:
    """
    Recherche une commune par nom (et optionnellement département).

    Retourne la première correspondance trouvée (pd.Series).
    Lève ValueError si introuvable.
    """
    communes = get_communes()

    mask = communes["nom_standard"].str.normalize("NFKD").str.lower() == (
        nom.strip().lower()
    )
    if not mask.any():
        # Repli : recherche partielle
        mask = communes["nom_standard"].str.lower().str.contains(
            nom.strip().lower(), na=False
        )

    if dept is not None:
        dept_norm = str(dept).zfill(2).upper()
        mask = mask & (communes["dep_code"] == dept_norm)

    if not mask.any():
        raise ValueError(
            f"Commune '{nom}'"
            + (f" (dept {dept})" if dept else "")
            + " introuvable dans le référentiel."
        )

    results = communes[mask]
    if len(results) > 1 and dept is None:
        # Retourne la première, mais avertit
        print(
            f"[geo_matching] Attention : {len(results)} communes trouvées pour '{nom}'. "
            f"Utilisation de '{results.iloc[0]['nom_standard']}' "
            f"(dept {results.iloc[0]['dep_code']}). "
            "Précisez le département pour lever l'ambiguïté."
        )
    return results.iloc[0]


# ---------------------------------------------------------------------------
# Matching commune → N stations candidates (tous départements confondus)
# ---------------------------------------------------------------------------

def get_candidate_stations(
    lat: float,
    lon: float,
    n: int = config.NUM_NEAREST_STATIONS,
) -> pd.DataFrame:
    """
    Retourne les n stations les plus proches (tous départements) triées par
    distance Haversine croissante.

    Colonnes du DataFrame retourné :
        NUM_POSTE, NOM_USUEL, LAT, LON, dept, tn_scale, dist_km
    """
    stations = get_stations()
    lats = stations["LAT"].values.astype(float)
    lons = stations["LON"].values.astype(float)

    dists = haversine_km_vec(lat, lon, lats, lons)
    idx_sorted = np.argsort(dists)[:n]
    candidates = stations.iloc[idx_sorted].copy()
    candidates["dist_km"] = dists[idx_sorted]
    return candidates.reset_index(drop=True)