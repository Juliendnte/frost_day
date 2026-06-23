"""
Configuration centrale du projet Frost Days.
"""

import os

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
METEO_RAW_DIR = os.path.join(RAW_DIR, "meteo")
COMMUNES_RAW_DIR = os.path.join(RAW_DIR, "communes")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

for d in (DATA_DIR, RAW_DIR, METEO_RAW_DIR, COMMUNES_RAW_DIR, PROCESSED_DIR):
    os.makedirs(d, exist_ok=True)

STATIONS_CACHE_PATH = os.path.join(PROCESSED_DIR, "stations.parquet")
COMMUNES_CACHE_PATH = os.path.join(PROCESSED_DIR, "communes.parquet")

# --------------------------------------------------------------------------
# Période d'intérêt par défaut (cf. sujet du défi)
# --------------------------------------------------------------------------
DEFAULT_START_YEAR = 2014
DEFAULT_END_YEAR = 2023

# --------------------------------------------------------------------------
# data.gouv.fr : datasets
# --------------------------------------------------------------------------
# "Données climatologiques de base - quotidiennes" (Météo-France)
METEO_DATASET_ID = "6569b51ae64326786e4e8e1a"
METEO_DATASET_API_URL = f"https://www.data.gouv.fr/api/1/datasets/{METEO_DATASET_ID}/"

# Fichier de métadonnées des champs (RR-T-Vent)
METADATA_FIELDS_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/"
    "BASE/QUOT/Q_descriptif_champs_RR-T-Vent.csv"
)

# "Communes et villes de France en CSV, Excel, Json, Parquet et Feather"
COMMUNES_DATASET_ID = "6745d9ae4524d845d2138193"
COMMUNES_DATASET_API_URL = f"https://www.data.gouv.fr/api/1/datasets/{COMMUNES_DATASET_ID}/"

# --------------------------------------------------------------------------
# Liste des codes départements (métropole + DOM)
# --------------------------------------------------------------------------
METROPOLE_DEPTS = [f"{i:02d}" for i in range(1, 96) if i != 20] + ["2A", "2B"]
DOM_DEPTS = ["971", "972", "973", "974", "975", "984", "986", "987", "988"]
ALL_DEPTS = METROPOLE_DEPTS + DOM_DEPTS

# --------------------------------------------------------------------------
# Coordonnées GPS manquantes pour certaines communes (cf. sujet du défi)
# --------------------------------------------------------------------------
MISSING_CITIES_LAT_LON = {
    "Marseille": [43.295, 5.372],
    "Paris": [48.866, 2.333],
    "Culey": [48.755, 5.266],
    "Les Hauts-Talican": [49.3436, 2.0193],
    "Lyon": [45.75, 4.85],
    "Bihorel": [49.4542, 1.1162],
    "Saint-Lucien": [48.6480, 1.6229],
    "L'Oie": [46.7982, -1.1302],
    "Sainte-Florence": [46.7965, -1.1520],
}

# --------------------------------------------------------------------------
# Paramètres métier
# --------------------------------------------------------------------------
MAX_MISSING_PERCENT = 35.0   # % de valeurs manquantes max pour garder une station
FROST_THRESHOLD_C = 0.0      # Seuil de gel : TN <= 0°C
NUM_NEAREST_STATIONS = 5     # Nombre de stations candidates les plus proches à examiner
EARTH_RADIUS_KM = 6371.0