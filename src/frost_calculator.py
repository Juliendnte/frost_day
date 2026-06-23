"""
frost_calculator.py
-------------------
Calcul des jours de gel pour une commune et une plage de dates données.

Pipeline :
  1. Résolution commune → candidats stations (via geo_matching)
  2. Pour chaque candidat (trié par distance), lecture du fichier météo du
     département correspondant + filtrage qualité 35%
  3. Dès qu'une station valide est trouvée, calcul des agrégations :
       a) Nombre total de jours de gel sur la période
       b) Nombre moyen de jours de gel par année calendaire complète
       c) Pour chaque jour-de-l'année (MM-DD hors 02-29) :
            - nombre de fois où il a gelé sur la période
            - taux = nb_gel / nb_années_où_ce_jour_existe

Gestion de l'unité de TN :
  La colonne TN est lue telle quelle depuis le CSV, puis multipliée par
  `tn_scale` (0.1 si dixièmes de degré, 1.0 sinon) pour obtenir des °C.
  La détection est faite par geo_matching.detect_tn_scale() et stockée dans
  le cache stations.
  Le seuil de gel est toujours exprimé en °C dans config.FROST_THRESHOLD_C
  (0.0 °C) ; c'est TN qui est convertie, pas le seuil.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import config
import geo_matching as gm

# ---------------------------------------------------------------------------
# Types de sortie
# ---------------------------------------------------------------------------

@dataclass
class FrostResult:
    commune_name: str
    dept: str
    station_num: str
    station_name: str
    dist_km: float
    tn_scale: float                      # 0.1 si dixièmes, 1.0 si °C directs
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    # Agrégations
    total_frost_days: int = 0
    avg_frost_days_per_year: float = 0.0
    # DataFrame (index = "MM-DD", colonnes = ["frost_count", "years_present", "frost_rate"])
    daily_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"Commune          : {self.commune_name} ({self.dept})",
            f"Station retenue  : {self.station_name} ({self.station_num})",
            f"Distance         : {self.dist_km:.1f} km",
            f"Échelle TN       : {'×0.1 (dixièmes→°C)' if self.tn_scale == 0.1 else '×1.0 (déjà en °C)'}",
            f"Période          : {self.start_date.date()} → {self.end_date.date()}",
            "",
            f"Jours de gel (total)         : {self.total_frost_days}",
            f"Jours de gel (moy./an)       : {self.avg_frost_days_per_year:.1f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lecture d'un fichier météo département
# ---------------------------------------------------------------------------

def _find_meteo_file(dept: str) -> Optional[str]:
    """Retourne le chemin du fichier météo pour un département donné, ou None."""
    # Format : Q_{DEPT}_previous-..._RR-T-Vent.csv.gz
    pattern = os.path.join(config.METEO_RAW_DIR, f"Q_{dept}_*RR-T-Vent*.csv.gz")
    files = glob.glob(pattern)
    return files[0] if files else None


def _load_station_data(
    filepath: str,
    num_poste: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tn_scale: float,
) -> Optional[pd.DataFrame]:
    """
    Charge les données d'une seule station pour la période demandée.

    Retourne un DataFrame avec colonnes [date, tn_celsius] ou None si
    le fichier ne contient pas cette station.
    """
    try:
        # Lecture minimale : on lit uniquement les colonnes nécessaires
        header = pd.read_csv(
            filepath, sep=";", compression="gzip", nrows=0
        ).columns.tolist()

        usecols = ["NUM_POSTE", "AAAAMMJJ", "TN"]
        usecols = [c for c in usecols if c in header]  # sécurité

        df = pd.read_csv(
            filepath,
            sep=";",
            compression="gzip",
            usecols=usecols,
            dtype={"NUM_POSTE": str},
        )
    except Exception as exc:
        print(f"[frost_calculator] Erreur lecture {filepath} : {exc}")
        return None

    df = df[df["NUM_POSTE"] == num_poste].copy()
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["AAAAMMJJ"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    if df.empty:
        return None

    # Conversion en °C
    df["tn_celsius"] = pd.to_numeric(df["TN"], errors="coerce") * tn_scale
    return df[["date", "tn_celsius"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calcul du taux de valeurs manquantes d'une station sur une période
# ---------------------------------------------------------------------------

def _missing_rate(df_station: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Taux de valeurs manquantes (%) en tenant compte :
      - des NaN dans tn_celsius
      - des jours absents du fichier dans la plage [min_date, max_date] de la station
    """
    if df_station.empty:
        return 100.0

    nb_present = len(df_station)
    nb_tn_nan = df_station["tn_celsius"].isna().sum()

    date_min = df_station["date"].min()
    date_max = df_station["date"].max()
    nb_expected = (date_max - date_min).days + 1
    nb_absent_rows = max(0, nb_expected - nb_present)
    nb_missing_total = int(nb_tn_nan) + nb_absent_rows

    return 100.0 * nb_missing_total / nb_expected if nb_expected > 0 else 100.0


# ---------------------------------------------------------------------------
# Agrégations jours de gel
# ---------------------------------------------------------------------------

def _compute_frost_stats(
    df_station: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[int, float, pd.DataFrame]:
    """
    À partir d'un DataFrame [date, tn_celsius] (déjà filtré sur la période),
    calcule :
      - total_frost_days  : int
      - avg_per_year      : float
      - daily_stats       : DataFrame indexé sur "MM-DD" (hors 29 fév.)

    daily_stats columns : frost_count, years_present, frost_rate
    """
    df = df_station.dropna(subset=["tn_celsius"]).copy()
    df["is_frost"] = df["tn_celsius"] <= config.FROST_THRESHOLD_C
    df["year"] = df["date"].dt.year
    df["mmdd"] = df["date"].dt.strftime("%m-%d")

    # Total
    total_frost = int(df["is_frost"].sum())

    # Moyenne annuelle (sur les années civiles complètes)
    years_with_data = sorted(df["year"].unique())
    if years_with_data:
        frost_per_year = df.groupby("year")["is_frost"].sum()
        avg_per_year = float(frost_per_year.mean())
    else:
        avg_per_year = 0.0

    # Statistiques par jour de l'année (hors 29 fév.)
    df_no_leap = df[df["mmdd"] != "02-29"].copy()

    # Nombre de fois où chaque MM-DD est gélif
    frost_count = df_no_leap.groupby("mmdd")["is_frost"].sum().rename("frost_count")

    # Nombre d'années où ce MM-DD est présent dans les données (pour le taux)
    years_present = (
        df_no_leap.groupby("mmdd")["year"].nunique().rename("years_present")
    )

    daily = pd.concat([frost_count, years_present], axis=1)
    daily["frost_count"] = daily["frost_count"].fillna(0).astype(int)
    daily["years_present"] = daily["years_present"].fillna(0).astype(int)
    daily["frost_rate"] = np.where(
        daily["years_present"] > 0,
        daily["frost_count"] / daily["years_present"],
        np.nan,
    )
    daily.index.name = "day_of_year"

    return total_frost, avg_per_year, daily


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def compute_frost_days(
    commune_name: str,
    dept: Optional[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    n_candidates: int = config.NUM_NEAREST_STATIONS,
    max_missing_pct: float = config.MAX_MISSING_PERCENT,
    verbose: bool = True,
) -> FrostResult:
    """
    Calcule les statistiques de jours de gel pour une commune et une période.

    Parameters
    ----------
    commune_name : str
        Nom de la commune (ex : "Dijon").
    dept : str | None
        Code département (ex : "21") pour lever les ambiguïtés. Peut être None.
    start_date, end_date : str ou Timestamp
        Plage de dates inclusivement (format ISO 'YYYY-MM-DD' accepté).
    n_candidates : int
        Nombre de stations candidates à examiner (triées par distance).
    max_missing_pct : float
        Seuil d'exclusion (%) de valeurs manquantes.
    verbose : bool
        Affiche les informations de progression.

    Returns
    -------
    FrostResult

    Raises
    ------
    ValueError
        Si aucune station valide n'est trouvée parmi les candidats.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # 1. Résolution de la commune
    commune_row = gm.find_commune(commune_name, dept)
    lat = float(commune_row["lat"])
    lon = float(commune_row["lon"])
    commune_dept = str(commune_row.get("dep_code", dept or ""))

    if verbose:
        print(
            f"[frost_calculator] Commune : {commune_row['nom_standard']} "
            f"(dept {commune_dept})  lat={lat:.4f} lon={lon:.4f}"
        )

    # 2. Stations candidates (tous départements)
    candidates = gm.get_candidate_stations(lat, lon, n=n_candidates)

    if verbose:
        print(
            f"[frost_calculator] {len(candidates)} station(s) candidate(s) examinée(s) :"
        )
        for _, row in candidates.iterrows():
            print(
                f"   {row['NOM_USUEL']:30s} ({row['NUM_POSTE']})  "
                f"dept={row['dept']}  dist={row['dist_km']:.1f} km  "
                f"tn_scale={row['tn_scale']}"
            )

    # 3. Boucle sur les candidats jusqu'à trouver une station valide
    for _, cand in candidates.iterrows():
        num_poste = str(cand["NUM_POSTE"])
        dept_station = str(cand["dept"])
        tn_scale = float(cand["tn_scale"])
        dist_km = float(cand["dist_km"])
        station_name = str(cand["NOM_USUEL"])

        filepath = _find_meteo_file(dept_station)
        if filepath is None:
            if verbose:
                print(
                    f"   [skip] {station_name} : fichier dept {dept_station} absent localement."
                )
            continue

        df_station = _load_station_data(filepath, num_poste, start, end, tn_scale)
        if df_station is None or df_station.empty:
            if verbose:
                print(f"   [skip] {station_name} : aucune donnée sur la période.")
            continue

        miss_rate = _missing_rate(df_station, start, end)
        if miss_rate > max_missing_pct:
            if verbose:
                print(
                    f"   [skip] {station_name} : {miss_rate:.1f}% manquants "
                    f"(> {max_missing_pct}%)."
                )
            continue

        # Station valide !
        if verbose:
            print(
                f"   [OK]   {station_name} ({num_poste})  "
                f"dist={dist_km:.1f} km  manquants={miss_rate:.1f}%  "
                f"tn_scale={tn_scale}"
            )

        total, avg, daily = _compute_frost_stats(df_station, start, end)

        return FrostResult(
            commune_name=str(commune_row["nom_standard"]),
            dept=commune_dept,
            station_num=num_poste,
            station_name=station_name,
            dist_km=dist_km,
            tn_scale=tn_scale,
            start_date=start,
            end_date=end,
            total_frost_days=total,
            avg_frost_days_per_year=avg,
            daily_stats=daily,
        )

    raise ValueError(
        f"Aucune station valide trouvée pour '{commune_name}' parmi "
        f"les {n_candidates} candidats les plus proches. "
        "Essayez d'augmenter NUM_NEAREST_STATIONS dans config.py ou de "
        "télécharger plus de départements."
    )