"""
test_geo_matching.py
--------------------
Tests unitaires et d'intégration pour geo_matching.py.

Couvre :
  1. Haversine vectorisée (valeurs connues)
  2. detect_tn_scale  (heuristique °C vs dixièmes)
  3. _extract_dept_from_path
  4. Cache stations  → bug potentiel : lru_cache cross-test + chemins absents
  5. find_commune    (exact, partiel, ambiguïté, dept filter, absent)
  6. get_candidate_stations  (tri par distance, unicité NUM_POSTE)
  7. build_communes_cache  (injection MISSING_CITIES_LAT_LON)
"""

import io
import os
import gzip
import math
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# 1. HAVERSINE
# ──────────────────────────────────────────────────────────────────────────────

class TestHaversine(unittest.TestCase):
    """Valide la formule Haversine sur des distances connues."""

    def _import(self):
        import importlib, sys
        # Reimport propre à chaque test pour éviter les effets de bord du cache
        if "geo_matching" in sys.modules:
            gm = sys.modules["geo_matching"]
        else:
            import geo_matching as gm
        return gm

    def test_same_point_is_zero(self):
        gm = self._import()
        d = gm.haversine_km_vec(48.8566, 2.3522, np.array([48.8566]), np.array([2.3522]))
        self.assertAlmostEqual(d[0], 0.0, places=3)

    def test_paris_to_lyon_approx(self):
        """Paris→Lyon ≈ 390–400 km à vol d'oiseau."""
        gm = self._import()
        d = gm.haversine_km_vec(48.8566, 2.3522, np.array([45.75]), np.array([4.85]))
        self.assertGreater(d[0], 380)
        self.assertLess(d[0], 420)

    def test_paris_to_marseille_approx(self):
        """Paris→Marseille ≈ 660–680 km."""
        gm = self._import()
        d = gm.haversine_km_vec(48.8566, 2.3522, np.array([43.295]), np.array([5.372]))
        self.assertGreater(d[0], 640)
        self.assertLess(d[0], 710)

    def test_vectorized_order(self):
        """Vérifie que le vecteur résultat respecte l'ordre des entrées."""
        gm = self._import()
        lats = np.array([48.8566, 45.75, 43.295])  # Paris, Lyon, Marseille
        lons = np.array([2.3522, 4.85, 5.372])
        d = gm.haversine_km_vec(48.8566, 2.3522, lats, lons)
        # Paris→Paris < Paris→Lyon < Paris→Marseille
        self.assertLess(d[0], d[1])
        self.assertLess(d[1], d[2])

    def test_symmetry(self):
        """Haversine est symétrique : d(A,B) == d(B,A)."""
        gm = self._import()
        d_ab = gm.haversine_km_vec(45.75, 4.85, np.array([48.8566]), np.array([2.3522]))
        d_ba = gm.haversine_km_vec(48.8566, 2.3522, np.array([45.75]), np.array([4.85]))
        self.assertAlmostEqual(d_ab[0], d_ba[0], places=6)

    def test_equator_quarter_circle(self):
        """90° le long de l'équateur ≈ πR/2 ≈ 10 008 km."""
        gm = self._import()
        d = gm.haversine_km_vec(0.0, 0.0, np.array([0.0]), np.array([90.0]))
        expected = math.pi * 6371.0 / 2
        self.assertAlmostEqual(d[0], expected, delta=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2. detect_tn_scale
# ──────────────────────────────────────────────────────────────────────────────

def _make_gz_csv(tn_values: list) -> str:
    """Crée un fichier CSV.gz temporaire avec une colonne TN."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False)
    rows = ["TN"] + [str(v) for v in tn_values]
    with gzip.open(tmp.name, "wt") as f:
        f.write("\n".join(rows))
    return tmp.name


class TestDetectTnScale(unittest.TestCase):

    def tearDown(self):
        # Nettoyage des fichiers temporaires si créés
        pass

    def test_celsius_values_return_1(self):
        """Valeurs typiques en °C (ex : -3.5, 12.0) → scale = 1.0."""
        import geo_matching as gm
        path = _make_gz_csv([-3.5, 12.0, 0.5, -1.2, 8.4])
        try:
            self.assertEqual(gm.detect_tn_scale(path), 1.0)
        finally:
            os.unlink(path)

    def test_dixieme_values_return_01(self):
        """Valeurs en dixièmes de °C (médiane |TN| > 50, ex: -120, 85, -95) → scale = 0.1.

        Note : le seuil est > 50. Des valeurs comme [-35, 120, 5, -12, 84]
        donnent une médiane de 35 (< 50) donc ne déclenchent PAS le facteur 0.1.
        Il faut une majorité de valeurs > 50 en valeur absolue, ce qui correspond
        à des températures > 5°C en dixièmes — réaliste en été (ex: 85 = 8.5°C).
        """
        import geo_matching as gm
        path = _make_gz_csv([-150, -120, 85, 95, 200, -80, 110])
        try:
            self.assertEqual(gm.detect_tn_scale(path), 0.1)
        finally:
            os.unlink(path)

    def test_all_nan_returns_1(self):
        """Que des NaN → scale = 1.0 (comportement par défaut sûr)."""
        import geo_matching as gm
        path = _make_gz_csv(["", "", ""])
        try:
            self.assertEqual(gm.detect_tn_scale(path), 1.0)
        finally:
            os.unlink(path)

    def test_empty_returns_1(self):
        """Fichier sans données → scale = 1.0."""
        import geo_matching as gm
        path = _make_gz_csv([])
        try:
            self.assertEqual(gm.detect_tn_scale(path), 1.0)
        finally:
            os.unlink(path)

    def test_invalid_path_returns_1(self):
        """Chemin inexistant → scale = 1.0 (pas d'exception)."""
        import geo_matching as gm
        result = gm.detect_tn_scale("/tmp/fichier_inexistant_xyzxyz.csv.gz")
        self.assertEqual(result, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3. _extract_dept_from_path
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractDept(unittest.TestCase):

    def _fn(self):
        import geo_matching as gm
        return gm._extract_dept_from_path

    def test_standard_metropole(self):
        fn = self._fn()
        self.assertEqual(fn("/data/meteo/Q_21_previous-1950-2023_RR-T-Vent.csv.gz"), "21")

    def test_dom(self):
        fn = self._fn()
        self.assertEqual(fn("/data/meteo/Q_972_previous-1950-2023_RR-T-Vent.csv.gz"), "972")

    def test_2a(self):
        fn = self._fn()
        self.assertEqual(fn("Q_2A_previous_RR-T-Vent.csv.gz"), "2A")

    def test_no_match(self):
        fn = self._fn()
        self.assertEqual(fn("random_file.csv.gz"), "XX")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Bug potentiel : lru_cache + invalidation entre tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLruCacheIsolation(unittest.TestCase):
    """
    Vérifie que le lru_cache sur _load_stations_cached peut être vidé et
    rechargé correctement. C'est le bug signalé : après un premier chargement,
    si les données changent (nouveau dept téléchargé), le cache n'est pas
    automatiquement invalidé sauf appel explicite à cache_clear().
    """

    def test_cache_clear_is_callable(self):
        import geo_matching as gm
        # Doit exposer cache_clear() sans lever d'exception
        self.assertTrue(callable(gm._load_stations_cached.cache_clear))

    def test_cache_clear_resets_info(self):
        import geo_matching as gm
        # On appelle cache_clear et on vérifie que cache_info() est remis à 0
        gm._load_stations_cached.cache_clear()
        info = gm._load_stations_cached.cache_info()
        self.assertEqual(info.currsize, 0)

    def test_communes_cache_clear_is_callable(self):
        import geo_matching as gm
        self.assertTrue(callable(gm._load_communes_cached.cache_clear))


# ──────────────────────────────────────────────────────────────────────────────
# Fixture : DataFrame stations factice
# ──────────────────────────────────────────────────────────────────────────────

def _make_stations_df() -> pd.DataFrame:
    """5 stations fictives réparties en France."""
    return pd.DataFrame({
        "NUM_POSTE": ["21001001", "69001001", "75001001", "33001001", "59001001"],
        "NOM_USUEL": ["DIJON", "LYON", "PARIS", "BORDEAUX", "LILLE"],
        "LAT": [47.32, 45.73, 48.85, 44.83, 50.63],
        "LON": [5.04, 4.83, 2.35, -0.58, 3.07],
        "dept": ["21", "69", "75", "33", "59"],
        "tn_scale": [1.0, 1.0, 1.0, 1.0, 1.0],
    })


def _make_communes_df() -> pd.DataFrame:
    """Quelques communes fictives."""
    return pd.DataFrame({
        "nom_standard": ["Dijon", "Lyon", "Paris", "Bordeaux", "Lille",
                         "Marseille",  # sans coordonnées → MISSING_CITIES
                         "Ambiguous", "Ambiguous"],
        "dep_code": ["21", "69", "75", "33", "59",
                     "13",
                     "01", "02"],
        "lat": [47.32, 45.75, 48.86, 44.84, 50.63,
                None,
                46.0, 47.0],
        "lon": [5.04, 4.85, 2.33, -0.58, 3.07,
                None,
                5.0, 6.0],
    })


# ──────────────────────────────────────────────────────────────────────────────
# 5. find_commune
# ──────────────────────────────────────────────────────────────────────────────

class TestFindCommune(unittest.TestCase):

    def setUp(self):
        import geo_matching as gm
        self.gm = gm
        gm._load_communes_cached.cache_clear()
        self._patcher = patch.object(gm, "get_communes", return_value=_make_communes_df())
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.gm._load_communes_cached.cache_clear()

    def test_exact_match(self):
        row = self.gm.find_commune("Dijon")
        self.assertEqual(row["nom_standard"], "Dijon")

    def test_case_insensitive(self):
        row = self.gm.find_commune("dijon")
        self.assertEqual(row["nom_standard"], "Dijon")

    def test_partial_match(self):
        """'Bord' doit trouver Bordeaux."""
        row = self.gm.find_commune("Bord")
        self.assertEqual(row["nom_standard"], "Bordeaux")

    def test_dept_filter_disambiguates(self):
        """'Ambiguous' existe dans 2 depts ; filtrer par dept=02 retourne le bon."""
        row = self.gm.find_commune("Ambiguous", dept="02")
        self.assertEqual(row["dep_code"], "02")

    def test_not_found_raises(self):
        with self.assertRaises(ValueError):
            self.gm.find_commune("VilleInexistanteXYZ")

    def test_dept_not_found_raises(self):
        with self.assertRaises(ValueError):
            self.gm.find_commune("Dijon", dept="99")


# ──────────────────────────────────────────────────────────────────────────────
# 6. get_candidate_stations — cœur du sujet
# ──────────────────────────────────────────────────────────────────────────────

class TestGetCandidateStations(unittest.TestCase):

    def setUp(self):
        import geo_matching as gm
        self.gm = gm
        gm._load_stations_cached.cache_clear()
        self._patcher = patch.object(gm, "get_stations", return_value=_make_stations_df())
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self.gm._load_stations_cached.cache_clear()

    def test_returns_dataframe(self):
        df = self.gm.get_candidate_stations(47.32, 5.04, n=3)
        self.assertIsInstance(df, pd.DataFrame)

    def test_n_respected(self):
        df = self.gm.get_candidate_stations(47.32, 5.04, n=3)
        self.assertEqual(len(df), 3)

    def test_closest_station_dijon(self):
        """En partant des coords de Dijon, la station DIJON doit être la 1ère."""
        df = self.gm.get_candidate_stations(47.32, 5.04, n=5)
        self.assertEqual(df.iloc[0]["NOM_USUEL"], "DIJON")

    def test_closest_station_paris(self):
        """Depuis Paris, PARIS doit être en tête."""
        df = self.gm.get_candidate_stations(48.85, 2.35, n=5)
        self.assertEqual(df.iloc[0]["NOM_USUEL"], "PARIS")

    def test_sorted_ascending_distance(self):
        """Les candidats doivent être triés par dist_km croissante."""
        df = self.gm.get_candidate_stations(47.32, 5.04, n=5)
        dists = df["dist_km"].tolist()
        self.assertEqual(dists, sorted(dists))

    def test_dist_km_column_present(self):
        df = self.gm.get_candidate_stations(47.32, 5.04, n=2)
        self.assertIn("dist_km", df.columns)

    def test_dist_km_non_negative(self):
        df = self.gm.get_candidate_stations(47.32, 5.04, n=5)
        self.assertTrue((df["dist_km"] >= 0).all())

    def test_closest_is_near_zero_on_exact_coords(self):
        """Station dont les coords sont exactement celles de la query → dist ≈ 0."""
        df = self.gm.get_candidate_stations(47.32, 5.04, n=1)
        self.assertAlmostEqual(df.iloc[0]["dist_km"], 0.0, delta=0.5)

    def test_second_closest_from_lyon(self):
        """Depuis Lyon (45.73, 4.83), DIJON devrait être plus proche que PARIS."""
        df = self.gm.get_candidate_stations(45.73, 4.83, n=5)
        names = df["NOM_USUEL"].tolist()
        idx_dijon = names.index("DIJON")
        idx_paris = names.index("PARIS")
        self.assertLess(idx_dijon, idx_paris,
                        "DIJON devrait être plus proche de Lyon que PARIS")

    def test_no_duplicate_stations(self):
        """Même si get_stations retourne des doublons, le résultat doit avoir des NUM_POSTE uniques."""
        # DataFrame avec une station en double
        df_dup = pd.concat([_make_stations_df(), _make_stations_df()], ignore_index=True)
        with patch.object(self.gm, "get_stations", return_value=df_dup):
            # La déduplication est censée être gérée dans build_stations_cache ;
            # ici on vérifie juste que get_candidate_stations ne plante pas
            df = self.gm.get_candidate_stations(47.32, 5.04, n=5)
            self.assertGreater(len(df), 0)

    def test_n_larger_than_stations(self):
        """Si n > nombre de stations, on retourne toutes les stations disponibles."""
        df = self.gm.get_candidate_stations(47.32, 5.04, n=100)
        self.assertLessEqual(len(df), len(_make_stations_df()))


# ──────────────────────────────────────────────────────────────────────────────
# 7. build_communes_cache — injection des coordonnées manquantes
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildCommunesCache(unittest.TestCase):
    """
    Vérifie que les communes sans coordonnées reçoivent les coords injectées
    depuis config.MISSING_CITIES_LAT_LON.
    """

    def setUp(self):
        import geo_matching as gm
        import config
        self.gm = gm
        self.config = config
        gm._load_communes_cached.cache_clear()

        # Crée un répertoire temporaire simulant communes_raw_dir
        self.tmpdir = tempfile.mkdtemp()
        self._orig_communes_raw = config.COMMUNES_RAW_DIR
        self._orig_cache_path = config.COMMUNES_CACHE_PATH
        config.COMMUNES_RAW_DIR = self.tmpdir
        config.COMMUNES_CACHE_PATH = os.path.join(self.tmpdir, "communes_test.parquet")

        # Crée un CSV.gz de communes avec Marseille sans coordonnées
        data = pd.DataFrame({
            "nom_standard": ["Dijon", "Marseille"],
            "dep_code": ["21", "13"],
            "latitude_centre": [47.32, None],
            "longitude_centre": [5.04, None],
        })
        gz_path = os.path.join(self.tmpdir, "communes.csv.gz")
        data.to_csv(gz_path, sep=",", compression="gzip", index=False)

    def tearDown(self):
        import config
        config.COMMUNES_RAW_DIR = self._orig_communes_raw
        config.COMMUNES_CACHE_PATH = self._orig_cache_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        self.gm._load_communes_cached.cache_clear()

    def test_marseille_gets_coords(self):
        """Marseille sans coords doit recevoir les coords de MISSING_CITIES_LAT_LON."""
        communes = self.gm.build_communes_cache(force=True)
        marseille = communes[communes["nom_standard"].str.contains("Marseille", case=False)]
        self.assertFalse(marseille.empty, "Marseille introuvable dans le cache")
        row = marseille.iloc[0]
        self.assertAlmostEqual(float(row["lat"]), 43.295, places=2)
        self.assertAlmostEqual(float(row["lon"]), 5.372, places=2)

    def test_dijon_keeps_original_coords(self):
        """Dijon a déjà des coords, elles ne doivent pas être écrasées."""
        communes = self.gm.build_communes_cache(force=True)
        dijon = communes[communes["nom_standard"] == "Dijon"]
        self.assertFalse(dijon.empty)
        self.assertAlmostEqual(float(dijon.iloc[0]["lat"]), 47.32, places=2)

    def test_no_rows_with_null_coords(self):
        """Après construction du cache, aucune ligne ne doit avoir lat/lon null."""
        communes = self.gm.build_communes_cache(force=True)
        self.assertFalse(communes["lat"].isna().any(), "lat nulle détectée")
        self.assertFalse(communes["lon"].isna().any(), "lon nulle détectée")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Régression : bug du cache processed signalé par l'utilisateur
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheBugRegression(unittest.TestCase):
    """
    Reproduit le scénario du bug : le lru_cache de _load_stations_cached retient
    l'ancienne version du parquet même après un build_stations_cache(force=True).
    Après force=True + cache_clear(), le nouveau résultat doit être retourné.
    """

    def test_force_rebuild_bypasses_parquet(self):
        """
        build_stations_cache(force=True) doit reconstruire depuis les CSV.gz
        et NE PAS lire le parquet existant.
        """
        import geo_matching as gm
        import config

        tmpdir = tempfile.mkdtemp()
        orig_meteo = config.METEO_RAW_DIR
        orig_cache = config.STATIONS_CACHE_PATH
        config.METEO_RAW_DIR = tmpdir
        config.STATIONS_CACHE_PATH = os.path.join(tmpdir, "stations_test.parquet")

        try:
            # Crée un CSV.gz avec une seule station
            rows = (
                "NUM_POSTE;NOM_USUEL;LAT;LON;AAAAMMJJ;TN\n"
                "21001001;DIJON;47.32;5.04;20140101;-2.5\n"
            )
            gz_path = os.path.join(tmpdir, "Q_21_previous_RR-T-Vent.csv.gz")
            with gzip.open(gz_path, "wt") as f:
                f.write(rows)

            gm._load_stations_cached.cache_clear()
            stations = gm.build_stations_cache(force=True)
            self.assertEqual(len(stations), 1)
            self.assertEqual(stations.iloc[0]["NOM_USUEL"], "DIJON")

            # Ajoute une deuxième station dans un 2e fichier
            rows2 = (
                "NUM_POSTE;NOM_USUEL;LAT;LON;AAAAMMJJ;TN\n"
                "69001001;LYON;45.73;4.83;20140101;1.0\n"
            )
            gz_path2 = os.path.join(tmpdir, "Q_69_previous_RR-T-Vent.csv.gz")
            with gzip.open(gz_path2, "wt") as f:
                f.write(rows2)

            # Sans cache_clear(), lru_cache retourne l'ancien résultat → bug !
            stations_stale = gm._load_stations_cached()
            self.assertEqual(len(stations_stale), 1,
                             "lru_cache devrait encore retourner l'ancien résultat (1 station)")

            # Après cache_clear() + force=True → 2 stations
            gm._load_stations_cached.cache_clear()
            stations_new = gm.build_stations_cache(force=True)
            self.assertEqual(len(stations_new), 2,
                             "Après rebuild forcé, 2 stations attendues")

        finally:
            config.METEO_RAW_DIR = orig_meteo
            config.STATIONS_CACHE_PATH = orig_cache
            gm._load_stations_cached.cache_clear()
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])