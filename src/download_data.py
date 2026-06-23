"""
download_data.py
=================

Télécharge les données nécessaires au projet Frost Days :

1. Le référentiel des communes françaises (coordonnées GPS, département, etc.)
2. Les fichiers météo quotidiens (RR-T-Vent) par département, depuis data.gouv.fr
   (organisation Météo-France).

Les fichiers météo sont volumineux (plusieurs centaines de Mo à quelques Go pour
toute la France) : par défaut, ce script ne télécharge QUE les départements demandés
en argument, et met en cache localement ce qui a déjà été récupéré.

Usage
-----
    # Télécharger les communes + la météo pour les départements 13 et 75
    python src/download_data.py --depts 13 75

    # Télécharger pour toute la France métropolitaine (très volumineux, ~plusieurs Go)
    python src/download_data.py --depts all

    # Ne télécharger que le référentiel des communes
    python src/download_data.py --communes-only

Le script interroge l'API data.gouv.fr à chaque exécution pour obtenir les URLs de
téléchargement à jour (elles changent au fil des mises à jour des jeux de données,
par ex. "latest-2023-2024" devient "latest-2025-2026" l'année suivante).
"""

import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from tqdm import tqdm
except ImportError:  # fallback si tqdm n'est pas installé
    def tqdm(iterable, **kwargs):
        return iterable


HEADERS = {"User-Agent": "frost-days-defi/1.0 (+https://www.data.gouv.fr)"}
TIMEOUT = 60


# --------------------------------------------------------------------------
# Utilitaires réseau
# --------------------------------------------------------------------------

def _get_json_with_retry(url: str, max_retries: int = 3, backoff: float = 2.0) -> dict:
    """GET une URL et renvoie le JSON, avec quelques tentatives en cas d'échec réseau."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            print(f"  [!] Tentative {attempt}/{max_retries} échouée ({exc}), nouvelle tentative...")
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Impossible de récupérer {url}") from last_exc


def _download_file(url: str, dest_path: str, force: bool = False) -> str:
    """Télécharge un fichier avec barre de progression, sauf s'il existe déjà."""
    if os.path.exists(dest_path) and not force:
        size = os.path.getsize(dest_path)
        if size > 0:
            print(f"  [skip] {os.path.basename(dest_path)} déjà présent ({size / 1e6:.1f} Mo)")
            return dest_path

    tmp_path = dest_path + ".part"
    with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=os.path.basename(dest_path), leave=False,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    os.replace(tmp_path, dest_path)
    return dest_path


# --------------------------------------------------------------------------
# Données météo (Météo-France via data.gouv.fr)
# --------------------------------------------------------------------------

def list_meteo_resources() -> list[dict]:
    """
    Interroge l'API v2 (paginée) de data.gouv.fr et renvoie la liste complète des
    ressources du dataset météo. Le dataset contient plusieurs centaines de
    ressources (toutes paginées par lots de `page_size`), donc on boucle sur les
    pages jusqu'à épuisement.
    """
    api_v2_base = f"https://www.data.gouv.fr/api/2/datasets/{config.METEO_DATASET_ID}/resources/"
    print(f"Interrogation de l'API data.gouv.fr (v2, paginée) ...")

    all_resources: list[dict] = []
    page = 1
    page_size = 100
    while True:
        url = f"{api_v2_base}?page={page}&page_size={page_size}&type=main"
        payload = _get_json_with_retry(url)
        data = payload.get("data", [])
        all_resources.extend(data)
        if len(data) < page_size:
            break  # dernière page atteinte
        page += 1

    print(f"  -> {len(all_resources)} ressources trouvées dans le dataset.")
    return all_resources


def filter_resources_for_depts(resources: list[dict], depts: list[str]) -> dict:
    """
    Filtre les ressources RR-T-Vent pour les départements demandés.

    Renvoie un dict {dept: [urls...]} (un département peut avoir plusieurs fichiers,
    un par tranche de période : avant-1950, 1950-2024, 2025-2026 (la tranche la plus
    récente est glissante et son nom change au fil des mises à jour), etc.)

    Le département est extrait du *titre* de la ressource (format
    "QUOT_departement_{DEP}_periode_{PERIODE}_RR-T-Vent"), plus fiable que de parser
    l'URL.
    """
    depts_set = set(depts)
    matches: dict[str, list[str]] = {d: [] for d in depts}

    for res in resources:
        url = res.get("url", "") or ""
        title = res.get("title", "") or ""
        if res.get("type") == "documentation":
            continue
        if "RR-T-Vent" not in title:
            continue
        if "departement_" not in title:
            continue

        # title ex: "QUOT_departement_13_periode_1950-2024_RR-T-Vent"
        try:
            dep_code = title.split("departement_", 1)[1].split("_periode_", 1)[0]
        except IndexError:
            continue

        if dep_code in depts_set:
            matches[dep_code].append(url)

    return matches


def download_meteo_for_depts(depts: list[str], force: bool = False) -> list[str]:
    """Télécharge les fichiers météo RR-T-Vent pour la liste de départements donnée."""
    resources = list_meteo_resources()
    matches = filter_resources_for_depts(resources, depts)

    downloaded_paths = []
    not_found = []
    for dep in depts:
        urls = matches.get(dep, [])
        if not urls:
            not_found.append(dep)
            continue
        for url in urls:
            filename = url.rsplit("/", 1)[-1]
            dest_path = os.path.join(config.METEO_RAW_DIR, filename)
            print(f"Téléchargement département {dep} : {filename}")
            _download_file(url, dest_path, force=force)
            downloaded_paths.append(dest_path)

    if not_found:
        print(f"\n[!] Aucun fichier RR-T-Vent trouvé pour les départements : {not_found}")
        print("    Vérifiez le code département (ex: '2A'/'2B' pour la Corse, '971'-'988' pour les DOM).")

    return downloaded_paths


# --------------------------------------------------------------------------
# Référentiel des communes
# --------------------------------------------------------------------------

def list_communes_resources() -> list[dict]:
    """Récupère (en paginant) toutes les ressources du dataset des communes."""
    api_v2_base = f"https://www.data.gouv.fr/api/2/datasets/{config.COMMUNES_DATASET_ID}/resources/"
    all_resources: list[dict] = []
    page = 1
    page_size = 50
    while True:
        url = f"{api_v2_base}?page={page}&page_size={page_size}"
        payload = _get_json_with_retry(url)
        data = payload.get("data", [])
        all_resources.extend(data)
        if len(data) < page_size:
            break
        page += 1
    return all_resources


def download_communes(force: bool = False) -> str:
    """Télécharge le référentiel des communes françaises (CSV)."""
    print(f"Interrogation de l'API data.gouv.fr pour le référentiel des communes...")
    resources = list_communes_resources()

    # On cherche le csv.gz "principal" sans polygone (le plus léger, suffisant pour nos coords)
    candidates = [
        r for r in resources
        if r.get("url", "").endswith(".csv.gz")
        and r.get("type") != "documentation"
        and "polygone" not in (r.get("title") or "").lower()
        and "geographie" not in (r.get("title") or "").lower()
    ]

    if not candidates:
        # repli : n'importe quel csv.gz
        candidates = [
            r for r in resources
            if r.get("url", "").endswith(".csv.gz") and r.get("type") != "documentation"
        ]

    if not candidates:
        raise RuntimeError("Aucune ressource csv.gz trouvée pour le dataset des communes.")

    # On privilégie le fichier le plus récent (souvent le premier listé par l'API,
    # mais on trie par last_modified pour être sûr)
    candidates.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
    chosen = candidates[0]
    url = chosen["url"]
    filename = url.rsplit("/", 1)[-1]
    dest_path = os.path.join(config.COMMUNES_RAW_DIR, filename)

    print(f"  -> Fichier retenu : {chosen.get('title', filename)}")
    _download_file(url, dest_path, force=force)
    return dest_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_depts_arg(raw: list[str]) -> list[str]:
    """Transforme l'argument --depts (ex: ['13', '75'] ou ['all']) en liste de codes."""
    if len(raw) == 1 and raw[0].lower() == "all":
        return config.ALL_DEPTS
    if len(raw) == 1 and raw[0].lower() == "metropole":
        return config.METROPOLE_DEPTS
    # normalise les codes sur 2 caractères pour la métropole (ex: "1" -> "01")
    normalized = []
    for d in raw:
        d = d.strip().upper()
        if d.isdigit() and len(d) == 1:
            d = f"0{d}"
        normalized.append(d)
    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge les données météo (Météo-France) et le référentiel des communes."
    )
    parser.add_argument(
        "--depts", nargs="+", default=None,
        help="Liste de départements à télécharger (ex: 13 75 2A), 'all' pour tout, "
             "'metropole' pour la métropole uniquement.",
    )
    parser.add_argument(
        "--communes-only", action="store_true",
        help="Ne télécharge que le référentiel des communes.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-télécharge même si les fichiers existent déjà localement.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Téléchargement des données — Défi Frost Days")
    print("=" * 70)

    # 1. Communes (toujours, sauf désactivation explicite future)
    communes_path = download_communes(force=args.force)
    print(f"[OK] Référentiel des communes : {communes_path}\n")

    if args.communes_only:
        return

    # 2. Météo
    if not args.depts:
        print("Aucun département spécifié (--depts). Rien à télécharger côté météo.")
        print("Exemple : python src/download_data.py --depts 13 75")
        return

    depts = parse_depts_arg(args.depts)
    print(f"Départements demandés : {depts}\n")
    paths = download_meteo_for_depts(depts, force=args.force)

    print("\n" + "=" * 70)
    print(f"[OK] {len(paths)} fichier(s) météo téléchargé(s) dans {config.METEO_RAW_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()