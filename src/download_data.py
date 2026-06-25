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

Seule la tranche de période "1950 -> année-2" (ex: fichier
Q_{DEP}_previous-1950-2024_RR-T-Vent.csv.gz) est téléchargée : c'est la seule à
couvrir la période d'intérêt du défi (2014-2023). Les tranches "avant 1950"
(historique très ancien) et "latest-20XX-20YY" (deux dernières années glissantes)
sont volontairement exclues.

Usage
-----
    # Télécharger les communes + la météo (tranche 1950-20XX) pour les départements 13 et 75
    python src/download_data.py --depts 13 75

    # Télécharger pour toute la France métropolitaine (volumineux, ~plusieurs Go)
    python src/download_data.py --depts all

    # Ne télécharger que le référentiel des communes
    python src/download_data.py --communes-only

Le script interroge l'API data.gouv.fr à chaque exécution pour obtenir les URLs de
téléchargement à jour (le nom de la tranche "1950-20XX" change chaque année, par ex.
"1950-2024" deviendra "1950-2025").
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from tqdm import tqdm
except ImportError:  # fallback si tqdm n'est pas installé
    def tqdm(iterable, **kwargs):
        return iterable


HEADERS = {"User-Agent": "frost-days-defi/1.0 (+https://www.data.gouv.fr)"}
CONNECT_TIMEOUT = 15   # secondes pour établir la connexion
READ_TIMEOUT = 300     # secondes entre deux paquets reçus (fichiers volumineux + serveur parfois lent)
REQUEST_TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)
MAX_DOWNLOAD_RETRIES = 5


# --------------------------------------------------------------------------
# Utilitaires réseau
# --------------------------------------------------------------------------

def _get_json_with_retry(url: str, max_retries: int = 3, backoff: float = 2.0) -> dict:
    """GET une URL et renvoie le JSON, avec quelques tentatives en cas d'échec réseau."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            print(f"  [!] Tentative {attempt}/{max_retries} échouée ({exc}), nouvelle tentative...")
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Impossible de récupérer {url}") from last_exc


def _download_file(url: str, dest_path: str, force: bool = False) -> str:
    """
    Télécharge un fichier avec barre de progression, sauf s'il existe déjà.

    Gère :
    - la reprise (resume) d'un téléchargement partiel via l'en-tête HTTP Range,
      ce qui évite de tout recommencer si la connexion a été coupée ;
    - plusieurs tentatives en cas d'erreur réseau (timeout, connexion réinitialisée...),
      avec un backoff progressif ;
    - un timeout de lecture généreux (les fichiers météo "previous-*" peuvent peser
      plusieurs centaines de Mo et le serveur de data.gouv.fr est parfois lent à
      répondre entre deux paquets).
    """
    if os.path.exists(dest_path) and not force:
        size = os.path.getsize(dest_path)
        if size > 0:
            print(f"  [skip] {os.path.basename(dest_path)} déjà présent ({size / 1e6:.1f} Mo)")
            return dest_path

    tmp_path = dest_path + ".part"
    if force and os.path.exists(tmp_path):
        os.remove(tmp_path)

    last_exc = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        resume_from = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = dict(HEADERS)
        mode = "wb"
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"

        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT
            ) as resp:
                # 416 = on a déjà tout le fichier (la reprise demandait plus que la taille réelle)
                if resp.status_code == 416:
                    break
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()

                # Si le serveur ignore le Range et renvoie tout le fichier (200 au lieu de 206),
                # on doit repartir de zéro pour ne pas dupliquer le contenu déjà écrit.
                if resume_from > 0 and resp.status_code == 200:
                    resume_from = 0
                    mode = "wb"

                total = int(resp.headers.get("content-length", 0)) + resume_from
                with open(tmp_path, mode) as f, tqdm(
                    total=total or None, initial=resume_from, unit="B", unit_scale=True,
                    desc=os.path.basename(dest_path), leave=False,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            os.replace(tmp_path, dest_path)
            return dest_path

        except requests.RequestException as exc:
            last_exc = exc
            wait = min(60, 5 * attempt)
            print(
                f"  [!] Tentative {attempt}/{MAX_DOWNLOAD_RETRIES} échouée pour "
                f"{os.path.basename(dest_path)} ({exc}). Reprise dans {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Échec du téléchargement de {url} après {MAX_DOWNLOAD_RETRIES} tentatives"
    ) from last_exc


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
    Filtre les ressources RR-T-Vent pour les départements demandés, en ne gardant
    que la tranche de période "1950 -> année-2" (fichier nommé
    Q_{DEP}_previous-1950-20XX_RR-T-Vent.csv.gz côté URL, et dont le titre contient
    "1950-20XX"). C'est la seule tranche couvrant 2014-2023 ; on exclut donc
    volontairement :
      - la tranche historique avant 1950 (ex: "avant-1949"), trop ancienne ;
      - la tranche "latest-20XX-20YY" des deux dernières années, hors de notre période.

    Renvoie un dict {dept: [urls...]}. Avec ce filtre, chaque département a au plus
    un seul fichier.

    Le département et la période sont extraits du *titre* de la ressource (format
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
        if "departement_" not in title or "_periode_" not in title:
            continue

        # title ex: "QUOT_departement_13_periode_1950-2024_RR-T-Vent"
        try:
            after_dep = title.split("departement_", 1)[1]
            dep_code, rest = after_dep.split("_periode_", 1)
            periode = rest.split("_RR-T-Vent", 1)[0]  # ex: "1950-2024"
        except (IndexError, ValueError):
            continue

        # On ne garde que la tranche qui commence en 1950 (la seule à couvrir 2014-2023).
        # On exclut donc "avant-1949"/"1852-1949"/etc. (trop ancienne) et la tranche
        # glissante des deux dernières années (ex: "2025-2026"), qui ne commence pas en 1950.
        if not periode.startswith("1950-"):
            continue

        if dep_code in depts_set:
            matches[dep_code].append(url)

    return matches


def download_meteo_for_depts(
    depts: list[str], force: bool = False, max_workers: int = 5
) -> list[str]:
    """
    Télécharge les fichiers météo RR-T-Vent pour la liste de départements donnée.

    Les téléchargements (limités par le réseau, pas par le CPU) sont effectués en
    parallèle via un pool de threads (`max_workers` téléchargements simultanés).
    On limite volontairement le nombre de threads pour rester poli avec le serveur
    de data.gouv.fr et éviter de saturer la bande passante.

    Un échec de téléchargement sur un fichier (après épuisement des tentatives de
    reprise) n'interrompt pas le reste : il est consigné et le script continue avec
    les fichiers/départements suivants. Un récapitulatif des échecs est affiché à
    la fin pour permettre de relancer uniquement ce qui manque (le script reprend
    automatiquement les fichiers partiels grâce au cache + Range HTTP).
    """
    resources = list_meteo_resources()
    matches = filter_resources_for_depts(resources, depts)

    # On construit d'abord la liste des téléchargements à faire (un par fichier),
    # en notant au passage les départements sans aucune ressource.
    tasks = []  # (dep, filename, url, dest_path)
    not_found = []
    for dep in depts:
        urls = matches.get(dep, [])
        if not urls:
            not_found.append(dep)
            continue
        for url in urls:
            filename = url.rsplit("/", 1)[-1]
            dest_path = os.path.join(config.METEO_RAW_DIR, filename)
            tasks.append((dep, filename, url, dest_path))

    downloaded_paths = []
    failed = []

    # Chaque thread télécharge un fichier. `_download_file` est déjà autonome
    # (cache, reprise via Range HTTP, tentatives multiples), donc thread-safe :
    # chaque appel écrit dans son propre fichier ".part" / fichier de destination.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_download_file, url, dest_path, force=force): (dep, filename, url)
            for dep, filename, url, dest_path in tasks
        }
        for future in as_completed(future_to_task):
            dep, filename, url = future_to_task[future]
            try:
                downloaded_paths.append(future.result())
                print(f"  [OK] département {dep} : {filename}")
            except RuntimeError as exc:
                print(f"  [ÉCHEC] {filename} : {exc}")
                failed.append((dep, filename, url))

    if not_found:
        print(f"\n[!] Aucun fichier RR-T-Vent trouvé pour les départements : {not_found}")
        print("    Vérifiez le code département (ex: '2A'/'2B' pour la Corse, '971'-'988' pour les DOM).")

    if failed:
        print(f"\n[!] {len(failed)} fichier(s) n'ont pas pu être téléchargés après plusieurs tentatives :")
        for dep, filename, url in failed:
            print(f"    - dept {dep} : {filename}")
        print("    Relancez simplement la même commande : les fichiers déjà complets sont")
        print("    ignorés et les téléchargements interrompus reprennent où ils s'étaient arrêtés.")

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
    start_time = time.time()
    paths = download_meteo_for_depts(depts, force=args.force)
    print(f"[etl] Temps de téléchargement (multithread) : {time.time() - start_time:.2f}s")

    print("\n" + "=" * 70)
    print(f"[OK] {len(paths)} fichier(s) météo téléchargé(s) dans {config.METEO_RAW_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()