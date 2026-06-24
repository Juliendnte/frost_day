"""
app.py — Interface Streamlit pour le calculateur de jours de gel
"""

from __future__ import annotations

import glob
import os
import sys
import threading
import time
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Chemin vers src/ ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import download_data as dl
import frost_calculator as fc
import geo_matching as gm

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jours de Gel",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ---- palette ---- */
    :root {
        --ice:    #E8F4FD;
        --frost:  #1565C0;
        --sky:    #42A5F5;
        --cold:   #0D2B4E;
        --neutral:#F7FAFC;
        --muted:  #6B7E8C;
        --ok:     #2E7D32;
        --warn:   #E65100;
        --card:   #FFFFFF;
    }

    /* ---- global ---- */
    body, .stApp {
        background: linear-gradient(160deg, #EBF4FD 0%, #F7FAFC 100%);
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--cold);
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] {
        background: var(--cold) !important;
        border-right: 1px solid #1A3A5C;
    }
    section[data-testid="stSidebar"] * { color: #D6E8F7 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stDateInput label {
        font-weight: 600;
        font-size: 0.82rem;
        letter-spacing: .05em;
        text-transform: uppercase;
    }

    /* ---- hero ---- */
    .hero {
        background: linear-gradient(120deg, var(--cold) 0%, #1B4F87 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -.02em; }
    .hero p  { font-size: 1rem; opacity: .75; margin: .4rem 0 0; }

    /* ---- KPI cards ---- */
    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        flex: 1;
        background: var(--card);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 8px rgba(21,101,192,.10);
        border-top: 4px solid var(--sky);
    }
    .kpi-card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .07em;
                       color: var(--muted); margin-bottom: .3rem; }
    .kpi-card .value { font-size: 2rem; font-weight: 800; color: var(--frost); line-height: 1; }
    .kpi-card .sub   { font-size: .82rem; color: var(--muted); margin-top: .2rem; }

    /* ---- station badge ---- */
    .station-badge {
        background: var(--ice);
        border: 1px solid #BBDEFB;
        border-radius: 8px;
        padding: .6rem 1rem;
        font-size: .85rem;
        margin-bottom: 1rem;
        color: var(--cold);
    }
    .station-badge strong { color: var(--frost); }

    /* ---- section header ---- */
    .section-title {
        font-size: 1rem; font-weight: 700; letter-spacing: .04em;
        text-transform: uppercase; color: var(--frost);
        border-left: 4px solid var(--sky); padding-left: .7rem;
        margin: 1.8rem 0 .8rem;
    }

    /* ---- download info ---- */
    .dl-info {
        background: #FFF3E0; border-left: 4px solid var(--warn);
        border-radius: 6px; padding: .8rem 1rem;
        font-size: .88rem; color: #5D2E00; margin-bottom: 1rem;
    }
    .dl-ok {
        background: #E8F5E9; border-left: 4px solid var(--ok);
        border-radius: 6px; padding: .8rem 1rem;
        font-size: .88rem; color: #1B5E20; margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def dept_meteo_file_exists(dept: str) -> bool:
    """Retourne True si le fichier météo pour ce département est déjà téléchargé."""
    pattern = os.path.join(config.METEO_RAW_DIR, f"Q_{dept}_*RR-T-Vent*.csv.gz")
    return bool(glob.glob(pattern))


def communes_file_exists() -> bool:
    pattern = os.path.join(config.COMMUNES_RAW_DIR, "*.csv.gz")
    return bool(glob.glob(pattern))


@st.cache_data(show_spinner=False)
def get_communes_list() -> list[str]:
    """Retourne la liste triée des noms de communes disponibles."""
    try:
        df = gm.get_communes()
        return sorted(df["nom_standard"].dropna().unique().tolist())
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_dept_list() -> list[str]:
    """Retourne la liste des codes de département."""
    return config.METROPOLE_DEPTS + config.DOM_DEPTS


# ── Sidebar : formulaire de saisie ──────────────────────────────────────────

with st.sidebar:
    st.markdown("## ❄️ Frost Days")
    st.markdown("---")

    commune_input = st.text_input(
        "Commune",
        placeholder="ex : Oraison",
        help="Nom de la commune. En cas d'homonymie, précisez le département.",
    )

    dept_options = [""] + config.METROPOLE_DEPTS + config.DOM_DEPTS
    dept_input = st.selectbox(
        "Département",
        options=dept_options,
        format_func=lambda d: d if d else "— Tous —",
    )

    st.markdown("**Plage de dates**")
    col_a, col_b = st.columns(2)
    with col_a:
        start_input = st.date_input(
            "Du",
            value=date(2014, 1, 1),
            min_value=date(1950, 1, 1),
            max_value=date(2024, 12, 31),
        )
    with col_b:
        end_input = st.date_input(
            "Au",
            value=date(2023, 12, 31),
            min_value=date(1950, 1, 1),
            max_value=date(2024, 12, 31),
        )

    st.markdown("---")
    run_btn = st.button("🔍 Calculer", use_container_width=True, type="primary")

    st.markdown("---")
    with st.expander("⚙️ Options avancées"):
        n_stations = st.slider(
            "Stations candidates (max)",
            min_value=1, max_value=20,
            value=config.NUM_NEAREST_STATIONS,
            help="Nombre de stations météo les plus proches à examiner.",
        )
        max_missing = st.slider(
            "Seuil de données manquantes (%)",
            min_value=5, max_value=80,
            value=int(config.MAX_MISSING_PERCENT),
            help="Une station avec plus de X % de données manquantes est exclue.",
        )

# ── Hero ────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero">
  <h1>❄️ Jours de Gel</h1>
  <p>Statistiques de gel pour n'importe quelle commune française — données Météo-France (data.gouv.fr)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Logique principale ───────────────────────────────────────────────────────

if not run_btn:
    # État d'accueil
    st.info(
        "👈 Renseignez une commune et une plage de dates dans le panneau de gauche, "
        "puis cliquez sur **Calculer**.",
        icon="💡",
    )
    st.stop()

# Validation minimale
if not commune_input.strip():
    st.error("Veuillez saisir un nom de commune.")
    st.stop()

if start_input >= end_input:
    st.error("La date de début doit être antérieure à la date de fin.")
    st.stop()

dept_norm = dept_input.strip() if dept_input else None

# ── Téléchargement automatique si nécessaire ────────────────────────────────

needs_communes = not communes_file_exists()
needs_meteo = dept_norm and not dept_meteo_file_exists(dept_norm)

if needs_communes or needs_meteo:
    missing_parts = []
    if needs_communes:
        missing_parts.append("le référentiel des communes")
    if needs_meteo:
        missing_parts.append(f"les données météo du département **{dept_norm}**")

    st.markdown(
        f"<div class='dl-info'>📥 Téléchargement automatique de {' et '.join(missing_parts)} "
        f"depuis data.gouv.fr…</div>",
        unsafe_allow_html=True,
    )

    download_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_box = st.empty()
    log_lines: list[str] = []

    def download_thread():
        if needs_communes:
            log_lines.append("⬇️ Communes en cours…")
            try:
                dl.download_communes()
                log_lines.append("✅ Communes téléchargées.")
            except Exception as e:
                log_lines.append(f"❌ Erreur communes : {e}")

        if needs_meteo and dept_norm:
            log_lines.append(f"⬇️ Météo dept {dept_norm} en cours (peut prendre plusieurs minutes)…")
            try:
                dl.download_meteo_for_depts([dept_norm])
                log_lines.append(f"✅ Météo dept {dept_norm} téléchargée.")
            except Exception as e:
                log_lines.append(f"❌ Erreur météo dept {dept_norm} : {e}")

    t = threading.Thread(target=download_thread, daemon=True)
    t.start()

    steps = 0
    while t.is_alive():
        steps = (steps + 1) % 100
        progress_bar.progress(steps)
        if log_lines:
            log_box.markdown("\n\n".join(log_lines))
        time.sleep(0.3)

    t.join()
    progress_bar.progress(100)
    if log_lines:
        log_box.markdown("\n\n".join(log_lines))

    if any("❌" in l for l in log_lines):
        st.error("Le téléchargement a rencontré une erreur. Consultez les logs ci-dessus.")
        st.stop()

    st.markdown(
        "<div class='dl-ok'>✅ Données disponibles localement.</div>",
        unsafe_allow_html=True,
    )
    # Invalider le cache communes si on vient de les télécharger
    get_communes_list.clear()

# ── Calcul ───────────────────────────────────────────────────────────────────

try:
    meteo_file = fc.find_meteo_file(gm.find_commune(commune_input.strip(), dept_norm))
except ValueError as exc:
    st.error(f"❌ Commune introuvable : {exc}")
    st.stop()

on_missing_dept = None if meteo_file else dl.download_meteo_for_depts

with st.spinner("Calcul des jours de gel en cours…"):
    try:
        result = fc.compute_frost_days(
            commune_name=commune_input.strip(),
            dept=dept_norm,
            start_date=str(start_input),
            end_date=str(end_input),
            n_candidates=n_stations,
            max_missing_pct=float(max_missing),
            verbose=True,
            on_missing_dept=on_missing_dept,
        )
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

# ── Résultats ─────────────────────────────────────────────────────────────────

nb_years = result.end_date.year - result.start_date.year + 1

# -- Station badge --
st.markdown(
    f"<div class='station-badge'>"
    f"📡 Station retenue : <strong>{result.station_name}</strong> "
    f"(n° {result.station_num}) — "
    f"{result.dist_km:.1f} km de <strong>{result.commune_name}</strong> "
    f"(dépt {result.dept})"
    f"</div>",
    unsafe_allow_html=True,
)

# -- KPI cards --
st.markdown(
    f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="label">Jours de gel total</div>
    <div class="value">{result.total_frost_days}</div>
    <div class="sub">{result.start_date.date()} → {result.end_date.date()}</div>
  </div>
  <div class="kpi-card">
    <div class="label">Moyenne annuelle</div>
    <div class="value">{result.avg_frost_days_per_year:.1f}</div>
    <div class="sub">jours de gel par an</div>
  </div>
  <div class="kpi-card">
    <div class="label">Période couverte</div>
    <div class="value">{nb_years}</div>
    <div class="sub">années dans la plage sélectionnée</div>
  </div>
  <div class="kpi-card">
    <div class="label">Station la plus proche</div>
    <div class="value">{result.dist_km:.0f} km</div>
    <div class="sub">{result.station_name}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Graphiques ────────────────────────────────────────────────────────────────

daily = result.daily_stats.copy()

if daily.empty:
    st.warning("Aucune statistique journalière disponible pour cette période.")
    st.stop()

daily = daily.reset_index()  # "day_of_year" comme colonne
daily["frost_pct"] = (daily["frost_rate"] * 100).round(1)

# Ajouter une colonne mois pour le tri et la couleur
daily["month_num"] = daily["day_of_year"].str[:2].astype(int)
daily["month_label"] = pd.to_datetime(
    daily["day_of_year"], format="%m-%d", errors="coerce"
).dt.strftime("%b")

MONTH_COLORS = {
    1: "#2979FF", 2: "#448AFF", 3: "#40C4FF",
    4: "#80DEEA", 5: "#A5D6A7", 6: "#FFF59D",
    7: "#FFCC80", 8: "#FFAB40", 9: "#FF8A65",
    10: "#F48FB1", 11: "#CE93D8", 12: "#90CAF9",
}
daily["color"] = daily["month_num"].map(MONTH_COLORS)

# ---- Graphique 1 : Taux de gel par jour de l'année ----
st.markdown("<div class='section-title'>Risque de gel par jour de l'année</div>", unsafe_allow_html=True)

fig_rate = px.bar(
    daily.dropna(subset=["frost_pct"]),
    x="day_of_year",
    y="frost_pct",
    color="month_num",
    color_continuous_scale=[
        [0,   "#2979FF"],
        [0.25,"#80DEEA"],
        [0.5, "#FFF59D"],
        [0.75,"#FF8A65"],
        [1,   "#2979FF"],
    ],
    labels={"day_of_year": "Jour de l'année", "frost_pct": "Risque de gel (%)", "month_num": "Mois"},
    hover_data={"frost_count": True, "years_present": True},
    custom_data=["frost_count", "years_present"],
)
fig_rate.update_traces(
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Risque : <b>%{y:.1f}%</b><br>"
        "Jours gelés : %{customdata[0]} / %{customdata[1]} ans"
        "<extra></extra>"
    )
)
fig_rate.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, Segoe UI, sans-serif",
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=10, b=0),
    height=320,
    xaxis=dict(
        tickmode="array",
        tickvals=["01-01","02-01","03-01","04-01","05-01","06-01",
                  "07-01","08-01","09-01","10-01","11-01","12-01"],
        ticktext=["Jan","Fév","Mar","Avr","Mai","Juin",
                  "Juil","Août","Sep","Oct","Nov","Déc"],
        gridcolor="#E3EDF7",
    ),
    yaxis=dict(gridcolor="#E3EDF7", ticksuffix="%"),
)
st.plotly_chart(fig_rate, use_container_width=True)

# ---- Graphique 2 : Carte thermique mensuelle ----
st.markdown("<div class='section-title'>Carte thermique — Risque de gel par mois</div>", unsafe_allow_html=True)

# Construire un pivot : lignes = jours du mois (1–31), colonnes = mois
daily["day_num"] = daily["day_of_year"].str[3:].astype(int)

pivot = daily.pivot_table(
    index="day_num", columns="month_num", values="frost_pct", aggfunc="mean"
)
pivot.index.name = "Jour"
pivot.columns = ["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"]

fig_hm = go.Figure(
    go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[
            [0.0,  "#FFFFFF"],
            [0.01, "#E3F2FD"],
            [0.3,  "#64B5F6"],
            [0.6,  "#1565C0"],
            [1.0,  "#0D2B4E"],
        ],
        zmin=0, zmax=100,
        hovertemplate="<b>%{x} — jour %{y}</b><br>Risque : %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Risque (%)", ticksuffix="%"),
    )
)
fig_hm.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, Segoe UI, sans-serif",
    margin=dict(l=0, r=0, t=10, b=0),
    height=380,
    yaxis=dict(autorange="reversed", title="Jour du mois", gridcolor="#E3EDF7"),
    xaxis=dict(title="Mois"),
)
st.plotly_chart(fig_hm, use_container_width=True)

# ---- Graphique 3 : Nombre absolu de jours de gel par mois ----
st.markdown("<div class='section-title'>Jours de gel totaux par mois</div>", unsafe_allow_html=True)

monthly = (
    daily.groupby(["month_num", "month_label"])["frost_count"]
    .sum()
    .reset_index()
    .sort_values("month_num")
)

fig_monthly = px.bar(
    monthly,
    x="month_label",
    y="frost_count",
    color="frost_count",
    color_continuous_scale=["#E3F2FD", "#1565C0", "#0D2B4E"],
    labels={"month_label": "Mois", "frost_count": "Jours de gel (total)"},
)
fig_monthly.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, Segoe UI, sans-serif",
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=0, r=0, t=10, b=0),
    height=280,
    yaxis=dict(gridcolor="#E3EDF7"),
    xaxis=dict(categoryorder="array",
               categoryarray=["Jan","Fév","Mar","Avr","Mai","Juin",
                               "Juil","Août","Sep","Oct","Nov","Déc"]),
)
st.plotly_chart(fig_monthly, use_container_width=True)

# ── Tableau détaillé ──────────────────────────────────────────────────────────

with st.expander("📋 Tableau détaillé jour par jour"):
    display_df = daily[["day_of_year", "frost_count", "years_present", "frost_pct"]].copy()
    display_df.columns = ["Jour (MM-JJ)", "Nb jours gelés", "Années présentes", "Risque (%)"]
    st.dataframe(
        display_df.style.background_gradient(
            subset=["Risque (%)"],
            cmap="Blues",
            vmin=0,
            vmax=100,
        ).format({"Risque (%)": "{:.1f}"}),
        use_container_width=True,
        height=400,
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Données : [Météo-France via data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) · "
    f"Seuil de gel : TN ≤ {config.FROST_THRESHOLD_C} °C · "
    "Stations exclues si > 35% de données manquantes"
)