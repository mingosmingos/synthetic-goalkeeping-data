import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

from data.loaders import load_shots, load_players

st.title("Guarda-redes (GK)")

# --- Load data ---
ROOT = Path(__file__).parent.parent
players_attr_path = ROOT / "Synthetic Data.xlsx"

@st.cache_data
def load_player_attributes(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df[[
        "player_id",
        "name",
        "reflexes",
        "agility",
        "presence",
        "flexibility",
    ]]

attr_df = load_player_attributes(players_attr_path)
perf_df = load_players()  # player_id, saved, conceded

# Merge to have everything numérico por GK
perf_df = perf_df.copy()
perf_df["shots_faced"] = perf_df["saved"] + perf_df["conceded"]
perf_df["save_pct"] = perf_df["saved"] / perf_df["shots_faced"]
full_df = attr_df.merge(perf_df, on="player_id", how="left", suffixes=("", "_perf"))

# Small table, escondida por defeito
with st.expander("Ver tabela completa de guarda-redes"):
    st.dataframe(full_df, use_container_width=True)

# --- Selecção de jogadores (2 dropdowns) ---
player_options = full_df["player_id"].tolist()
labels = {pid: f"GK {pid} - {row['name']}" for pid, row in full_df.set_index("player_id").iterrows()}

col1, col2 = st.columns(2)
with col1:
    gk1 = st.selectbox(
        "Jogador 1 (GK)",
        options=player_options,
        index=0 if len(player_options) > 0 else None,
        format_func=lambda pid: labels.get(pid, str(pid)),
    )
with col2:
    default_idx = 1 if len(player_options) > 1 else 0
    gk2 = st.selectbox(
        "Jogador 2 (GK)",
        options=player_options,
        index=default_idx,
        format_func=lambda pid: labels.get(pid, str(pid)),
    )

selected_ids = [gk1, gk2] if gk1 != gk2 else [gk1]

# --- Radar 1: Atributos técnicos ---
attr_metrics = ["reflexes", "agility", "presence", "flexibility"]
attr_labels = ["Reflexos", "Agilidade", "Presença", "Flexibilidade"]

colors = ["#9abfe7", "#C0E742", "#2cc199"]  # azul claro / azul / verde

fig_attr = go.Figure()

for idx, pid in enumerate(selected_ids):
    row = full_df[full_df["player_id"] == pid].iloc[0]
    values = [row[m] for m in attr_metrics]
    values += values[:1]

    fig_attr.add_trace(
        go.Scatterpolar(
            r=values,
            theta=attr_labels + [attr_labels[0]],
            name=f"GK {pid}",
            fill="toself",
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.6,
        )
    )

fig_attr.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100]),
    ),
    showlegend=True,
    template="plotly_dark",
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig_attr, use_container_width=True)


# --- Radar 2: Performance (defesas / golos / remates / %defesa) ---
perf_metrics = ["saved", "conceded", "shots_faced", "save_pct"]
perf_labels = ["Defesas", "Golos sofridos", "Remates sofridos", "Taxa de defesa"]

norm_df = full_df.set_index("player_id")[perf_metrics].copy()
for col in ["saved", "conceded", "shots_faced"]:
    col_max = norm_df[col].max()
    if col_max and col_max > 0:
        norm_df[col] = norm_df[col] / col_max

fig_perf = go.Figure()

for idx, pid in enumerate(selected_ids):
    row = norm_df.loc[pid]
    values = [row[m] for m in perf_metrics]
    values += values[:1]

    fig_perf.add_trace(
        go.Scatterpolar(
            r=values,
            theta=perf_labels + [perf_labels[0]],
            name=f"GK {pid}",
            fill="toself",
            line=dict(color=colors[idx % len(colors)], width=2),
            opacity=0.6,
        )
    )

fig_perf.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1]),
    ),
    showlegend=True,
    template="plotly_dark",
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig_perf, use_container_width=True)
