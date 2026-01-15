import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_injuries

st.title("Lesões")


BODY_REGIONS = ["head", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]


def map_body_part_to_region(body_part: str) -> str:
    """Mapeia o texto de body_part para uma região simplificada do corpo.

    O objetivo não é ser perfeito clinicamente, mas agrupar em
    cabeça, tronco, braços e pernas para o boneco.
    """

    if not isinstance(body_part, str):
        return "torso"

    text = body_part.lower()

    # Cabeça
    if any(k in text for k in ["head", "cabe", "face", "concuss"]):
        return "head"

    # Braços
    if any(k in text for k in ["ombro", "shoulder", "arm", "cotovelo", "elbow", "pulso", "wrist", "mao", "mão", "hand"]):
        if any(k in text for k in ["left", "esq"]):
            return "left_arm"
        if any(k in text for k in ["right", "dir"]):
            return "right_arm"
        # Se não der para distinguir, contam nos dois lados de forma igual depois
        return "arms"

    # Pernas
    if any(k in text for k in ["leg", "coxa", "thigh", "joelho", "knee", "tornozelo", "ankle", "pé", "pe", "foot", "calf"]):
        if any(k in text for k in ["left", "esq"]):
            return "left_leg"
        if any(k in text for k in ["right", "dir"]):
            return "right_leg"
        return "legs"

    # Costas / tronco
    if any(k in text for k in ["back", "costas", "spine", "coluna", "peito", "chest", "abd", "torso"]):
        return "torso"

    return "torso"


def build_body_figure(region_counts: dict) -> go.Figure:
    """Cria um boneco simples em Plotly com cores por região.

    region_counts deve ter chaves em BODY_REGIONS e valores inteiros
    (número de lesões nessa região).
    """

    # Escala de cor simples: sem lesões = cinzento claro, com lesões = laranja
    max_count = max(region_counts.values()) if region_counts else 0

    def color_for(count: int) -> str:
        if count <= 0:
            return "#f2f2f2"
        # Normaliza para uma paleta de laranjas
        palette = px.colors.sequential.Oranges
        idx = min(len(palette) - 1, max(1, int((count / max(max_count, 1)) * (len(palette) - 1))))
        return palette[idx]

    shapes = []
    annotations = []

    def add_region(x0, x1, y0, y1, key, label):
        count = region_counts.get(key, 0)
        shapes.append(
            dict(
                type="rect",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                line=dict(color="#555555"),
                fillcolor=color_for(count),
            )
        )
        text = f"{label}<br>{count} lesões" if count > 0 else label
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=text,
                showarrow=False,
                font=dict(size=10, color="#000000"),
                align="center",
            )
        )

    # Coordenadas de um boneco estilizado
    add_region(1.25, 2.75, 4.2, 5.2, "head", "Cabeça")  # cabeça
    add_region(1.0, 3.0, 2.2, 4.2, "torso", "Tronco")  # tronco
    add_region(0.0, 1.0, 2.3, 4.0, "left_arm", "Braço Esq.")  # braço esq
    add_region(3.0, 4.0, 2.3, 4.0, "right_arm", "Braço Dir.")  # braço dir
    add_region(1.3, 2.0, 0.0, 2.2, "left_leg", "Perna Esq.")  # perna esq
    add_region(2.0, 2.7, 0.0, 2.2, "right_leg", "Perna Dir.")  # perna dir

    fig = go.Figure()
    fig.update_layout(
        width=320,
        height=520,
        xaxis=dict(visible=False, range=[-0.5, 4.5]),
        yaxis=dict(visible=False, range=[-0.5, 5.8]),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

try:
    injuries_df = load_injuries()
except Exception as e:
    st.error(
        "Não foi possível carregar a sheet 'Injuries' do data/output.xlsx. "
        "Corre a secção 'Injuries Generation' no Data Generation.ipynb primeiro."
    )
    st.exception(e)
    st.stop()

if injuries_df.empty:
    st.info("Não há lesões geradas (Injuries está vazio).")
    st.dataframe(injuries_df, use_container_width=True)
    st.stop()

# Opcional: se o dataset tiver poucas lesões / partes do corpo,
# criamos alguns exemplos adicionais sintéticos só para visualização
if injuries_df["body_part"].nunique(dropna=False) <= 2:
    max_id = injuries_df["injury_id"].max() if "injury_id" in injuries_df.columns else 0
    max_id = int(max_id) if pd.notna(max_id) else 0

    base_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=120)

    extra = [
        # player_id, injury_type, body_part, severity, start_offset_days, duration
        (1, "Concussion", "Head", "short", 10, 7),
        (1, "Shoulder strain", "Shoulder", "short", 40, 14),
        (3, "Back spasm", "Back", "long", 60, 30),
        (3, "Wrist sprain", "Wrist", "short", 80, 10),
        (4, "Chest contusion", "Chest", "short", 20, 5),
        (4, "Thigh strain", "Upper Leg", "long", 90, 21),
    ]

    rows = []
    for i, (player_id, injury_type, body_part, severity, offset, duration) in enumerate(extra, start=1):
        start_date = base_start + pd.Timedelta(days=offset)
        end_date = start_date + pd.Timedelta(days=duration)
        rows.append(
            {
                "injury_id": max_id + i,
                "player_id": player_id,
                "injury_type": injury_type,
                "body_part": body_part,
                "severity": severity,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": duration,
                "recurrent": False,
                "recurrence_of": pd.NA,
            }
        )

    injuries_df = pd.concat([injuries_df, pd.DataFrame(rows)], ignore_index=True)

# Parse dates defensivamente
for col in ["start_date", "end_date"]:
    if col in injuries_df.columns:
        injuries_df[col] = pd.to_datetime(injuries_df[col], errors="coerce")

# Sidebar filters
with st.sidebar:
    st.header("Filtros")

    player_ids = sorted(injuries_df["player_id"].dropna().unique().tolist())
    selected_players = st.multiselect(
        "Guarda-redes",
        options=player_ids,
        default=player_ids,
        format_func=lambda pid: f"GK {int(pid)}",
    )

    severities = ["All"] + sorted(injuries_df["severity"].dropna().unique().tolist())
    severity = st.selectbox("Severidade", severities, index=0)

    body_parts = ["All"] + sorted(injuries_df["body_part"].dropna().unique().tolist())
    body_part = st.selectbox("Parte do corpo", body_parts, index=0)

    # Date range
    min_date = injuries_df["start_date"].min()
    max_date = injuries_df["end_date"].max()
    date_range = st.date_input(
        "Período",
        value=(min_date.date() if pd.notna(min_date) else None, max_date.date() if pd.notna(max_date) else None),
    )

filtered = injuries_df.copy()

if selected_players:
    filtered = filtered[filtered["player_id"].isin(selected_players)]

if severity != "All":
    filtered = filtered[filtered["severity"] == severity]

if body_part != "All":
    filtered = filtered[filtered["body_part"] == body_part]

if isinstance(date_range, tuple) and len(date_range) == 2 and all(date_range):
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    # overlap filter: [start_date,end_date] intersects [start,end]
    filtered = filtered[(filtered["start_date"] <= end) & (filtered["end_date"] >= start)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Lesões", int(len(filtered)))
with col2:
    st.metric("Recorrentes", int(filtered.get("recurrent", False).sum()) if "recurrent" in filtered else 0)
with col3:
    total_days = int(filtered.get("duration_days", pd.Series([0] * len(filtered))).fillna(0).sum())
    st.metric("Dias fora (total)", total_days)
with col4:
    # Current injured in filtered scope
    today = pd.Timestamp.today().normalize()
    current = filtered[(filtered["start_date"] <= today) & (filtered["end_date"] >= today)]
    st.metric("Atualmente lesionados", int(current["player_id"].nunique()))

st.subheader("Timeline de lesões")
if not filtered.empty:
    timeline = filtered.copy()
    timeline["player"] = timeline["player_id"].apply(lambda x: f"GK {int(x)}")
    timeline["label"] = timeline.apply(
        lambda r: f"{r.get('injury_type','')} · {r.get('body_part','')}" + (" (rec.)" if bool(r.get("recurrent", False)) else ""),
        axis=1,
    )

    fig = px.timeline(
        timeline,
        x_start="start_date",
        x_end="end_date",
        y="player",
        color="severity" if "severity" in timeline.columns else None,
        hover_name="label",
        hover_data={
            "injury_id": True,
            "body_part": True,
            "injury_type": True,
            "duration_days": True,
            "recurrent": True,
            "recurrence_of": True,
        },
        template="plotly_dark",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem lesões com estes filtros.")

st.subheader("Recorrência por parte do corpo")
if not filtered.empty:
    by_part = (
        filtered.groupby(["player_id", "body_part"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    by_part["player"] = by_part["player_id"].apply(lambda x: f"GK {int(x)}")

    heat = px.density_heatmap(
        by_part,
        x="body_part",
        y="player",
        z="count",
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    heat.update_layout(margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(heat, use_container_width=True)

st.subheader("Mapa corporal por guarda-redes")

if not filtered.empty:
    players_body = sorted(filtered["player_id"].dropna().unique().tolist())

    if not players_body:
        st.info("Sem lesões para o mapa corporal com estes filtros.")
    else:
        selected_player_body = st.selectbox(
            "Seleciona um guarda-redes para ver o boneco",
            options=players_body,
            format_func=lambda pid: f"GK {int(pid)}",
            key="bodymap_player",
        )

        player_injuries = filtered[filtered["player_id"] == selected_player_body]

        if player_injuries.empty:
            st.info("Este guarda-redes não tem lesões no período/filtros atuais.")
        else:
            # Conta lesões por região simplificada
            region_counts = {r: 0 for r in BODY_REGIONS}

            for _, row in player_injuries.iterrows():
                region = map_body_part_to_region(row.get("body_part"))

                if region == "arms":
                    region_counts["left_arm"] += 0.5
                    region_counts["right_arm"] += 0.5
                elif region == "legs":
                    region_counts["left_leg"] += 0.5
                    region_counts["right_leg"] += 0.5
                else:
                    if region not in region_counts:
                        region_counts[region] = 0
                    region_counts[region] += 1

            # Garantir que são inteiros ou floats aceitáveis
            body_fig = build_body_figure(region_counts)

            st.caption(
                "Cada zona do boneco fica mais laranja quanto mais lesões o guarda-redes teve nessa região (segundo o filtro atual)."
            )
            st.plotly_chart(body_fig, use_container_width=False)

st.subheader("Dados (Injuries)")
with st.expander("Ver tabela"):
    st.dataframe(filtered.sort_values(["player_id", "start_date"], na_position="last"), use_container_width=True)
