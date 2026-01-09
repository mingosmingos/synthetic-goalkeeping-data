import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output.xlsx"

st.set_page_config(page_title="Digital Twin Guarda-Redes", layout="wide")

st.title("Digital Twin para Guarda-Redes – Interface de Exploração")
st.markdown(
    "Interface para explorar os dados sintéticos de guarda-redes (Shots, Appearances, Players, Matches)."
)

if not DATA_PATH.exists():
    st.error(
        f"Ficheiro de dados não encontrado em '{DATA_PATH}'. "
        "Gera primeiro os dados correndo o notebook 'Generating.ipynb' para criar o ficheiro 'output.xlsx'."
    )
    st.stop()

xls = pd.ExcelFile(DATA_PATH)
shots = pd.read_excel(xls, "Shots")
appearances = pd.read_excel(xls, "Appearances")
players = pd.read_excel(xls, "Players")
matches = pd.read_excel(xls, "Matches")

# Ligar shots a jogador e jogo através da coluna "appearance"
shots_ext = shots.merge(
    appearances[["appearance", "player_id", "match_id"]],
    on="appearance",
    how="left",
)

# Coluna de minuto (se existir timestamp) para gráficos temporais
if "timestamp" in shots_ext.columns:
    shots_ext["minute"] = pd.to_datetime(shots_ext["timestamp"]).dt.floor("1min").dt.minute

with st.sidebar:
    st.header("Filtros globais")

    player_options = sorted(shots_ext["player_id"].dropna().unique())
    selected_players = st.multiselect("Player ID", player_options, default=player_options)

    match_options = sorted(shots_ext["match_id"].dropna().unique())
    selected_matches = st.multiselect("Match ID", match_options, default=match_options)

    type_options = sorted(shots_ext["type"].dropna().unique())
    selected_types = st.multiselect("Tipo de remate", type_options, default=type_options)

    goal_filter = st.selectbox(
        "Filtrar por golo?",
        ["Todos", "Apenas golos", "Apenas não golo"],
        index=0,
    )

filtered = shots_ext.copy()
if selected_players:
    filtered = filtered[filtered["player_id"].isin(selected_players)]
if selected_matches:
    filtered = filtered[filtered["match_id"].isin(selected_matches)]
if selected_types:
    filtered = filtered[filtered["type"].isin(selected_types)]
if goal_filter == "Apenas golos":
    filtered = filtered[filtered["isgoal"] == True]
elif goal_filter == "Apenas não golo":
    filtered = filtered[filtered["isgoal"] == False]

st.subheader("Resumo geral (dados filtrados)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Nº de remates", len(filtered))
with col2:
    st.metric("Nº de golos", int(filtered["isgoal"].sum()))
with col3:
    st.metric("Jogadores distintos", filtered["player_id"].nunique())
with col4:
    st.metric("Jogos distintos", filtered["match_id"].nunique())

cenario_tab, diferencia_tab, explorador_tab = st.tabs(
    [
        "Cuidado com a Direita!",
        "Diferencia-te!",
        "Explorador Geral",
    ]
)

with cenario_tab:
    st.markdown("## Cuidado com a Direita! – Resposta táctica em tempo real")
    st.markdown(
        "Foca-se na leitura de padrões ofensivos e zonas de maior perigo, "
        "através de mapas de calor e barras empilhadas dos golos sofridos.",
    )

    goals = filtered[filtered["isgoal"] == True]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Mapa de calor de golos sofridos (x, y)")
        if goals.empty:
            st.info("Nenhum golo com os filtros actuais.")
        else:
            fig_heat = px.density_heatmap(
                goals,
                x="x",
                y="y",
                nbinsx=27,
                nbinsy=27,
                color_continuous_scale="Reds",
                labels={"x": "Coordenada X", "y": "Coordenada Y"},
                title="Zonas de maior incidência de golo",
            )
            fig_heat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        st.markdown("### Stacked bar – golos por minuto e tipo de remate")
        if "minute" not in goals.columns or goals.empty:
            st.info("Sem informação temporal suficiente para construir o gráfico.")
        else:
            goals_min = goals.copy()
            goals_min["minuto"] = goals_min["minute"]
            fig_stack = px.bar(
                goals_min,
                x="minuto",
                color="type",
                barmode="stack",
                labels={"minuto": "Minuto de jogo", "count": "Nº de golos"},
                title="Distribuição temporal dos golos por tipologia",
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown("### Índice de 'quase' – proximidade ao ideal")
    if len(filtered) == 0:
        st.info("Sem remates nos filtros actuais para calcular o índice.")
    else:
        total_shots = len(filtered)
        goals_count = int(filtered["isgoal"].sum())
        save_rate = 1.0 - goals_count / total_shots

        # Normalizar em relação à média global de todo o dataset
        global_total = len(shots_ext)
        global_goals = int(shots_ext["isgoal"].sum())
        global_save_rate = 1.0 - global_goals / global_total if global_total else 0.0

        if global_save_rate > 0:
            indice_quase = max(0.0, min(1.5 * save_rate / global_save_rate, 1.0)) * 100
        else:
            indice_quase = save_rate * 100

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=indice_quase,
                number={"suffix": "%"},
                title={"text": "Índice de 'quase' (adesão ao posicionamento ideal)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "#ffcccc"},
                        {"range": [50, 75], "color": "#fff0b3"},
                        {"range": [75, 100], "color": "#ccffcc"},
                    ],
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

with diferencia_tab:
    st.markdown("## Diferencia-te! – Especialização estratégica do plantel")
    st.markdown(
        "Compara guarda-redes através de métricas agregadas e gráficos de radar, "
        "para suportar decisões de convocatória e papéis complementares.",
    )

    # Tabela de desempenho (remates/golos) baseada nos filtros actuais
    if filtered.empty:
        st.info("Sem remates nos filtros actuais para a tabela de desempenho.")
    else:
        grouped = filtered.groupby("player_id").agg(
            shots_faced=("isgoal", "count"),
            goals_conceded=("isgoal", "sum"),
        )
        grouped["shots_faced"] = grouped["shots_faced"].astype(float)
        grouped["goals_conceded"] = grouped["goals_conceded"].astype(float)
        grouped["save_pct"] = 100.0 * (
            1.0 - grouped["goals_conceded"] / grouped["shots_faced"].clip(lower=1)
        )

        st.markdown("### Tabela comparativa de desempenho")
        st.dataframe(grouped)

    # Radar com stats físicas/mentais vindas da tabela Players,
    # separados por posição e permitindo comparar 2 jogadores da mesma posição
    st.markdown("### Radar plot – perfis físico/mentais por posição")

    gk_stat_cols = [
        "reflexes",
        "handling",
        "aerial_command",
        "one_v_one",
        "communication",
    ]
    st_stat_cols = [
        "finishing",
        "off_ball",
        "pace",
        "strength",
        "pressing",
    ]

    needed_cols = set(gk_stat_cols) | set(st_stat_cols)
    if not needed_cols.issubset(players.columns):
        st.info(
            "As estatísticas físicas/mentais ainda não estão disponíveis em 'Players'. "
            "Volta a gerar os dados correndo o notebook 'Generating.ipynb'."
        )
    else:
        base_players = players.copy()
        # respeitar filtro global de jogadores, se existir
        if selected_players:
            base_players = base_players[
                base_players["player_id"].isin(selected_players)
            ]

        if base_players.empty:
            st.info("Nenhum jogador disponível com os filtros actuais.")
        else:
            # Bloco para guarda-redes (GK)
            st.markdown("#### Guarda-redes (GK)")
            gk_players = base_players[base_players["position"] == "GK"]
            if gk_players.empty:
                st.info("Nenhum guarda-redes disponível para comparação.")
            else:
                gk_ids = sorted(gk_players["player_id"].unique())
                default_ids = gk_ids[:2] if len(gk_ids) >= 2 else gk_ids
                col_gk1, col_gk2 = st.columns(2)
                with col_gk1:
                    gk1 = st.selectbox(
                        "Jogador 1 (GK)",
                        gk_ids,
                        index=0,
                        key="gk_player_1",
                    )
                with col_gk2:
                    idx2 = 1 if len(gk_ids) > 1 else 0
                    gk2 = st.selectbox(
                        "Jogador 2 (GK)",
                        gk_ids,
                        index=idx2,
                        key="gk_player_2",
                    )

                sel_gk_ids = sorted({gk1, gk2})
                gk_comp = gk_players[gk_players["player_id"].isin(sel_gk_ids)]

                max_vals_gk = {
                    col: max(gk_players[col].max(), 1) for col in gk_stat_cols
                }

                fig_gk = go.Figure()
                for _, row in gk_comp.iterrows():
                    values = [
                        float(row[col]) / max_vals_gk[col]
                        for col in gk_stat_cols
                    ]
                    fig_gk.add_trace(
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=[
                                "Reflexos",
                                "Handling",
                                "Comando da área",
                                "1v1",
                                "Comunicação",
                                "Reflexos",
                            ],
                            fill="toself",
                            name=f"GK {int(row['player_id'])}",
                        )
                    )

                fig_gk.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                    showlegend=True,
                )

                st.plotly_chart(fig_gk, use_container_width=True)

            # Bloco para avançados (ST)
            st.markdown("#### Avançados (ST)")
            st_players = base_players[base_players["position"] == "ST"]
            if st_players.empty:
                st.info("Nenhum avançado disponível para comparação.")
            else:
                st_ids = sorted(st_players["player_id"].unique())
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st1 = st.selectbox(
                        "Jogador 1 (ST)",
                        st_ids,
                        index=0,
                        key="st_player_1",
                    )
                with col_st2:
                    idx2 = 1 if len(st_ids) > 1 else 0
                    st2 = st.selectbox(
                        "Jogador 2 (ST)",
                        st_ids,
                        index=idx2,
                        key="st_player_2",
                    )

                sel_st_ids = sorted({st1, st2})
                st_comp = st_players[st_players["player_id"].isin(sel_st_ids)]

                max_vals_st = {
                    col: max(st_players[col].max(), 1) for col in st_stat_cols
                }

                fig_st = go.Figure()
                for _, row in st_comp.iterrows():
                    values = [
                        float(row[col]) / max_vals_st[col]
                        for col in st_stat_cols
                    ]
                    fig_st.add_trace(
                        go.Scatterpolar(
                            r=values + [values[0]],
                            theta=[
                                "Finalização",
                                "Jogo sem bola",
                                "Velocidade",
                                "Força",
                                "Pressão",
                                "Finalização",
                            ],
                            fill="toself",
                            name=f"ST {int(row['player_id'])}",
                        )
                    )

                fig_st.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 1]}},
                    showlegend=True,
                )

                st.plotly_chart(fig_st, use_container_width=True)

with explorador_tab:
    st.markdown("## Explorador geral de tabelas")

    st.markdown("### Shots (após filtros globais)")
    st.dataframe(filtered)

    st.markdown("### Appearances")
    st.dataframe(appearances)

    st.markdown("### Players")
    st.dataframe(players)

    st.markdown("### Matches")
    st.dataframe(matches)
