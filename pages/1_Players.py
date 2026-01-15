import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_appearances, load_matches, load_players, load_shots

st.title("Players")

ROOT = Path(__file__).parent.parent
ATTR_PATH = ROOT / "Synthetic Data.xlsx"


def _player_display_name(player_row: pd.Series) -> str:
	for col in ("name", "player_name", "keeper_name"):
		if col in player_row.index and pd.notna(player_row[col]):
			return str(player_row[col])
	if "player_id" in player_row.index and pd.notna(player_row["player_id"]):
		return f"Player {player_row['player_id']}"
	if player_row.name is not None:
		return f"Player {player_row.name}"
	return "Player"


@st.cache_data
def _load_player_attributes(path: Path) -> pd.DataFrame:
	df = pd.read_excel(path)
	needed = ["player_id", "name", "reflexes", "agility", "presence", "flexibility"]
	existing = [c for c in needed if c in df.columns]
	return df[existing]


@st.cache_data
def _load_shots_with_keeper() -> pd.DataFrame:
	shots_df = load_shots()
	appearances_df = load_appearances()
	shots_df = shots_df.copy() if shots_df is not None else pd.DataFrame()
	appearances_df = appearances_df.copy() if appearances_df is not None else pd.DataFrame()

	if shots_df.empty:
		return shots_df

	if "player_id" in shots_df.columns:
		return shots_df

	if not {"appearance_id"}.issubset(shots_df.columns) or appearances_df.empty:
		return shots_df

	if not {"appearance_id", "player_id"}.issubset(appearances_df.columns):
		return shots_df

	apps = appearances_df[["appearance_id", "player_id"]].dropna()
	return shots_df.merge(apps, on="appearance_id", how="left")


@st.cache_data
def _matches_played_for_player(player_id) -> pd.DataFrame:
	appearances_df = load_appearances()
	matches_df = load_matches()
	appearances_df = appearances_df.copy() if appearances_df is not None else pd.DataFrame()
	matches_df = matches_df.copy() if matches_df is not None else pd.DataFrame()

	if appearances_df.empty or matches_df.empty:
		return pd.DataFrame()
	if not {"player_id", "match_id"}.issubset(appearances_df.columns):
		return pd.DataFrame()

	match_ids = (
		appearances_df.loc[appearances_df["player_id"] == player_id, "match_id"]
		.dropna()
		.unique()
		.tolist()
	)
	if not match_ids:
		return pd.DataFrame()

	if "match_id" in matches_df.columns:
		out = matches_df[matches_df["match_id"].isin(match_ids)].copy()
	else:
		# Fallback: match_id might be the index
		out = matches_df.loc[matches_df.index.isin(match_ids)].copy()
		if "match_id" not in out.columns:
			out = out.reset_index().rename(columns={"index": "match_id"})

	if "date" in out.columns:
		out = out.sort_values("date", ascending=False)
	return out


def _coerce_int(value, default: int = 0) -> int:
	n = pd.to_numeric(value, errors="coerce")
	if pd.isna(n):
		return default
	return int(n)


players_df = load_players()
players_df = players_df.copy() if players_df is not None else pd.DataFrame()
if players_df.empty:
	st.info("No player data available.")
	st.stop()

if "player_id" in players_df.columns:
	player_ids = sorted([pid for pid in players_df["player_id"].dropna().unique().tolist()])
else:
	player_ids = sorted([pid for pid in players_df.index.dropna().unique().tolist()])

id_to_name = {}
for _, row in players_df.iterrows():
	pid = row.get("player_id", row.name)
	if pd.notna(pid):
		id_to_name[pid] = _player_display_name(row)

with st.sidebar:
	st.header("Filters")
	selected_players = st.multiselect(
		"Compare players",
		options=player_ids,
		default=[],
		format_func=lambda pid: id_to_name.get(pid, f"Player {pid}"),
	)


def _pid_series(df: pd.DataFrame) -> pd.Series:
	return df["player_id"] if "player_id" in df.columns else df.index


filtered_players_df = players_df


def _player_matches_from_row(player_row: pd.Series) -> int:
	# Prefer explicit matches columns if present, otherwise derive from W/D/L
	if "matches" in player_row.index or "Matches" in player_row.index:
		return _coerce_int(player_row.get("matches", player_row.get("Matches", 0)))
	win = _coerce_int(player_row.get("wins", player_row.get("Win", 0)))
	draw = _coerce_int(player_row.get("draws", player_row.get("Draw", 0)))
	loss = _coerce_int(player_row.get("losses", player_row.get("Loss", 0)))
	return win + draw + loss


_matches_by_pid = {}
if "player_id" in players_df.columns:
	for _, r in players_df.iterrows():
		pid = r.get("player_id")
		if pd.notna(pid):
			_matches_by_pid[pid] = _player_matches_from_row(r)


# Players selected in the sidebar drive the comparison graphs
comparison_pids = [pid for pid in selected_players if _matches_by_pid.get(pid, 0) > 0]


st.subheader("Player Comparison")
if not comparison_pids:
	st.info("Select one or more players (with matches) in the sidebar to compare.")
else:
	perf_df = players_df.copy()
	if "shots_faced" not in perf_df.columns and {"saved", "conceded"}.issubset(perf_df.columns):
		perf_df["shots_faced"] = perf_df["saved"] + perf_df["conceded"]
	if "save_pct" not in perf_df.columns and {"saved", "shots_faced"}.issubset(perf_df.columns):
		perf_df["save_pct"] = perf_df["saved"] / perf_df["shots_faced"].replace({0: pd.NA})

	attr_df = pd.DataFrame()
	if ATTR_PATH.exists():
		attr_df = _load_player_attributes(ATTR_PATH)
	elif {"reflexes", "agility", "presence", "flexibility"}.issubset(players_df.columns):
		cols = [c for c in ["player_id", "name", "reflexes", "agility", "presence", "flexibility"] if c in players_df.columns]
		attr_df = players_df[cols].copy()

	full_df = perf_df
	if not attr_df.empty and "player_id" in attr_df.columns and "player_id" in perf_df.columns:
		full_df = attr_df.merge(perf_df, on="player_id", how="left", suffixes=("", "_perf"))
	elif "player_id" in players_df.columns:
		full_df = players_df.copy()

	colors = ["#9abfe7", "#C0E742", "#2cc199", "#ff7f0e", "#9467bd", "#d62728"]

	tab_attr, tab_perf = st.tabs(["Attributes", "Performance"])

	with tab_attr:
		attr_metrics = ["reflexes", "agility", "presence", "flexibility"]
		attr_labels = ["Reflexes", "Agility", "Presence", "Flexibility"]
		if not set(attr_metrics).issubset(full_df.columns):
			st.warning("Attribute columns not found (reflexes/agility/presence/flexibility).")
		else:
			fig_attr = go.Figure()
			for idx, pid in enumerate(comparison_pids):
				row = full_df[full_df["player_id"] == pid]
				if row.empty:
					continue
				row = row.iloc[0]
				values = [row[m] for m in attr_metrics]
				values = values + values[:1]
				fig_attr.add_trace(
					go.Scatterpolar(
						r=values,
						theta=attr_labels + [attr_labels[0]],
						name=id_to_name.get(pid, f"Player {pid}"),
						fill="toself",
						line=dict(color=colors[idx % len(colors)], width=2),
						opacity=0.6,
					)
				)
			fig_attr.update_layout(
				title="Attributes",
				polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
				showlegend=True,
				template="plotly_dark",
				margin=dict(l=40, r=40, t=40, b=40),
			)
			st.plotly_chart(fig_attr, use_container_width=True, key="players_radar_attr")

	with tab_perf:
		if "saved" not in full_df.columns:
			st.warning("Performance column not found: saved")
		else:
			# KPI row: Save % per selected keeper
			kpi_df = full_df.copy()
			if "shots_faced" not in kpi_df.columns and {"saved", "conceded"}.issubset(kpi_df.columns):
				kpi_df["shots_faced"] = pd.to_numeric(kpi_df["saved"], errors="coerce").fillna(0) + pd.to_numeric(
					kpi_df["conceded"], errors="coerce"
				).fillna(0)
			if "save_pct" not in kpi_df.columns and {"saved", "shots_faced"}.issubset(kpi_df.columns):
				shots_faced = pd.to_numeric(kpi_df["shots_faced"], errors="coerce")
				saved = pd.to_numeric(kpi_df["saved"], errors="coerce")
				kpi_df["save_pct"] = (saved / shots_faced.replace({0: pd.NA})).astype(float)

			if "save_pct" in kpi_df.columns and "shots_faced" in kpi_df.columns:
				overall = kpi_df.copy()
				overall["shots_faced"] = pd.to_numeric(overall["shots_faced"], errors="coerce").fillna(0)
				overall["save_pct"] = pd.to_numeric(overall["save_pct"], errors="coerce")
				overall_avg = overall.loc[overall["shots_faced"] > 0, "save_pct"].mean()
				if pd.isna(overall_avg):
					overall_avg = 0.0

				kpi_cols = st.columns(max(1, len(comparison_pids)))
				for i, pid in enumerate(comparison_pids):
					row = kpi_df[kpi_df["player_id"] == pid]
					label = id_to_name.get(pid, f"Player {pid}")
					if row.empty:
						kpi_cols[i].metric(label, "—")
						continue
					row = row.iloc[0]
					shots = pd.to_numeric(row.get("shots_faced"), errors="coerce")
					save_pct = pd.to_numeric(row.get("save_pct"), errors="coerce")
					if pd.isna(shots) or shots <= 0 or pd.isna(save_pct):
						kpi_cols[i].metric(label, "—", help="No shots faced")
						continue
					delta_pp = (float(save_pct) - float(overall_avg)) * 100.0
					kpi_cols[i].metric(label, f"{float(save_pct) * 100.0:.1f}%", delta=f"{delta_pp:+.1f} pp")
				st.caption("Delta compares an individual's percentage with the overall average.")
			else:
				st.warning("Cannot compute save% KPI (needs shots faced).")

			# Bar plot: successful saves per selected keeper
			df_sel = full_df[full_df["player_id"].isin(comparison_pids)].copy()
			df_sel["saved"] = pd.to_numeric(df_sel["saved"], errors="coerce").fillna(0)
			if "conceded" in df_sel.columns:
				df_sel["conceded"] = pd.to_numeric(df_sel["conceded"], errors="coerce").fillna(0)
			else:
				df_sel["conceded"] = 0
			df_sel["shots_faced"] = df_sel["saved"] + df_sel["conceded"]
			df_sel["label"] = df_sel["player_id"].map(lambda pid: id_to_name.get(pid, f"Player {pid}"))
			df_sel = df_sel.sort_values("shots_faced", ascending=True)

			fig_perf = go.Figure(
				data=[
					go.Bar(
						x=df_sel["saved"],
						y=df_sel["label"],
						orientation="h",
						name="Saves",
						marker_color="#2cc199",
						customdata=df_sel[["conceded", "shots_faced"]].to_numpy(),
						hovertemplate="%{y}<br>Saves: %{x}<br>Conceded: %{customdata[0]}<br>Shots faced: %{customdata[1]}<extra></extra>",
					),
					go.Bar(
						x=df_sel["conceded"],
						y=df_sel["label"],
						orientation="h",
						name="Conceded",
						marker_color="#c62828",
						hovertemplate="%{y}<br>Conceded: %{x}<extra></extra>",
					),
				]
			)
			fig_perf.update_layout(
				template="plotly_dark",
				title="Saves & Goals",
				xaxis_title="Shots faced",
				yaxis_title="",
				barmode="stack",
				margin=dict(l=60, r=20, t=50, b=40),
				legend=dict(orientation="h", y=-0.15, x=0.0),
			)
			fig_perf.update_xaxes(rangemode="tozero")
			st.plotly_chart(fig_perf, use_container_width=True, key="players_perf_bar")


st.divider()
st.subheader(f"Players ({len(filtered_players_df)})")

cols = st.columns(2)
for idx, (_, player) in enumerate(filtered_players_df.iterrows()):
	pid = player.get("player_id", player.name)
	player_name = id_to_name.get(pid, _player_display_name(player))

	# Your Players sheet uses 'Win'/'Draw'/'Loss' (capitalized)
	wins = _coerce_int(player.get("wins", player.get("Win", 0)))
	draws = _coerce_int(player.get("draws", player.get("Draw", 0)))
	losses = _coerce_int(player.get("losses", player.get("Loss", 0)))
	matches = _coerce_int(player.get("matches", player.get("Matches", wins + draws + losses)), wins + draws + losses)

	col = cols[idx % 2]
	with col:
		with st.container(border=True):
			st.write(f"**{player_name}**")
			st.write(f"Matches: {matches} • W-D-L: {wins}-{draws}-{losses}")

			show_donut = st.toggle(
				"Show more details",
				value=st.session_state.get(f"player_donut_show_{pid}", False),
				disabled=matches == 0,
				key=f"player_donut_show_{pid}",
			)
			if matches == 0:
				st.caption("No matches recorded for this player yet.")
			elif show_donut:
				fig = go.Figure(
					data=[
						go.Pie(
							labels=["Wins", "Draws", "Losses"],
							values=[wins, draws, losses],
							hole=0.65,
							marker={
								"colors": ["#2e7d32", "#f9a825", "#c62828"],
								"line": {"color": "white", "width": 2},
							},
							sort=False,
							direction="clockwise",
							textinfo="none",
						)
					]
				)
				fig.update_layout(
					template="plotly_white",
					height=240,
					margin={"l": 10, "r": 10, "t": 10, "b": 10},
					showlegend=True,
					legend={"orientation": "h", "y": -0.05, "x": 0.2},
				)
				st.plotly_chart(fig, use_container_width=True, key=f"player_donut_{pid}")

				shots_with_keeper = _load_shots_with_keeper()
				if (
					shots_with_keeper is None
					or shots_with_keeper.empty
					or "player_id" not in shots_with_keeper.columns
					or not {"x", "y"}.issubset(shots_with_keeper.columns)
				):
					st.caption("No shot location data available for heatmap.")
				else:
					player_shots = shots_with_keeper[shots_with_keeper["player_id"] == pid]
					player_shots = player_shots.copy()
					player_shots["x"] = pd.to_numeric(player_shots["x"], errors="coerce")
					player_shots["y"] = pd.to_numeric(player_shots["y"], errors="coerce")
					player_shots = player_shots.dropna(subset=["x", "y"])

					if player_shots.empty:
						st.caption("No shots faced recorded for this player yet.")
					else:
						fig_hm = go.Figure(
							data=[
								go.Histogram2d(
									x=player_shots["x"],
									y=player_shots["y"],
									nbinsx=14,
									nbinsy=14,
									colorscale="YlOrRd",
									showscale=False,
									hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>count=%{z}<extra></extra>",
								)
							]
						)
						fig_hm.update_layout(
							template="plotly_white",
							height=240,
							margin={"l": 10, "r": 10, "t": 28, "b": 10},
							title="Shots faced heatmap",
							xaxis=dict(range=[0, 27], showgrid=False, zeroline=False, visible=False),
							yaxis=dict(range=[0, 27], showgrid=False, zeroline=False, visible=False),
						)
						st.plotly_chart(fig_hm, use_container_width=True, key=f"player_shots_heatmap_{pid}")

				st.markdown("**Matches played**")
				played_df = _matches_played_for_player(pid)
				if played_df is None or played_df.empty or "match_id" not in played_df.columns:
					st.caption("No matches found for this player.")
				else:
					max_show = 8
					for _, m in played_df.head(max_show).iterrows():
						mid = m.get("match_id")
						opponent = m.get("opponent", "Unknown")
						date_val = m.get("date")
						date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)
						label = f"{date_str} vs {opponent} (Match {mid})"
						if st.button(label, key=f"player_{pid}_match_{mid}"):
							st.session_state["selected_match"] = mid
							st.switch_page("pages/3_Match_Details.py")
					if len(played_df) > max_show:
						st.caption(f"Showing latest {max_show} of {len(played_df)} matches")


st.divider()
with st.expander("Player table"):
	st.dataframe(players_df, use_container_width=True)