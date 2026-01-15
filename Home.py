import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.loaders import load_matches, load_shots

st.set_page_config(page_title="Home", layout="wide")

st.title("ABC de Braga")


st.subheader("Last 5 Matches Overview")

matches_df = load_matches()
matches_df = matches_df.copy() if matches_df is not None else pd.DataFrame()

required_cols = {"date", "scored", "conceded"}
if matches_df.empty or not required_cols.issubset(matches_df.columns):
	st.info("Match data not available yet.")
	st.stop()

df = matches_df.copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["scored"] = pd.to_numeric(df["scored"], errors="coerce")
df["conceded"] = pd.to_numeric(df["conceded"], errors="coerce")
df = df.dropna(subset=["date", "scored", "conceded"])

if df.empty:
	st.info("No valid match rows found to plot.")
	st.stop()

df = df.sort_values("date", ascending=False).head(5)
df = df.sort_values("date", ascending=True)

label_parts = []
for _, r in df.iterrows():
	date_str = r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])
	opponent = r.get("opponent")
	opp_str = str(opponent) if pd.notna(opponent) else "Opponent"
	label_parts.append(f"{date_str}\nvs {opp_str}")

x = label_parts
scored = df["scored"].astype(float)
conceded = df["conceded"].astype(float) * -1.0
match_ids = df["match_id"].tolist() if "match_id" in df.columns else [None] * len(df)
conceded_pos = df["conceded"].astype(float)

shots_last5_df = None

# Save % per match (based on shots faced in that match)
save_pct = None
if "match_id" in df.columns:
	shots_df = load_shots()
	shots_df = shots_df.copy() if shots_df is not None else pd.DataFrame()
	if not shots_df.empty and {"match_id", "saved"}.issubset(shots_df.columns):
		shots_last5_df = shots_df[shots_df["match_id"].isin(df["match_id"].tolist())].copy()
		shots_last5_df["saved"] = shots_last5_df["saved"].astype(bool)
		agg = shots_last5_df.groupby("match_id")["saved"].agg(["sum", "count"]).reset_index()
		agg = agg.rename(columns={"sum": "saves", "count": "shots"})
		agg["save_pct"] = (agg["saves"] / agg["shots"].replace({0: pd.NA})) * 100.0
		pct_by_match = dict(zip(agg["match_id"].tolist(), agg["save_pct"].tolist()))
		save_pct = [pct_by_match.get(mid) for mid in df["match_id"].tolist()]

fig = go.Figure(
	data=[
		go.Bar(
			x=x,
			y=scored,
			name="Scored",
			marker_color="#2e7d32",
			customdata=list(zip(match_ids, scored.tolist())),
			hovertemplate="%{x}<br>Scored: %{y}<extra></extra>",
		),
		go.Bar(
			x=x,
			y=conceded,
			name="Conceded",
			marker_color="#c62828",
			customdata=list(zip(match_ids, conceded_pos.tolist())),
			hovertemplate="%{x}<br>Conceded: %{customdata[1]:.0f}<extra></extra>",
		),
	]
)

if save_pct is not None:
	fig.add_trace(
		go.Scatter(
			x=x,
			y=save_pct,
			name="Save %",
			yaxis="y2",
			mode="lines+markers",
			line=dict(color="#1f77b4", width=3),
			marker=dict(size=9, color="#1f77b4"),
			customdata=list(zip(match_ids, save_pct)),
			hovertemplate="%{x}<br>Save %: %{y:.1f}%<extra></extra>",
		)
	)

max_val = float(max(scored.max(), df["conceded"].astype(float).max()))
pad = max(2.0, max_val * 0.15)

fig.update_layout(
	template="plotly_white",
	barmode="relative",
	title="Goals Scored & Conceded",
	yaxis_title="Goals",
	xaxis_title="",
	clickmode="event+select",
	margin=dict(l=10, r=10, t=60, b=10),
	legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right"),
	# Secondary axis for save %
	yaxis2=dict(
		title="Save %",
		overlaying="y",
		side="right",
		range=[0, 100],
		ticksuffix="%",
		showgrid=False,
	),
)
fig.update_yaxes(range=[-(max_val + pad), max_val + pad], zeroline=True, zerolinecolor="rgba(0,0,0,0.25)")

# Click a bar to jump to Match Details
plotly_params = inspect.signature(st.plotly_chart).parameters
if "on_select" in plotly_params and "selection_mode" in plotly_params and "match_id" in df.columns:
	event = st.plotly_chart(
		fig,
		use_container_width=True,
		on_select="rerun",
		selection_mode=["points"],
	)
	selection = getattr(event, "selection", None)
	points = getattr(selection, "points", None) if selection is not None else None
	if points:
		cd = points[0].get("customdata")
		mid = None
		if isinstance(cd, (list, tuple)) and len(cd) > 0:
			mid = cd[0]
		else:
			mid = cd
		if mid is not None:
			st.session_state["selected_match"] = int(mid)
			st.switch_page("pages/3_Match_Details.py")
else:
	st.plotly_chart(fig, use_container_width=True)

if shots_last5_df is None or shots_last5_df.empty:
	st.info("Shot data not available for these matches yet.")
else:
	col_a, col_b = st.columns(2, gap="large")

	# 1) Stacked bar: Saves vs Goals conceded (from shots)
	with col_a:
		st.markdown("**Shots faced (Saves vs Goals)**")
		match_id_order = df["match_id"].tolist() if "match_id" in df.columns else []
		mid_to_label = dict(zip(match_id_order, x))

		saves_vals = []
		goals_vals = []
		total_vals = []
		labels = []
		custom = []
		for mid in match_id_order:
			ms = shots_last5_df[shots_last5_df["match_id"] == mid]
			shots_n = int(len(ms))
			saves_n = int(ms["saved"].sum()) if shots_n else 0
			goals_n = int(shots_n - saves_n)
			labels.append(mid_to_label.get(mid, str(mid)))
			total_vals.append(shots_n)
			saves_vals.append(saves_n)
			goals_vals.append(goals_n)
			custom.append((mid, shots_n, saves_n, goals_n))

		fig_shots = go.Figure(
			data=[
				go.Bar(
					x=labels,
					y=saves_vals,
					name="Saves",
					marker_color="#1f77b4",
					customdata=custom,
					hovertemplate="%{x}<br>Saves: %{y}<br>Shots: %{customdata[1]}<extra></extra>",
				),
				go.Bar(
					x=labels,
					y=goals_vals,
					name="Goals",
					marker_color="#c62828",
					customdata=custom,
					hovertemplate="%{x}<br>Goals: %{y}<br>Shots: %{customdata[1]}<extra></extra>",
				),
			]
		)
		fig_shots.update_layout(
			template="plotly_white",
			barmode="stack",
			height=380,
			margin=dict(l=10, r=10, t=30, b=10),
			legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right"),
			yaxis_title="Shots",
			xaxis_title="",
		)
		st.plotly_chart(fig_shots, use_container_width=True)

	# 2) Velocity distribution (fallback to Save vs Goal pie)
	with col_b:
		vel_col = "velocity" if "velocity" in shots_last5_df.columns else None
		if vel_col is not None:
			st.markdown("**Shot velocity (Save vs Goal)**")
			vel_df = shots_last5_df.copy()
			vel_df[vel_col] = pd.to_numeric(vel_df[vel_col], errors="coerce")
			vel_df = vel_df.dropna(subset=[vel_col])
			if vel_df.empty:
				st.info("No velocity values found in Shots.")
			else:
				vel_saved = vel_df.loc[vel_df["saved"] == True, vel_col].astype(float).tolist()
				vel_goal = vel_df.loc[vel_df["saved"] == False, vel_col].astype(float).tolist()
				fig_vel = go.Figure(
					data=[
						go.Histogram(x=vel_saved, name="Save", marker_color="#1f77b4", opacity=0.65),
						go.Histogram(x=vel_goal, name="Goal", marker_color="#c62828", opacity=0.65),
					]
				)
				fig_vel.update_layout(
					template="plotly_white",
					barmode="overlay",
					height=380,
					margin=dict(l=10, r=10, t=30, b=10),
					xaxis_title="Velocity",
					yaxis_title="Count",
					legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right"),
				)
				st.plotly_chart(fig_vel, use_container_width=True)
		else:
			st.markdown("**Saves vs Goals (All shots)**")
			total_shots = int(len(shots_last5_df))
			total_saves = int(shots_last5_df["saved"].sum()) if total_shots else 0
			total_goals = int(total_shots - total_saves)
			fig_pie = go.Figure(
				data=[
					go.Pie(
						labels=["Saves", "Goals"],
						values=[total_saves, total_goals],
						hole=0.55,
						marker=dict(colors=["#1f77b4", "#c62828"]),
						hovertemplate="%{label}: %{value}<extra></extra>",
					),
				]
			)
			fig_pie.update_layout(
				template="plotly_white",
				height=380,
				margin=dict(l=10, r=10, t=30, b=10),
				showlegend=True,
				legend=dict(orientation="h", y=1.12, x=1.0, xanchor="right"),
			)
			st.plotly_chart(fig_pie, use_container_width=True)