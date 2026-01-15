import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_matches, load_shots, load_appearances

st.title("Match Details")

if 'selected_match' not in st.session_state:
    st.info("No match selected. Pick a match on the Matches page to see details.")
    if st.button("Go to Matches"):
        st.switch_page("pages/2_Matches.py")
    st.stop()

match_id = st.session_state['selected_match']

matches_df = load_matches()
shots_df = load_shots()
appearances_df = load_appearances()

match = matches_df[matches_df['match_id'] == match_id]

if match.empty:
    st.error(f"Match {match_id} not found.")
    st.stop()

match = match.iloc[0]

col1, col2 = st.columns(2)
with col1:
    st.metric("Opponent", match['opponent'])
    st.metric("Result", match['result'])
with col2:
    st.metric("Date", match['date'].strftime('%Y-%m-%d'))
    st.metric("Score", f"{int(match['scored'])} - {int(match['conceded'])}")
st.divider()

# Get appearances for this match
match_appearances = appearances_df[appearances_df['match_id'] == match_id]

if not match_appearances.empty:
    # Get shots from those appearances
    match_shots = shots_df[shots_df['appearance_id'].isin(match_appearances.index)]
    
    # Filter for goals and sort by timestamp
    goals = match_shots[match_shots['saved'] == False].copy()
    goals = goals.sort_values('timestamp')
    
    if not goals.empty:
        # Create cumulative count
        goals['cumulative_goals'] = range(1, len(goals) + 1)
        
        # Calculate time in seconds from start
        start_time = goals['timestamp'].min()
        goals['time_seconds'] = (goals['timestamp'] - start_time).dt.total_seconds()
        goals['time_minutes'] = goals['time_seconds'] / 60.0

        # Format x-axis as MM:SS
        def seconds_to_mmss(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        # Plotly step chart (cleaner visuals + hover tooltips)
        max_minutes = float(goals['time_minutes'].max())
        last_tick_min = int(max_minutes // 5 + 1) * 5
        tick_vals = list(range(0, last_tick_min + 1, 5))

        # Keep the original shot index so we can navigate on point selection
        goals = goals.copy()
        goals['shot_idx'] = goals.index

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=goals['time_minutes'],
                    y=goals['cumulative_goals'],
                    customdata=goals['shot_idx'],
                    mode='lines+markers',
                    line={'width': 3, 'color': '#1f77b4', 'shape': 'hv'},
                    marker={'size': 10, 'color': '#1f77b4'},
                    hovertemplate='Time: %{x:.0f} min<extra></extra>',
                )
            ]
        )
        fig.update_layout(
            template='plotly_white',
            title='Goals Over Time',
            xaxis_title='Time (MM:SS)',
            yaxis_title='Cumulative Goals',
            clickmode='event+select',
            margin={'l': 10, 'r': 10, 't': 60, 'b': 10},
        )
        fig.update_xaxes(
            range=[0, max_minutes],
            tickmode='array',
            tickvals=tick_vals,
            ticktext=[seconds_to_mmss(v * 60) for v in tick_vals],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
        )
        fig.update_yaxes(
            range=[0, float(goals['cumulative_goals'].max()) + 0.5],
            dtick=5 if int(goals['cumulative_goals'].max()) >= 5 else 1,
            tick0=0,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
        )

        # Streamlit selection API: click a point to jump to that shot
        plotly_params = inspect.signature(st.plotly_chart).parameters
        if 'on_select' in plotly_params and 'selection_mode' in plotly_params:
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                on_select='rerun',
                selection_mode=['points'],
            )

            selection = getattr(event, 'selection', None)
            points = getattr(selection, 'points', None) if selection is not None else None
            if points:
                shot_idx = points[0].get('customdata')
                if shot_idx is not None:
                    st.session_state['selected_shot'] = int(shot_idx)
                    st.switch_page('pages/4_Shots.py')
        else:
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "Tip: upgrade Streamlit to enable click-to-navigate on the chart "
                "(needs `st.plotly_chart(..., on_select=..., selection_mode=...)`)."
            )
    else:
        st.info("No goals in this match (clean sheet!)")
else:
    st.warning("No shot data available for this match")

st.divider()
st.subheader("Full Match Details")
st.json(match.to_dict())

if st.button("Back to Matches"):
    st.switch_page("pages/2_Matches.py")
