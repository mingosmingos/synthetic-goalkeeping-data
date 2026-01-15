import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_matches, load_shots, load_appearances

st.title("Match Details")

if 'selected_match' not in st.session_state:
    st.error("No match selected. Please go back to Matches page.")
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
        
        st.write("**Goals Over Time:**")
        
        # Calculate time in seconds from start
        start_time = goals['timestamp'].min()
        goals['time_seconds'] = (goals['timestamp'] - start_time).dt.total_seconds()
        
        # Create step chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.step(goals['time_seconds'], goals['cumulative_goals'], where='post', linewidth=2, color='#1f77b4')
        ax.scatter(goals['time_seconds'], goals['cumulative_goals'], s=100, color='#1f77b4', zorder=5)
        
        # Format x-axis as MM:SS
        def seconds_to_mmss(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        
        max_seconds = goals['time_seconds'].max()
        ax.set_xticks([i * 300 for i in range(0, int(max_seconds // 300) + 2)])
        ax.set_xticklabels([seconds_to_mmss(i * 300) for i in range(0, int(max_seconds // 300) + 2)])
        
        ax.set_xlabel('Time (MM:SS)')
        ax.set_ylabel('Cumulative Goals')
        ax.set_title('Goals Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(goals['cumulative_goals']) + 0.5)
        
        st.pyplot(fig)
        
        st.write("**Goal Timeline:**")
        with st.container(height=260, border=True):
            cols = st.columns(3)
            for idx, (shot_idx, goal) in enumerate(goals.iterrows(), start=1):
                time_str = seconds_to_mmss(goal['time_seconds'])
                with cols[(idx - 1) % 3]:
                    if st.button(
                        f"Goal {idx} • {time_str}",
                        key=f"goal_{match_id}_{shot_idx}",
                        use_container_width=True,
                    ):
                        st.session_state['selected_shot'] = shot_idx
                        st.switch_page("pages/3_Shots.py")
    else:
        st.info("No goals in this match (clean sheet!)")
else:
    st.warning("No shot data available for this match")

st.divider()
st.subheader("Full Match Details")
st.json(match.to_dict())

if st.button("Back to Matches"):
    st.switch_page("pages/2_Matches.py")
