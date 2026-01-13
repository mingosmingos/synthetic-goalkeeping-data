import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_matches, load_shots, load_appearances

st.title("Matches")

matches_df = load_matches()
shots_df = load_shots()
appearances_df = load_appearances()

# Add filtering options in the sidebar
with st.sidebar:
    st.header("Filters")
    
    # Filter by opponent
    opponents = ["All"] + matches_df['opponent'].unique().tolist()
    selected_opponent = st.selectbox("Opponent", opponents)
    
    # Filter by result
    st.subheader("Result")
    show_won = st.checkbox("Won", value=True)
    show_lost = st.checkbox("Lost", value=True)
    show_draw = st.checkbox("Draw", value=True)

# Apply filters
filtered_df = matches_df.copy()

if selected_opponent != "All":
    filtered_df = filtered_df[filtered_df['opponent'] == selected_opponent]

# Filter by result
results = []
if show_won:
    results.append(filtered_df[filtered_df['scored'] > filtered_df['conceded']])
if show_lost:
    results.append(filtered_df[filtered_df['scored'] < filtered_df['conceded']])
if show_draw:
    results.append(filtered_df[filtered_df['scored'] == filtered_df['conceded']])

if results:
    filtered_df = pd.concat(results, ignore_index=True)
else:
    filtered_df = pd.DataFrame()  # Empty if no filters selected

# Display match count
st.subheader(f"Matches ({len(filtered_df)})")

# Display all matches with expand/collapse details
for idx, (_, match) in enumerate(filtered_df.iterrows()):
    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    
    with col1:
        st.metric("Match ID", match['match_id'])
    with col2:
        st.metric("Opponent", match['opponent'])
    with col3:
        st.metric("Score", f"{int(match['scored'])} - {int(match['conceded'])}")
    with col4:
        if st.button("View Details", key=f"details_{match['match_id']}"):
            st.session_state[f"show_details_{match['match_id']}"] = not st.session_state.get(f"show_details_{match['match_id']}", False)
    
    # Display selected match details
    if st.session_state.get(f"show_details_{match['match_id']}", False):
        with st.container(border=True):
            # Get appearances for this match to find shot IDs
            match_appearances = appearances_df[appearances_df['match_id'] == match['match_id']]
            
            if not match_appearances.empty:
                # Get shots from those appearances
                match_shots = shots_df[shots_df['appearance'].isin(match_appearances.index)]
                
                # Filter for goals and sort by timestamp
                goals = match_shots[match_shots['isgoal'] == True].copy()
                goals = goals.sort_values('timestamp')
                
                if not goals.empty:
                    # Create cumulative count
                    goals['cumulative_goals'] = range(1, len(goals) + 1)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Goals Over Time:**")
                        
                        # Create step chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.step(goals['timestamp'], goals['cumulative_goals'], where='post', linewidth=2, color='#1f77b4')
                        ax.scatter(goals['timestamp'], goals['cumulative_goals'], s=100, color='#1f77b4', zorder=5)
                        
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Cumulative Goals')
                        ax.set_title('Goals Over Time')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(0, max(goals['cumulative_goals']) + 0.5)
                        
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("**Goal Timeline:**")
                        for idx, (shot_idx, goal) in enumerate(goals.iterrows(), 1):
                            time_str = goal['timestamp'].strftime('%H:%M:%S')
                            if st.button(f"Goal {idx} - {time_str}", key=f"goal_{match['match_id']}_{shot_idx}"):
                                st.session_state['selected_shot'] = shot_idx
                                st.switch_page("pages/3_Shots.py")
                else:
                    st.info("No goals in this match (clean sheet!)")
            else:
                st.warning("No shot data available for this match")
            
            st.divider()
            st.write("**Full Match Details:**")
            st.json(match.to_dict())
    
    st.divider()
st.subheader("Match Statistics")
st.dataframe(matches_df, use_container_width=True)