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

# Display all matches in 3-column grid with cards
cols = st.columns(2)
for idx, (_, match) in enumerate(filtered_df.iterrows()):
    col = cols[idx % 2]
    
    with col:
        with st.container(border=True):
            #st.write(f"**Match {int(match['match_id'])}**")
            st.write(f"vs {match['opponent']}")
            st.write(f"📊 {int(match['scored'])} - {int(match['conceded'])}")
            st.write(f"📅 {match['date'].strftime('%Y-%m-%d')}")
            
            if st.button("View Details", key=f"details_{match['match_id']}"):
                st.session_state['selected_match'] = match['match_id']
                st.switch_page("pages/4_Match_Details.py")
    
st.divider()
st.subheader("Match Statistics")
st.dataframe(matches_df, use_container_width=True)