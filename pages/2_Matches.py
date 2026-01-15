import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_matches

st.title("Matches")

matches_df = load_matches()

matches_df = matches_df.copy() if matches_df is not None else pd.DataFrame()
if matches_df.empty:
    st.info("No match data available.")
    st.stop()


def _result_label(scored: int, conceded: int) -> str:
    if scored > conceded:
        return "Win"
    if scored < conceded:
        return "Loss"
    return "Draw"


def _truncate(text: str, max_len: int = 34) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

# Add filtering options in the sidebar
with st.sidebar:
    st.header("Filters")
    
    # Filter by opponent
    opponents = ["All"] + sorted(matches_df["opponent"].dropna().unique().tolist())
    selected_opponent = st.selectbox("Opponent", opponents)
    
    # Filter by result
    st.subheader("Result")
    show_win = st.checkbox("Win", value=True)
    show_loss = st.checkbox("Loss", value=True)
    show_draw = st.checkbox("Draw", value=True)

# Apply filters
filtered_df = matches_df.copy()

if selected_opponent != "All":
    filtered_df = filtered_df[filtered_df["opponent"] == selected_opponent]

# Filter by result
results = []
if show_win:
    results.append(filtered_df[filtered_df['scored'] > filtered_df['conceded']])
if show_loss:
    results.append(filtered_df[filtered_df['scored'] < filtered_df['conceded']])
if show_draw:
    results.append(filtered_df[filtered_df['scored'] == filtered_df['conceded']])

if results:
    filtered_df = pd.concat(results, ignore_index=True)
else:
    filtered_df = pd.DataFrame()  # Empty if no filters selected

if not filtered_df.empty and "date" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("date", ascending=False)

# Display match count
st.subheader(f"Matches ({len(filtered_df)})")

if filtered_df.empty:
    st.info("No matches found for the current filters.")
    st.stop()

# Display all matches in a 4-column grid with cards
cols = st.columns(4)
for idx, (_, match) in enumerate(filtered_df.iterrows()):
    scored = int(match.get("scored", 0))
    conceded = int(match.get("conceded", 0))
    result_label = _result_label(scored, conceded)

    accent_color = {
        "Win": "#2e7d32",  # green
        "Loss": "#c62828",  # red
        "Draw": "#f9a825",  # amber
    }.get(result_label, "#9e9e9e")

    col = cols[idx % 4]
    
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='height:6px; background:{accent_color}; border-radius:6px; margin-bottom:0.6rem;'></div>",
                unsafe_allow_html=True,
            )
            opponent = _truncate(match.get("opponent", "Unknown"), max_len=16)
            date_val = match.get("date")
            date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)

            st.markdown(f"**vs {opponent}**")
            st.caption(date_str)

            st.markdown(f"### {scored} - {conceded}")
            
            if st.button("View Details", key=f"details_{match['match_id']}", use_container_width=True):
                st.session_state['selected_match'] = match['match_id']
                st.switch_page("pages/3_Match_Details.py")
    
st.divider()
st.subheader("Match Statistics")
st.dataframe(matches_df, use_container_width=True, hide_index=True)