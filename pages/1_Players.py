import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_players

st.title("Players")

players_df = load_players()
st.subheader("Player Statistics")
st.dataframe(players_df, use_container_width=True)