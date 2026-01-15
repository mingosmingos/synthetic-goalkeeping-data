import pandas as pd
import streamlit as st

data = "output.xlsx"

@st.cache_data
def load_shots():
    return pd.read_excel(data, sheet_name="Shots")

@st.cache_data
def load_appearances():
    return pd.read_excel(data, sheet_name="Appearances")

@st.cache_data
def load_players():
    return pd.read_excel(data, sheet_name="Players")

@st.cache_data
def load_matches():
    return pd.read_excel(data, sheet_name="Matches")

#@st.cache_data
#def load_shooters():
#    return pd.read_excel(data, sheet_name="Shooters")

@st.cache_data
def load_injuries():
    return pd.read_excel("data/injuries.xlsx", sheet_name="Injuries")

@st.cache_data
def load_optimal_poses():
    return pd.read_excel("data\output_with_optimal_poses.xlsx", sheet_name=0)
