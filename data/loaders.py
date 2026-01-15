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
