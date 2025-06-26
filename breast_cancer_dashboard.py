# ── Streamlit Dashboard: Final Version with Sidebar Awareness Message ──
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joypy
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os, json, time, folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ── Import Tab Modules ──
import tab1_kaggle_patient
import tab2_national_benchmark_full
import tab3_hospital_map

# ───────────────────── Page Config ─────────────────────
st.set_page_config(page_title="Breast Cancer Awareness", layout="wide")

# ───────────────────── Centralized Paths ─────────────────────
main_path = "."  # relative path

# Data files
data_file = os.path.join(main_path, "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv")
hospital_coords_file = os.path.join(main_path, "demo_hospitals_with_coordinates.csv")
full_hospitals_file = os.path.join(main_path, "lebanon_private_hospitals_complete.csv")
geo_cache_file = os.path.join(main_path, "geo_cache.json")

# Images
pink_ribbon_img = os.path.join(main_path, "pink_ribbon.png")
man_and_woman_img = os.path.join(main_path, "manandwoman.png")

# ───────────────────── Load and Clean Main Dataset ─────────────────────
df = pd.read_csv(data_file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date_of_surgery"] = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"] = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# ───────────────────── Sidebar Ribbon ─────────────────────
st.sidebar.image(pink_ribbon_img, width=60)

# ───────────────────── Sidebar Filters ─────────────────────
st.sidebar.title("Filters")
gender = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
stages = st.sidebar.multiselect("Tumor Stage", df["tumour_stage"].dropna().unique(), default=list(df["tumour_stage"].dropna().unique()))
status = st.sidebar.multiselect("Patient Status", df["patient_status"].dropna().unique(), default=list(df["patient_status"].dropna().unique()))
age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))

# ───────────────────── Sidebar Awareness Section ─────────────────────
st.sidebar.image(man_and_woman_img, use_column_width=True)
st.sidebar.markdown("### Breast Cancer in Lebanon")
st.sidebar.write("Everyone, men and women, should know their risk and get screened.")
st.sidebar.info("**What you can do today:** Talk to your doctor about screening options or find a clinic near you.")

# ───────────────────── Sidebar Footer Quote and Credit ─────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:13px'><strong>Early detection saves lives.</strong><br>"
    "Awareness is the first step to prevention.</p>",
    unsafe_allow_html=True
)
st.sidebar.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")

# ───────────────────── Filtered Data ─────────────────────
filtered = df[
    (df["gender"].isin(gender)) &
    (df["tumour_stage"].isin(stages)) &
    (df["patient_status"].isin(status)) &
    (df["age"].between(age_range[0], age_range[1]))
]

# ───────────────────── Header ─────────────────────
st.title("Breast Cancer Awareness")
st.markdown("Developed by **Carine Bichara**")

# ───────────────────── Tabs ─────────────────────
tab1, tab2, tab3 = st.tabs(["Patient-Level (Kaggle)", "National Benchmarks", "Hospitals"])

with tab1:
    tab1_kaggle_patient.display_tab(filtered)

with tab2:
    tab2_national_benchmark_full.display_tab(main_path)

with tab3:
    tab3_hospital_map.display_tab(main_path, hospital_coords_file, full_hospitals_file, geo_cache_file)
