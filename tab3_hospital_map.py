import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import os

# ───────────────────── Display Function ─────────────────────
def display_tab(main_path, hospital_coords_file, full_hospitals_file, geo_cache_file):
    st.subheader("Hospital Screening Locations in Lebanon")

    # Load hospital data
    try:
        df_hospitals = pd.read_csv(hospital_coords_file)
        full_df = pd.read_csv(full_hospitals_file)
    except FileNotFoundError:
        st.error("Hospital data files not found.")
        return

    df_hospitals.columns = df_hospitals.columns.str.lower().str.strip().str.replace(" ", "_")
    full_df.columns = full_df.columns.str.lower().str.strip().str.replace(" ", "_")

    region = st.selectbox("Choose a Region (Caza):", sorted(full_df["caza"].dropna().unique()))

    filtered_df = df_hospitals[df_hospitals["caza"] == region]

    # ─────────────── Compact Layout: Map and Table Side by Side ───────────────
    col1, col2 = st.columns([1.5, 1])

    with col1:
        m = folium.Map(location=[33.8547, 35.8623], zoom_start=8)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in filtered_df.iterrows():
            popup_text = f"<b>{row['name']}</b><br>Phone: {row['phone']}"
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=popup_text,
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(marker_cluster)

        st_folium(m, width=600, height=380)  # Reduced height

    with col2:
        st.markdown("#### Matching Hospitals")
        st.dataframe(filtered_df[["name", "phone", "caza"]].reset_index(drop=True), height=380)
