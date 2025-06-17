# Grid-based Layout Version of Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

st.set_page_config(page_title="Breast Cancer Dashboard", layout="wide")

# Load Data
df = pd.read_csv("Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date_of_surgery"] = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"] = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# Sidebar Filters
st.sidebar.header("Filters")
genders = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
stages = st.sidebar.multiselect("Tumor Stage", df["tumour_stage"].dropna().unique(), default=list(df["tumour_stage"].dropna().unique()))
status = st.sidebar.multiselect("Patient Status", df["patient_status"].dropna().unique(), default=list(df["patient_status"].dropna().unique()))
age_range = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))

filtered = df[
    df["gender"].isin(genders)
    & df["tumour_stage"].isin(stages)
    & df["patient_status"].isin(status)
    & df["age"].between(age_range[0], age_range[1])
]

# Header
st.title("🩺 Breast Cancer Awareness in Lebanon")
st.caption("By Carine Bichara")

# Top KPIs and Pie Chart
kpi1, kpi2, kpi3, pie = st.columns([1, 1, 1, 2])
kpi1.metric("Total Patients", f"{len(filtered):,}")
kpi2.metric("Deaths", f"{int(filtered['mortality'].sum()):,}")
kpi3.metric("Total DALYs", f"{int(filtered['daly'].sum()):,}")
with pie:
    gender_counts = filtered["gender"].value_counts()
    fig_gender = px.pie(
        names=gender_counts.index,
        values=gender_counts.values,
        title="Gender Breakdown",
        color_discrete_sequence=["#8B0000", "#FFC1C1"]
    )
    st.plotly_chart(fig_gender, use_container_width=True)

# Age & Surgery Chart Grid
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    fig_age, ax = plt.subplots()
    filtered["age"].hist(bins=12, color="#F4CCCC", edgecolor="white", ax=ax)
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig_age)

with row1_col2:
    surgery_counts = filtered["surgery_type"].value_counts()
    fig_surg, ax2 = plt.subplots()
    ax2.pie(
        surgery_counts,
        labels=surgery_counts.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=sns.color_palette("Reds")[:len(surgery_counts)]
    )
    ax2.axis("equal")
    st.pyplot(fig_surg)

# Summary Table
st.subheader("Summary by Gender and Tumor Stage")
summary = filtered.groupby(["gender", "tumour_stage"]).agg(
    avg_daly=("daly", "mean"),
    avg_yll=("yll", "mean"),
    avg_yld=("yld", "mean")
).reset_index()
st.dataframe(summary)

# GCO Incidence & Mortality Side-by-Side
st.subheader("National Benchmarks from GCO")
rates = pd.read_csv("GCO_Lebanon_rates.csv")
latest = rates[rates.year == rates.year.max()]
colG1, colG2 = st.columns(2)
with colG1:
    fig1 = px.bar(latest, x="gender", y="incidence_rate", title="Incidence Rate", text_auto=True)
    st.plotly_chart(fig1, use_container_width=True)
with colG2:
    fig2 = px.bar(latest, x="gender", y="mortality_rate", title="Mortality Rate", text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

# Time Series & Forecast
st.subheader("3-Year Forecast: Incidence Rate")
ts = rates.groupby("year")[["incidence_rate"]].mean().reset_index()
y = ts.set_index("year")["incidence_rate"]
model = SARIMAX(y, order=(1, 1, 1))
res = model.fit(disp=False)
forecast = res.get_forecast(steps=3)
pred = forecast.predicted_mean
ci = forecast.conf_int()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=y.index, y=y, name="Historical"))
fig3.add_trace(go.Scatter(x=pred.index, y=pred, name="Forecast"))
fig3.add_trace(go.Scatter(
    x=list(pred.index) + list(pred.index[::-1]),
    y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
    fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"),
    name="95% CI", showlegend=False
))
fig3.update_layout(title="Forecasted Incidence Rate")
st.plotly_chart(fig3, use_container_width=True)

# Hospital Map
st.subheader("🗺️ Screening Hospital Finder")
CSV_PATH = "demo_hospitals_with_coordinates.csv"
if os.path.exists(CSV_PATH):
    hosp = pd.read_csv(CSV_PATH)
    st.write(f"{len(hosp)} hospitals found.")
    map_center = [hosp["latitude"].mean(), hosp["longitude"].mean()]
    m = folium.Map(location=map_center, zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in hosp.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=f"{row['Name']}<br>{row['Caza']}<br>{row['Phone']}",
            icon=folium.Icon(color="red")
        ).add_to(marker_cluster)
    st_folium(m, width=700, height=500)
else:
    st.error("Hospital coordinates file not found.")

# Footer
st.markdown("---")
st.markdown(
    '<p style="font-size:18px; font-style:italic; color:#8B0000;">'
    '“Early detection saves lives. Awareness is the first step to prevention.”'
    '</p>', unsafe_allow_html=True
)
st.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")
