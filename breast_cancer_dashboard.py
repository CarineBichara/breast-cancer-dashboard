import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joypy
from statsmodels.tsa.statespace.sarimax import SARIMAX
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium

# ────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breast Cancer Awareness", layout="wide")

# ────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ────────────────────────────────────────────────────────────────
DATA_PATH = "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
GCO_PATH  = "GCO_Lebanon_rates.csv"
HOSP_PATH = "demo_hospitals_with_coordinates.csv"

# Patient‑level dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["date_of_surgery"]    = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"]      = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# GCO national rates
gco = pd.read_csv(GCO_PATH)
latest_gco = gco[gco.year == gco.year.max()]

# ────────────────────────────────────────────────────────────────
# Sidebar – interactive filters
# ────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")

sel_gender = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
sel_stage  = st.sidebar.multiselect("Tumor Stage", df["tumour_stage"].dropna().unique(), default=list(df["tumour_stage"].dropna().unique()))
sel_status = st.sidebar.multiselect("Patient Status", df["patient_status"].dropna().unique(), default=list(df["patient_status"].dropna().unique()))
min_age, max_age = int(df["age"].min()), int(df["age"].max())
sel_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))

# Apply filters
fltr = (
    df["gender"].isin(sel_gender)
    & df["tumour_stage"].isin(sel_stage)
    & df["patient_status"].isin(sel_status)
    & df["age"].between(sel_age[0], sel_age[1])
)
filtered = df.loc[fltr].copy()

# ────────────────────────────────────────────────────────────────
# Header & intro
# ────────────────────────────────────────────────────────────────
st.image("pink_ribbon.png", width=80)
st.title("Breast Cancer Awareness Dashboard – Lebanon")
st.caption("Developed by Carine Bichara")
st.image("manandwoman.png", width=600)

st.markdown(
    "### Breast cancer in Lebanon:\n"
    "- **Women** ≈ 75.5 cases / 100 000\n"
    "- **Men**   ≈ 0.8 cases / 100 000\n\n"
    "Both should know their risk and get screened."
)

st.info("**What you can do today:** Talk to your doctor about screening options or find a clinic near you.")

with st.expander("What Do These Metrics Mean?"):
    st.markdown(
        "- **YLL**: Years of Life Lost due to early death  \n"
        "- **YLD**: Years Lived with Disability  \n"
        "- **DALY**: Disability‑Adjusted Life Years (YLL + YLD)  \n"
        "- **%**: Share of the total within this cohort"
    )

# ────────────────────────────────────────────────────────────────
# KPI Row
# ────────────────────────────────────────────────────────────────
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Filtered Patients", len(filtered))
col_kpi2.metric("Deaths", int(filtered["mortality"].sum()))
col_kpi3.metric("DALYs", int(filtered["daly"].sum()))

# ────────────────────────────────────────────────────────────────
# Row 1 – Gender‑based burden
# ────────────────────────────────────────────────────────────────
st.markdown("### Gender‑Based Burden")

agg = (
    filtered.groupby("gender").agg(
        cases=("gender", "count"),
        deaths=("mortality", "sum"),
        daly=("daly", "sum"),
    ).reset_index()
)
for col in ["cases", "deaths", "daly"]:
    agg[f"{col}_pct"] = agg[col] / agg[col].sum() * 100

red_palette = sns.color_palette("Reds", 4)
fig_gender, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (metric, title) in enumerate(zip(["cases_pct", "deaths_pct", "daly_pct"], ["% Cases", "% Deaths", "% DALYs"])):
    ax = axes[i]
    sns.barplot(data=agg, x="gender", y=metric, palette=[red_palette[3], red_palette[1]], ax=ax)
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_yticks([])
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom")

st.pyplot(fig_gender)

with st.expander("Interpretation: Gender‑Based Burden"):
    st.markdown(
        "- **Women** represent ~98 % of cases, deaths, and DALYs.  \n"
        "- **Men** account for the remaining ~2 %.  \n\n"
        "Awareness and screening must focus on women **and** ensure men aren’t overlooked."
    )

# ────────────────────────────────────────────────────────────────
# Row 2 – Age distribution & Surgery type
# ────────────────────────────────────────────────────────────────
col_age, col_surg = st.columns(2)

# Age distribution
with col_age:
    st.markdown("### Age Distribution")
    bins = list(range(30, 95, 5))
    labels = [f"{b}-{b+4}" for b in bins[:-1]]
    fig_age, ax_age = plt.subplots(figsize=(6, 5))
    counts, _, patches = ax_age.hist(filtered["age"], bins=bins, rwidth=0.9, edgecolor="white")
    for i, p in enumerate(patches):
        p.set_facecolor("#F4CCCC" if bins[i] < 45 or bins[i] >= 65 else "#B22222")
    for rect, lbl in zip(patches, counts):
        if lbl:
            ax_age.text(rect.get_x() + rect.get_width()/2, lbl+1, int(lbl), ha="center")
    ax_age.set_xticks([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    ax_age.set_xticklabels(labels, fontsize=8)
    ax_age.set_yticks([])
    ax_age.set_title("Patient Age Distribution")
    st.pyplot(fig_age)

    with st.expander("Interpretation: Age Distribution"):
        st.markdown("Most patients cluster between **45 – 65 years** – prime target group for screening.")

# Surgery breakdown
with col_surg:
    st.markdown("### Surgery Type Breakdown")
    with st.expander("Surgery Definitions"):
        st.markdown("- **Lumpectomy**: Tumour removed, breast preserved  \n- **Simple Mastectomy**: Whole breast removed  \n- **Modified Radical Mastectomy**: Breast + lymph nodes removed")

    surg_counts = filtered["surgery_type"].value_counts()
    surg_colors = {"Simple Mastectomy": red_palette[1], "Lumpectomy": red_palette[2], "Modified Radical Mastectomy": red_palette[3]}
    colors = [surg_colors.get(x, red_palette[0]) for x in surg_counts.index]
    fig_surg, ax_surg = plt.subplots()
    ax_surg.pie(surg_counts, labels=surg_counts.index, autopct="%1.0f%%", startangle=90, colors=colors)
    ax_surg.axis("equal")
    st.pyplot(fig_surg)

    with st.expander("Interpretation: Surgery Types"):
        st.markdown("Early detection can shift more patients toward **lumpectomy** (breast‑conserving).")

# ────────────────────────────────────────────────────────────────
# Row 3 – DALY heat‑map
# ────────────────────────────────────────────────────────────────
st.markdown("### Average DALY by Age × Tumor Stage")
filtered["age_group"] = pd.cut(filtered["age"], bins=bins, labels=labels, right=False)
pivot = filtered.groupby(["age_group", "tumour_stage"])["daly"].mean().unstack().fillna(0)
fig_heat = px.imshow(pivot, labels=dict(x="Tumor Stage", y="Age Group", color="Avg DALY"), text_auto=".1f", color_continuous_scale="Reds", aspect="auto")
fig_heat.update_layout(yaxis_autorange="reversed")
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Interpretation: DALY Heat‑Map"):
    st.markdown("Younger Stage I patients show the highest average DALYs – emphasising prompt detection.")

# ────────────────────────────────────────────────────────────────
# Row 4 – YLL distribution
# ────────────────────────────────────────────────────────────────
st.markdown("### Years of Life Lost by Tumor Stage")
fig_yll, ax_yll = plt.subplots(figsize=(8, 6))
joypy.joyplot(data=filtered, by="tumour_stage", column="yll", ax=ax_yll, kind="counts", colormap=plt.cm.Reds, legend=False, bins=40, fade=True)
ax_yll.set_xlabel("Years of Life Lost (YLL)")
ax_yll.set_ylabel("Tumor Stage")
ax_yll.set_title("Distribution of YLL by Tumor Stage")
st.pyplot(fig_yll)

with st.expander("Interpretation: YLL Distribution"):
    st.markdown("Stage I still shows wide YLL range – delayed diagnosis matters even in early stage.")

# ────────────────────────────────────────────────────────────────
# Row 5 – National incidence & mortality
# ────────────────────────────────────────────────────────────────
col_nat_inc, col_nat_mort = st.columns(2)

# Incidence
with col_nat_inc:
    fig_inc = px.bar(
        latest_gco,
        x="gender",
        y="incidence_rate",
        text_auto=".1f",
        labels={"incidence_rate": "Incidence / 100 000", "gender": "Gender"},
        title=f"Incidence Rate – {int(latest_gco.year.iloc[0])}",
        color="gender",
        color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
    )
    fig_inc.update_layout(
        showlegend=False,
        yaxis=dict(showticklabels=False),
        xaxis=dict(title="Gender"),
    )
    fig_inc.update_traces(textposition="outside")
    st.plotly_chart(fig_inc, use_container_width=True)

    with st.expander("Interpretation: Incidence"):
        inc_f = latest_gco.query("gender == 'Female'").incidence_rate.iloc[0]
        inc_m = latest_gco.query("gender == 'Male'").incidence_rate.iloc[0]
        st.markdown(f"Women: **{inc_f:.1f}** / 100 000   |   Men: **{inc_m:.1f}** / 100 000")

# Mortality
with col_nat_mort:
    fig_mort = px.bar(
        latest_gco,
        x="gender",
        y="mortality_rate",
        text_auto=".1f",
        labels={"mortality_rate": "Mortality / 100 000", "gender": "Gender"},
        title=f"Mortality Rate – {int(latest_gco.year.iloc[0])}",
        color="gender",
        color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
    )
    fig_mort.update_layout(
        showlegend=False,
        yaxis=dict(showticklabels=False),
        xaxis=dict(title="Gender"),
    )
    fig_mort.update_traces(textposition="outside")
    st.plotly_chart(fig_mort, use_container_width=True)

    with st.expander("Interpretation: Mortality"):
        mort_f = latest_gco.query("gender == 'Female'").mortality_rate.iloc[0]
        mort_m = latest_gco.query("gender == 'Male'").mortality_rate.iloc[0]
        st.markdown(f"Women: **{mort_f:.1f}** / 100 000   |   Men: **{mort_m:.1f}** / 100 000")

# ────────────────────────────────────────────────────────────────
# Row 6 – Trends & short‑term forecast
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Trends & Short‑Term Forecast")

# Prepare yearly series
ts = gco.groupby("year")[["incidence_rate", "mortality_rate"]].mean().reset_index()
ts_plot = ts.copy()
ts_plot["year_dt"] = pd.to_datetime(ts_plot["year"], format="%Y")

fig_trend = px.line(
    ts_plot,
    x="year_dt",
    y=["incidence_rate", "mortality_rate"],
    labels={"value": "Rate / 100 000", "year_dt": "Year", "variable": "Metric"},
    title="Incidence vs Mortality Over Time",
    color_discrete_sequence=["#8B0000", "#FFC1C1"],
)
fig_trend.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig_trend, use_container_width=True)

# 3‑year ARIMA forecast for incidence
inc_series = ts.set_index("year")["incidence_rate"]
model = SARIMAX(inc_series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

horizon = 3
forecast = res.get_forecast(steps=horizon)
pred = forecast.predicted_mean
ci = forecast.conf_int(alpha=0.05)

future_years = list(range(ts["year"].max() + 1, ts["year"].max() + 1 + horizon))
fc_df = pd.DataFrame({"year": future_years, "forecast": pred.values, "lower": ci.iloc[:, 0].values, "upper": ci.iloc[:, 1].values})

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=ts["year"], y=ts["incidence_rate"], mode="lines+markers", name="Historical", line=dict(color="#8B0000")))
fig_fc.add_trace(go.Scatter(x=fc_df["year"], y=fc_df["forecast"], mode="lines+markers", name="Forecast", line=dict(color="#FFC1C1")))
fig_fc.add_trace(go.Scatter(x=fc_df["year"].tolist() + fc_df["year"][::-1].tolist(), y=fc_df["upper"].tolist() + fc_df["lower"][::-1].tolist(), fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% CI"))
fig_fc.update_layout(title="3‑Year Forecast: Incidence", xaxis_title="Year", yaxis_title="Incidence / 100 000")

st.plotly_chart(fig_fc, use_container_width=True)

with st.expander("Interpretation: Forecast"):
    y0, y1, y2 = fc_df["forecast"][:3].round(1).tolist()
    st.markdown(f"Incidence is projected to rise from **{y0}** to **{y2}** per 100 000 over the next three years, warranting strengthened screening efforts.")

# ────────────────────────────────────────────────────────────────
# Row 7 – Hospital locator map
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗺️ Find a Breast‑Cancer Screening Hospital")

if not os.path.exists(HOSP_PATH):
    st.error("Hospital CSV file not found.")
else:
    hosp = pd.read_csv(HOSP_PATH)

    # Geocode missing coordinates once, cache to JSON
    if {"latitude", "longitude"}.issubset(hosp.columns) is False or hosp[["latitude", "longitude"]].isna().any().any():
        geolocator = Nominatim(user_agent="bc-screening-map")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        cache_path = "geo_cache.json"
        geo_cache = json.load(open(cache_path, "r", encoding="utf-8")) if os.path.exists(cache_path) else {}

        def fetch_latlon(row):
            if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                return row["latitude"], row["longitude"]
            key = f"{row['Name']}_{row['Caza']}"
            if key in geo_cache:
                return geo_cache[key]
            loc = geocode(f"{row['Name']}, {row['Caza']}, Lebanon")
            latlon = (loc.latitude, loc.longitude) if loc else (None, None)
            geo_cache[key] = latlon
            time.sleep(1)
            return latlon

        hosp[["latitude", "longitude"]] = hosp.apply(fetch_latlon, axis=1, result_type="expand")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(geo_cache, f, ensure_ascii=False, indent=2)

    query = st.text_input("Search city, caza, or hospital", "").strip().lower()
    mask = (
        hosp["Caza"].astype(str).str.lower().str.contains(query)
        | hosp.get("governorate", hosp["Caza"]).astype(str).str.lower().str.contains(query)
        | hosp["Name"].str.lower().str.contains(query)
    ) if query else slice(None)
    data = hosp.loc[mask].copy()

    st.write(f"**{len(data)} hospital(s) found**")

    if data.empty:
        st.warning("No hospitals match that search. Try another term.")
    else:
        m = folium.Map(location=[data["latitude"].mean(), data["longitude"].mean()], zoom_start=9, tiles="CartoDB positron")
        cluster = MarkerCluster().add_to(m)
        for _, row in data.dropna(subset=["latitude", "longitude"]).iterrows():
            popup = f"<b>{row['Name']}</b><br>Caza: {row['Caza']}<br>Phone: {row['Phone']}"
            folium.Marker([row["latitude"], row["longitude"]], popup=popup, icon=folium.Icon(color="red", icon="plus-sign"),).add_to(cluster)
        st_folium(m, width=750, height=500)
        with st.expander("↕️ See results in a table"):
            st.dataframe(data[["Name", "Caza", "Phone"]].reset_index(drop=True))

# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='font-style:italic; font-size:18px;'>\"Early detection saves lives. Awareness is the first step to prevention.\"</p>", unsafe_allow_html=True)
st.caption("© 2025 Carine Bichara | Breast Cancer Awareness Dashboard")
