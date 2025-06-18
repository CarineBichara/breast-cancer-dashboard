import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joypy
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pytrends.request import TrendReq
import os, json, time
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ────────────────────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Breast Cancer Awareness", layout="wide")

# ────────────────────────────────────────────────────────────────
# Data loading & preprocessing
# ────────────────────────────────────────────────────────────────
FILE_PATH = "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["date_of_surgery"]    = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"]      = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# ────────────────────────────────────────────────────────────────
# Sidebar - Filters
# ────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")
gender = st.sidebar.multiselect(
    "Gender", df["gender"].unique(), default=list(df["gender"].unique())
)
stages = st.sidebar.multiselect(
    "Tumor Stage",
    df["tumour_stage"].dropna().unique(),
    default=list(df["tumour_stage"].dropna().unique()),
)
status = st.sidebar.multiselect(
    "Patient Status",
    df["patient_status"].dropna().unique(),
    default=list(df["patient_status"].dropna().unique()),
)
age_range = st.sidebar.slider(
    "Age Range",
    int(df["age"].min()),
    int(df["age"].max()),
    (int(df["age"].min()), int(df["age"].max())),
)

# Apply filters
filtered = df[
    df["gender"].isin(gender)
    & df["tumour_stage"].isin(stages)
    & df["patient_status"].isin(status)
]
filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

# ────────────────────────────────────────────────────────────────
# Header, logos, and intro text
# ────────────────────────────────────────────────────────────────
st.image("pink_ribbon.png", width=80)
st.title("Breast Cancer Awareness")
st.markdown("Developed by **Carine Bichara**")
st.image("manandwoman.png", width=600)

st.markdown(
    "### Breast cancer in Lebanon:\n"
    "- Women face ~75.5 cases per 100 000 population  \n"
    "- Men face   ~0.8  cases per 100 000 population  \n\n"
    "Both should know their risk and get screened."
)
st.info("**What you can do today:** Talk to your doctor about screening options or find a clinic near you.")

# ────────────────────────────────────────────────────────────────
# Data overview & glossary
# ────────────────────────────────────────────────────────────────
st.markdown("### Data Overview")
st.markdown(
    "We’ll start by exploring patient-level data from a public Kaggle dataset. "
    "Then we’ll layer in Lebanon’s population-standardized rates from the Global Cancer Observatory (GCO)."
)

with st.expander("What Do These Metrics Mean?"):
    st.markdown(
        "- **YLL** – Years of Life Lost due to early death  \n"
        "- **YLD** – Years Lived with Disability  \n"
        "- **DALY** – Disability-Adjusted Life Years (YLL + YLD)  \n"
        "- **%** – Share of total cases, deaths, YLL, YLD, or DALYs"
    )

col1, col2, col3 = st.columns(3)
col1.metric("Filtered Patients", len(filtered))
col2.metric("Deaths", int(filtered["mortality"].sum()))
col3.metric("DALYs", int(filtered["daly"].sum()))

# ────────────────────────────────────────────────────────────────
# Patient-level insights
# ────────────────────────────────────────────────────────────────
st.header("Patient-Level Insights (Kaggle Dataset)")

# Comparison by gender
grouped = (
    filtered.groupby("gender")
    .agg(
        cases=("gender", "count"),
        deaths=("mortality", "sum"),
        yll=("yll", "sum"),
        yld=("yld", "sum"),
        daly=("daly", "sum"),
    )
    .reset_index()
)
totals = grouped[["cases", "deaths", "yll", "yld", "daly"]].sum()
for col in ["cases", "deaths", "yll", "yld", "daly"]:
    grouped[f"{col}_pct"] = grouped[col] / totals[col] * 100

red_palette = sns.color_palette("Reds", n_colors=4)

st.subheader("Comparison by Gender")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (metric, title) in enumerate(
    zip(["cases_pct", "deaths_pct", "daly_pct"], ["% of Cases", "% of Deaths", "% of DALYs"])
):
    ax = axes[i]
    colors = [red_palette[3], red_palette[1]]
    sns.barplot(x="gender", y=metric, data=grouped, palette=colors, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_yticks([])
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )
st.pyplot(fig)

with st.expander("Interpretation: Comparison by Gender"):
    st.markdown(
        "- **Female patients represent ~98.8 % of all cases**; **males ~1.2 %**.  \n"
        "- **Women account for ~98.5 % of deaths**; **men 1.5 %**.  \n"
        "- **Women bear ~98.4 % of DALYs**; **men 1.6 %**.  \n\n"
        "**Key Message:** Breast-cancer burden is heavily concentrated in women, but men must not be overlooked."
    )

# Age distribution ---------------------------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Age Distribution")
    fig_age, ax = plt.subplots(figsize=(6, 5))
    bins = list(range(30, 95, 5))
    counts, _, patches = ax.hist(filtered["age"], bins=bins, edgecolor="white", rwidth=0.9)
    for i, patch in enumerate(patches):
        patch.set_facecolor("#F4CCCC" if bins[i] < 45 or bins[i] >= 65 else "#B22222")
    for rect, label in zip(patches, counts):
        if label > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                label + 1,
                f"{int(label)}",
                ha="center",
                va="bottom",
            )
    ax.set_title("Age Distribution of Breast Cancer Patients")
    ax.set_xlabel("Age Group")
    ax.set_xticks([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    ax.set_xticklabels(
        [
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80-84",
            "85-89",
        ],
        fontsize=8,
    )
    ax.set_yticks([])
    ax.grid(False)
    st.pyplot(fig_age)

    with st.expander("Interpretation: Age Distribution"):
        st.markdown(
            "- Most patients are **45–65 years** old.  \n"
            "- The **50–54** age group has the highest count.  \n\n"
            "**Key Message:** Target screening to ages 45–65."
        )

# Surgery-type breakdown ----------------------------------------
with colB:
    st.subheader("Surgery Type Breakdown")
    with st.expander("Click to view surgery procedure definitions"):
        st.markdown(
            "- **Lumpectomy** – Tumor removed, breast kept  \n"
            "- **Simple Mastectomy** – Whole breast removed, lymph nodes not removed  \n"
            "- **Modified Radical Mastectomy** – Breast + lymph nodes removed"
        )
    surgery_counts = filtered["surgery_type"].value_counts()
    surgery_colors = {
        "Simple Mastectomy": red_palette[1],
        "Lumpectomy": red_palette[2],
        "Modified Radical Mastectomy": red_palette[3],
        "Other": red_palette[0],
    }
    colors = [surgery_colors.get(x, red_palette[0]) for x in surgery_counts.index]
    fig_surg, ax = plt.subplots()
    ax.pie(
        surgery_counts,
        labels=surgery_counts.index,
        autopct="%1.0f%%",
        startangle=90,
        colors=colors,
    )
    ax.axis("equal")
    st.pyplot(fig_surg)

    with st.expander("Interpretation: Surgery Type Breakdown"):
        st.markdown(
            "- **Modified Radical Mastectomy (29 %)**  \n"
            "- **Lumpectomy (21 %)**  \n"
            "- **Simple Mastectomy (20 %)**  \n"
            "- **Other Surgeries (31 %)**  \n\n"
            "**Key Message:** Early detection reduces the need for invasive surgery."
        )

# DALY heat-map -------------------------------------------------
st.subheader("Average DALY by Age Group and Tumor Stage")
bins = list(range(30, 95, 5))
labels = [f"{b}-{b + 4}" for b in bins[:-1]]
filtered["age_group"] = pd.cut(filtered["age"], bins=bins, labels=labels, right=False)
pivot = (
    filtered.groupby(["age_group", "tumour_stage"])["daly"]
    .mean()
    .unstack()
    .fillna(0)
)
fig_heat = px.imshow(
    pivot,
    labels=dict(x="Tumor Stage", y="Age Group", color="Average DALY"),
    text_auto=".1f",
    color_continuous_scale="Reds",
    aspect="auto",
)
fig_heat.update_layout(
    height=600,
    width=800,
    title="Average DALY by Age Group and Tumor Stage",
    yaxis_autorange="reversed",
)
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Interpretation: DALY by Age and Tumor Stage"):
    st.markdown(
        "- Younger patients with Stage I show highest average DALYs.  \n"
        "- DALYs decline with advancing age.  \n\n"
        "Use this to prioritize interventions by age and stage."
    )

# YLL distribution (joy-plot) ----------------------------------
st.subheader("Years of Life Lost by Tumor Stage")
fig, ax = plt.subplots(figsize=(8, 6))
joypy.joyplot(
    filtered,
    by="tumour_stage",
    column="yll",
    ax=ax,
    kind="counts",
    colormap=plt.cm.Reds,
    legend=False,
    bins=40,
    fade=True,
)
ax.set_xlabel("Years of Life Lost (YLL)")
ax.set_ylabel("Tumor Stage")
ax.set_title("Distribution of YLL by Tumor Stage")
st.pyplot(fig)

with st.expander("Interpretation: YLL by Tumor Stage"):
    st.markdown(
        "- Stage I shows wide variability in YLL.  \n"
        "- Stages II & III are more concentrated with lower YLL.  \n"
        "Highlights the importance of early detection."
    )

# ────────────────────────────────────────────────────────────────
# National benchmarks (GCO data)
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "While patient-level insights are useful, they don’t account for population size. "
    "We now turn to population-standardized rates from the Global Cancer Observatory."
)

st.header("National Benchmarks (GCO Data)")
rates = pd.read_csv("GCO_Lebanon_rates.csv")
latest = rates.loc[rates.year == rates.year.max()]

# Incidence bar
fig1 = px.bar(
    latest,
    x="gender",
    y="incidence_rate",
    text_auto=".1f",
    labels=dict(incidence_rate="Incidence per 100 000", gender="Gender"),
    title=f"Lebanon Incidence Rate — {int(latest.year.iloc[0])}",
    color="gender",
    color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
)
fig1.update_layout(showlegend=False, yaxis=dict(showticklabels=False))
fig1.update_traces(textposition="outside")
st.plotly_chart(fig1, use_container_width=True)

with st.expander("Interpretation: Incidence Rate"):
    st.markdown(
        f"- **Women:** {latest.query('gender == \"Female\"')['incidence_rate'].values[0]:.1f} per 100 000  \n"
        f"- **Men:** {latest.query('gender == \"Male\"')['incidence_rate'].values[0]:.1f} per 100 000  \n\n"
        "Screening must focus on women but remain inclusive of men."
    )

# Mortality bar
fig2 = px.bar(
    latest,
    x="gender",
    y="mortality_rate",
    text_auto=".1f",
    labels=dict(mortality_rate="Mortality per 100 000", gender="Gender"),
    title=f"Lebanon Mortality Rate — {int(latest.year.iloc[0])}",
    color="gender",
    color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
)
fig2.update_layout(showlegend=False, yaxis=dict(showticklabels=False))
fig2.update_traces(textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Interpretation: Mortality Rate"):
    st.markdown(
        f"- **Women:** {latest.query('gender == \"Female\"')['mortality_rate'].values[0]:.1f} per 100 000  \n"
        f"- **Men:** {latest.query('gender == \"Male\"')['mortality_rate'].values[0]:.1f} per 100 000  \n\n"
        "Again, burden is much higher in women but men must not be ignored."
    )

# ────────────────────────────────────────────────────────────────
# Time-trend analysis & short-term forecast
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Time Trends & Short-Term Forecast")

ts = rates.groupby("year")[["incidence_rate", "mortality_rate"]].mean().reset_index()
ts_plot = ts.copy()
ts_plot["year_dt"] = pd.to_datetime(ts_plot["year"], format="%Y")

fig_trend = px.line(
    ts_plot,
    x="year_dt",
    y=["incidence_rate", "mortality_rate"],
    labels=dict(value="Rate per 100 000", year_dt="Year", variable="Metric"),
    title="Lebanon: Incidence & Mortality Trends",
    color_discrete_sequence=["#8B0000", "#FFC1C1"],
)
fig_trend.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig_trend, use_container_width=True)

with st.expander("Interpretation: Time Trends"):
    st.markdown(
        f"- Incidence rose from **{ts.incidence_rate.iloc[0]:.1f}** to **{ts.incidence_rate.iloc[-1]:.1f}** per 100 000.  \n"
        f"- Mortality grew from **{ts.mortality_rate.iloc[0]:.1f}** to **{ts.mortality_rate.iloc[-1]:.1f}** per 100 000.  \n"
        "The gap remains wide, indicating better survival relative to incidence."
    )

# 3-year ARIMA forecast (incidence)
y = ts.set_index("year")["incidence_rate"]
model = SARIMAX(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

horizon = 3
fc = res.get_forecast(steps=horizon)
pred = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)

years_fc = list(range(ts["year"].max() + 1, ts["year"].max() + 1 + horizon))
df_fc = pd.DataFrame(
    {"year": years_fc, "forecast": pred.values, "lower": ci.iloc[:, 0], "upper": ci.iloc[:, 1]}
)

fig_fc = go.Figure()
fig_fc.add_trace(
    go.Scatter(
        x=ts["year"], y=ts["incidence_rate"], name="Historical",
        mode="lines+markers", line=dict(color="#8B0000"), marker=dict(color="#8B0000")
    )
)
fig_fc.add_trace(
    go.Scatter(
        x=df_fc["year"], y=df_fc["forecast"], name="Forecast",
        mode="lines+markers", line=dict(color="#FFC1C1"), marker=dict(color="#FFC1C1")
    )
)
fig_fc.add_trace(
    go.Scatter(
        x=df_fc["year"].tolist() + df_fc["year"][::-1].tolist(),
        y=df_fc["upper"].tolist() + df_fc["lower"][::-1].tolist(),
        fill="toself", fillcolor="rgba(255,0,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% CI"
    )
)
fig_fc.update_layout(
    title="3-Year Forecast: Incidence Rate",
    xaxis_title="Year",
    yaxis_title="Incidence per 100 000",
)
st.subheader("Short-Term Forecast")
st.plotly_chart(fig_fc, use_container_width=True)

with st.expander("Interpretation: 3-Year Forecast"):
    st.markdown(
        f"The forecast shows incidence rising from **{df_fc.forecast.iloc[0]:.1f}** "
        f"to **{df_fc.forecast.iloc[-1]:.1f}** per 100 000 by {df_fc.year.iloc[-1]}. "
        "Confidence intervals widen, underscoring uncertainty and the need for proactive action."
    )

# ────────────────────────────────────────────────────────────────
# Hospital map
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🗺️ Find a Breast-Cancer Screening Hospital")

CSV_PATH = "lebanon_private_hospitals_complete.csv"

if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()

hosp = pd.read_csv(CSV_PATH)

# Normalize text columns for reliable search
hosp["name_clean"] = hosp["Name"].str.strip().str.lower()
hosp["caza_clean"] = hosp["Caza"].astype(str).str.strip().str.lower()
hosp["gov_clean"] = hosp.get("governorate", hosp["Caza"]).astype(str).str.strip().str.lower()

# Geocode if coordinates are missing
if {"latitude", "longitude"}.issubset(hosp.columns) is False or hosp[["latitude", "longitude"]].isna().any().any():
    geolocator = Nominatim(user_agent="bc-screening-map")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    cache_path = os.path.join(os.path.dirname(CSV_PATH) or ".", "geo_cache.json")
    geo_cache = json.load(open(cache_path, "r", encoding="utf-8")) if os.path.exists(cache_path) else {}

    def fetch_latlon(row):
        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            return row["latitude"], row["longitude"]
        key = f'{row["Name"]}_{row["Caza"]}'
        if key in geo_cache:
            return geo_cache[key]
        loc = geocode(f'{row["Name"]}, {row["Caza"]}, Lebanon')
        latlon = (loc.latitude, loc.longitude) if loc else (None, None)
        geo_cache[key] = latlon
        time.sleep(1)
        return latlon

    hosp[["latitude", "longitude"]] = hosp.apply(fetch_latlon, axis=1, result_type="expand")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(geo_cache, f, ensure_ascii=False, indent=2)

# Search bar
query = st.text_input("🔎 Search by City / Caza / Hospital").strip().lower()
mask = (
    hosp["name_clean"].str.contains(query)
    | hosp["caza_clean"].str.contains(query)
    | hosp["gov_clean"].str.contains(query)
) if query else slice(None)
data = hosp.loc[mask].copy()

st.write(f"**{len(data)} hospital(s) found**")

if data.empty:
    st.warning("No hospitals match that search.")
else:
    # Generate map
    m = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        zoom_start=9,
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google",
    )
    cluster = MarkerCluster().add_to(m)

    for _, row in data.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        popup_html = (
            f"<b>{row['Name']}</b><br>"
            f"Caza: {row['Caza']}<br>"
            f"Phone: {row['Phone']}<br>"
            f"Investment #: {row['Investment_Authorization_Nb']}"
        )

        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=popup_html,
            tooltip=row["Name"],  # ← this shows name on hover
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(cluster)

    st_folium(m, width=750, height=500)

    with st.expander("↕️ See hospitals in a table"):
        st.dataframe(
            data[["Name", "Caza", "Phone", "Investment_Authorization_Nb"]].reset_index(drop=True)
        )
# ────────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-weight:bold; font-size:18px; font-style:italic;">'
    "Early detection saves lives. Awareness is the first step to prevention."
    "</p>",
    unsafe_allow_html=True,
)
st.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")
