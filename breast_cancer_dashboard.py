# Breast-Cancer Dashboard — compact visual layout
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joypy
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os, json, time, folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ───────────────────── page setup
st.set_page_config(page_title="Breast Cancer Awareness", layout="wide")

# ───────────────────── data
FILE_PATH = "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df["date_of_surgery"]    = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"]      = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# ───────────────────── sidebar filters
st.sidebar.title("Filters")
gender     = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
stages     = st.sidebar.multiselect("Tumor Stage", df["tumour_stage"].dropna().unique(),
                                    default=list(df["tumour_stage"].dropna().unique()))
status     = st.sidebar.multiselect("Patient Status", df["patient_status"].dropna().unique(),
                                    default=list(df["patient_status"].dropna().unique()))
age_range  = st.sidebar.slider("Age Range", int(df["age"].min()), int(df["age"].max()),
                               (int(df["age"].min()), int(df["age"].max())))

filtered = df[(df["gender"].isin(gender))
              & (df["tumour_stage"].isin(stages))
              & (df["patient_status"].isin(status))]
filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

# ───────────────────── header
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

# ───────────────────── quick KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Patients", len(filtered))
col2.metric("Deaths", int(filtered["mortality"].sum()))
col3.metric("DALYs",  int(filtered["daly"].sum()))

# ───────────────────── Patient-level insights
st.header("Patient-Level Insights (Kaggle Dataset)")

# ―― Comparison by gender (compact 3-panel bar plot)
grouped = (filtered.groupby("gender")
           .agg(cases=("gender", "count"),
                deaths=("mortality", "sum"),
                yll=("yll","sum"), yld=("yld","sum"), daly=("daly","sum"))
           .reset_index())
totals = grouped[["cases","deaths","yll","yld","daly"]].sum()
for c in ["cases","deaths","yll","yld","daly"]:
    grouped[f"{c}_pct"] = grouped[c] / totals[c] * 100

red_palette = sns.color_palette("Reds", n_colors=4)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), dpi=100)
fig.subplots_adjust(wspace=0.4, left=0.05, right=0.97, top=0.85, bottom=0.2)
for i, (metric, title) in enumerate(zip(
        ["cases_pct","deaths_pct","daly_pct"],
        ["% of Cases","% of Deaths","% of DALYs"])):
    ax = axes[i]
    sns.barplot(x="gender", y=metric, data=grouped,
                palette=[red_palette[3], red_palette[1]], ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(""); ax.set_yticks([])
    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
st.pyplot(fig)
with st.expander("Interpretation: Gender Comparison"):
    st.markdown("""
- **Female patients represent ~98.8% of all cases**, while **males account for only ~1.2%**.  
- **Women account for ~98.5% of all deaths**, compared to **1.5% for men**, indicating mortality is overwhelmingly among female patients.  
- **Similarly, female patients bear ~98.4% of total DALYs**, versus **1.6% for males**, reflecting the proportional disease burden.

**Key Message:**  
Breast cancer burden is heavily concentrated in women, underscoring the need for gender-targeted awareness and screening.  
Yet **men must not be overlooked**, as even a small number of cases and deaths represent important opportunities for early detection and care.
""")

# ―― Age histogram (compact)
colA, colB = st.columns(2)
with colA:
    st.subheader("Age Distribution")
    fig_age, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    bins = list(range(30, 95, 5))
    counts, _, patches = ax.hist(filtered["age"], bins=bins, edgecolor="white", rwidth=0.9)
    for i, p in enumerate(patches):
        p.set_facecolor("#F4CCCC" if bins[i]<45 or bins[i]>=65 else "#B22222")
    for rect, lbl in zip(patches, counts):
        if lbl:
            ax.text(rect.get_x()+rect.get_width()/2, lbl+1, f"{int(lbl)}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Age Distribution of Patients", fontsize=10)
    ax.set_xlabel("Age Group"); ax.set_yticks([])
    ax.set_xticks([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])
    ax.set_xticklabels(
        ["30-34","35-39","40-44","45-49","50-54","55-59","60-64",
         "65-69","70-74","75-79","80-84","85-89"],
        fontsize=7)
    st.pyplot(fig_age)

with st.expander("Interpretation: Age Distribution"):
    st.markdown("""
    - Most patients are between **45 and 65** years old.  
    - The **50–54** age group has the highest count.  
    - Lower counts at the extremes may reflect screening patterns or risk differences.

    **Key Message:**  
    Focus screening efforts on ages **45–65** for maximum impact.
    """)

# ―― Surgery pie (compact)
with colB:
    st.subheader("Surgery Type Breakdown")
    surgery_counts = filtered["surgery_type"].value_counts()
    colors = {"Simple Mastectomy":red_palette[1], "Lumpectomy":red_palette[2],
              "Modified Radical Mastectomy":red_palette[3], "Other":red_palette[0]}
    fig_surg, ax = plt.subplots(figsize=(5.5, 3.5), dpi=100)
    ax.pie(surgery_counts, labels=surgery_counts.index, autopct="%1.0f%%",
           startangle=90, colors=[colors.get(x,red_palette[0]) for x in surgery_counts.index])
    ax.axis("equal"); st.pyplot(fig_surg)
with st.expander("Interpretation: Surgery Type Breakdown"):
    st.markdown("""
    - **Modified Radical Mastectomy (29%)**  
    - **Lumpectomy (21%)**  
    - **Simple Mastectomy (20%)**  
    - **Other Surgeries (31%)**  

    **Key Message:**  
    Early detection can reduce the need for invasive procedures.
    """)


# ―― DALY heat-map (height/width trimmed)
st.subheader("Average DALY by Age Group and Tumor Stage")
labels = [f"{b}-{b+4}" for b in bins[:-1]]
filtered["age_group"] = pd.cut(filtered["age"], bins=bins, labels=labels, right=False)
pivot = (filtered.groupby(["age_group","tumour_stage"])["daly"]
         .mean().unstack().fillna(0))
fig_heat = px.imshow(
    pivot, labels=dict(x="Tumor Stage", y="Age Group", color="Average DALY"),
    text_auto=".1f", color_continuous_scale="Reds", aspect="auto")
fig_heat.update_layout(height=350, width=600, margin=dict(l=10,r=10,t=40,b=20),
                       yaxis_autorange="reversed")
st.plotly_chart(fig_heat, use_container_width=True)
with st.expander("Interpretation: Average DALY by Age & Tumor Stage"):
    st.markdown("""
- Younger patients with **Stage I** have the highest average DALYs.  
- DALYs decline with advancing age and vary by tumor stage.  

**Key Message:**  
Use this to prioritize interventions by **age group and stage** — younger patients with early-stage diagnoses lose the most healthy years.
""")

# ―― Joyplot (compact)
st.subheader("Years of Life Lost by Tumor Stage")
fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=100)
joypy.joyplot(filtered, by="tumour_stage", column="yll", ax=ax, kind="counts",
              colormap=plt.cm.Reds, legend=False, bins=40, fade=True)
ax.set_xlabel("Years of Life Lost (YLL)"); ax.set_ylabel("Tumor Stage")
ax.set_title("Distribution of YLL by Tumor Stage", fontsize=10)
st.pyplot(fig)
with st.expander("Interpretation: YLL by Tumor Stage"):
    st.markdown("""
- **Stage I** shows wide variability in YLL.  
- **Stages II and III** have more clustered, lower YLL distributions.  

**Key Message:**  
Highlights the importance of **early detection** to reduce life years lost.
""")

# ───────────────────── National benchmarks (GCO)
rates = pd.read_csv("GCO_Lebanon_rates.csv")

# 1️⃣  Latest-year slice
latest = rates.loc[rates.year == rates.year.max()].copy()

# 2️⃣  ▸ Gender-share variables (for the bar chart that must be comparable to Kaggle %)
latest["total_incidence"] = latest["incidence_rate"].sum()
latest["total_mortality"] = latest["mortality_rate"].sum()
latest["incidence_pct_of_cases"]  = latest["incidence_rate"]  / latest["total_incidence"]  * 100
latest["mortality_pct_of_cases"]  = latest["mortality_rate"]  / latest["total_mortality"]  * 100

# 3️⃣  ▸ Population-rate variables (for time-trend & forecast, if you still want them)
rates["incidence_pct_pop"]  = rates["incidence_rate"]  / 1_000 * 100   # per-100 000 → %
rates["mortality_pct_pop"]  = rates["mortality_rate"]  / 1_000 * 100
latest["incidence_pct_pop"] = latest["incidence_rate"] / 1_000 * 100
latest["mortality_pct_pop"] = latest["mortality_rate"] / 1_000 * 100

# National benchmarks — gender share of incidence & mortality (in % of cases)
for metric, title in [("incidence_pct_of_cases", "Share of Incidence"),
                      ("mortality_pct_of_cases", "Share of Mortality")]:
    fig_tmp = px.bar(
        latest,
        x="gender",
        y=metric,
        text_auto=".1f",
        labels={metric: f"{title} (%)", "gender": "Gender"},
        title=f"Lebanon {title} — {int(latest.year.iloc[0])}",
        color="gender",
        color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
    )
    fig_tmp.update_layout(
        height=330,
        width=500,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=20),
        yaxis=dict(title=f"{title} (%)", showticklabels=False)
    )
    st.plotly_chart(fig_tmp, use_container_width=True)

    # Interpretation
    female_pct = latest.loc[latest.gender == "Female", metric].values[0]
    male_pct   = latest.loc[latest.gender == "Male",   metric].values[0]
    expander_title = ("Interpretation: Incidence Share"
                      if "incidence" in metric
                      else "Interpretation: Mortality Share")
    with st.expander(expander_title):
        st.markdown(f"""
- Women: **{female_pct:.1f}%**  
- Men&nbsp;&nbsp;: **{male_pct:.1f}%**

**Key Insight:**  
This shows *how cases and deaths are distributed between genders* in Lebanon’s population (share of total cases).  
That metric is directly comparable to the Kaggle chart above.
""")
# ───────────────────── Time-trend & forecast (compact)
st.markdown("---"); st.subheader("Time Trends & Short-Term Forecast")
# ───────────────────── Time-trend & short-term forecast  —  PERCENT VALUES
# (make sure `rates` already contains incidence_pct & mortality_pct as shown earlier)

# 1. Prep the time-series dataframe
ts = (
    rates.groupby("year")[["incidence_pct", "mortality_pct"]]
         .mean()
         .reset_index()
)
ts["year_dt"] = pd.to_datetime(ts["year"].astype(str))

# 2. Line chart: historical trends (%)
fig_trend = px.line(
    ts, x="year_dt", y=["incidence_pct", "mortality_pct"],
    labels=dict(value="Rate (%)", year_dt="Year", variable="Metric"),
    title="Lebanon: Incidence & Mortality Trends",
    color_discrete_sequence=["#8B0000", "#FFC1C1"],
)
fig_trend.update_layout(
    height=350, width=700,
    xaxis=dict(rangeslider=dict(visible=True)),
    margin=dict(l=10, r=10, t=40, b=20),
    yaxis_tickformat=".2f"
)
st.plotly_chart(fig_trend, use_container_width=True)
with st.expander("Interpretation: Incidence & Mortality Trends"):
    st.markdown(f"""
- Incidence rose from **{ts['incidence_pct'].iloc[0]:.2f}%** to **{ts['incidence_pct'].iloc[-1]:.2f}%** between {ts['year'].iloc[0]} and {ts['year'].iloc[-1]}.  
- Mortality rose from **{ts['mortality_pct'].iloc[0]:.2f}%** to **{ts['mortality_pct'].iloc[-1]:.2f}%** in the same period.  

**Key Message:**  
The **gap between incidence and mortality** suggests survival is improving — but not fast enough to match rising case numbers.
""")

# 3. Forecast on % series (SARIMAX)
y = ts.set_index("year")["incidence_pct"]         # ← use % column
res = SARIMAX(
    y, order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

h = 3                               # forecast horizon
pred = res.get_forecast(h).predicted_mean
ci   = res.get_forecast(h).conf_int(alpha=0.05)
years_fc = list(range(ts["year"].max() + 1, ts["year"].max() + 1 + h))

df_fc = pd.DataFrame({
    "year":  years_fc,
    "forecast": pred.values,
    "lower":    ci.iloc[:, 0],
    "upper":    ci.iloc[:, 1],
})
with st.expander("Interpretation: Forecast of Incidence"):
    st.markdown(f"""
The forecast shows Lebanon’s breast cancer incidence may rise from **{df_fc['forecast'].iloc[0]:.2f}%**  
to **{df_fc['forecast'].iloc[1]:.2f}%** and **{df_fc['forecast'].iloc[2]:.2f}%** in the coming years.  

**Key Message:**  
This signals a need to scale up **screening, prevention, and early intervention** immediately.
""")

# 4. Plot the forecast
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=ts["year"], y=ts["incidence_pct"],
    name="Historical", mode="lines+markers",
    line=dict(color="#8B0000"), marker=dict(color="#8B0000")
))
fig_fc.add_trace(go.Scatter(
    x=df_fc["year"], y=df_fc["forecast"],
    name="Forecast", mode="lines+markers",
    line=dict(color="#FFC1C1"), marker=dict(color="#FFC1C1")
))
fig_fc.add_trace(go.Scatter(
    x=df_fc["year"].tolist() + df_fc["year"][::-1].tolist(),
    y=df_fc["upper"].tolist() + df_fc["lower"][::-1].tolist(),
    fill="toself", fillcolor="rgba(255,0,0,0.2)",
    line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
    name="95% CI"
))
fig_fc.update_layout(
    title="3-Year Forecast: Incidence (%)",
    xaxis_title="Year", yaxis_title="Incidence (%)",
    height=350, width=700,
    margin=dict(l=10, r=10, t=40, b=20),
    yaxis_tickformat=".2f"
)
st.subheader("Short-Term Forecast")
st.plotly_chart(fig_fc, use_container_width=True)
# ────────────────────────────────────────────────────────────────
# Hospital map
# ────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🗺️ Find a Breast-Cancer Screening Hospital")

CSV_PATH = "demo_hospitals_with_coordinates.csv"
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()

hosp = pd.read_csv(CSV_PATH)
hosp.columns = hosp.columns.str.strip().str.lower()

# Normalize for search
hosp["name_clean"] = hosp["name"].str.strip().str.lower()
hosp["caza_clean"] = hosp["caza"].astype(str).str.strip().str.lower()

# Search
query = st.text_input("🔎 Search by City / Caza / Hospital").strip().lower()
mask = (
    hosp["name_clean"].str.contains(query, regex=False)
    | hosp["caza_clean"].str.contains(query, regex=False)
) if query else slice(None)
data = hosp.loc[mask].copy()

# Result summary
if query:
    st.write(f"**{len(data)} hospital(s) found for '{query}'**")
else:
    st.write(f"**{len(data)} hospital(s) found**")

# Map
if data.empty:
    st.warning("No hospitals match that search.")
else:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium

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
            f"<b>{row['name']}</b><br>"
            f"Caza: {row['caza']}<br>"
            f"Phone: {row['phone']}<br>"
            f"Investment #: {row['investment_authorization_nb']}"
        )
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=popup_html,
            tooltip=row["name"],
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(cluster)

    st_folium(m, width=750, height=500)

    with st.expander("↕️ See hospitals in a table"):
        st.dataframe(
            data[["name", "caza", "phone", "investment_authorization_nb"]].reset_index(drop=True),
            use_container_width=True,
        )
# Footer
st.markdown("---")
st.markdown(
    '<p style="font-weight:bold; font-size:18px; font-style:italic;">'
    'Early detection saves lives. Awareness is the first step to prevention.'
    '</p>',
    unsafe_allow_html=True,
)
st.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")

