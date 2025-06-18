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

# ―― Joyplot (compact)
st.subheader("Years of Life Lost by Tumor Stage")
fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=100)
joypy.joyplot(filtered, by="tumour_stage", column="yll", ax=ax, kind="counts",
              colormap=plt.cm.Reds, legend=False, bins=40, fade=True)
ax.set_xlabel("Years of Life Lost (YLL)"); ax.set_ylabel("Tumor Stage")
ax.set_title("Distribution of YLL by Tumor Stage", fontsize=10)
st.pyplot(fig)

# ───────────────────── National benchmarks (GCO) — bar plots
st.markdown("---"); st.header("National Benchmarks (GCO Data)")
rates   = pd.read_csv("GCO_Lebanon_rates.csv")
latest  = rates.loc[rates.year == rates.year.max()]

for metric, title in [("incidence_rate","Incidence"), ("mortality_rate","Mortality")]:
    fig_tmp = px.bar(
        latest, x="gender", y=metric, text_auto=".1f",
        labels={metric:f"{title} per 100 000", "gender":"Gender"},
        title=f"Lebanon {title} Rate — {int(latest.year.iloc[0])}",
        color="gender", color_discrete_map={"Female":"#8B0000","Male":"#FFC1C1"})
    fig_tmp.update_layout(height=330, width=500, showlegend=False,
                          margin=dict(l=10,r=10,t=40,b=20),
                          yaxis=dict(showticklabels=False))
    fig_tmp.update_traces(textposition="outside")
    st.plotly_chart(fig_tmp, use_container_width=True)

# ───────────────────── Time-trend & forecast (compact)
st.markdown("---"); st.subheader("Time Trends & Short-Term Forecast")
ts = rates.groupby("year")[["incidence_rate","mortality_rate"]].mean().reset_index()
ts["year_dt"] = pd.to_datetime(ts["year"].astype(str))
fig_trend = px.line(ts, x="year_dt", y=["incidence_rate","mortality_rate"],
                    labels=dict(value="Rate per 100 000", year_dt="Year", variable="Metric"),
                    title="Lebanon: Incidence & Mortality Trends",
                    color_discrete_sequence=["#8B0000","#FFC1C1"])
fig_trend.update_layout(height=350, width=700, xaxis=dict(rangeslider=dict(visible=True)),
                        margin=dict(l=10,r=10,t=40,b=20))
st.plotly_chart(fig_trend, use_container_width=True)

# forecast
y = ts.set_index("year")["incidence_rate"]
res = SARIMAX(y, order=(1,1,1), enforce_stationarity=False,
              enforce_invertibility=False).fit(disp=False)
h, ci = 3, res.get_forecast(3).conf_int(alpha=0.05)
pred  = res.get_forecast(h).predicted_mean
years_fc = list(range(ts["year"].max()+1, ts["year"].max()+1+h))
df_fc = pd.DataFrame({"year":years_fc, "forecast":pred.values,
                      "lower":ci.iloc[:,0], "upper":ci.iloc[:,1]})

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=ts["year"], y=ts["incidence_rate"],
                            name="Historical", mode="lines+markers",
                            line=dict(color="#8B0000"), marker=dict(color="#8B0000")))
fig_fc.add_trace(go.Scatter(x=df_fc["year"], y=df_fc["forecast"],
                            name="Forecast", mode="lines+markers",
                            line=dict(color="#FFC1C1"), marker=dict(color="#FFC1C1")))
fig_fc.add_trace(go.Scatter(
    x=df_fc["year"].tolist()+df_fc["year"][::-1].tolist(),
    y=df_fc["upper"].tolist()+df_fc["lower"][::-1].tolist(),
    fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(0,0,0,0)"),
    hoverinfo="skip", name="95% CI"))
fig_fc.update_layout(title="3-Year Forecast: Incidence Rate",
                     xaxis_title="Year", yaxis_title="Incidence per 100 000",
                     height=350, width=700, margin=dict(l=10,r=10,t=40,b=20))
st.subheader("Short-Term Forecast"); st.plotly_chart(fig_fc, use_container_width=True)

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

# Caza summary
if not data.empty:
    caza_counts = (
        data["caza"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Caza", "caza": "Hospital Count"})
    )
    st.dataframe(caza_counts, use_container_width=True)

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

