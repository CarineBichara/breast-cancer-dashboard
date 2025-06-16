import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joypy
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pytrends.request import TrendReq

# Page configuration
st.set_page_config(page_title="Breast Cancer Awareness", layout="wide")

# Load and clean data
file_path = "C:/Users/User/Desktop/Carine/AUB/Summer 2024-25/Health Care/Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df['date_of_surgery'] = pd.to_datetime(df['date_of_surgery'], errors='coerce')
df['date_of_last_visit'] = pd.to_datetime(df['date_of_last_visit'], errors='coerce')
df['followup_days'] = (df['date_of_last_visit'] - df['date_of_surgery']).dt.days

# Sidebar filters
st.sidebar.title("Filters")
gender = st.sidebar.multiselect(
    "Gender", df['gender'].unique(), default=list(df['gender'].unique())
)
stages = st.sidebar.multiselect(
    "Tumor Stage", df['tumour_stage'].dropna().unique(),
    default=list(df['tumour_stage'].dropna().unique())
)
status = st.sidebar.multiselect(
    "Patient Status", df['patient_status'].dropna().unique(),
    default=list(df['patient_status'].dropna().unique())
)
age_range = st.sidebar.slider(
    "Age Range", int(df['age'].min()), int(df['age'].max()),
    (int(df['age'].min()), int(df['age'].max()))
)

# Filter data
filtered = df[
    df['gender'].isin(gender) &
    df['tumour_stage'].isin(stages) &
    df['patient_status'].isin(status)
]
filtered = filtered[
    (filtered['age'] >= age_range[0]) &
    (filtered['age'] <= age_range[1])
]

# Display logos and titles
st.image("C:/Users/User/Desktop/Carine/AUB/Summer 2024-25/Health Care/pink_ribbon.png", width=80)
st.title("Breast Cancer Awareness")
st.markdown("Developed by **Carine Bichara**")
st.image("C:/Users/User/Desktop/Carine/AUB/Summer 2024-25/Health Care/manandwoman.png", width=600)
# Core message & call-to-action (hard-coded values)
st.markdown(
    "### Breast cancer in Lebanon:\n"
    "- Women face ~75.5 cases per 100 000 population  \n"
    "- Men face   ~ 0.8 cases per 100 000 population  \n\n"
    "Both should know their risk and get screened."
)
st.info("**What you can do today:** Talk to your doctor about screening options or find a clinic near you.")

# Data Overview
st.markdown("### Data Overview")
st.markdown(
    "We’ll start by exploring patient-level data drawn from a public Kaggle dataset, covering demographics, clinical features, and outcomes. "
    "Then, to benchmark Lebanon’s national burden, we’ll layer in population-standardized rates from the Global Cancer Observatory (GCO)."
)

# Glossary
with st.expander("What Do These Metrics Mean?"):
    st.markdown("""
- **YLL**: Years of Life Lost due to early death  
- **YLD**: Years Lived with Disability  
- **DALY**: Disability-Adjusted Life Years (YLL + YLD)  
- **Percentages (%)**: Share of total cases, deaths, YLL, YLD or DALYs within this cohort
""")

# Summary metrics
col1, col2, col3 = st.columns(3)
col1.metric("Filtered Patients", len(filtered))
col2.metric("Deaths", int(filtered['mortality'].sum()))
col3.metric("DALYs", int(filtered['daly'].sum()))

# Section: Patient-Level Insights
st.header("Patient-Level Insights (Kaggle Dataset)")

# Aggregate by gender
grouped = filtered.groupby('gender').agg(
    cases=('gender', 'count'),
    deaths=('mortality', 'sum'),
    yll=('yll', 'sum'),
    yld=('yld', 'sum'),
    daly=('daly', 'sum')
).reset_index()

# Compute percentages of total
totals = grouped[['cases', 'deaths', 'yll', 'yld', 'daly']].sum()
for col in ['cases', 'deaths', 'yll', 'yld', 'daly']:
    grouped[f"{col}_pct"] = grouped[col] / totals[col] * 100

# Prepare palette
red_palette = sns.color_palette("Reds", n_colors=4)

# Comparison by Gender
st.subheader("Comparison by Gender")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (metric, title) in enumerate(zip(
        ['cases_pct', 'deaths_pct', 'daly_pct'],
        ['% of Cases', '% of Deaths', '% of DALYs']
    )):
    ax = axes[i]
    colors = [red_palette[3], red_palette[1]]  # dark red for female, lighter red for male
    sns.barplot(x='gender', y=metric, data=grouped, palette=colors, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("")       # no y-axis label
    ax.set_yticks([])         # hide ticks
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", 
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha='center', va='bottom')
st.pyplot(fig)

with st.expander("Interpretation: Comparison by Gender"):
    st.markdown("""
- **Female patients represent ~98.8% of all cases**, while **males account for only ~1.2%**.  
- **Women account for ~98.5% of all deaths**, compared to **1.5% for men**, indicating mortality is overwhelmingly among female patients.  
- **Similarly, female patients bear ~98.4% of total DALYs**, versus **1.6% for males**, reflecting the proportional disease burden.

**Key Message:**  
Breast cancer burden is heavily concentrated in women, underscoring the need for gender-targeted awareness and screening. Yet **men must not be overlooked**, as even a small number of cases and deaths represent important opportunities for early detection and care.
""")

# Age Distribution
colA, colB = st.columns(2)
with colA:
    st.subheader("Age Distribution")
    fig_age, ax = plt.subplots(figsize=(6, 5))
    bins = [30,35,40,45,50,55,60,65,70,75,80,85,90]
    counts, _, patches = ax.hist(
        filtered['age'], bins=bins, edgecolor="white", rwidth=0.9
    )
    for i, patch in enumerate(patches):
        patch.set_facecolor('#F4CCCC' if bins[i] < 45 or bins[i] >= 65 else '#B22222')
    for rect, label in zip(patches, counts):
        if label > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                label + 1,
                f"{int(label)}",
                ha='center', va='bottom'
            )
    ax.set_title("Age Distribution of Breast Cancer Patients")
    ax.set_xlabel("Age Group")
    ax.set_xticks([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    ax.set_xticklabels(
        ['30-34','35-39','40-44','45-49','50-54','55-59','60-64',
         '65-69','70-74','75-79','80-84','85-89'],
        fontsize=8
    )
    ax.set_yticks([])
    ax.grid(False)
    st.pyplot(fig_age)

    with st.expander("Interpretation: Age Distribution"):
        st.markdown("""
- Most patients are between **45 and 65** years old.  
- The **50–54** age group has the highest count.  
- Lower counts at the extremes may reflect screening patterns or risk differences.

**Key Message:**  
Focus screening efforts on ages **45–65**.
""")

# Surgery Type Breakdown
with colB:
    st.subheader("Surgery Type Breakdown")
    with st.expander("Click to view surgery procedure definitions"):
        st.markdown("""
- **Lumpectomy**: Tumor only removed, breast is kept  
- **Simple Mastectomy**: Whole breast removed, lymph nodes not removed  
- **Modified Radical Mastectomy**: Breast and lymph nodes removed
        """)
    surgery_counts = filtered['surgery_type'].value_counts()
    surgery_colors = {
        'Simple Mastectomy': red_palette[1],
        'Lumpectomy': red_palette[2],
        'Modified Radical Mastectomy': red_palette[3],
        'Other': red_palette[0]
    }
    colors = [surgery_colors.get(x, red_palette[0]) for x in surgery_counts.index]
    fig_surgery, ax = plt.subplots()
    ax.pie(
        surgery_counts,
        labels=surgery_counts.index,
        autopct='%1.0f%%',
        startangle=90,
        colors=colors
    )
    ax.axis('equal')
    st.pyplot(fig_surgery)

    with st.expander("Interpretation: Surgery Type Breakdown"):
        st.markdown("""
- **Modified Radical Mastectomy (29%)**  
- **Lumpectomy (21%)**  
- **Simple Mastectomy (20%)**  
- **Other Surgeries (31%)**  

**Key Message:**  
Early detection can reduce need for invasive procedures.
""")

# Average DALY Heatmap
st.subheader("Average DALY by Age Group and Tumor Stage")
bins = list(range(30, 95, 5))
labels = [f"{b}-{b+4}" for b in bins[:-1]]
filtered['age_group'] = pd.cut(filtered['age'], bins=bins, labels=labels, right=False)
pivot = (
    filtered
    .groupby(['age_group','tumour_stage'])['daly']
    .mean()
    .unstack()
    .fillna(0)
)
fig_heat = px.imshow(
    pivot,
    labels=dict(x="Tumor Stage", y="Age Group", color="Average DALY"),
    x=pivot.columns,
    y=pivot.index,
    text_auto='.1f',
    color_continuous_scale='Reds',
    aspect="auto"
)
fig_heat.update_layout(
    height=600,
    width=800,
    title="Average DALY by Age Group and Tumor Stage",
    xaxis_title="Tumor Stage",
    yaxis_title="Age Group",
    yaxis_autorange='reversed'
)
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Interpretation: DALY by Age and Tumor Stage"):
    st.markdown("""
- Younger patients with Stage I have the highest average DALYs.  
- DALYs decline with advancing age and vary by stage.  
- Use this to prioritize interventions by age/stage.
""")

# YLL Distribution Joyplot
st.subheader("Years of Life Lost by Tumor Stage")
fig, ax = plt.subplots(figsize=(8, 6))
joypy.joyplot(
    filtered,
    by='tumour_stage',
    column='yll',
    ax=ax,
    kind='counts',
    colormap=plt.cm.Reds,
    legend=False,
    bins=40,
    fade=True
)
ax.set_xlabel("Years of Life Lost (YLL)")
ax.set_ylabel("Tumor Stage")
ax.set_title("Distribution of YLL by Tumor Stage")
st.pyplot(fig)

with st.expander("Interpretation: YLL by Tumor Stage"):
    st.markdown("""
- Stage I shows wide variability in YLL.  
- Stages II and III have more clustered, lower YLL distributions.  
- Highlights need for early detection to reduce life years lost.
""")

# Bridge to benchmarks
st.markdown("---")
st.markdown(
    "While these patient-level insights reveal key trends, they don’t account for population size." 
    " Next, we’ll incorporate GCO’s standardized rates to contextualize Lebanon’s national burden."
)

# Section: National Benchmarks (GCO Data)
st.header("National Benchmarks (GCO Data)")

# ─── Population-Level Rates & Benchmarks ───
rates = pd.read_csv(
    r"C:\Users\User\Desktop\Carine\AUB\Summer 2024-25\Health Care\GCO_Lebanon_rates.csv"
)
latest = rates[rates.year == rates.year.max()]

# bar chart: incidence, styled like your matplotlib version
fig1 = px.bar(
    latest,
    x="gender",
    y="incidence_rate",
    text_auto='.1f',  # label each bar with its value
    labels={"incidence_rate": "Incidence per 100 000", "gender": "Gender"},
    title=f"Lebanon Incidence Rate (per 100 000) — {int(latest.year.iloc[0])}",
    color="gender",
    color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
)

# remove legend
fig1.update_layout(showlegend=False)

# hide the y-axis tick labels but keep the axis title
fig1.update_layout(
    yaxis=dict(showticklabels=False, title_text="Incidence per 100 000"),
    xaxis=dict(title_text="Gender")
)

# move labels above each bar
fig1.update_traces(textposition="outside")

st.plotly_chart(fig1, use_container_width=True)

with st.expander("Interpretation: Incidence Rate"):
    st.markdown(f"""
- Women have an incidence rate of {latest.query("gender == 'Female'").incidence_rate.values[0]:.1f} per 100 000  
- Men have an incidence rate of {latest.query("gender == 'Male'").incidence_rate.values[0]:.1f} per 100 000  
- Incidence is far higher in women, highlighting the need for gender-focused screening while keeping access open to men
""")

# bar chart: mortality, styled like the incidence chart
fig2 = px.bar(
    latest,
    x="gender",
    y="mortality_rate",
    text_auto='.1f',  # show value above each bar
    labels={"mortality_rate": "Mortality per 100 000", "gender": "Gender"},
    title=f"Lebanon Mortality Rate (per 100 000) — {int(latest.year.iloc[0])}",
    color="gender",
    color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"},
)

# hide legend
fig2.update_layout(showlegend=False)

# remove y-axis numbers but keep its title, and label the x-axis
fig2.update_layout(
    yaxis=dict(showticklabels=False, title_text="Mortality per 100 000"),
    xaxis=dict(title_text="Gender")
)

# move the text labels just above the bars
fig2.update_traces(textposition="outside")

st.plotly_chart(fig2, use_container_width=True)

with st.expander("Interpretation: Mortality Rate"):
    st.markdown(f"""
- Women have a mortality rate of {latest.query("gender == 'Female'").mortality_rate.values[0]:.1f} per 100 000  
- Men have a mortality rate of {latest.query("gender == 'Male'").mortality_rate.values[0]:.1f} per 100 000  

**Key point:** Lebanon’s breast cancer burden is far higher in women, but even a small male rate signals the need not to overlook men in screening programs.
""")

# ─── Time-Trend Analysis & Forecasting ───
st.markdown("---")
st.subheader("Time Trends & Short-Term Forecast")

# Prepare year-level series
ts = rates.groupby('year')[['incidence_rate','mortality_rate']].mean().reset_index()
ts_plot = ts.copy()
ts_plot['year_dt'] = pd.to_datetime(ts_plot['year'], format='%Y')

# ─── Time-Trend Analysis & Forecasting ───
# Prepare year-level series
ts = rates.groupby('year')[['incidence_rate','mortality_rate']].mean().reset_index()
ts_plot = ts.copy()
ts_plot['year_dt'] = pd.to_datetime(ts_plot['year'], format='%Y')

# 1) Trend chart with slider
fig_trend = px.line(
    ts_plot,
    x='year_dt',
    y=['incidence_rate','mortality_rate'],
    labels={'value':'Rate per 100 000','year_dt':'Year','variable':'Metric'},
    title="Lebanon: Incidence & Mortality Trends",
    color_discrete_sequence=['#8B0000', '#FFC1C1']   
)
fig_trend.update_layout(
    xaxis=dict(rangeslider=dict(visible=True))
)
st.plotly_chart(fig_trend, use_container_width=True)

with st.expander("Interpretation: Time Trends & Short-Term Forecast"):
    st.markdown(f"""
- Incidence rose from **{ts['incidence_rate'].iloc[0]:.1f}** to **{ts['incidence_rate'].iloc[-1]:.1f}** per 100 000 between {ts['year'].iloc[0]} and {ts['year'].iloc[-1]}.  
- Mortality grew from **{ts['mortality_rate'].iloc[0]:.1f}** to **{ts['mortality_rate'].iloc[-1]:.1f}** per 100 000 over the same period.  
- The gap between incidence and mortality remains wide, showing survival gains have not outpaced rising case numbers.  
- If these trends continue, both rates will inch upward, underscoring the need to reinforce prevention, screening, and treatment efforts.
""")

# 2) 3-Year ARIMA forecast of incidence
y = ts.set_index('year')['incidence_rate']
model = SARIMAX(y, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

horizon = 3
fc = res.get_forecast(steps=horizon)
pred = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)

years_fc = list(range(ts['year'].max()+1, ts['year'].max()+1+horizon))
df_fc = pd.DataFrame({
    'year':   years_fc,
    'forecast': pred.values,
    'lower':  ci.iloc[:,0].values,
    'upper':  ci.iloc[:,1].values
})

fig_fc = go.Figure()
# Historical in red
fig_fc.add_trace(go.Scatter(
    x=ts['year'], y=ts['incidence_rate'],
    name='Historical', mode='lines+markers',
    line=dict(color='#8B0000'),
    marker=dict(color='#8B0000')
))
# Forecast in red
fig_fc.add_trace(go.Scatter(
    x=df_fc['year'], y=df_fc['forecast'],
    name='Forecast', mode='lines+markers',
    line=dict(color='#FFC1C1'),
    marker=dict(color='#FFC1C1')
))
# Confidence band in semi-transparent red
fig_fc.add_trace(go.Scatter(
    x=df_fc['year'].tolist() + df_fc['year'][::-1].tolist(),
    y=df_fc['upper'].tolist() + df_fc['lower'][::-1].tolist(),
    fill='toself',
    fillcolor='rgba(255,0,0,0.2)',    # ← red with 20% opacity
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip", name='95% CI'
))
fig_fc.update_layout(
    title="3-Year Forecast: Incidence Rate",
    xaxis_title="Year",
    yaxis_title="Incidence per 100 000"
)
st.subheader("Short-Term Forecast")
st.plotly_chart(fig_fc, use_container_width=True)

with st.expander("Interpretation: 3-Year ARIMA Forecast of Incidence"):
    st.markdown(f"""
    The three-year forecast shows Lebanon’s breast-cancer incidence rising from about **{df_fc['forecast'].iloc[0]:.1f}** per 100 000 in {df_fc['year'].iloc[0]}  
    to **{df_fc['forecast'].iloc[1]:.1f}** in {df_fc['year'].iloc[1]} and **{df_fc['forecast'].iloc[2]:.1f}** in {df_fc['year'].iloc[2]},  
    with a widening 95% confidence band. Continued growth at this pace highlights the need to scale up screening and prevention programs now.
    """)
    # ── Breast-Cancer Screening Facilities Map ────────────────────────────────────
import os, json, time
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.markdown("---")
st.header("🗺️ Find a Breast-Cancer Screening Hospital")

# Your CSV with all hospitals
CSV_PATH = r"C:\Users\User\Desktop\Carine\AUB\Summer 2024-25\Health Care\lebanon_private_hospitals_complete.csv"

# 1️⃣ Load data or fail gracefully
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found at\n{CSV_PATH}")
    st.stop()

hosp = pd.read_csv(CSV_PATH)

# 2️⃣ Ensure we have coordinates (will geocode the first time) ────────────────
if {"latitude", "longitude"}.issubset(hosp.columns) is False or hosp[["latitude","longitude"]].isna().any().any():
    geolocator = Nominatim(user_agent="bc-screening-map")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    cache_path = os.path.join(os.path.dirname(CSV_PATH), "geo_cache.json")
    geo_cache  = json.load(open(cache_path, "r", encoding="utf-8")) if os.path.exists(cache_path) else {}

    def fetch_latlon(row):
        # skip if already present
        if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
            return row["latitude"], row["longitude"]
        # use cache first
        key = f'{row["Name"]}_{row["Caza"]}'
        if key in geo_cache:
            return geo_cache[key]
        # query Nominatim once
        loc = geocode(f'{row["Name"]}, {row["Caza"]}, Lebanon')
        latlon = (loc.latitude, loc.longitude) if loc else (None, None)
        geo_cache[key] = latlon
        time.sleep(1)  # be polite
        return latlon

    hosp[["latitude", "longitude"]] = hosp.apply(fetch_latlon, axis=1, result_type="expand")

    # save cache for future runs
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(geo_cache, f, ensure_ascii=False, indent=2)

# 3️⃣ Live search box (city, Caza, or hospital) ───────────────────────────────
query = st.text_input(
    "🔎 Search by City / Caza / Hospital   (e.g. Beirut, Aley, Akkar, West Bekaa, Aley)",
    ""
).strip().lower()

if query:
    mask = (
        hosp["Caza"].astype(str).str.lower().str.contains(query)
        | hosp.get("governorate", hosp["Caza"]).astype(str).str.lower().str.contains(query)
        | hosp["Name"].str.lower().str.contains(query)
    )
    data = hosp[mask].copy()
else:
    data = hosp.copy()

st.write(f"**{len(data)} hospital(s) found**")

# 4️⃣ Build & show Folium map ─────────────────────────────────────────────────
if data.empty:
    st.warning("No hospitals match that search. Try another term.")
else:
    # centre map on visible hospitals
    m = folium.Map(
        location=[data["latitude"].mean(), data["longitude"].mean()],
        zoom_start=9,
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google"
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
            icon=folium.Icon(color="red", icon="plus-sign"),
        ).add_to(cluster)

    st_folium(m, width=750, height=500)

    with st.expander("↕️ See results in a table"):
        st.dataframe(
            data[["Name", "Caza", "Phone", "Investment_Authorization_Nb"]]
            .reset_index(drop=True)
        )

st.markdown("---")
st.markdown(
    '<p style="font-weight:bold; font-size:18px; font-style:italic;">'
    'Early detection saves lives. Awareness is the first step to prevention.'
    '</p>', unsafe_allow_html=True
)
st.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")
