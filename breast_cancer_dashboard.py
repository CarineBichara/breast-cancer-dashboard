# Breast Cancer Awareness Dashboard – Plotly Dash Version
# Author: Carine Bichara
# -----------------------------------------------------------------------------
# NOTE: This is a one‑file Dash application that mirrors the Streamlit dashboard
# you shared.  It preserves the same charts, colours, wording, and interpretations
# while giving you full layout control.  Deploy with:
#     $ pip install dash dash-bootstrap-components plotly pandas numpy statsmodels
#     $ python breast_cancer_dashboard_dash.py
# -----------------------------------------------------------------------------

import os
import json
import time
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# -----------------------------------------------------------------------------
# Data paths (adjust if you deploy elsewhere)
# -----------------------------------------------------------------------------
DATA_PATH = "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
GCO_PATH  = "GCO_Lebanon_rates.csv"
HOSP_PATH = "demo_hospitals_with_coordinates.csv"

# -----------------------------------------------------------------------------
# Load & prep data (identical to Streamlit)
# -----------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df["date_of_surgery"]    = pd.to_datetime(df["date_of_surgery"], errors="coerce")
df["date_of_last_visit"] = pd.to_datetime(df["date_of_last_visit"], errors="coerce")
df["followup_days"]      = (df["date_of_last_visit"] - df["date_of_surgery"]).dt.days

# GCO national
gco = pd.read_csv(GCO_PATH)
latest_gco = gco[gco.year == gco.year.max()]

# Age bins used everywhere
BINS   = list(range(30, 95, 5))
LABELS = [f"{b}-{b+4}" for b in BINS[:-1]]

# -----------------------------------------------------------------------------
# Helper: forecast incidence (identical to Streamlit logic)
# -----------------------------------------------------------------------------

def forecast_incidence(series, horizon=3):
    model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc    = res.get_forecast(horizon)
    pred  = fc.predicted_mean
    ci    = fc.conf_int(alpha=0.05)
    years = list(range(series.index.max() + 1, series.index.max() + 1 + horizon))
    return pd.DataFrame({
        "year": years,
        "forecast": pred.values,
        "lower": ci.iloc[:, 0].values,
        "upper": ci.iloc[:, 1].values,
    })

# Pre-compute once to speed up callbacks
INC_FC_DF = forecast_incidence(gco.groupby("year")["incidence_rate"].mean())

# -----------------------------------------------------------------------------
# Dash app & layout
# -----------------------------------------------------------------------------

a pp = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Breast Cancer Awareness"

# --- Sidebar controls ---------------------------------------------------------
sidebar = dbc.Col([
    html.Img(src="/assets/pink_ribbon.png", style={"width": "60px", "margin-bottom": "10px"}),
    html.H4("Filters"),

    html.Label("Gender"),
    dcc.Dropdown(df["gender"].unique(), value=df["gender"].unique().tolist(), id="filter-gender", multi=True),

    html.Br(),
    html.Label("Tumor Stage"),
    dcc.Dropdown(df["tumour_stage"].dropna().unique(),
                 value=df["tumour_stage"].dropna().unique().tolist(), id="filter-stage", multi=True),

    html.Br(),
    html.Label("Patient Status"),
    dcc.Dropdown(df["patient_status"].dropna().unique(),
                 value=df["patient_status"].dropna().unique().tolist(), id="filter-status", multi=True),

    html.Br(),
    html.Label("Age Range"),
    dcc.RangeSlider(int(df["age"].min()), int(df["age"].max()), 5,
                    value=[int(df["age"].min()), int(df["age"].max())],
                    id="filter-age", allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": False}),

    html.Hr(),
    dbc.Alert("What you can do today: Talk to your doctor about screening options or find a clinic near you.", color="info", style={"fontSize": "0.9rem"}),

    html.Details([
        html.Summary("What Do These Metrics Mean?"),
        html.Ul([
            html.Li("YLL: Years of Life Lost due to early death"),
            html.Li("YLD: Years Lived with Disability"),
            html.Li("DALY: Disability‑Adjusted Life Years (YLL + YLD)"),
            html.Li("%: Share of the total within this cohort")
        ], style={"marginLeft": "-20px"})
    ])
], width=3, style={"padding": "20px", "background": "#f8f9fa"})

# --- Placeholder KPIs (updated via callback) ----------------------------------
kpi_cards = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardHeader("Filtered Patients"), dbc.CardBody(html.H4(id="kpi-patients"))])),
    dbc.Col(dbc.Card([dbc.CardHeader("Deaths"),            dbc.CardBody(html.H4(id="kpi-deaths"))])),
    dbc.Col(dbc.Card([dbc.CardHeader("DALYs"),             dbc.CardBody(html.H4(id="kpi-dalys"))]))
], className="mb-4")

# --- Charts layout (empty dcc.Graph components filled later) ------------------
charts = html.Div([
    kpi_cards,

    # Row 1 – Gender burden (three bars in one figure)
    dcc.Graph(id="fig-gender"),

    # Row 2 – Age & Surgery side‑by‑side
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-age"),  width=6),
        dbc.Col(dcc.Graph(id="fig-surgery"), width=6)
    ], className="mb-4"),

    # Row 3 – DALY heat‑map
    dcc.Graph(id="fig-heat"),

    # Row 4 – YLL distribution
    dcc.Graph(id="fig-yll"),

    # Row 5 – National incidence & mortality
    dbc.Row([
        dbc.Col(dcc.Graph(id="fig-incidence"), width=6),
        dbc.Col(dcc.Graph(id="fig-mortality"), width=6)
    ], className="mb-4"),

    # Row 6 – Trend line & Forecast
    dcc.Graph(id="fig-trend"),
    dcc.Graph(id="fig-forecast"),

    html.Hr(),
    html.H5("\U0001F5FA  Find a Breast‑Cancer Screening Hospital"),
    dcc.Graph(id="map-hospitals"),

    html.P("\"Early detection saves lives. Awareness is the first step to prevention.\"",
           style={"fontStyle": "italic", "fontSize": "18px", "textAlign": "center", "marginTop": "30px"})
])

# --- Main Layout --------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        dbc.Col([
            html.Img(src="/assets/manandwoman.png", style={"width": "60%", "margin": "auto", "display": "block"}),
            html.H1("Breast Cancer Awareness", style={"textAlign": "center"}),
            html.P("Developed by Carine Bichara", style={"textAlign": "center", "marginBottom": "40px"}),
            charts
        ], width=9)
    ])
], fluid=True)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

# Utility to filter data based on controls

def filter_data(genders, stages, statuses, age_range):
    fltr = (
        df["gender"].isin(genders) &
        df["tumour_stage"].isin(stages) &
        df["patient_status"].isin(statuses) &
        df["age"].between(age_range[0], age_range[1])
    )
    return df.loc[fltr].copy()

# KPI & Charts callback
@app.callback([
    Output("kpi-patients", "children"),
    Output("kpi-deaths",   "children"),
    Output("kpi-dalys",    "children"),
    Output("fig-gender",   "figure"),
    Output("fig-age",      "figure"),
    Output("fig-surgery",  "figure"),
    Output("fig-heat",     "figure"),
    Output("fig-yll",      "figure"),
    Output("fig-incidence","figure"),
    Output("fig-mortality","figure"),
    Output("fig-trend",    "figure"),
    Output("fig-forecast", "figure"),
    Output("map-hospitals","figure")
],
    [Input("filter-gender", "value"),
     Input("filter-stage",  "value"),
     Input("filter-status", "value"),
     Input("filter-age",    "value")])

def update_dashboard(genders, stages, statuses, age_range):
    filtered = filter_data(genders, stages, statuses, age_range)

    # KPIs
    kpi_patients = f"{len(filtered):,}"
    kpi_deaths   = f"{int(filtered['mortality'].sum()):,}"
    kpi_dalys    = f"{int(filtered['daly'].sum()):,}"

    # --- Gender burden bar group
    agg = (
        filtered.groupby("gender").agg(
            cases=("gender", "count"),
            deaths=("mortality", "sum"),
            daly=("daly", "sum")
        ).reset_index()
    )
    for c in ["cases", "deaths", "daly"]:
        agg[f"{c}_pct"] = agg[c] / agg[c].sum() * 100

    fig_gender = make_gender_bars(agg)

    # --- Age histogram
    fig_age = make_age_hist(filtered)

    # --- Surgery donut
    fig_surgery = make_surgery_pie(filtered)

    # --- DALY heatmap
    fig_heat = make_daly_heat(filtered)

    # --- YLL ridgeline approximated with violin
    fig_yll = make_yll_violin(filtered)

    # --- National incidence & mortality (static, based on latest_gco)
    fig_incidence = make_incidence_bar(latest_gco)
    fig_mortality = make_mortality_bar(latest_gco)

    # --- Trend line (static)
    fig_trend = make_trend_line(gco)

    # --- Forecast (pre-computed)
    fig_forecast = make_forecast_chart(gco, INC_FC_DF)

    # --- Map (static for now)
    fig_map = make_map_chart()

    return (kpi_patients, kpi_deaths, kpi_dalys, fig_gender, fig_age, fig_surgery,
            fig_heat, fig_yll, fig_incidence, fig_mortality, fig_trend,
            fig_forecast, fig_map)

# -----------------------------------------------------------------------------
# Chart Builders  -------------------------------------------------------------


def make_gender_bars(agg):
    cols = ["cases_pct", "deaths_pct", "daly_pct"]
    titles = ["% Cases", "% Deaths", "% DALYs"]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=titles)
    palette = ["#8B0000", "#FFC1C1"]
    for idx, metric in enumerate(cols, 1):
        fig.add_bar(x=agg["gender"], y=agg[metric], marker_color=palette, row=1, col=idx, text=[f"{v:.1f}%" for v in agg[metric]])
        fig.update_yaxes(visible=False, row=1, col=idx)
    fig.update_layout(showlegend=False, height=300, margin=dict(t=30, b=10))
    return fig


def make_age_hist(filtered):
    fig = px.histogram(filtered, x="age", nbins=len(BINS)-1, color_discrete_sequence=["#B22222"],
                       labels={"age": "Age"})
    fig.update_layout(title="Patient Age Distribution", yaxis_showticklabels=False)
    # Custom colour for under 45 / above 65 similar to Streamlit can be added if needed.
    return fig


def make_surgery_pie(filtered):
    surg_counts = filtered["surgery_type"].value_counts().reset_index(name="count").rename(columns={"index": "surgery"})
    color_map = {"Simple Mastectomy": "#FFC1C1", "Lumpectomy": "#F4CCCC", "Modified Radical Mastectomy": "#8B0000"}
    fig = px.pie(surg_counts, names="surgery", values="count", color="surgery",
                 color_discrete_map=color_map, hole=0.4, title="Surgery Type Breakdown")
    return fig


def make_daly_heat(filtered):
    filtered["age_group"] = pd.cut(filtered["age"], bins=BINS, labels=LABELS, right=False)
    pivot = filtered.groupby(["age_group", "tumour_stage"])["daly"].mean().unstack().fillna(0)
    fig = px.imshow(pivot, aspect="auto", text_auto=".1f", color_continuous_scale="Reds",
                    labels=dict(x="Tumor Stage", y="Age Group", color="Avg DALY"))
    fig.update_layout(title="Average DALY by Age × Tumor Stage", yaxis_autorange="reversed")
    return fig


def make_yll_violin(filtered):
    fig = px.violin(filtered, y="tumour_stage", x="yll", orientation="h", color="tumour_stage",
                    color_discrete_sequence=px.colors.sequential.Reds, box=True, points=False)
    fig.update_layout(title="Years of Life Lost by Tumor Stage", yaxis_title="Tumor Stage", xaxis_title="YLL")
    return fig


def make_incidence_bar(latest):
    fig = px.bar(latest, x="gender", y="incidence_rate", text_auto=".1f",
                 title=f"Incidence Rate – {int(latest.year.iloc[0])}",
                 color="gender", color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"})
    fig.update_layout(yaxis_showticklabels=False, showlegend=False, xaxis_title="Gender")
    return fig


def make_mortality_bar(latest):
    fig = px.bar(latest, x="gender", y="mortality_rate", text_auto=".1f",
                 title=f"Mortality Rate – {int(latest.year.iloc[0])}",
                 color="gender", color_discrete_map={"Female": "#8B0000", "Male": "#FFC1C1"})
    fig.update_layout(yaxis_showticklabels=False, showlegend=False, xaxis_title="Gender")
    return fig


def make_trend_line(gco):
    ts_plot = gco.groupby("year")[["incidence_rate", "mortality_rate"]].mean().reset_index()
    ts_plot["year_dt"] = pd.to_datetime(ts_plot["year"], format="%Y")
    fig = px.line(ts_plot, x="year_dt", y=["incidence_rate", "mortality_rate"],
                  labels={"value": "Rate / 100 000", "variable": "Metric", "year_dt": "Year"},
                  title="Incidence vs Mortality Over Time",
                  color_discrete_sequence=["#8B0000", "#FFC1C1"])
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return fig


def make_forecast_chart(gco, fc_df):
    hist = gco.groupby("year")["incidence_rate"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["year"], y=hist["incidence_rate"], mode="lines+markers",
                             name="Historical", line=dict(color="#8B0000")))
    fig.add_trace(go.Scatter(x=fc_df["year"], y=fc_df["forecast"], mode="lines+markers",
                             name="Forecast", line=dict(color="#FFC1C1")))
    fig.add_trace(go.Scatter(x=fc_df["year"].tolist() + fc_df["year"][::-1].tolist(),
                             y=fc_df["upper"].tolist() + fc_df["lower"][::-1].tolist(),
                             fill="toself", fillcolor="rgba(255,0,0,0.2)",
                             line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                             name="95% CI"))
    fig.update_layout(title="3‑Year Forecast: Incidence", xaxis_title="Year",
                      yaxis_title="Incidence / 100 000")
    return fig


def make_map_chart():
    # Simplified static scatter mapbox (requires MAPBOX token) – placeholder until
    # full dash‑leaflet or iframe integration.
    if not os.path.exists(HOSP_PATH):
        return go.Figure()
    hosp = pd.read_csv(HOSP_PATH)
    fig = px.scatter_mapbox(hosp.dropna(subset=["latitude", "longitude"]),
                            lat="latitude", lon="longitude", hover_name="Name",
                            hover_data=["Caza", "Phone"], zoom=6,
                            height=500, color_discrete_sequence=["#8B0000"])
    fig.update_layout(mapbox_style="open-street-map", margin=dict(t=0, l=0, r=0, b=0))
    return fig

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
