# breast_cancer_dashboard_dash.py
# -------------------------------------------------------------
# Breast Cancer Awareness – Plotly Dash
# Author: Carine Bichara
# -------------------------------------------------------------
# One-file Dash app mirroring your Streamlit dashboard.
#   • Incidence & Mortality charts = percent share
#   • All other visuals, colours, wording retained
# -------------------------------------------------------------
# Run locally:
#   pip install dash dash-bootstrap-components plotly pandas numpy statsmodels geopy
#   python breast_cancer_dashboard_dash.py
# -------------------------------------------------------------

import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# --- DATA PATHS ---
DATA_PATH = "Breast_Cancer_Survival_Data_with_YLL_YLD_DALY.csv"
GCO_PATH  = "GCO_Lebanon_rates.csv"
HOSP_PATH = "demo_hospitals_with_coordinates.csv"

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
gco = pd.read_csv(GCO_PATH)
latest_gco = gco[gco.year.eq(gco.year.max())].copy()
latest_gco["incidence_pct"] = latest_gco["incidence_rate"]  / latest_gco["incidence_rate"].sum()  * 100
latest_gco["mortality_pct"] = latest_gco["mortality_rate"] / latest_gco["mortality_rate"].sum() * 100
BINS   = list(range(30, 95, 5))
LABELS = [f"{b}-{b+4}" for b in BINS[:-1]]

# --- FORECAST ---
def _forecast(s, h=3):
    m = SARIMAX(s, order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    f  = m.get_forecast(h);  years = range(s.index.max()+1, s.index.max()+1+h)
    ci = f.conf_int(alpha=0.05)
    return pd.DataFrame({"year": years,
                         "forecast": f.predicted_mean.values,
                         "lower": ci.iloc[:,0].values,
                         "upper": ci.iloc[:,1].values})
INC_FC_DF = _forecast(gco.groupby("year")["incidence_rate"].mean())

# --- DASH APP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP]);  app.title = "Breast Cancer Awareness"

# === SIDEBAR ===
sidebar = dbc.Col([
    html.Img(src="/assets/pink_ribbon.png", style={"width":"60px","marginBottom":"10px"}),
    html.H4("Filters"),
    html.Label("Gender"),  dcc.Dropdown(df["gender"].unique(), df["gender"].unique().tolist(), id="f-gender", multi=True),
    html.Br(),
    html.Label("Tumor Stage"), dcc.Dropdown(df["tumour_stage"].dropna().unique(), df["tumour_stage"].dropna().unique().tolist(), id="f-stage", multi=True),
    html.Br(),
    html.Label("Patient Status"), dcc.Dropdown(df["patient_status"].dropna().unique(), df["patient_status"].dropna().unique().tolist(), id="f-status", multi=True),
    html.Br(),
    html.Label("Age Range"),
    dcc.RangeSlider(int(df["age"].min()), int(df["age"].max()), 5,
                    value=[int(df["age"].min()), int(df["age"].max())], id="f-age"),
    html.Hr(),
    dbc.Alert("What you can do today: Talk to your doctor about screening options or find a clinic near you.",
              color="info", style={"fontSize":"0.85rem"})
], width=3, style={"padding":"20px","background":"#f8f9fa"})

# === KPI CARDS ===
kpi_row = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardHeader("Filtered Patients"), dbc.CardBody(html.H4(id="kpi-pat"))])),
    dbc.Col(dbc.Card([dbc.CardHeader("Deaths"),            dbc.CardBody(html.H4(id="kpi-death"))])),
    dbc.Col(dbc.Card([dbc.CardHeader("DALYs"),             dbc.CardBody(html.H4(id="kpi-daly"))]))
], className="mb-4")

# === PLACEHOLDERS ===
main_graphs = html.Div([
    kpi_row,
    dcc.Graph(id="g-gender"),
    dbc.Row([dbc.Col(dcc.Graph(id="g-age"), width=6),
             dbc.Col(dcc.Graph(id="g-surgery"), width=6)], className="mb-4"),
    dcc.Graph(id="g-heat"),
    dcc.Graph(id="g-yll"),
    dbc.Row([dbc.Col(dcc.Graph(id="g-inc"), width=6),
             dbc.Col(dcc.Graph(id="g-mort"), width=6)], className="mb-4"),
    dcc.Graph(id="g-trend"),
    dcc.Graph(id="g-forecast"),
])

app.layout = dbc.Container([
    dbc.Row([sidebar,
        dbc.Col([
            html.Img(src="/assets/manandwoman.png", style={"width":"60%","display":"block","margin":"0 auto"}),
            html.H1("Breast Cancer Awareness", style={"textAlign":"center"}),
            html.P("Developed by Carine Bichara", style={"textAlign":"center","marginBottom":"30px"}),
            main_graphs
        ], width=9)])
], fluid=True)

# === HELPERS ===
def _filter(g, s, st, ar):
    return df.loc[df["gender"].isin(g) &
                  df["tumour_stage"].isin(s) &
                  df["patient_status"].isin(st) &
                  df["age"].between(ar[0], ar[1])].copy()

def _bars_percent(latest, col, title):
    fig = px.bar(latest, x="gender", y=col, text_auto=".1f",
                 title=title, labels={col:"Share (%)","gender":"Gender"},
                 color="gender", color_discrete_map={"Female":"#8B0000","Male":"#FFC1C1"})
    fig.update_layout(yaxis_range=[0,100], showlegend=False)
    return fig

# === CALLBACK ===
@app.callback(
    [Output("kpi-pat","children"), Output("kpi-death","children"), Output("kpi-daly","children"),
     Output("g-gender","figure"), Output("g-age","figure"), Output("g-surgery","figure"),
     Output("g-heat","figure"), Output("g-yll","figure"),
     Output("g-inc","figure"), Output("g-mort","figure"),
     Output("g-trend","figure"), Output("g-forecast","figure")],
    [Input("f-gender","value"), Input("f-stage","value"),
     Input("f-status","value"), Input("f-age","value")]
)
def update(g, stg, sts, ar):
    sub = _filter(g, stg, sts, ar)
    k_pat, k_death, k_daly = f"{len(sub):,}", f"{int(sub['mortality'].sum()):,}", f"{int(sub['daly'].sum()):,}"
    # gender burden
    agg = sub.groupby("gender").agg(cases=("gender","count"), deaths=("mortality","sum"), daly=("daly","sum")).reset_index()
    for c in ("cases","deaths","daly"): agg[f"{c}_pct"]=agg[c]/agg[c].sum()*100
    g_gender = make_subplots(rows=1,cols=3,shared_yaxes=True,subplot_titles=["% Cases","% Deaths","% DALYs"])
    for i,p in enumerate(["cases_pct","deaths_pct","daly_pct"],1):
        g_gender.add_bar(x=agg["gender"],y=agg[p],marker_color=["#8B0000","#FFC1C1"],text=[f"{v:.1f}%" for v in agg[p]],row=1,col=i)
        g_gender.update_yaxes(visible=False,row=1,col=i)
    g_gender.update_layout(showlegend=False,height=300,margin=dict(t=30,b=30))
    # age hist
    g_age = px.histogram(sub,x="age",nbins=len(BINS)-1,color_discrete_sequence=["#B22222"])
    g_age.update_layout(title="Patient Age Distribution",yaxis_showticklabels=False)
    # surgery donut
    sc=sub["surgery_type"].value_counts()
    g_surg = px.pie(values=sc.values,names=sc.index,color=sc.index,
                    color_discrete_map={"Simple Mastectomy":"#FFC1C1","Lumpectomy":"#F4CCCC",
                                        "Modified Radical Mastectomy":"#8B0000"},
                    hole=0.4,title="Surgery Type Breakdown")
    # daly heat
    sub["age_group"]=pd.cut(sub["age"],bins=BINS,labels=LABELS,right=False)
    pv=sub.groupby(["age_group","tumour_stage"])["daly"].mean().unstack().fillna(0)
    g_heat=px.imshow(pv,labels=dict(x="Tumor Stage",y="Age Group",color="Avg DALY"),
                     text_auto=".1f",color_continuous_scale="Reds"); g_heat.update_layout(yaxis_autorange="reversed")
    # yll violin
    g_yll=px.violin(sub,y="tumour_stage",x="yll",orientation="h",color="tumour_stage",
                   color_discrete_sequence=px.colors.sequential.Reds,box=True,points=False)
    g_yll.update_layout(title="Years of Life Lost by Tumor Stage",xaxis_title="YLL",yaxis_title="Stage")
    # national bars %
    g_inc  = _bars_percent(latest_gco,"incidence_pct",f"Incidence Share – {int(latest_gco.year.iloc[0])}")
    g_mort = _bars_percent(latest_gco,"mortality_pct",f"Mortality Share – {int(latest_gco.year.iloc[0])}")
    # trend
    ts=gco.groupby("year")[["incidence_rate","mortality_rate"]].mean().reset_index(); ts["year_dt"]=pd.to_datetime(ts["year"],format="%Y")
    g_trend=px.line(ts,x="year_dt",y=["incidence_rate","mortality_rate"],
                    labels={"value":"Rate / 100 000","variable":"Metric","year_dt":"Year"},
                    title="Incidence vs Mortality Over Time",
                    color_discrete_sequence=["#8B0000","#FFC1C1"])
    # forecast
    hist=gco.groupby("year")["incidence_rate"].mean().reset_index()
    g_fc=go.Figure(); g_fc.add_scatter(x=hist["year"],y=hist["incidence_rate"],mode="lines+markers",name="Historical",line_color="#8B0000")
    g_fc.add_scatter(x=INC_FC_DF["year"],y=INC_FC_DF["forecast"],mode="lines+markers",name="Forecast",line_color="#FFC1C1")
    g_fc.add_scatter(x=INC_FC_DF["year"].tolist()+INC_FC_DF["year"][::-1].tolist(),
                     y=INC_FC_DF["upper"].tolist()+INC_FC_DF["lower"][::-1].tolist(),
                     fill="toself",fillcolor="rgba(255,0,0,0.2)",hoverinfo="skip",line=dict(color="rgba(255,255,255,0)"),
                     name="95% CI")
    g_fc.update_layout(title="3-Year Forecast of Incidence",xaxis_title="Year",yaxis_title="Incidence / 100 000")
    return k_pat, k_death, k_daly, g_gender, g_age, g_surg, g_heat, g_yll, g_inc, g_mort, g_trend, g_fc

# --- MAIN ---
if __name__ == "__main__":
    app.run_server(debug=True,port=8050)
