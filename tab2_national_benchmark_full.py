import streamlit as st
import pandas as pd
import plotly.express as px
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

def display_tab(main_path):
    st.subheader("National Benchmarks: Lebanon Breast Cancer Trends (Female)")

    # Load GCO data
    gco_file = os.path.join(main_path, "GCO_Lebanon_rates.csv")
    try:
        gco_df = pd.read_csv(gco_file)
    except FileNotFoundError:
        st.error("GCO_Lebanon_rates.csv not found.")
        return

    gco_df.columns = gco_df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Filter female data
    gco_df_female = gco_df[gco_df["gender"] == "Female"]

    col1, col2 = st.columns(2)

    # ────────────── Chart 1: Historical Trends ──────────────
    with col1:
        st.markdown("##### Incidence and Mortality (2015–2020)")

        df_long = gco_df_female.melt(
            id_vars="year",
            value_vars=["incidence_rate", "mortality_rate"],
            var_name="Metric",
            value_name="Rate"
        )

        fig1 = px.line(
            df_long,
            x="year",
            y="Rate",
            color="Metric",
            markers=True,
            line_shape="linear",
            color_discrete_map={
                "incidence_rate": "#8B0000",   # Dark red
                "mortality_rate": "#FFC1C1"    # Light red
            },
            labels={"year": "Year", "Rate": "Rate per 100,000"},
        )
        fig1.update_layout(
            height=350, margin=dict(l=10, r=10, t=30, b=10),
            title_text="",showlegend=True
        )
        fig1.update_traces(line=dict(width=2))
        st.plotly_chart(fig1, use_container_width=True)

    # ────────────── Chart 2: Forecast ──────────────
    with col2:
        st.markdown("##### Forecast of Incidence Rate (Next 3 Years)")

        y = gco_df_female.set_index("year")["incidence_rate"]

        model = SARIMAX(y, order=(1, 1, 0), seasonal_order=(0, 1, 1, 3),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=3)
        forecast_values = forecast.predicted_mean
        forecast_years = list(range(y.index.max() + 1, y.index.max() + 4))

        actual = pd.DataFrame({
            "year": y.index,
            "Rate": y.values.ravel(),
            "Source": ["Actual"] * len(y)
        })

        forecast_df = pd.DataFrame({
            "year": forecast_years,
            "Rate": forecast_values.values.ravel(),
            "Source": ["Forecast"] * len(forecast_years)
        })

        combined = pd.concat([actual, forecast_df], ignore_index=True)

        fig2 = px.line(
            combined,
            x="year",
            y="Rate",
            color="Source",
            markers=True,
            line_shape="linear",
            color_discrete_map={"Actual": "#8B0000", "Forecast": "#FFA07A"},
        )
        fig2.update_layout(
            height=350, margin=dict(l=10, r=10, t=30, b=10),
            title_text="", showlegend=True
        )
        fig2.update_traces(line=dict(width=2))
        st.plotly_chart(fig2, use_container_width=True)

    # ────────────── Source ──────────────
    st.caption("Source: Global Cancer Observatory (GCO), IARC 2023")
