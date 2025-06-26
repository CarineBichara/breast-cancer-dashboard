import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def display_tab(filtered):
    st.subheader("Patient-Level Insights from Kaggle Dataset")

    # ────────────── Row 1: Age Distribution + Surgery Pie ──────────────
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown("#### Age Distribution of Patients")
        bins = list(range(30, 100, 5))
        labels = [f"{b}-{b+4}" for b in bins[:-1]]
        filtered["age_group"] = pd.cut(filtered["age"], bins=bins, labels=labels, right=False)
        age_counts = filtered["age_group"].value_counts().sort_index()
        bar_colors = ['#FFC1C1' if count < age_counts.max() * 0.8 else '#8B0000' for count in age_counts]

        fig_age, ax = plt.subplots(figsize=(5.5, 2.2))
        bars = ax.bar(age_counts.index.astype(str), age_counts.values, color=bar_colors)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=7)

        ax.set_xlabel("Age Group", fontsize=8)
        ax.set_ylabel("Patients", fontsize=8)
        ax.set_title("Patients by Age Group", fontsize=9)
        plt.xticks(rotation=45, fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig_age)

    with col2:
        st.markdown("#### Surgery Type Breakdown")
        surgery_counts = filtered["surgery_type"].value_counts()
        color_map = {
            "Simple Mastectomy": "#FFC1C1",
            "Lumpectomy": "#FF9999",
            "Modified Radical Mastectomy": "#8B0000",
            "Other": "#FFA07A"
        }
        fig_surgery, ax = plt.subplots(figsize=(2.8, 2.8), dpi=100)
        ax.pie(
            surgery_counts,
            labels=surgery_counts.index,
            autopct="%1.0f%%",
            startangle=90,
            colors=[color_map.get(x, "#D3D3D3") for x in surgery_counts.index]
        )
        ax.axis("equal")
        st.pyplot(fig_surgery)

    # ────────────── Row 2: YLL and DALY Heatmap ──────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### YLL by Tumor Stage")
        fig_yll = px.box(
            filtered,
            x="tumour_stage",
            y="yll",
            labels={"yll": "Years of Life Lost", "tumour_stage": "Tumor Stage"},
            title="YLL by Tumor Stage"
        )
        fig_yll.update_traces(marker_color="#8B0000")
        fig_yll.update_layout(
            height=320,
            title_x=0.3,
            plot_bgcolor="#FAFAFA",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_yll, use_container_width=True)

    with col4:
        st.markdown("#### Average DALY by Age & Tumor Stage")
        pivot = filtered.groupby(['age_group', 'tumour_stage'])['daly'].mean().unstack().fillna(0)

        fig_heatmap = px.imshow(
            pivot,
            text_auto=".1f",
            height=320,
            aspect="auto",
            color_continuous_scale='Reds',
            labels={"color": "Avg DALY", "x": "Tumor Stage", "y": "Age Group"},
            title="DALY Heatmap"
        )
        fig_heatmap.update_layout(
            title_x=0.3,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="#FAFAFA"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
