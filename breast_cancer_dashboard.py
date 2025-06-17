# (continued from above)

# --------------------------------------------------------------------------------
# SECTION C · MAP OF BREAST‑CANCER SCREENING HOSPITALS (continued)
# --------------------------------------------------------------------------------

st.header("🗺️ Find a Breast‑Cancer Screening Hospital")
CSV_PATH = "lebanon_private_hospitals_complete.csv"
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found at\n{CSV_PATH}")
    st.stop()

hosp = pd.read_csv(CSV_PATH)

# 1️⃣ Search box (by city, Caza, or hospital name)
query = st.text_input(
    "🔎 Search by City / Caza / Hospital (e.g. Beirut, Aley, Akkar, Tripoli)", ""
).strip().lower()

if query:
    mask = (
        hosp["Caza"].astype(str).str.lower().str.contains(query) |
        hosp.get("governorate", hosp["Caza"]).astype(str).str.lower().str.contains(query) |
        hosp["Name"].str.lower().str.contains(query)
    )
    data = hosp[mask].copy()
else:
    data = hosp.copy()

st.write(f"**{len(data)} hospital(s) found**")

# 2️⃣ Show results on an interactive map
if data.empty:
    st.warning("No hospitals match that search. Try another keyword.")
else:
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
            icon=folium.Icon(color="red", icon="plus-sign")
        ).add_to(cluster)

    st_folium(m, width=750, height=500)

    with st.expander("↕️ See results in a table"):
        st.dataframe(
            data[["Name", "Caza", "Phone", "Investment_Authorization_Nb"]].reset_index(drop=True)
        )

# --------------------------------------------------------------------------------
# SECTION D · CLOSING MESSAGE
# --------------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    '<p style="font-weight:bold; font-size:18px; font-style:italic;">
    Early detection saves lives. Awareness is the first step to prevention.
    </p>', unsafe_allow_html=True
)

st.caption("© Carine Bichara | Breast Cancer Awareness Dashboard")
