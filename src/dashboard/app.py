import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import os
import sys
import json
import requests
import datetime

# ─── Phase 3 model imports ──────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model_training"))
try:
    from predict import predict_flood_risk, models_available
    PHASE3_IMPORT_OK = True
except Exception:
    PHASE3_IMPORT_OK = False

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaShield — VVN Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1a6eb5 0%, #0a9396 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #888;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f7ff, #e8f4fd);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1a6eb5;
        box-shadow: 0 2px 8px rgba(26,110,181,0.1);
    }
    .monsoon-alert {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 0.85rem 1.4rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
        animation: pulse-alert 2s infinite;
        box-shadow: 0 4px 18px rgba(192,57,43,0.4);
    }
    .offseason-banner {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 0.7rem 1.4rem;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(39,174,96,0.3);
    }
    .forecast-row { display:flex; gap:10px; margin-top:0.6rem; }
    .fcard {
        flex:1; background:linear-gradient(160deg,#0d1b2a,#1a3a5c);
        border-radius:12px; padding:0.7rem 0.5rem; color:white;
        text-align:center; border:1px solid rgba(255,255,255,0.1);
        min-width:0;
    }
    .fcard .day  { font-size:0.72rem; opacity:0.7; margin-bottom:2px; }
    .fcard .rain { font-size:1.1rem; font-weight:700; }
    .fcard .badge{ display:inline-block; margin-top:4px; padding:1px 8px;
                   border-radius:20px; font-size:0.68rem; font-weight:600; }
    .badge-High   { background:#c0392b; }
    .badge-Medium { background:#e67e22; }
    .badge-Low    { background:#27ae60; }
    .risk-high   { color: #c0392b; font-weight: 700; }
    .risk-medium { color: #e67e22; font-weight: 700; }
    .risk-low    { color: #27ae60; font-weight: 700; }
    @keyframes pulse-alert {
        0%   { box-shadow: 0 4px 18px rgba(192,57,43,0.4); }
        50%  { box-shadow: 0 4px 28px rgba(192,57,43,0.7); }
        100% { box-shadow: 0 4px 18px rgba(192,57,43,0.4); }
    }
    div[data-testid="stSidebar"] { background-color: #0d1b2a; }
    div[data-testid="stSidebar"] * { color: #e0e8f0 !important; }
    [data-testid="stMetricValue"] { font-size:1.55rem !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data loading ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "..", "data", "processed")

@st.cache_data
def load_flood():
    path = os.path.join(DATA_DIR, "flood_risk_dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
    return df

@st.cache_data
def load_roads():
    path = os.path.join(DATA_DIR, "road_risk_dataset.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# ─── Live forecast from Open-Meteo (free, no API key) ──────────────────────
@st.cache_data(ttl=600)  # refresh every 10 minutes
def fetch_live_forecast():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=19.45&longitude=72.82"
            "&daily=precipitation_sum,weathercode"
            "&timezone=Asia%2FKolkata&forecast_days=7"
        )
        r = requests.get(url, timeout=8)
        d = r.json()["daily"]
        df = pd.DataFrame({"date": d["time"], "precip_mm": d["precipitation_sum"]})
        df["date"] = pd.to_datetime(df["date"])
        df["risk"] = df["precip_mm"].apply(
            lambda x: "High" if (x or 0) >= 60 else ("Medium" if (x or 0) >= 20 else "Low")
        )
        df["precip_mm"] = df["precip_mm"].fillna(0.0)
        return df
    except Exception:
        return None

# Detect risk column name (handles slight naming differences)
def get_risk_col(df):
    for column in ["flood_risk", "infra_risk"]:
        if column in df.columns:
            return column

    for column in df.columns:
        if "risk" in column and "score" not in column:
            return column

    return None

def get_score_col(df):
    for c in df.columns:
        if "score" in c:
            return c
    return None

try:
    flood_df = load_flood()
    roads_df = load_roads()
    data_ok = True
except Exception as e:
    data_ok = False
    err_msg = str(e)

# ─── VVN Landmarks for Safe Routing ──────────────────────────────────────────
LANDMARKS = {
    "🚉 Virar Station":            (19.4640, 72.8115),
    "🚉 Nalasopara Station":       (19.4162, 72.7980),
    "🚉 Vasai Road Station":        (19.3795, 72.8350),
    "🚉 Naigaon Station":           (19.3625, 72.8520),
    "🏘️ Pelhar, Vasai":           (19.4500, 72.7900),
    "🏘️ Nalasopara East Market":  (19.4200, 72.8100),
    "🛣️ Manor Road Junction":     (19.4900, 72.8750),
    "🏆 Vasai Fort":               (19.3990, 72.8300),
    "🌊 Virar West Beach":          (19.4700, 72.8000),
    "🏥 Apex Hospital, Virar":     (19.4580, 72.8080),
}

@st.cache_resource
def build_route_graph_v2(_roads_df):
    """
    Build a NetworkX graph from road segment midpoints.
    v2: larger connect radius + largest-connected-component extraction
    guarantees Dijkstra always finds a path within the main network.
    """
    import networkx as nx
    from scipy.spatial import cKDTree
    import numpy as np

    df = _roads_df.reset_index(drop=True)
    coords      = df[["mid_lat", "mid_lon"]].values.astype(float)
    risk_scores = df["infra_risk_score"].fillna(50).values.astype(float)
    names       = df["name"].fillna("Unknown").values
    risks       = df["infra_risk"].fillna("Medium").values

    tree = cKDTree(coords)
    G = nx.Graph()

    for i in range(len(df)):
        G.add_node(i,
                   lat=float(coords[i, 0]),
                   lon=float(coords[i, 1]),
                   risk_score=float(risk_scores[i]),
                   risk_label=str(risks[i]),
                   name=str(names[i]))

    # Connect each segment to its 12 nearest neighbours within 2.0 km
    # Larger radius bridges gaps across creeks / sparse areas
    for i in range(len(df)):
        dists, idxs = tree.query(coords[i], k=13)
        for dist_deg, j in zip(dists[1:], idxs[1:]):
            if j == i:
                continue
            dist_km = float(dist_deg) * 111.0
            if dist_km > 2.0:
                continue
            risk_mult = 1.0 + 4.5 * (risk_scores[j] / 100.0)
            weight = dist_km * risk_mult
            if not G.has_edge(i, j):
                G.add_edge(i, j,
                           weight=float(weight),
                           dist_km=float(dist_km),
                           risk_score=float(risk_scores[j]))

    # Keep only the largest connected component so Dijkstra always succeeds
    largest_cc  = max(nx.connected_components(G), key=len)
    G_cc        = G.subgraph(largest_cc).copy()
    node_ids    = list(G_cc.nodes())                       # original node indices
    cc_coords   = np.array([[G_cc.nodes[n]["lat"],
                              G_cc.nodes[n]["lon"]] for n in node_ids])
    cc_tree     = cKDTree(cc_coords)                       # KD-tree on CC nodes only
    return G_cc, node_ids, cc_coords, cc_tree

# ─── Sidebar nav ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌊 AquaShield")
    st.markdown("**Urban Flood Intelligence**")
    st.markdown("*Vasai-Virar-Nalasopara*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🗺️ Flood Risk Map", "📊 Analytics", "🤖 ML Model",
         "🛣️ Road Risk Map", "🛡️ Safe Routes", "📡 IoT Sensor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Team AquaShield**")
    st.caption("Harshita • Mufis • Vidhan • Dhana")

if not data_ok:
    st.error(f"❌ Data load karne mein error: {err_msg}")
    st.info("Check karo ki `data/processed/` folder mein dono CSV files hain.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">🌊 AquaShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Urban Flood Intelligence System — Vasai-Virar-Nalasopara, Maharashtra</p>',
                unsafe_allow_html=True)

    risk_col = get_risk_col(flood_df)
    score_col = get_score_col(flood_df)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    total = len(flood_df)
    if risk_col:
        high   = (flood_df[risk_col].str.lower() == "high").sum()
        medium = (flood_df[risk_col].str.lower() == "medium").sum()
        low    = (flood_df[risk_col].str.lower() == "low").sum()
    else:
        high = medium = low = 0

    c1.metric("📍 Total records", f"{total:,}", "12 locations × 5 yrs")
    c2.metric("🔴 High-risk days", f"{high:,}", f"{high/total*100:.1f}% of total", delta_color="inverse")
    c3.metric("🟡 Medium-risk days", f"{medium:,}", f"{medium/total*100:.1f}%")
    c4.metric("🟢 Low-risk days", f"{low:,}", f"{low/total*100:.1f}%")

    # ── Monsoon alert banner ──────────────────────────────────────────────────
    current_month = datetime.datetime.now().month
    if current_month in [6, 7, 8, 9]:
        st.markdown(
            '<div class="monsoon-alert">⚠️ MONSOON SEASON ACTIVE — June to September is the peak flood risk period '
            'for Vasai-Virar-Nalasopara. Monitor rainfall closely and avoid flood-prone roads.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="offseason-banner">✅ Off-monsoon season — Flood risk is currently low. '
            'Monsoon season runs June–September.</div>',
            unsafe_allow_html=True
        )

    # ── Live 7-day rainfall forecast ─────────────────────────────────────────
    forecast_df = fetch_live_forecast()
    if forecast_df is not None:
        st.subheader("📡 Live 7-Day Rainfall Forecast — VVN")
        st.caption("Source: Open-Meteo API · Auto-refreshes every 10 min")
        day_names = ["Today", "Tomorrow"] + [
            (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%a %d")
            for i in range(2, 7)
        ]
        cols = st.columns(7)
        badge_colors = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#27ae60"}
        for i, (col, (_, row)) in enumerate(zip(cols, forecast_df.iterrows())):
            with col:
                st.markdown(
                    f'<div class="fcard">'
                    f'<div class="day">{day_names[i]}</div>'
                    f'<div class="rain">{row["precip_mm"]:.0f}<span style="font-size:0.65rem">mm</span></div>'
                    f'<span class="badge badge-{row["risk"]}">{row["risk"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown("")
    else:
        st.info("📡 Live forecast unavailable (check internet connection). Showing historical data below.")

    st.markdown("---")

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("📅 Risk distribution over time")
        if risk_col and "year" in flood_df.columns:
            yearly = flood_df.groupby(["year", risk_col]).size().reset_index(name="count")
            fig = px.bar(yearly, x="year", y="count", color=risk_col,
                         color_discrete_map={"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"},
                         barmode="stack", height=320)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title="Risk level")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date/risk column nahi mila CSV mein.")

    with col_b:
        st.subheader("⚠️ About VVN")
        st.info("""
**Vasai-Virar-Nalasopara** is one of the most flood-prone
urban regions near Mumbai.

- Area: ~310–380 sq. km
- Floods almost every monsoon
- Verified flood event: **18 July 2021**
- Low elevation + blocked drainage + rapid construction
        """)
        st.subheader("📋 Project phases")
        st.success("✅ Phase 0-1-2: Data collection & preprocessing")
        st.success("✅ Phase 3: ML model training (Mufis)")
        st.warning("🔄 Phase 4: Route engine (Vidhan)")
        st.info("🚧 Phase 5: Dashboard (Dhana — this!)")
        st.info("⏳ Phase 6: IoT sensor integration")

    st.markdown("---")
    st.subheader("🗂️ Raw data preview")
    tab1, tab2 = st.tabs(["Flood risk dataset", "Road risk dataset"])
    with tab1:
        st.dataframe(flood_df.head(20), use_container_width=True)
        st.caption(f"Total rows: {len(flood_df):,} | Columns: {list(flood_df.columns)}")
    with tab2:
        st.dataframe(roads_df.head(20), use_container_width=True)
        st.caption(f"Total rows: {len(roads_df):,} | Columns: {list(roads_df.columns)}")


# ════════════════════════════════════════════════════════════════
# PAGE 2 — FLOOD RISK MAP
# ════════════════════════════════════════════════════════════════
elif page == "🗺️ Flood Risk Map":
    st.markdown('<p class="main-header">🗺️ Flood Risk Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Location-wise flood risk across Vasai-Virar-Nalasopara</p>',
                unsafe_allow_html=True)

    risk_col  = get_risk_col(flood_df)
    score_col = get_score_col(flood_df)

    # Find lat/lon columns
    lat_col = next((c for c in flood_df.columns if "lat" in c), None)
    lon_col = next((c for c in flood_df.columns if "lon" in c or "lng" in c), None)

    col_filter, col_info = st.columns([1, 3])
    with col_filter:
        if "year" in flood_df.columns:
            years = sorted(flood_df["year"].dropna().unique().astype(int).tolist())
            sel_year = st.selectbox("Select year", years, index=len(years)-1)
            filtered = flood_df[flood_df["year"] == sel_year]
        else:
            filtered = flood_df
            st.info("No date column found.")

        if risk_col:
            levels = ["All"] + sorted(flood_df[risk_col].dropna().unique().tolist())
            sel_risk = st.selectbox("Filter by risk", levels)
            if sel_risk != "All":
                filtered = filtered[filtered[risk_col].str.lower() == sel_risk.lower()]

        st.metric("Showing records", f"{len(filtered):,}")

    with col_info:
        st.markdown("""
        **Color coding:**
        🟢 **Green** = Low risk &nbsp;&nbsp; 🟡 **Orange** = Medium risk &nbsp;&nbsp; 🔴 **Red** = High risk

        Each dot = one monitoring location on one day. Click any dot for details.
        """)

    # Map
    if lat_col and lon_col:
        map_cols = [c for c in [lat_col, lon_col, risk_col, score_col] if c]
        map_data = filtered[map_cols].dropna().copy()

        # Sample for performance (max 3000 points)
        if len(map_data) > 3000:
            map_data = map_data.sample(3000, random_state=42)

        view_mode = st.radio(
            "🗺️ Map view mode",
            ["🔥 Heatmap (intensity)", "🔵 Dots (individual points)"],
            horizontal=True
        )

        m = folium.Map(location=[19.45, 72.82], zoom_start=12, tiles="OpenStreetMap")

        if "Heatmap" in view_mode and score_col:
            heat_data = [
                [row[lat_col], row[lon_col], float(row[score_col])]
                for _, row in map_data.iterrows()
                if pd.notna(row.get(score_col, None))
            ]
            HeatMap(
                heat_data, radius=18, blur=22,
                min_opacity=0.35,
                gradient={"0.0": "#27ae60", "0.5": "#e67e22", "1.0": "#c0392b"}
            ).add_to(m)
            st.caption("🔥 Heatmap: Red = high flood risk intensity · Green = low risk")
        else:
            color_map = {"low": "green", "medium": "orange", "high": "red"}
            for _, row in map_data.iterrows():
                try:
                    risk_val  = str(row[risk_col]).lower() if risk_col else "low"
                    color     = color_map.get(risk_val, "blue")
                    score_val = f"Score: {row[score_col]:.1f}" if score_col and pd.notna(row[score_col]) else ""
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=6, color=color, fill=True,
                        fill_color=color, fill_opacity=0.7,
                        popup=folium.Popup(
                            f"<b>Risk: {risk_val.title()}</b><br>{score_val}<br>"
                            f"Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}",
                            max_width=200
                        )
                    ).add_to(m)
                except Exception:
                    continue

        st_folium(m, width=None, height=520, returned_objects=[])
    else:
        st.warning("CSV mein latitude/longitude columns nahi mile. Column names check karo.")
        st.write("Available columns:", list(flood_df.columns))


# ════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.markdown('<p class="main-header">📊 Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Historical flood data analysis — VVN 2019–2023</p>',
                unsafe_allow_html=True)

    risk_col  = get_risk_col(flood_df)
    score_col = get_score_col(flood_df)

    # KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric("Total data points", f"{len(flood_df):,}", "12 locations × 5 years daily")
    if risk_col:
        high = (flood_df[risk_col].str.lower() == "high").sum()
        c2.metric("High-risk records", f"{high:,}", f"{high/len(flood_df)*100:.1f}% of all data",
                  delta_color="inverse")
    road_risk_col = get_risk_col(roads_df)
    if road_risk_col:
        road_high = (roads_df[road_risk_col].str.lower() == "high").sum()
        c3.metric("High-risk roads", f"{road_high:,}",
                  f"{road_high/len(roads_df)*100:.0f}% of {len(roads_df):,} roads",
                  delta_color="inverse")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Monthly rainfall", "🎯 Risk distribution",
                                       "📈 Year trend", "🏘️ Area hotspots"])

    # Tab 1: Monthly rainfall
    with tab1:
        rain_col = next((c for c in flood_df.columns if "rain" in c), None)
        if rain_col and "month" in flood_df.columns:
            monthly = flood_df.groupby("month")[rain_col].mean().reset_index()
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["month_name"] = monthly["month"].map(month_names)
            monthly["is_monsoon"] = monthly["month"].isin([6,7,8,9])
            fig = px.bar(monthly, x="month_name", y=rain_col,
                         color="is_monsoon",
                         color_discrete_map={True: "#1a6eb5", False: "#b0c4de"},
                         labels={rain_col: "Avg rainfall (mm)", "month_name": "Month"},
                         title="Average monthly rainfall — VVN (2019–2023)",
                         height=380)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Blue bars = monsoon months (Jun–Sep). July is consistently the highest-risk month.")
        else:
            st.info("Rainfall ya month column nahi mila CSV mein.")

    # Tab 2: Risk distribution
    with tab2:
        if risk_col:
            col_pie, col_bar = st.columns(2)
            with col_pie:
                counts = flood_df[risk_col].value_counts().reset_index()
                counts.columns = ["Risk level", "Count"]
                fig = px.pie(counts, names="Risk level", values="Count",
                             color="Risk level",
                             color_discrete_map={"Low":"#27ae60","Medium":"#e67e22","High":"#c0392b"},
                             title="Flood risk zone distribution",
                             hole=0.4, height=340)
                fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig, use_container_width=True)
            with col_bar:
                if road_risk_col:
                    road_counts = roads_df[road_risk_col].value_counts().reset_index()
                    road_counts.columns = ["Risk level", "Count"]
                    fig2 = px.bar(road_counts, x="Risk level", y="Count",
                                  color="Risk level",
                                  color_discrete_map={"Low":"#27ae60","Medium":"#e67e22","High":"#c0392b"},
                                  title="Road infrastructure risk",
                                  height=340)
                    fig2.update_layout(showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Risk label column nahi mila.")

    # Tab 3: Year trend
    with tab3:
        if risk_col and "year" in flood_df.columns:
            trend = flood_df[flood_df[risk_col].str.lower() == "high"].groupby("year").size().reset_index(name="high_risk_days")
            fig = px.line(trend, x="year", y="high_risk_days",
                          markers=True,
                          labels={"high_risk_days": "High-risk day count", "year": "Year"},
                          title="High-risk days per year (2019–2023)",
                          height=360)
            # Annotate historical flood events
            FLOOD_EVENTS = [
                (2019, "Aug 2019\nVVN flooding",     "top left"),
                (2021, "18 Jul 2021\nRecord flood",  "top right"),
                (2022, "Oct 2022\nWaterlogging",      "top left"),
            ]
            for yr, lbl, pos in FLOOD_EVENTS:
                fig.add_vline(x=yr, line_dash="dash", line_color="#c0392b",
                              annotation_text=lbl, annotation_position=pos,
                              annotation_font_size=10, annotation_font_color="#c0392b")
            fig.update_traces(line_color="#1a6eb5", marker_size=10)
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📍 Red dashed lines = real historical flood events. 2021 spike validates model accuracy.")
        else:
            st.info("Year ya risk column nahi mila.")

    # Tab 4: Area hotspots
    with tab4:
        lat_col = next((c for c in flood_df.columns if "lat" in c), None)
        lon_col = next((c for c in flood_df.columns if "lon" in c or "lng" in c), None)
        if score_col and lat_col and lon_col:
            hotspots = (flood_df.groupby([lat_col, lon_col])[score_col]
                        .mean()
                        .reset_index()
                        .sort_values(score_col, ascending=False)
                        .head(12))
            hotspots.columns = ["Latitude", "Longitude", "Avg risk score"]
            hotspots["Avg risk score"] = hotspots["Avg risk score"].round(1)
            hotspots["Risk level"] = hotspots["Avg risk score"].apply(
                lambda x: "🔴 High" if x >= 60 else ("🟡 Medium" if x >= 30 else "🟢 Low"))
            def _ward(lat, lon):
                if lat > 19.45:
                    return "Virar West" if lon < 72.81 else "Virar East"
                elif lat > 19.40:
                    return "Nalasopara West" if lon < 72.80 else "Nalasopara East"
                elif lat > 19.37:
                    return "Vasai West" if lon < 72.83 else "Vasai East"
                else:
                    return "Naigaon / South Vasai"

            hotspots["Ward"] = [
                _ward(r["Latitude"], r["Longitude"]) for _, r in hotspots.iterrows()
            ]
            hotspots.insert(0, "Rank", range(1, len(hotspots) + 1))
            hotspots = hotspots[["Rank", "Ward", "Avg risk score", "Risk level", "Latitude", "Longitude"]]
            st.dataframe(hotspots, use_container_width=True, hide_index=True)

            # Ward-level leaderboard
            ward_tbl = (
                hotspots.groupby("Ward")["Avg risk score"]
                .mean().round(1)
                .sort_values(ascending=False)
                .reset_index()
            )
            ward_tbl.columns = ["Ward / Locality", "Avg Risk Score"]
            ward_tbl["Status"] = ward_tbl["Avg Risk Score"].apply(
                lambda x: "🔴 High" if x >= 60 else ("🟡 Medium" if x >= 30 else "🟢 Low")
            )
            st.markdown("")
            st.subheader("🏘️ Ward-Level Vulnerability Leaderboard")
            st.dataframe(ward_tbl, use_container_width=True, hide_index=True)
            st.caption("📈 Aggregate flood vulnerability by municipal ward. Data for civic planning & resource allocation.")
        else:
            st.info("Score ya lat/lon columns nahi mile.")


# ════════════════════════════════════════════════════════════════
# PAGE — ML MODEL (Phase 3)
# ════════════════════════════════════════════════════════════════
elif page == "🤖 ML Model":
    st.markdown('<p class="main-header">🤖 ML Flood Risk Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phase 3 — trained classifier + regressor on flood_risk_dataset.csv</p>',
                unsafe_allow_html=True)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    METRICS_PATH = os.path.join(BASE_DIR, "..", "..", "models", "metrics_summary.json")

    tab_metrics, tab_predict, tab_whatif = st.tabs(
        ["📈 Model performance", "🔮 Try a prediction", "🎲 What If Simulator"]
    )

    # ---- Tab 1: metrics from training ----
    with tab_metrics:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics = json.load(f)

            st.info(
                "Two evaluation splits were used. **Temporal** (train 2019-2022, test 2023) scores "
                "near-perfect because the original risk label was itself a rule-based formula — the "
                "model mostly re-derives it. **Spatial** (3 of 12 locations held out entirely) is the "
                "honest generalization test and is the number reported as the project's real result."
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Temporal split (diagnostic)")
                temp_df = pd.DataFrame(metrics["temporal_split"]).T
                temp_df.columns = ["Macro F1", "High-risk recall"]
                st.dataframe(temp_df.style.format("{:.3f}"), use_container_width=True)
            with col2:
                st.subheader("Spatial split (real result)")
                spat_df = pd.DataFrame(metrics["spatial_split"]).T
                spat_df.columns = ["Macro F1", "High-risk recall"]
                st.dataframe(spat_df.style.format("{:.3f}"), use_container_width=True)

            st.subheader("Regression — predicting the 0-100 risk score")
            col3, col4 = st.columns(2)
            with col3:
                st.caption("Temporal split")
                st.dataframe(pd.DataFrame(metrics["regression_temporal"]).T.style.format("{:.3f}"),
                             use_container_width=True)
            with col4:
                st.caption("Spatial split")
                st.dataframe(pd.DataFrame(metrics["regression_spatial"]).T.style.format("{:.3f}"),
                             use_container_width=True)

            st.success(f"✅ Best classifier: **{metrics['best_classifier']}** &nbsp;|&nbsp; "
                       f"Best regressor: **{metrics['best_regressor']}**")
        else:
            st.warning("⚠️ `models/metrics_summary.json` not found. Run "
                       "`src/model_training/train_flood_risk_model.py` first to generate it.")

    # ---- Tab 2: live prediction ----
    with tab_predict:
        if not PHASE3_IMPORT_OK or not models_available():
            st.warning("⚠️ Trained model files not found in `models/`. Run "
                       "`python src/model_training/train_flood_risk_model.py --data data/processed/"
                       "flood_risk_dataset.csv --out models` first, then reload this page.")
        else:
            st.caption("Enter conditions for a location and get a live risk prediction from the trained model.")
            c1, c2, c3 = st.columns(3)
            with c1:
                precip = st.slider("Rainfall today (mm)", 0.0, 300.0, 25.0)
                elevation = st.slider("Elevation (m)", 0.0, 150.0, 10.0)
            with c2:
                drainage_dist = st.slider("Distance to nearest drainage (km)", 0.0, 10.0, 1.0)
                building_count = st.slider("Building count (density proxy)", 0, 500, 100)
            with c3:
                lat = st.number_input("Latitude", value=19.40, format="%.4f")
                lon = st.number_input("Longitude", value=72.81, format="%.4f")
                month = st.selectbox("Month", list(range(1, 13)), index=6)

            if st.button("🔮 Predict flood risk", type="primary"):
                result = predict_flood_risk(
                    precipitation_mm=precip,
                    elevation_m=elevation,
                    drainage_dist_km=drainage_dist,
                    building_count=building_count,
                    latitude=lat,
                    longitude=lon,
                    month=month,
                )
                if result:
                    label      = result["risk_label"]
                    score      = result["risk_score"]
                    confidence = result.get("confidence")
                    css_color  = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}.get(label, "")
                    gauge_color = {"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"}.get(label, "#1a6eb5")

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.markdown(
                        f'<h3>Predicted risk: <span class="{css_color}">{label}</span></h3>',
                        unsafe_allow_html=True
                    )
                    rc2.metric("Risk score (0–100)", score)
                    if confidence:
                        rc3.metric("Model confidence", f"{confidence}%")

                    # Confidence gauge chart
                    if confidence:
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence,
                            number={"suffix": "%", "font": {"size": 32, "color": gauge_color}},
                            title={"text": f"Model Confidence — <b>{label}</b>",
                                   "font": {"size": 15}},
                            gauge={
                                "axis": {"range": [0, 100], "tickwidth": 1},
                                "bar": {"color": gauge_color, "thickness": 0.28},
                                "bgcolor": "white",
                                "borderwidth": 2,
                                "bordercolor": "#ddd",
                                "steps": [
                                    {"range": [0,  40], "color": "#eafaf1"},
                                    {"range": [40, 70], "color": "#fef9e7"},
                                    {"range": [70, 100], "color": "#fdedec"},
                                ],
                                "threshold": {
                                    "line": {"color": gauge_color, "width": 4},
                                    "thickness": 0.8,
                                    "value": confidence,
                                },
                            },
                        ))
                        gauge_fig.update_layout(
                            height=260, margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(gauge_fig, use_container_width=True)
                else:
                    st.error("Prediction failed — check model files in `models/`.")

    # ---- Tab 3: What If Scenario Simulator ----
    with tab_whatif:
        st.subheader("🎲 What If Scenario Simulator")
        st.caption("Adjust rainfall and see how flood risk shifts across all 12 VVN monitoring locations.")

        if not PHASE3_IMPORT_OK or not models_available():
            st.warning("⚠️ Train models first (run `train_flood_risk_model.py`) to use the simulator.")
        else:
            SIM_LOCATIONS = [
                {"name": "Virar West",      "lat": 19.465, "lon": 72.799, "elev": 8,  "drain": 0.8, "bldg": 200},
                {"name": "Virar East",      "lat": 19.462, "lon": 72.822, "elev": 12, "drain": 1.2, "bldg": 180},
                {"name": "Nalasopara W",    "lat": 19.418, "lon": 72.790, "elev": 5,  "drain": 0.5, "bldg": 300},
                {"name": "Nalasopara E",    "lat": 19.420, "lon": 72.808, "elev": 6,  "drain": 0.9, "bldg": 250},
                {"name": "Vasai West",      "lat": 19.395, "lon": 72.800, "elev": 7,  "drain": 1.0, "bldg": 150},
                {"name": "Vasai East",      "lat": 19.382, "lon": 72.837, "elev": 15, "drain": 1.5, "bldg": 120},
                {"name": "Naigaon",         "lat": 19.362, "lon": 72.852, "elev": 20, "drain": 2.0, "bldg": 90},
                {"name": "Pelhar",          "lat": 19.450, "lon": 72.790, "elev": 30, "drain": 3.0, "bldg": 60},
                {"name": "Manor Road",      "lat": 19.490, "lon": 72.875, "elev": 25, "drain": 2.5, "bldg": 70},
                {"name": "Vasai Fort",      "lat": 19.400, "lon": 72.831, "elev": 10, "drain": 1.1, "bldg": 100},
                {"name": "Arnala",          "lat": 19.430, "lon": 72.780, "elev": 4,  "drain": 0.4, "bldg": 160},
                {"name": "Bolinj",          "lat": 19.412, "lon": 72.783, "elev": 9,  "drain": 1.3, "bldg": 130},
            ]

            c_sa, c_sb = st.columns([1, 2])
            with c_sa:
                st.markdown("**Adjust scenario parameters:**")
                delta_rain = st.slider(
                    "🌧️ Rainfall change (mm)",
                    min_value=-50, max_value=200, value=0, step=10,
                    help="Positive = more rain than 25mm baseline. Negative = drier conditions."
                )
                sim_rain  = float(max(0, 25 + delta_rain))
                sim_month = st.selectbox(
                    "📅 Simulate for month", list(range(1, 13)), index=6,
                    format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                           "Jul","Aug","Sep","Oct","Nov","Dec"][m - 1]
                )
                run_sim = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
                st.metric("🌧️ Simulated rainfall",
                          f"{sim_rain:.0f} mm",
                          f"{'+' if delta_rain >= 0 else ''}{delta_rain} mm vs baseline")

            with c_sb:
                if run_sim or "sim_results" not in st.session_state:
                    sim_out = []
                    for loc in SIM_LOCATIONS:
                        try:
                            r = predict_flood_risk(
                                precipitation_mm=sim_rain,
                                elevation_m=loc["elev"],
                                drainage_dist_km=loc["drain"],
                                building_count=loc["bldg"],
                                latitude=loc["lat"],
                                longitude=loc["lon"],
                                month=sim_month,
                            )
                            if r:
                                sim_out.append({
                                    "Location":   loc["name"],
                                    "Risk Score": r["risk_score"],
                                    "Risk Label": r["risk_label"],
                                })
                        except Exception:
                            pass
                    st.session_state["sim_results"] = sim_out

                sim_res = st.session_state.get("sim_results", [])
                if sim_res:
                    sim_df = pd.DataFrame(sim_res).sort_values("Risk Score", ascending=False)
                    clr_map = {"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"}
                    bar_colors = [clr_map.get(l, "#888") for l in sim_df["Risk Label"]]
                    month_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
                                 "Jul","Aug","Sep","Oct","Nov","Dec"][sim_month - 1]

                    fig_sim = go.Figure(go.Bar(
                        x=sim_df["Location"],
                        y=sim_df["Risk Score"],
                        marker_color=bar_colors,
                        text=sim_df["Risk Label"],
                        textposition="auto",
                    ))
                    fig_sim.add_hline(y=60, line_dash="dash", line_color="red",
                                      annotation_text="High threshold (60)")
                    fig_sim.add_hline(y=30, line_dash="dot",  line_color="orange",
                                      annotation_text="Medium threshold (30)")
                    fig_sim.update_layout(
                        title=f"Flood risk @ {sim_rain:.0f} mm rainfall — {month_lbl}",
                        yaxis_title="Risk Score (0–100)",
                        xaxis_tickangle=-35,
                        height=370,
                        margin=dict(l=0, r=0, t=45, b=0),
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

                    high_n   = sum(1 for r in sim_res if r["Risk Label"] == "High")
                    medium_n = sum(1 for r in sim_res if r["Risk Label"] == "Medium")
                    low_n    = sum(1 for r in sim_res if r["Risk Label"] == "Low")
                    sa2, sb2, sc2 = st.columns(3)
                    sa2.metric("🔴 High risk zones",   high_n,   f"of {len(sim_res)}", delta_color="inverse")
                    sb2.metric("🟡 Medium risk zones", medium_n, f"of {len(sim_res)}")
                    sc2.metric("🟢 Low risk zones",    low_n,    f"of {len(sim_res)}", delta_color="off")


# ════════════════════════════════════════════════════════════════
# PAGE 4 — ROAD RISK MAP
# ════════════════════════════════════════════════════════════════
elif page == "🛣️ Road Risk Map":
    st.markdown('<p class="main-header">🛣️ Road Risk Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Infrastructure risk — which roads flood first</p>',
                unsafe_allow_html=True)

    road_risk_col = get_risk_col(roads_df)
    lat_col = next((c for c in roads_df.columns if "lat" in c), None)
    lon_col = next((c for c in roads_df.columns if "lon" in c or "lng" in c), None)

    col_f, col_i = st.columns([1, 3])
    with col_f:
        if road_risk_col:
            levels = ["All"] + sorted(roads_df[road_risk_col].dropna().unique().tolist())
            sel = st.selectbox("Filter roads by risk", levels)
            road_filtered = roads_df if sel == "All" else roads_df[roads_df[road_risk_col].str.lower() == sel.lower()]
        else:
            road_filtered = roads_df
        st.metric("Roads shown", f"{len(road_filtered):,}")

    with col_i:
        st.markdown("🟢 Low-risk roads = safe to use &nbsp;&nbsp; 🟡 Medium = caution &nbsp;&nbsp; 🔴 High = avoid during floods")

    if lat_col and lon_col:
        sample = road_filtered.sample(min(2000, len(road_filtered)), random_state=42)
        m = folium.Map(location=[19.45, 72.82], zoom_start=12, tiles="OpenStreetMap")
        color_map = {"low": "green", "medium": "orange", "high": "red"}
        for _, row in sample.iterrows():
            try:
                risk_val = str(row[road_risk_col]).lower() if road_risk_col else "low"
                color    = color_map.get(risk_val, "blue")
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=4,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"Road risk: {risk_val.title()}"
                ).add_to(m)
            except Exception:
                continue
        st_folium(m, width=None, height=500, returned_objects=[])
    else:
        st.warning("Latitude/longitude columns roads dataset mein nahi mile.")
        st.write("Available columns:", list(roads_df.columns))


# ════════════════════════════════════════════════════════════════
# PAGE 5 — SAFE ROUTES (Dijkstra routing engine)
# ════════════════════════════════════════════════════════════════
elif page == "🛡️ Safe Routes":
    import networkx as nx

    st.markdown('<p class="main-header">🛡️ Safe Route Finder</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dijkstra’s algorithm — routes that avoid high flood-risk roads — VVN</p>',
                unsafe_allow_html=True)

    # Build / fetch cached graph (v2 = larger radius + connected component)
    with st.spinner("⏳ Building road graph (first load only, ~15 s)…"):
        try:
            G, node_ids, cc_coords, cc_tree = build_route_graph_v2(roads_df)
            graph_ok = True
        except Exception as ge:
            graph_ok = False
            st.error(f"Graph build failed: {ge}")

    if graph_ok:
        col_ctrl, col_map = st.columns([1, 2.8])

        with col_ctrl:
            st.subheader("📍 Select Route")
            landmark_names = list(LANDMARKS.keys())
            start_name = st.selectbox("🟢 Start location", landmark_names, index=0)
            end_name   = st.selectbox("🔴 End location",   landmark_names, index=2)

            st.markdown("---")
            show_roads = st.checkbox("🗺️ Show road risk layer", value=True)
            st.markdown("---")

            find_btn = st.button("🔍 Find Safest Route", type="primary", use_container_width=True)

            st.markdown("""
            **How it works:**
            - Each road segment is a graph node
            - High-risk roads get **4.5×** weight penalty
            - Dijkstra’s algorithm finds the lowest-cost path
            - Result = shortest path that avoids flooded roads
            """)
            st.markdown("")
            st.markdown("🟢 Low risk &nbsp; 🟡 Medium &nbsp; 🔴 High risk")

        with col_map:
            start_coord = LANDMARKS[start_name]
            end_coord   = LANDMARKS[end_name]

            m = folium.Map(location=[19.42, 72.83], zoom_start=12,
                           tiles="OpenStreetMap")

            # Background road risk layer
            if show_roads:
                sample_r = roads_df.sample(min(1800, len(roads_df)), random_state=7)
                c_map = {"Low": "#27ae60", "Medium": "#e67e22", "High": "#c0392b"}
                for _, row in sample_r.iterrows():
                    try:
                        clr = c_map.get(str(row["infra_risk"]), "#888")
                        folium.CircleMarker(
                            location=[row["mid_lat"], row["mid_lon"]],
                            radius=3, color=clr, fill=True, fill_color=clr,
                            fill_opacity=0.45, weight=0,
                        ).add_to(m)
                    except Exception:
                        continue

            # Start / End markers
            folium.Marker(
                start_coord,
                popup=folium.Popup(f"<b>🟢 START</b><br>{start_name}", max_width=180),
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
                tooltip=f"Start: {start_name}"
            ).add_to(m)
            folium.Marker(
                end_coord,
                popup=folium.Popup(f"<b>🔴 END</b><br>{end_name}", max_width=180),
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
                tooltip=f"End: {end_name}"
            ).add_to(m)

            # Run Dijkstra when button clicked OR on first page load
            run_route = find_btn or ("safe_route" not in st.session_state)

            if run_route and start_name != end_name:
                # Find nearest node inside the connected component
                _, si = cc_tree.query(start_coord)
                _, ei = cc_tree.query(end_coord)
                start_node = node_ids[si]
                end_node   = node_ids[ei]
                try:
                    path_nodes = nx.shortest_path(
                        G, int(start_node), int(end_node), weight="weight"
                    )
                    path_coords = [
                        (G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path_nodes
                    ]
                    total_dist = sum(
                        G.edges[path_nodes[i], path_nodes[i + 1]].get("dist_km", 0)
                        for i in range(len(path_nodes) - 1)
                    )
                    avg_risk = (
                        sum(G.nodes[n]["risk_score"] for n in path_nodes)
                        / max(len(path_nodes), 1)
                    )
                    high_segs = sum(
                        1 for n in path_nodes if G.nodes[n]["risk_score"] >= 60
                    )
                    st.session_state["safe_route"] = {
                        "path_coords": path_coords,
                        "total_dist":  round(total_dist, 2),
                        "avg_risk":    round(avg_risk, 1),
                        "high_segs":   high_segs,
                        "n_nodes":     len(path_nodes),
                        "start":       start_name,
                        "end":         end_name,
                    }
                except nx.NetworkXNoPath:
                    st.session_state["safe_route"] = None
                    st.error("❌ No connected path found. Try different locations.")
                except Exception as re:
                    st.session_state["safe_route"] = None
                    st.error(f"Routing error: {re}")

            # Draw stored route
            route = st.session_state.get("safe_route")
            if route and route["start"] == start_name and route["end"] == end_name:
                folium.PolyLine(
                    route["path_coords"],
                    color="#2471a3", weight=7, opacity=0.9,
                    tooltip="🛡️ Safest Route",
                    dash_array=None,
                ).add_to(m)
                # Waypoint dots along route (every 5th node)
                for pt in route["path_coords"][::5]:
                    folium.CircleMarker(
                        pt, radius=3, color="#2471a3",
                        fill=True, fill_color="white", fill_opacity=0.9, weight=2
                    ).add_to(m)

            st_folium(m, width=None, height=510, returned_objects=[])

        # Route stats panel
        route = st.session_state.get("safe_route")
        if route and route["start"] == start_name and route["end"] == end_name:
            st.markdown("---")
            st.subheader("📊 Route Analysis")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("📍 Route from",  route["start"].split()[-1])
            s2.metric("📍 Route to",    route["end"].split()[-1])
            s3.metric("⚠️ Avg risk score",  f"{route['avg_risk']}/100",
                      "Lower = safer" if route["avg_risk"] < 40 else "Moderate risk")
            s4.metric("🔴 High-risk segments",  str(route["high_segs"]),
                      delta="on safest path", delta_color="inverse")

            safety_pct = max(0, 100 - route["avg_risk"])
            st.markdown(f"""
            > **Route safety score: {safety_pct:.0f}/100** &nbsp;
            {'✅ Good — mostly low-risk roads' if safety_pct >= 65
             else '⚠️ Moderate — some risk unavoidable on this corridor'
             if safety_pct >= 45
             else '🔴 High risk — consider waiting for flood to recede'}
            """)
        elif start_name == end_name:
            st.warning("⚠️ Please select different start and end locations.")


# ════════════════════════════════════════════════════════════════
# PAGE 6 — IoT SENSOR (Placeholder)
# ════════════════════════════════════════════════════════════════
elif page == "📡 IoT Sensor":
    import random
    st.markdown('<p class="main-header">📡 IoT Sensor Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Simulated live sensor stream — ESP32 + Firebase integration (Phase 6)</p>',
                unsafe_allow_html=True)

    # Seed with current minute — values change every 60 seconds
    _seed = int(datetime.datetime.now().timestamp() // 60)
    _rng  = random.Random(_seed)

    water_level = _rng.uniform(28, 75)
    soil_moist  = _rng.uniform(40, 95)
    sensor_temp = _rng.uniform(26, 34)
    rain_now    = _rng.uniform(0, 45)
    ALERT_LVL   = 70
    WARN_LVL    = 50

    if water_level >= ALERT_LVL:
        _status_txt, _status_clr = "🚨 FLOOD ALERT",  "#c0392b"
    elif water_level >= WARN_LVL:
        _status_txt, _status_clr = "⚠️  WARNING",      "#e67e22"
    else:
        _status_txt, _status_clr = "✅ SAFE",           "#27ae60"

    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    # Status banner
    st.markdown(
        f'<div style="background:{_status_clr};color:white;padding:12px 20px;'
        f'border-radius:10px;font-size:1.25rem;font-weight:700;text-align:center;'
        f'margin-bottom:1rem;">'
        f'{_status_txt} &nbsp;—&nbsp; Water level: {water_level:.1f} cm'
        f' &nbsp;|&nbsp; Updated: {now_str}</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    _delta  = _rng.uniform(0.3, 2.8)
    _up     = _rng.random() > 0.45
    _arrow  = "▲" if _up else "▼"
    m1.metric("💧 Water Level",
              f"{water_level:.1f} cm",
              f"{_arrow} {_delta:.1f} cm",
              delta_color="inverse" if _up and water_level > WARN_LVL else "normal")
    m2.metric("🌡️ Sensor Temp",  f"{sensor_temp:.1f}°C")
    m3.metric("🌱 Soil Moisture", f"{soil_moist:.0f}%")
    m4.metric("🌧️ Rain Rate",    f"{rain_now:.1f} mm/hr")

    st.markdown("---")

    # 60-minute simulated history
    _hist = [_rng.uniform(max(5, water_level - 22), water_level) for _ in range(60)]
    _hist[-1] = water_level
    _x    = list(range(-59, 1))

    fig_iot = go.Figure()
    fig_iot.add_trace(go.Scatter(
        x=_x, y=_hist, mode="lines",
        line=dict(color="#1a6eb5", width=2.5),
        fill="tozeroy", fillcolor="rgba(26,110,181,0.10)",
        name="Water level (cm)"
    ))
    fig_iot.add_hline(y=ALERT_LVL, line_dash="dash", line_color="#c0392b",
                      annotation_text="🚨 Alert (70 cm)", annotation_font_color="#c0392b")
    fig_iot.add_hline(y=WARN_LVL,  line_dash="dot",  line_color="#e67e22",
                      annotation_text="⚠️ Warning (50 cm)", annotation_font_color="#e67e22")
    fig_iot.update_layout(
        title=f"Water level — last 60 minutes (as of {now_str})",
        xaxis_title="Minutes ago",
        yaxis_title="Water level (cm)",
        yaxis=dict(range=[0, 90]),
        height=340,
        margin=dict(l=0, r=0, t=45, b=0)
    )
    st.plotly_chart(fig_iot, use_container_width=True)

    col_btn, col_cap = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Refresh Sensor Data", use_container_width=True):
            st.rerun()
    with col_cap:
        st.caption(
            "📡 Values update every 60 s (seeded to current minute). "
            "Real ESP32 + Firebase stream planned for Phase 6. "
            "Station: Vasai Road canal monitoring point."
        )



