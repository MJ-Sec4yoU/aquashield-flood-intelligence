# AquaShield — Urban Flood Intelligence System
### Vasai-Virar-Nalasopara (VVN), Maharashtra

AquaShield is an AI-driven urban flood intelligence and climate resilience platform designed for hyper-local flood vulnerability mapping, road risk evaluation, and flood-resilient route navigation in the flood-prone Vasai-Virar-Nalasopara municipal region.

---

## 📌 Architecture & Modules

The platform is structured into modular pipelines:

1. **Phase 1: Data Acquisition & Preprocessing**
   - **Precipitation**: Multi-year daily precipitation from Open-Meteo Historical Weather API (ERA5 reanalysis).
   - **Elevation (DEM)**: Digital elevation models via OpenTopography SRTM APIs.
   - **Roads & Drainage**: Road network topology and drainage infrastructure extracted via OSMnx and OpenStreetMap.
   - **Features**: Accumulation metrics (3-day / 7-day rolling sums), day-over-day rainfall deltas, monsoon flags, and terrain interaction ratios.

2. **Phase 2: Flood Vulnerability Modeling**
   - **Classification Model**: Multi-class flood risk classifier (Low, Medium, High).
   - **Regression Model**: Continuous 0–100 flood vulnerability index.
   - **Evaluation**: Spatial cross-validation (unseen geographic cells) and temporal validation (unseen historical year), evaluating Macro-F1, minority class recall, and ROC-AUC.

3. **Phase 3: Flood-Aware Road Routing**
   - **Network Graph**: Road infrastructure graph enriched with elevation, drainage proximity, and live/historical flood risk weights.
   - **Dynamic Dijkstra Algorithm**: Evaluates shortest path vs. safest path penalizing inundated and high-risk road segments.

4. **Phase 4: Interactive Web Dashboard**
   - Built with Streamlit, Folium, and Plotly.
   - Includes 7-day live weather forecast integration, interactive geospatial flood heatmaps, ward vulnerability leaderboards, simulated IoT telemetry, and safe route visualization.

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Installation

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd urban-flood-intelligence
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Project

### 1. Data Pipeline
To collect and preprocess raw spatial and meteorological data:
```bash
# Data Collection
python src/data_pipeline/collect_rainfall_v2.py
python src/data_pipeline/collect_osm.py
python src/data_pipeline/collect_dem.py

# Preprocessing & Feature Engineering
python src/data_pipeline/preprocess_v2.py
```

### 2. Model Training & Evaluation
To train the Random Forest classification and regression models:
```bash
python src/model_training/train_flood_risk_model.py
```

### 3. Launch the Intelligence Dashboard
To launch the interactive web dashboard:
```bash
streamlit run src/dashboard/app.py
```

---

## ⚙️ Configuration

System parameters, geographical bounding boxes, rainfall risk thresholds, and directory paths are configured in [`config.yaml`](config.yaml).
