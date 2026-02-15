import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime

st.set_page_config(page_title="Wildfire Risk Prediction - Bayesian AI", layout="wide")

st.title("🔥 Wildfire Infrastructure Risk Prediction")
st.caption("Bayesian Neural Network • 42,309 Assets • Real-time Analysis")

# Sidebar
st.sidebar.header("⚙️ Configuration")

api_base = "http://127.0.0.1:8000"

# Coordinate input
st.sidebar.subheader("📍 Location")

location_preset = st.sidebar.selectbox(
    "Quick Location",
    [
        "Custom Coordinates",
        "Paradise (Camp Fire 2018)",
        "Los Angeles",
        "San Francisco",
        "San Diego",
        "Sacramento"
    ]
)

# Preset coordinates
presets = {
    "Paradise (Camp Fire 2018)": (39.76, -121.62, 11),
    "Los Angeles": (34.05, -118.24, 10),
    "San Francisco": (37.77, -122.42, 11),
    "San Diego": (32.72, -117.16, 11),
    "Sacramento": (38.58, -121.49, 11)
}

if location_preset != "Custom Coordinates":
    center_lat, center_lon, zoom = presets[location_preset]
else:
    center_lat = st.sidebar.number_input("Latitude", value=37.0, format="%.4f")
    center_lon = st.sidebar.number_input("Longitude", value=-120.0, format="%.4f")
    zoom = st.sidebar.slider("Zoom Level", 6, 15, 10)

st.sidebar.markdown("---")

# Fire settings
st.sidebar.subheader("🔥 Fire Data")
fire_source = st.sidebar.selectbox("FIRMS Source", ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT"], index=0)
firms_days = st.sidebar.selectbox("Lookback Days", [1, 2, 3], index=0)
min_fire_conf = st.sidebar.slider("Min Fire Confidence", 0, 100, 60)

# Risk settings  
st.sidebar.subheader("⚠️ Risk Analysis")
horizon = st.sidebar.selectbox("Risk Horizon (hours)", [24, 48], index=0)
min_risk_display = st.sidebar.slider("Min Risk to Display", 0.0, 1.0, 0.3, 0.05)

# Infrastructure filters
st.sidebar.subheader("🏢 Infrastructure Types")
infra_types = st.sidebar.multiselect(
    "Show Types",
    ["Hospitals", "Schools", "Residential Areas", "Power Substations", 
     "Cell Towers", "Medical Clinics", "Nursing Homes"],
    default=["Hospitals", "Schools", "Power Substations"]
)

# Initialize session
if "risk_result" not in st.session_state:
    st.session_state["risk_result"] = None
if "clicked_coords" not in st.session_state:
    st.session_state["clicked_coords"] = None

# Helper functions
def fetch_fires(bbox):
    resp = requests.get(
        f"{api_base}/fires",
        params={
            "bbox": bbox,
            "days": firms_days,
            "source": fire_source,
            "min_confidence": min_fire_conf
        },
        timeout=60
    )
    resp.raise_for_status()
    return resp.json().get("fires", [])

def run_risk_analysis(bbox):
    payload = {
        "bbox": bbox,
        "horizon_hours": horizon,
        "firms_days": firms_days,
        "fire_source": fire_source,
        "fire_confidence_threshold": float(min_fire_conf),
        "weather_source": "openmeteo"
    }
    resp = requests.post(f"{api_base}/risk", json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()

def bbox_from_center(lat, lon, zoom):
    """Create bbox from center point and zoom"""
    # Rough approximation
    if zoom >= 12:
        delta = 0.05
    elif zoom >= 10:
        delta = 0.15
    elif zoom >= 8:
        delta = 0.5
    else:
        delta = 2.0
    
    return f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

def risk_color(score):
    if score >= 0.7:
        return "red"
    if score >= 0.4:
        return "orange"
    return "green"

# Main map
st.subheader(f"📍 {location_preset}")

# Show current coordinates
col1, col2, col3 = st.columns(3)
col1.metric("Latitude", f"{center_lat:.4f}")
col2.metric("Longitude", f"{center_lon:.4f}")
col3.metric("Zoom Level", zoom)

# Try to fetch fires for current view
current_bbox = bbox_from_center(center_lat, center_lon, zoom)

try:
    fires = fetch_fires(current_bbox)
    st.success(f"✓ {len(fires)} active fires detected")
except Exception as e:
    st.error(f"Fire fetch failed: {e}")
    fires = []

# Build map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=zoom,
    tiles="OpenStreetMap"
)

# Add fires
for fire in fires:
    folium.CircleMarker(
        location=[fire["lat"], fire["lon"]],
        radius=6,
        color="darkred",
        fill=True,
        fill_color="red",
        fill_opacity=0.8,
        popup=f"""
        <b>Fire Detection</b><br>
        Date: {fire.get('acq_date')}<br>
        Time: {fire.get('acq_time')}<br>
        Confidence: {fire.get('confidence', 'N/A')}
        """
    ).add_to(m)

# Display map
map_data = st_folium(
    m,
    width=1400,
    height=600,
    returned_objects=["last_clicked", "bounds"]
)

# Show clicked coordinates
if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    st.session_state["clicked_coords"] = clicked
    
    st.info(f"📍 Clicked: Lat {clicked['lat']:.4f}, Lon {clicked['lng']:.4f}")

# Analyze button
if st.button("🔍 Analyze Infrastructure Risk", type="primary"):
    with st.spinner("Running Bayesian NN risk analysis..."):
        try:
            result = run_risk_analysis(current_bbox)
            st.session_state["risk_result"] = result
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}")

# Display results
result = st.session_state.get("risk_result")

if result:
    st.markdown("---")
    st.subheader("📊 Risk Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔥 Fires", result.get("fire_count", 0))
    col2.metric("🏢 Assets", result.get("asset_count", 0))
    
    weather = result.get("weather", {})
    col3.metric("🌬️ Wind", f"{weather.get('wind_speed_kmh', 0):.1f} km/h")
    col4.metric("💧 Humidity", f"{weather.get('humidity_pct', 0):.0f}%")
    
    # Filter and sort assets
    all_assets = result.get("assets", [])
    
    # Apply filters
    filtered = [
        a for a in all_assets
        if a.get("risk_score", 0) >= min_risk_display
    ]
    
    if infra_types:
        filtered = [a for a in filtered if a.get("asset_type") in infra_types]
    
    # Sort by risk
    top_20 = sorted(filtered, key=lambda x: x.get("risk_score", 0), reverse=True)[:20]
    
    st.markdown("### 🚨 Top 20 Highest Risk Assets")
    
    if top_20:
        df = pd.DataFrame([
            {
                "Asset": a.get("name", "Unknown"),
                "Type": a.get("asset_type"),
                "City": a.get("city", "N/A"),
                "Risk %": f"{a.get('risk_score', 0)*100:.1f}",
                "Confidence %": f"{a.get('confidence', 0)*100:.1f}",
                "Uncertainty": f"±{a.get('uncertainty', 0):.1f}",
                "Distance (km)": a.get("features", {}).get("min_dist_to_fire_km", "N/A"),
                "Status": "🚨 EVACUATE" if a.get("risk_score", 0) > 0.7 else "⚠️ PREPARE"
            }
            for a in top_20
        ])
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Detail map
        st.markdown("### 🗺️ Risk Heatmap")
        
        detail_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom+1
        )
        
        # Add fires
        for fire in fires:
            folium.CircleMarker(
                [fire["lat"], fire["lon"]],
                radius=8,
                color="darkred",
                fill=True,
                fill_opacity=0.8
            ).add_to(detail_map)
        
        # Add top 100 risky assets only
        top_100 = sorted(filtered, key=lambda x: x.get("risk_score", 0), reverse=True)[:100]
        
        for asset in top_100:
            score = asset.get("risk_score", 0)
            conf = asset.get("confidence", 0)
            
            folium.CircleMarker(
                location=[asset["lat"], asset["lon"]],
                radius=6,
                color=risk_color(score),
                fill=True,
                fill_opacity=0.7,
                popup=f"""
                <b>{asset.get('name')}</b><br>
                Type: {asset.get('asset_type')}<br>
                Risk: {score*100:.1f}%<br>
                Confidence: {conf*100:.1f}%<br>
                Distance: {asset.get('features', {}).get('min_dist_to_fire_km')} km
                """
            ).add_to(detail_map)
        
        st_folium(detail_map, width=1400, height=500)
    
    else:
        st.info("No assets above risk threshold")

else:
    st.info("👆 Click 'Analyze Infrastructure Risk' to run prediction")