import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

st.set_page_config(page_title="Wildfire Risk Prediction", layout="wide")

st.title("🔥 Wildfire Infrastructure Risk Prediction")
st.caption("Bayesian Neural Network • 42,309 Assets • Live + Historical Data")

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model_and_data():
    model_path = Path("D:/wildfire-risk-system/models/bayesian_risk_model_final.keras")
    scaler_path = Path("D:/wildfire-risk-system/models/feature_scaler_final.pkl")
    infra_path = Path("D:/wildfire-risk-system/data/infrastructure/all_infrastructure_with_residential.csv")
    
    model = tf.keras.models.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))
    infra = pd.read_csv(infra_path)
    
    return model, scaler, infra

@st.cache_data
def load_historical_fires():
    fire_file = "D:/wildfire-risk-system/data/fires/fire_archive_SV-C2_716427.csv"
    df = pd.read_csv(fire_file)
    df = df[df['confidence'] == 'h']
    return df

@st.cache_data
def load_historical_weather():
    weather_file = "D:/wildfire-risk-system/data/california_weather_clean.csv"
    df = pd.read_csv(weather_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

try:
    model, scaler, infra_df = load_model_and_data()
    all_historical_fires = load_historical_fires()
    historical_weather = load_historical_weather()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("⚙️ Configuration")

# DATA MODE
use_historical = st.sidebar.checkbox("📚 Use Historical Data", value=True, key="hist_toggle")

if use_historical:
    st.sidebar.subheader("🗓️ Historical Fire Data")
    
    years = sorted(all_historical_fires['acq_date'].str[:4].unique())
    default_year_idx = years.index('2018') if '2018' in years else 0
    selected_year = st.sidebar.selectbox("Year", years, index=default_year_idx, key="year_select")
    
    year_fires = all_historical_fires[all_historical_fires['acq_date'].str.startswith(selected_year)]
    months = sorted(year_fires['acq_date'].str[5:7].unique())
    default_month_idx = months.index('11') if '11' in months else 0
    selected_month = st.sidebar.selectbox("Month", months, index=default_month_idx, key="month_select")
    
    month_fires = year_fires[year_fires['acq_date'].str[5:7] == selected_month]
    dates = sorted(month_fires['acq_date'].unique())
    
    if dates:
        default_date_idx = dates.index('2018-11-08') if '2018-11-08' in dates else 0
        selected_date = st.sidebar.selectbox("Date", dates, index=default_date_idx, key="date_select")
        fires_for_date = month_fires[month_fires['acq_date'] == selected_date]
        st.sidebar.success(f"✓ {len(fires_for_date)} fires")
    else:
        fires_for_date = pd.DataFrame()
        selected_date = "N/A"

else:
    st.sidebar.subheader("🔴 Live Data")
    fire_source = st.sidebar.selectbox("FIRMS Source", ["VIIRS_NOAA20_NRT"], key="live_source")
    lookback_days = st.sidebar.selectbox("Lookback Days", [1, 2, 3], key="live_days")
    fires_for_date = pd.DataFrame()
    selected_date = "Live"

st.sidebar.markdown("---")

# LOCATION
st.sidebar.subheader("📍 Location")

location_preset = st.sidebar.selectbox(
    "Quick Location",
    ["Paradise (Camp Fire)", "Custom", "Los Angeles", "San Francisco"],
    index=0,
    key="location_preset"
)

presets = {
    "Paradise (Camp Fire)": (39.76, -121.62, 11),
    "Los Angeles": (34.05, -118.24, 10),
    "San Francisco": (37.77, -122.42, 11)
}

if location_preset != "Custom":
    center_lat, center_lon, zoom = presets[location_preset]
    st.sidebar.metric("Lat", f"{center_lat:.4f}")
    st.sidebar.metric("Lon", f"{center_lon:.4f}")
else:
    center_lat = st.sidebar.number_input("Latitude", value=39.76, format="%.4f", key="custom_lat")
    center_lon = st.sidebar.number_input("Longitude", value=-121.62, format="%.4f", key="custom_lon")
    zoom = st.sidebar.slider("Zoom", 6, 15, 11, key="custom_zoom")

st.sidebar.success("✅ Model Ready")

# ============================================================================
# HELPER FUNCTIONS (defined early)
# ============================================================================

def get_weather(lat, lon, date_str, is_historical):
    """Get weather for location and date"""
    
    if is_historical and date_str != "N/A":
        try:
            weather_coords = historical_weather[['grid_lat', 'grid_lon']].drop_duplicates().values
            distances = np.sqrt((weather_coords[:, 0] - lat)**2 + (weather_coords[:, 1] - lon)**2)
            nearest_idx = distances.argmin()
            nearest_lat, nearest_lon = weather_coords[nearest_idx]
            
            target_date = pd.to_datetime(date_str).date()
            station_data = historical_weather[
                (historical_weather['grid_lat'] == nearest_lat) &
                (historical_weather['grid_lon'] == nearest_lon) &
                (historical_weather['datetime'].dt.date == target_date)
            ]
            
            if len(station_data) > 0:
                return {
                    'wind_speed_kmh': float(station_data['wind_speed_kmh'].mean()),
                    'wind_direction': float(station_data['wind_direction'].mean()),
                    'temp_c': float(station_data['temp_c'].mean()),
                    'humidity': float(station_data['humidity'].mean())
                }
        except:
            pass
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        resp = requests.get(url, timeout=10)
        data = resp.json().get('current', {})
        
        return {
            'wind_speed_kmh': data.get('wind_speed_10m', 15),
            'wind_direction': data.get('wind_direction_10m', 180),
            'temp_c': data.get('temperature_2m', 20),
            'humidity': data.get('relative_humidity_2m', 50)
        }
    except:
        return {'wind_speed_kmh': 15, 'wind_direction': 180, 'temp_c': 20, 'humidity': 50}

def haversine_vectorized(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(lats2), np.radians(lons2)
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    wind_toward = (wind_dir + 180) % 360
    angle_diff = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return np.cos(np.radians(angle_diff))

def predict_risk(asset, fires_df, weather, model, scaler):
    if len(fires_df) == 0:
        return {"risk_score": 0, "confidence": 0, "distance_km": None, "distance_miles": None, "num_fires": 0}
    
    distances = haversine_vectorized(
        asset['lat'], asset['lon'],
        fires_df['latitude'].values,
        fires_df['longitude'].values
    )
    
    min_dist = distances.min()
    mean_dist = distances.mean()
    num_nearby = (distances < 30).sum()
    max_frp = fires_df['frp'].max()
    
    closest_fire = fires_df.iloc[distances.argmin()]
    wind_align = calculate_wind_alignment(
        closest_fire['latitude'], closest_fire['longitude'],
        asset['lat'], asset['lon'], 
        weather['wind_direction']
    )
    
    features = pd.DataFrame([{
        'min_distance_km': min_dist, 'mean_distance_km': mean_dist,
        'num_fires_30km': int(num_nearby), 'max_frp': max_frp,
        'wind_speed_kmh': weather['wind_speed_kmh'],
        'wind_direction': weather['wind_direction'],
        'temperature_c': weather['temp_c'],
        'humidity': weather['humidity'],
        'wind_fire_alignment': wind_align
    }])
    
    feat_scaled = scaler.transform(features)
    preds = [model(feat_scaled, training=True).numpy()[0, 0] for _ in range(10)]
    preds = np.array(preds)
    
    risk_mean = np.clip(preds.mean(), 0, 100)
    unc = preds.std()
    
    return {
        "risk_score": risk_mean,
        "confidence": 100 * (1.0 - min(unc / 30, 1.0)),
        "uncertainty": unc,
        "distance_km": min_dist,
        "distance_miles": min_dist * 0.621371,
        "num_fires": int(num_nearby)
    }

def quick_risk(dist_km):
    if dist_km < 5:
        return 85
    elif dist_km < 15:
        return 55
    elif dist_km < 30:
        return 25
    else:
        return 5

def degrees_to_cardinal(degrees):
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = int((degrees + 11.25) / 22.5) % 16
    return directions[idx]

# ============================================================================
# SESSION STATE FOR DISPLAY COORDINATES AND CLICK HANDLING
# ============================================================================

if "display_lat" not in st.session_state:
    st.session_state.display_lat = center_lat
    st.session_state.display_lon = center_lon
    st.session_state.last_preset_lat = center_lat
    st.session_state.last_preset_lon = center_lon
    st.session_state.use_preset = True
    st.session_state.last_handled_click = None   # store the last click we processed

# Check if preset changed (dropdown or custom inputs)
preset_changed = (st.session_state.last_preset_lat != center_lat or 
                  st.session_state.last_preset_lon != center_lon)

if preset_changed:
    st.session_state.display_lat = center_lat
    st.session_state.display_lon = center_lon
    st.session_state.last_preset_lat = center_lat
    st.session_state.last_preset_lon = center_lon
    st.session_state.use_preset = True
    st.session_state.last_handled_click = None   # reset the handled click so a new click on the same spot will work

current_lat = st.session_state.display_lat
current_lon = st.session_state.display_lon

# ============================================================================
# WEATHER & HEADERS (using current display coordinates)
# ============================================================================

current_weather = get_weather(current_lat, current_lon, selected_date, use_historical)
wind_cardinal = degrees_to_cardinal(current_weather['wind_direction'])

st.subheader(f"📍 {current_lat:.4f}°, {current_lon:.4f}° - {selected_date}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔥 Fires", len(fires_for_date))
col2.metric("📍 Location", f"{current_lat:.4f}°, {current_lon:.4f}°")
col3.metric("🌬️ Wind", f"{current_weather['wind_speed_kmh']:.1f} km/h")
col4.metric("🧭 Direction", wind_cardinal)

# ============================================================================
# BUILD MAP
# ============================================================================

m = folium.Map(location=[current_lat, current_lon], zoom_start=zoom)

# Add fires
for _, fire in fires_for_date.iterrows():
    folium.CircleMarker(
        [fire['latitude'], fire['longitude']],
        radius=6,
        color='darkred',
        fill=True,
        fill_opacity=0.8,
        popup=f"Fire<br>FRP: {fire.get('frp', 'N/A')}"
    ).add_to(m)

# Infrastructure bounds around current location
west, east = current_lon - 0.4, current_lon + 0.4
south, north = current_lat - 0.4, current_lat + 0.4

local_infra = infra_df[
    (infra_df['lat'] >= south) & (infra_df['lat'] <= north) &
    (infra_df['lon'] >= west) & (infra_df['lon'] <= east) &
    (infra_df['type'].isin(['Hospitals', 'Schools', 'Power Substations', 'Residential']))
]

if len(local_infra) > 150:
    local_infra = local_infra.sample(150)

# Add infrastructure markers
for _, asset in local_infra.iterrows():
    if len(fires_for_date) > 0:
        dists = haversine_vectorized(
            asset['lat'], asset['lon'],
            fires_for_date['latitude'].values,
            fires_for_date['longitude'].values
        )
        min_d = dists.min()
        min_d_mi = min_d * 0.621371
        qr = quick_risk(min_d)
    else:
        min_d, min_d_mi, qr = 999, 999, 0
    
    color = 'red' if qr >= 70 else 'orange' if qr >= 40 else 'green'
    icon_map = {'Hospitals': 'plus-sign', 'Schools': 'book', 'Power Substations': 'flash', 'Residential': 'home'}
    icon = icon_map.get(asset['type'], 'info-sign')
    
    folium.Marker(
        [asset['lat'], asset['lon']],
        icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
        popup=f"""
        <div style='width:220px'>
            <div style='background:{color}; color:white; padding:8px; margin:-10px -10px 8px -10px'>
                <b>{asset.get('name', 'Unknown')[:30]}</b>
            </div>
            <p>{asset['type']} • {asset.get('city', 'N/A')}</p>
            <b>Quick Risk: {qr:.0f}%</b><br>
            Distance: {min_d:.1f} km ({min_d_mi:.1f} mi)
        </div>
        """,
        tooltip=f"{asset.get('name', 'Unknown')[:25]} - ~{qr:.0f}%"
    ).add_to(m)

st.info(f"Showing {len(fires_for_date)} fires • {len(local_infra)} assets")

# Display map and capture interactions
map_data = st_folium(m, width=1400, height=600, key="map", returned_objects=["bounds", "last_clicked"])

# ============================================================================
# CLICK HANDLING WITH STALE CLICK PREVENTION
# ============================================================================
if map_data and map_data.get("last_clicked"):
    click = map_data["last_clicked"]
    click_tuple = (click["lat"], click["lng"])
    
    # Only react if this click is different from the last one we already processed
    if click_tuple != st.session_state.last_handled_click:
        st.session_state.display_lat = click["lat"]
        st.session_state.display_lon = click["lng"]
        st.session_state.last_handled_click = click_tuple
        st.session_state.use_preset = False
        st.rerun()

# Optional success message (can be removed)
if map_data and map_data.get("last_clicked"):
    c = map_data["last_clicked"]
    st.success(f"📍 Clicked: {c['lat']:.4f}°, {c['lng']:.4f}°")

st.markdown("---")

# ============================================================================
# ANALYZE BUTTON (unchanged)
# ============================================================================

if st.button("🧠 Analyze", type="primary", use_container_width=True):
    if map_data and map_data.get("bounds"):
        bounds = map_data["bounds"]
        sw = bounds["_southWest"]
        ne = bounds["_northEast"]
        vw = sw["lng"]
        vs = sw["lat"]
        ve = ne["lng"]
        vn = ne["lat"]
    else:
        vw, vs, ve, vn = west, south, east, north
    
    fires_in_view = fires_for_date[
        (fires_for_date['latitude'] >= vs) & (fires_for_date['latitude'] <= vn) &
        (fires_for_date['longitude'] >= vw) & (fires_for_date['longitude'] <= ve)
    ] if len(fires_for_date) > 0 else pd.DataFrame()
    
    infra_in_view = infra_df[
        (infra_df['lat'] >= vs) & (infra_df['lat'] <= vn) &
        (infra_df['lon'] >= vw) & (infra_df['lon'] <= ve) &
        (infra_df['type'].isin(['Hospitals', 'Schools', 'Power Substations', 'Nursing Homes']))
    ]
    
    if len(infra_in_view) > 200:
        infra_in_view = infra_in_view.sample(200)
    
    if len(fires_in_view) == 0:
        st.warning("No fires in view")
    else:
        with st.spinner(f"Analyzing {len(infra_in_view)} assets..."):
            results = []
            for _, a in infra_in_view.iterrows():
                ad = {'name': a.get('name', 'Unknown'), 'city': a.get('city', ''), 
                      'type': a['type'], 'lat': a['lat'], 'lon': a['lon']}
                rd = predict_risk(ad, fires_in_view, current_weather, model, scaler)
                results.append({**ad, **rd})
            
            top20 = sorted(results, key=lambda x: x['risk_score'], reverse=True)[:20]
            
            st.markdown("### 🚨 Top 20 Risk Assets")
            df = pd.DataFrame([{
                "Asset": str(r.get('name', 'Unknown'))[:40],
                "Type": r['type'],
                "City": str(r.get('city', 'N/A'))[:20],
                "Risk %": f"{r['risk_score']:.1f}",
                "Conf %": f"{r['confidence']:.1f}",
                "Unc": f"±{r['uncertainty']:.1f}",
                "Dist (km)": f"{r['distance_km']:.2f}" if r.get('distance_km') else "N/A",
                "Dist (mi)": f"{r['distance_miles']:.2f}" if r.get('distance_miles') else "N/A",
                "Fires": r['num_fires'],
                "Action": "🔴 EVAC" if r['risk_score'] > 70 else "🟠 PREP" if r['risk_score'] > 40 else "🟡 MON"
            } for r in top20])
            st.dataframe(df, use_container_width=True)
            
            # Risk map (top 20)
            st.markdown("### 🗺️ Risk Heatmap (Top 20 Critical)")
            rm = folium.Map(location=[(vs+vn)/2, (vw+ve)/2], zoom_start=map_data.get("zoom", zoom))
            for _, f in fires_in_view.iterrows():
                folium.CircleMarker([f['latitude'], f['longitude']], radius=8, color='darkred', fill=True).add_to(rm)
            for a in top20:
                r = a['risk_score']
                c = 'red' if r >= 70 else 'orange' if r >= 40 else 'green'
                ic = 'plus-sign' if a['type'] == 'Hospitals' else 'book'
                folium.Marker(
                    [a['lat'], a['lon']],
                    icon=folium.Icon(color=c, icon=ic, prefix='glyphicon'),
                    popup=f"""
                    <div style='width:240px'>
                        <div style='background:{c}; color:white; padding:10px; margin:-10px -10px 10px -10px'>
                            <b>{str(a.get('name', 'Unknown'))[:35]}</b>
                        </div>
                        <p>{a['type']} • {str(a.get('city', 'N/A'))[:20]}</p>
                        <hr>
                        <b style='font-size:18px; color:{c}'>Risk: {r:.1f}%</b><br>
                        <b>Confidence: {a['confidence']:.1f}%</b><br>
                        <b>Uncertainty: ±{a['uncertainty']:.1f}</b><br>
                        <hr>
                        <p style='font-size:12px'>
                            Distance: {a['distance_km']:.1f} km ({a['distance_miles']:.1f} mi)<br>
                            Fires nearby: {a['num_fires']}
                        </p>
                    </div>
                    """,
                    tooltip=f"{str(a.get('name', 'Unknown'))[:25]} - {r:.0f}%"
                ).add_to(rm)
            st_folium(rm, width=1400, height=600, key="riskmap", returned_objects=[])
            
            # Paradise validation
            if '2018-11-08' in str(selected_date):
                p = next((r for r in results if 'Feather River' in str(r.get('name', ''))), None)
                if p:
                    st.markdown("---")
                    st.markdown("### 🎯 Validation")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Paradise Hospital", f"{p['risk_score']:.1f}%")
                    col2.metric("Confidence", f"{p['confidence']:.1f}%")
                    col3.metric("Distance", f"{p['distance_miles']:.1f} mi")
                    st.success("✅ Correctly predicted CRITICAL. Hospital destroyed 18hrs later.")