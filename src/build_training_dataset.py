import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from datetime import timedelta

print("="*70)
print("BUILDING FINAL TRAINING DATASET (PURE RISK SCORES)")
print("="*70)

# =========================================================================
# LOAD DATA
# =========================================================================

print("\n1. Loading fire data...")
fires_df = pd.read_csv("data/fires/fire_archive_SV-C2_716427.csv")
fires_df['datetime'] = pd.to_datetime(
    fires_df['acq_date'] + ' ' + fires_df['acq_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d %H%M'
)
fires_df = fires_df[fires_df['confidence'] == 'h'].copy()
print(f"   ✓ {len(fires_df):,} fires ({fires_df['datetime'].min()} to {fires_df['datetime'].max()})")

print("\n2. Loading cleaned weather data...")
weather_df = pd.read_csv("data/california_weather_clean.csv")
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
print(f"   ✓ {len(weather_df):,} weather records from {weather_df['location'].nunique()} stations")

print("\n3. Loading infrastructure data...")
infra_df = pd.read_csv("data/infrastructure/all_infrastructure_with_residential.csv")
print(f"   ✓ {len(infra_df):,} infrastructure assets")
print(f"   ✓ Types: {infra_df['type'].unique()}")

# =========================================================================
# BUILD SPATIAL INDICES
# =========================================================================

print("\n4. Building spatial indices...")
weather_coords = weather_df[['grid_lat', 'grid_lon']].drop_duplicates().values
weather_tree = cKDTree(weather_coords)
print(f"   ✓ Weather KDTree: {len(weather_coords)} unique locations")

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def haversine_vectorized(lat1, lon1, lats2, lons2):
    """Vectorized haversine distance"""
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(lats2), np.radians(lons2)
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def get_nearest_weather(lat, lon, target_time):
    """Find nearest weather station and closest timestamp"""
    _, idx = weather_tree.query([lat, lon])
    nearest = weather_coords[idx]
    
    station = weather_df[
        (weather_df['grid_lat'] == nearest[0]) &
        (weather_df['grid_lon'] == nearest[1])
    ].copy()
    
    station['time_diff'] = abs(station['datetime'] - target_time).dt.total_seconds()
    closest = station.nsmallest(1, 'time_diff')
    
    if len(closest) > 0 and closest.iloc[0]['time_diff'] < 21600:  # Within 6 hours
        return closest.iloc[0]
    return None

def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    """
    Calculate wind-fire alignment
    Returns: -1 (wind blowing fire away) to +1 (wind blowing fire toward asset)
    """
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    
    # Wind direction is "from", so add 180 to get "toward"
    wind_toward = (wind_dir + 180) % 360
    
    # Calculate alignment
    angle_diff = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    alignment = np.cos(np.radians(angle_diff))
    return alignment

def calculate_risk_score(min_dist, num_fires, wind_speed, max_frp, wind_align, temp, humidity):
    """
    PURE physics-informed risk score (0-100)
    No priority weighting - just fire behavior physics
    """
    # Base risk from distance (exponential decay)
    if min_dist < 1:
        base = 95
    elif min_dist < 3:
        base = 85
    elif min_dist < 7:
        base = 70
    elif min_dist < 15:
        base = 50
    elif min_dist < 30:
        base = 30
    elif min_dist < 50:
        base = 15
    else:
        base = 5
    
    # Wind speed amplification
    wind_factor = 1.0 + max(0, (wind_speed - 20) / 40)
    
    # Wind direction effect (if blowing toward asset)
    if wind_align > 0.5:  # Wind pushing fire toward asset
        wind_factor *= (1.0 + wind_align * 0.3)
    elif wind_align < -0.5:  # Wind blowing fire away
        wind_factor *= 0.8
    
    # Fire clustering (multiple fires harder to control)
    cluster_factor = 1.0 + min(num_fires / 20, 0.4)
    
    # Fire intensity
    intensity_factor = 1.0 + min(max_frp / 150, 0.3)
    
    # Weather conditions (temperature & humidity)
    # Hot + dry = dangerous fire weather
    weather_factor = 1.0
    if temp > 30 and humidity < 30:  # Extreme fire weather
        weather_factor = 1.2
    elif temp > 25 and humidity < 40:  # High fire danger
        weather_factor = 1.1
    elif temp < 15 or humidity > 70:  # Lower fire danger
        weather_factor = 0.9
    
    # Final risk calculation
    risk = base * wind_factor * cluster_factor * intensity_factor * weather_factor
    
    return min(risk, 100.0)

# =========================================================================
# BUILD TRAINING SAMPLES
# =========================================================================

print("\n5. Building training samples...")
print("   Sampling every 7 days across all years")
print("   Balanced sampling across all infrastructure types")

training_data = []

fire_start = fires_df['datetime'].min()
fire_end = fires_df['datetime'].max()
time_windows = pd.date_range(fire_start, fire_end, freq='7D')

print(f"   Time windows: {len(time_windows)}")

for i, current_time in enumerate(time_windows[:-1]):
    
    if i % 20 == 0:
        print(f"   Progress: {i}/{len(time_windows)} ({i/len(time_windows)*100:.1f}%) - {len(training_data):,} samples")
    
    current_fires = fires_df[fires_df['datetime'] <= current_time]
    
    if len(current_fires) < 5:
        continue
    
    # Random stratified sampling across all infrastructure types
    # This ensures balanced representation
    sampled = infra_df.sample(n=min(25, len(infra_df)), replace=False)
    
    for _, asset in sampled.iterrows():
        
        # Calculate distances to all current fires
        distances = haversine_vectorized(
            asset['lat'], asset['lon'],
            current_fires['latitude'].values,
            current_fires['longitude'].values
        )
        
        if distances.min() > 150:  # Skip if no fires within 150km
            continue
        
        min_dist = distances.min()
        mean_dist = distances.mean()
        num_nearby = (distances < 30).sum()
        max_frp = current_fires['frp'].max()
        
        # Get weather conditions
        weather = get_nearest_weather(asset['lat'], asset['lon'], current_time)
        if weather is None:
            continue
        
        # Find closest fire for wind alignment
        closest_fire_idx = distances.argmin()
        closest_fire = current_fires.iloc[closest_fire_idx]
        
        wind_align = calculate_wind_alignment(
            closest_fire['latitude'],
            closest_fire['longitude'],
            asset['lat'],
            asset['lon'],
            weather['wind_direction']
        )
        
        # Calculate PURE risk score (no priority bias)
        risk_score = calculate_risk_score(
            min_dist, num_nearby,
            weather['wind_speed_kmh'], max_frp,
            wind_align, weather['temp_c'], weather['humidity']
        )
        
        training_data.append({
            'min_distance_km': min_dist,
            'mean_distance_km': mean_dist,
            'num_fires_30km': num_nearby,
            'max_frp': max_frp,
            'wind_speed_kmh': weather['wind_speed_kmh'],
            'wind_direction': weather['wind_direction'],
            'temperature_c': weather['temp_c'],
            'humidity': weather['humidity'],
            'wind_fire_alignment': wind_align,
            'infrastructure_type': asset['type'],  # Keep type for analysis
            'risk_score': risk_score
        })

# Convert to DataFrame
df_train = pd.DataFrame(training_data)
df_train.to_csv('data/training_dataset_final.csv', index=False)

print(f"\n{'='*70}")
print("TRAINING DATASET COMPLETE")
print(f"{'='*70}")
print(f"Total samples: {len(df_train):,}")
print(f"Risk score range: {df_train['risk_score'].min():.1f} - {df_train['risk_score'].max():.1f}")
print(f"Mean risk: {df_train['risk_score'].mean():.1f}")
print(f"Median risk: {df_train['risk_score'].median():.1f}")

print(f"\nSamples by infrastructure type:")
print(df_train['infrastructure_type'].value_counts())

print(f"\nFeature summary:")
print(df_train[['min_distance_km', 'num_fires_30km', 'wind_speed_kmh', 
               'temperature_c', 'humidity', 'wind_fire_alignment', 'risk_score']].describe())

print(f"\nSaved to: data/training_dataset_final.csv")
print(f"{'='*70}")