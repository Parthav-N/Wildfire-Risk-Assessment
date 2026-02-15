import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime

print("="*70)
print("DENSE CALIFORNIA WEATHER GRID (200 points)")
print("="*70)

# California bounding box
west, east = -124.5, -114.0
south, north = 32.5, 42.0

# Create 0.5 degree grid (~50km spacing)
lats = np.arange(south, north + 0.5, 0.5)
lons = np.arange(west, east + 0.5, 0.5)

grid_points = []
for lat in lats:
    for lon in lons:
        grid_points.append({'lat': round(lat, 2), 'lon': round(lon, 2)})

years = list(range(2017, 2025))

print(f"Grid points: {len(grid_points)}")
print(f"Years: {years}")
print(f"Total API calls: {len(grid_points) * len(years)}")
print(f"Estimated time: {len(grid_points) * len(years) * 0.6 / 60:.0f} minutes\n")

def fetch_weather(lat, lon, year):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': f'{year}-01-01',
        'end_date': f'{year}-12-31',
        'hourly': ['temperature_2m', 'relative_humidity_2m', 
                   'wind_speed_10m', 'wind_direction_10m'],
        'timezone': 'UTC'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'hourly' not in data:
            return None
        
        df = pd.DataFrame({
            'datetime': data['hourly']['time'],
            'temp_c': data['hourly']['temperature_2m'],
            'humidity': data['hourly']['relative_humidity_2m'],
            'wind_speed_kmh': data['hourly']['wind_speed_10m'],
            'wind_direction': data['hourly']['wind_direction_10m'],
            'grid_lat': lat,
            'grid_lon': lon
        })
        
        # Sample every 12 hours
        df = df.iloc[::12].reset_index(drop=True)
        return df
        
    except Exception as e:
        return None

all_weather = []
start_time = datetime.now()
completed = 0
total = len(grid_points) * len(years)

for point in grid_points:
    for year in years:
        completed += 1
        
        if completed % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = completed / elapsed
            remaining = (total - completed) / rate / 60
            print(f"[{completed}/{total}] {completed/total*100:.1f}% | ETA: {remaining:.0f} min")
        
        df = fetch_weather(point['lat'], point['lon'], year)
        
        if df is not None:
            all_weather.append(df)
        
        time.sleep(0.5)
        
        # Backup every 200
        if completed % 200 == 0 and len(all_weather) > 0:
            temp_df = pd.concat(all_weather, ignore_index=True)
            temp_df.to_csv('data/weather_grid_BACKUP.csv', index=False)

# Final save
weather_df = pd.concat(all_weather, ignore_index=True)
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.to_csv('data/california_weather_grid_dense.csv', index=False)

print(f"\n{'='*70}")
print(f"✅ SUCCESS")
print(f"   Records: {len(weather_df):,}")
print(f"   Grid points: {weather_df.groupby(['grid_lat', 'grid_lon']).ngroups}")
print(f"   Saved to: data/california_weather_grid_dense.csv")
print(f"{'='*70}")