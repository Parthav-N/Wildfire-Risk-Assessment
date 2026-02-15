import pandas as pd
import requests
import time

print("Fetching weather for 15 key California fire zones...")

weather_points = [
    {"name": "Paradise", "lat": 39.76, "lon": -121.62},
    {"name": "Redding", "lat": 40.59, "lon": -122.39},
    {"name": "Santa Rosa", "lat": 38.44, "lon": -122.71},
    {"name": "Napa", "lat": 38.30, "lon": -122.29},
    {"name": "Chico", "lat": 39.73, "lon": -121.84},
    {"name": "Sacramento", "lat": 38.58, "lon": -121.49},
    {"name": "Fresno", "lat": 36.75, "lon": -119.77},
    {"name": "Bakersfield", "lat": 35.37, "lon": -119.02},
    {"name": "LA_North", "lat": 34.50, "lon": -118.50},
    {"name": "San Diego", "lat": 32.90, "lon": -117.10},
    {"name": "Lake Tahoe", "lat": 39.10, "lon": -120.03},
    {"name": "Yosemite", "lat": 37.75, "lon": -119.59},
    {"name": "Big Sur", "lat": 36.27, "lon": -121.81},
    {"name": "Mendocino", "lat": 39.31, "lon": -123.80},
    {"name": "Shasta", "lat": 40.80, "lon": -122.49}
]

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
        
        # Sample every 12th row (keep every 12 hours instead of resampling)
        df = df.iloc[::12].reset_index(drop=True)
        
        return df
        
    except Exception as e:
        return None

all_weather = []
success = 0

for point in weather_points:
    print(f"{point['name']:15s}:", end=" ")
    
    for year in range(2017, 2025):
        df = fetch_weather(point['lat'], point['lon'], year)
        
        if df is not None and len(df) > 0:
            df['location'] = point['name']
            all_weather.append(df)
            print(f"✓", end="")
            success += 1
        else:
            print(f"✗", end="")
        
        time.sleep(0.5)
    
    print()

print(f"\n{'='*70}")

if len(all_weather) > 0:
    weather_df = pd.concat(all_weather, ignore_index=True)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df.to_csv('data/california_weather_15points.csv', index=False)
    
    print(f"✅ SUCCESS")
    print(f"   Records: {len(weather_df):,}")
    print(f"   Locations: {weather_df['location'].nunique()}")
    print(f"   Date range: {weather_df['datetime'].min()} to {weather_df['datetime'].max()}")
    print(f"   File: data/california_weather_15points.csv")
else:
    print(f"❌ No data retrieved")

print(f"{'='*70}")