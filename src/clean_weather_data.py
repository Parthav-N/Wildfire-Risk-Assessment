import pandas as pd
import numpy as np

print("="*70)
print("CLEANING CALCLIM WEATHER DATA")
print("="*70)

# Load raw weather data
print("\n1. Loading raw weather data...")
weather_raw = pd.read_csv("data/calmac.csv")
print(f"   ✓ Loaded {len(weather_raw):,} records")

# Show initial state
print(f"\n2. Initial data summary:")
print(f"   Columns: {list(weather_raw.columns)}")
print(f"   Date range: {weather_raw['year'].min()}/{weather_raw['month'].min()}/{weather_raw['day'].min()} to {weather_raw['year'].max()}/{weather_raw['month'].max()}/{weather_raw['day'].max()}")
print(f"   Unique stations: {weather_raw['station_name'].nunique()}")
print(f"   Unique coordinates: {weather_raw[['lat', 'lon']].drop_duplicates().shape[0]}")

# Create datetime
print(f"\n3. Creating datetime column...")
weather_raw['datetime'] = pd.to_datetime(
    weather_raw[['year', 'month', 'day']].assign(hour=weather_raw['hour_epw'])
)
print(f"   ✓ Datetime range: {weather_raw['datetime'].min()} to {weather_raw['datetime'].max()}")

# Check for missing values
print(f"\n4. Checking for missing values...")
missing = weather_raw.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print("   Missing values found:")
    for col, count in missing_cols.items():
        print(f"      {col}: {count} ({count/len(weather_raw)*100:.1f}%)")
else:
    print("   ✓ No missing values")

# Check for invalid values
print(f"\n5. Checking for invalid values...")

# Temperature check
temp_invalid = (weather_raw['dry_bulb_c'] < -50) | (weather_raw['dry_bulb_c'] > 60)
print(f"   Temperature: {temp_invalid.sum()} invalid records (<-50°C or >60°C)")

# Humidity check
humid_invalid = (weather_raw['relative_humidity_pct'] < 0) | (weather_raw['relative_humidity_pct'] > 100)
print(f"   Humidity: {humid_invalid.sum()} invalid records (<0% or >100%)")

# Wind speed check
wind_invalid = (weather_raw['wind_speed_kmh'] < 0) | (weather_raw['wind_speed_kmh'] > 200)
print(f"   Wind speed: {wind_invalid.sum()} invalid records (<0 or >200 km/h)")

# Wind direction check
wind_dir_invalid = (weather_raw['wind_direction_deg'] < 0) | (weather_raw['wind_direction_deg'] > 360)
print(f"   Wind direction: {wind_dir_invalid.sum()} invalid records (<0° or >360°)")

# Remove invalid records
invalid_mask = temp_invalid | humid_invalid | wind_invalid | wind_dir_invalid
weather_clean = weather_raw[~invalid_mask].copy()
print(f"\n6. Removed {invalid_mask.sum()} invalid records")
print(f"   Remaining: {len(weather_clean):,} records")

# Standardize column names
print(f"\n7. Standardizing column names...")
weather_clean = weather_clean.rename(columns={
    'lat': 'grid_lat',
    'lon': 'grid_lon',
    'dry_bulb_c': 'temp_c',
    'relative_humidity_pct': 'humidity',
    'wind_speed_kmh': 'wind_speed_kmh',
    'wind_direction_deg': 'wind_direction',
    'station_name': 'location'
})

# Keep only needed columns
weather_clean = weather_clean[[
    'datetime', 'grid_lat', 'grid_lon', 'temp_c', 'humidity',
    'wind_speed_kmh', 'wind_direction', 'location'
]].copy()

# Remove duplicates
print(f"\n8. Checking for duplicates...")
duplicates = weather_clean.duplicated(subset=['datetime', 'grid_lat', 'grid_lon'])
print(f"   Duplicates: {duplicates.sum()}")
if duplicates.sum() > 0:
    weather_clean = weather_clean[~duplicates]
    print(f"   Removed duplicates, remaining: {len(weather_clean):,}")

# Sort by datetime
weather_clean = weather_clean.sort_values(['location', 'datetime']).reset_index(drop=True)

# Save cleaned data
output_file = 'data/california_weather_clean.csv'
weather_clean.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print("CLEANING COMPLETE")
print(f"{'='*70}")
print(f"Output file: {output_file}")
print(f"Total records: {len(weather_clean):,}")
print(f"Stations: {weather_clean['location'].nunique()}")
print(f"Date range: {weather_clean['datetime'].min()} to {weather_clean['datetime'].max()}")
print(f"\nSummary statistics:")
print(weather_clean[['temp_c', 'humidity', 'wind_speed_kmh', 'wind_direction']].describe())
print(f"{'='*70}")