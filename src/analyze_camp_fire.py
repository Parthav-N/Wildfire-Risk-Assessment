import pandas as pd
import json
import numpy as np

# Load data
fires_df = pd.read_csv("data/fires/fire_archive_SV-C2_716423.csv")
with open("data/infrastructure/hospitals.geojson", 'r') as f:
    hospitals_data = json.load(f)

# Parse hospitals
hospitals = []
for feature in hospitals_data['features']:
    hospitals.append({
        'name': feature['properties']['NAME'],
        'city': feature['properties']['CITY'],
        'lat': feature['geometry']['coordinates'][1],
        'lon': feature['geometry']['coordinates'][0]
    })

# Parse fire dates
fires_df['datetime'] = pd.to_datetime(
    fires_df['acq_date'] + ' ' + fires_df['acq_time'].astype(str).str.zfill(4), 
    format='%Y-%m-%d %H%M'
)

# Filter high confidence fires only
fires_df = fires_df[fires_df['confidence'] == 'h'].copy()

# Distance calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print("="*70)
print("CAMP FIRE ANALYSIS")
print("="*70)

fire_start = fires_df['datetime'].min()
print(f"\n🔥 Fire started: {fire_start}")
print(f"🔥 Total detections: {len(fires_df)}")

# Analyze each hospital
print(f"\n🏥 TOP 10 CLOSEST HOSPITALS:\n")

hospital_risks = []

for h in hospitals:
    # Calculate distance to every fire detection
    distances = fires_df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], h['lat'], h['lon']),
        axis=1
    )
    
    min_distance = distances.min()
    
    # Find first time fire got within 20km
    nearby_fires = fires_df[distances < 20].sort_values('datetime')
    first_threat = nearby_fires['datetime'].min() if len(nearby_fires) > 0 else None
    
    hospital_risks.append({
        'name': h['name'],
        'city': h['city'],
        'min_distance_km': min_distance,
        'first_threat_time': first_threat
    })

# Sort by distance
hospital_risks.sort(key=lambda x: x['min_distance_km'])

# Print top 10
print(f"{'Rank':<6}{'Hospital':<40}{'City':<20}{'Closest (km)'}")
print("-"*80)
for i, h in enumerate(hospital_risks[:10], 1):
    print(f"{i:<6}{h['name']:<40}{h['city']:<20}{h['min_distance_km']:>10.1f}")

# Show warning times
print(f"\n⏱️  EARLY WARNING ANALYSIS (Top 5):\n")
print(f"{'Hospital':<40}{'Hours of Warning'}")
print("-"*60)

for h in hospital_risks[:5]:
    if h['first_threat_time']:
        hours_warning = (h['first_threat_time'] - fire_start).total_seconds() / 3600
        print(f"{h['name']:<40}{hours_warning:>6.1f} hours")
    else:
        print(f"{h['name']:<40}Never threatened")

print("\n" + "="*70)