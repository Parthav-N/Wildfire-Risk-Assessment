import json
import os
import pandas as pd

print("="*70)
print("CONSOLIDATING INFRASTRUCTURE DATA")
print("="*70)

infra_dir = 'data/infrastructure'
files = [f for f in os.listdir(infra_dir) if f.endswith('.geojson')]

print(f"\nFound {len(files)} infrastructure files\n")

all_assets = []

for filename in sorted(files):
    filepath = os.path.join(infra_dir, filename)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        features = data.get('features', [])
        
        for feature in features:
            geom = feature.get('geometry', {})
            props = feature.get('properties', {})
            
            if geom.get('type') == 'Point':
                coords = geom['coordinates']
                
                all_assets.append({
                    'type': props.get('infrastructure_type', filename.replace('.geojson', '')),
                    'name': props.get('name', props.get('NAME', 'Unknown')),
                    'city': props.get('city', props.get('CITY', '')),
                    'lon': coords[0],
                    'lat': coords[1]
                })
        
        print(f"{filename:<40} {len(features):>6} assets")
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Save consolidated CSV
df = pd.DataFrame(all_assets)
df.to_csv('data/infrastructure/all_infrastructure.csv', index=False)

print(f"\n{'='*70}")
print(f"✅ Total infrastructure points: {len(all_assets):,}")
print(f"   Saved to: data/infrastructure/all_infrastructure.csv")

# Summary by type
print(f"\nBreakdown by type:")
for infra_type, count in df['type'].value_counts().head(15).items():
    print(f"   {infra_type:<30} {count:>6,}")

print(f"{'='*70}")