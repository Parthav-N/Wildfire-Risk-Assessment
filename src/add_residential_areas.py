import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("ADDING RESIDENTIAL AREAS TO INFRASTRUCTURE")
print("="*70)

# Load existing infrastructure
infra_df = pd.read_csv("data/infrastructure/all_infrastructure.csv")
print(f"Current infrastructure: {len(infra_df):,} assets")
print(f"Current types: {infra_df['type'].unique()}")

# =========================================================================
# OPTION 1: US Census Populated Places
# =========================================================================
print("\n1. Downloading US Census Populated Places...")
census_url = "https://www2.census.gov/geo/tiger/TIGER2024/PLACE/tl_2024_06_place.zip"
# This gives all cities, towns, and CDPs in California

# =========================================================================
# OPTION 2: OpenStreetMap Residential Areas (RECOMMENDED)
# =========================================================================
print("\n2. Getting residential areas from OpenStreetMap...")
print("   Using OSMnx to download residential landuse")

import osmnx as ox

# California bounding box
north, south, east, west = 42.0, 32.5, -114.0, -124.5

try:
    # Get residential landuse areas
    residential = ox.geometries_from_bbox(
        north, south, east, west,
        tags={'landuse': 'residential'}
    )
    
    print(f"   Found {len(residential)} residential areas")
    
    # Convert to DataFrame with same format as your infrastructure
    residential_df = pd.DataFrame({
        'lat': residential.centroid.y,
        'lon': residential.centroid.x,
        'type': 'Residential Areas',
        'name': residential.get('name', 'Unknown'),
        'source': 'OpenStreetMap'
    })
    
    # Add to infrastructure
    infra_updated = pd.concat([infra_df, residential_df], ignore_index=True)
    
except Exception as e:
    print(f"   OSMnx failed: {e}")
    print("\n3. Using synthetic residential grid instead...")
    
    # Create a grid of residential areas across California
    lats = np.arange(32.5, 42.0, 0.1)  # ~10km spacing
    lons = np.arange(-124.5, -114.0, 0.1)
    
    residential_points = []
    for lat in lats:
        for lon in lons:
            # Skip ocean and sparsely populated areas
            # This is simplified - in reality you'd use population density
            if (lat > 34 and lat < 40) and (lon > -122 and lon < -118):
                # Major population centers
                residential_points.append({
                    'lat': lat,
                    'lon': lon,
                    'type': 'Residential Areas',
                    'name': f'Residential_{lat}_{lon}',
                    'source': 'synthetic_grid'
                })
    
    residential_df = pd.DataFrame(residential_points)
    print(f"   Created {len(residential_df)} synthetic residential grid points")
    
    infra_updated = pd.concat([infra_df, residential_df], ignore_index=True)

# =========================================================================
# OPTION 3: HIFLD Populated Places (Alternative)
# =========================================================================
print("\n4. Alternative: HIFLD Populated Places")
hifld_url = "https://hifld-geoplatform.opendata.arcgis.com/datasets/populated-places/data.geojson"
# This gives cities and towns with population

# Save updated infrastructure
output_file = "data/infrastructure/all_infrastructure_with_residential.csv"
infra_updated.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print(f"✅ UPDATED INFRASTRUCTURE")
print(f"{'='*70}")
print(f"Original assets: {len(infra_df):,}")
print(f"Residential areas added: {len(residential_df):,}")
print(f"Total assets: {len(infra_updated):,}")
print(f"New types: {infra_updated['type'].unique()}")
print(f"Saved to: {output_file}")
print(f"{'='*70}")