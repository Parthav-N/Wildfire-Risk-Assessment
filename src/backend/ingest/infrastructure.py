"""
Load infrastructure from CSV files
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Path to your infrastructure CSV
INFRA_FILE = Path("D:/wildfire-risk-system/data/infrastructure/all_infrastructure_with_residential.csv")

# Load once at startup
_infra_df = pd.read_csv(INFRA_FILE)
print(f"✓ Loaded {len(_infra_df):,} infrastructure assets")

def fetch_assets_from_csv(bbox: str, asset_types: List[str] = None) -> List[Dict]:
    """
    Fetch infrastructure from CSV within bounding box
    """
    west, south, east, north = [float(v) for v in bbox.split(",")]
    
    # Filter by bbox
    filtered = _infra_df[
        (_infra_df['lat'] >= south) &
        (_infra_df['lat'] <= north) &
        (_infra_df['lon'] >= west) &
        (_infra_df['lon'] <= east)
    ]
    
    # Filter by type if specified
    if asset_types:
        filtered = filtered[filtered['type'].isin(asset_types)]
    
    # Convert to list
    assets = []
    for idx, row in filtered.iterrows():
        assets.append({
            "id": f"{row['type']}_{idx}",
            "lat": row['lat'],
            "lon": row['lon'],
            "asset_type": row['type'],
            "name": row.get('name', row['type']),
            "city": row.get('city', '')
        })
    
    return assets