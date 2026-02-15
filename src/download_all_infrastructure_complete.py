import requests
import json
import os
from time import sleep

print("="*70)
print("DOWNLOADING CALIFORNIA INFRASTRUCTURE FROM OPENSTREETMAP")
print("="*70)

os.makedirs('data/infrastructure', exist_ok=True)

# California bounding box
CALIFORNIA_BBOX = {
    'south': 32.5,
    'west': -124.5,
    'north': 42.0,
    'east': -114.0
}

# Infrastructure types with OSM tags
INFRASTRUCTURE_TYPES = [
    {
        'name': 'hospitals',
        'tag': 'amenity',
        'value': 'hospital',
        'description': 'Hospitals'
    },
    {
        'name': 'fire_stations',
        'tag': 'amenity',
        'value': 'fire_station',
        'description': 'Fire Stations'
    },
    {
        'name': 'police_stations',
        'tag': 'amenity',
        'value': 'police',
        'description': 'Police Stations'
    },
    {
        'name': 'schools',
        'tag': 'amenity',
        'value': 'school',
        'description': 'Schools'
    },
    {
        'name': 'universities',
        'tag': 'amenity',
        'value': 'university',
        'description': 'Universities/Colleges'
    },
    {
        'name': 'clinics',
        'tag': 'amenity',
        'value': 'clinic',
        'description': 'Medical Clinics'
    },
    {
        'name': 'nursing_homes',
        'tag': 'amenity',
        'value': 'nursing_home',
        'description': 'Nursing Homes'
    },
    {
        'name': 'pharmacies',
        'tag': 'amenity',
        'value': 'pharmacy',
        'description': 'Pharmacies'
    },
    {
        'name': 'airports',
        'tag': 'aeroway',
        'value': 'aerodrome',
        'description': 'Airports'
    },
    {
        'name': 'cell_towers',
        'tag': 'man_made',
        'value': 'mast',
        'description': 'Cell Towers'
    },
    {
        'name': 'water_towers',
        'tag': 'man_made',
        'value': 'water_tower',
        'description': 'Water Towers'
    },
    {
        'name': 'power_substations',
        'tag': 'power',
        'value': 'substation',
        'description': 'Power Substations'
    },
    {
        'name': 'power_plants',
        'tag': 'power',
        'value': 'plant',
        'description': 'Power Plants'
    }
]

def build_overpass_query(tag, value, bbox):
    """Build Overpass API query"""
    query = f"""
    [out:json][timeout:180];
    (
      node["{tag}"="{value}"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      way["{tag}"="{value}"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
      relation["{tag}"="{value}"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out center;
    """
    return query

def osm_to_geojson(osm_data, infra_type):
    """Convert OSM JSON to GeoJSON"""
    features = []
    
    for element in osm_data.get('elements', []):
        # Get coordinates
        if element['type'] == 'node':
            lon, lat = element['lon'], element['lat']
        elif 'center' in element:
            lon, lat = element['center']['lon'], element['center']['lat']
        else:
            continue
        
        # Extract properties
        tags = element.get('tags', {})
        properties = {
            'osm_id': element.get('id'),
            'osm_type': element.get('type'),
            'infrastructure_type': infra_type,
            'name': tags.get('name', 'Unknown'),
            'address': tags.get('addr:full') or 
                      f"{tags.get('addr:housenumber', '')} {tags.get('addr:street', '')}".strip(),
            'city': tags.get('addr:city', ''),
            'state': tags.get('addr:state', 'CA'),
            'phone': tags.get('phone', ''),
            'website': tags.get('website', '')
        }
        
        # Remove empty fields
        properties = {k: v for k, v in properties.items() if v}
        
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [lon, lat]
            },
            'properties': properties
        }
        
        features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }

def download_infrastructure(infra_config, bbox):
    """Download one infrastructure type"""
    name = infra_config['name']
    description = infra_config['description']
    
    print(f"\n{description}...", end=" ", flush=True)
    
    # Build query
    query = build_overpass_query(
        infra_config['tag'],
        infra_config['value'],
        bbox
    )
    
    # Query Overpass API
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    try:
        response = requests.post(
            overpass_url,
            data={'data': query},
            timeout=180
        )
        response.raise_for_status()
        osm_data = response.json()
        
        # Convert to GeoJSON
        geojson = osm_to_geojson(osm_data, description)
        
        # Save
        output_file = f"data/infrastructure/{name}.geojson"
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        count = len(geojson['features'])
        print(f"✓ {count} locations")
        
        return count
        
    except requests.exceptions.Timeout:
        print("✗ Timeout (query too large)")
        return 0
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return 0

# Download all infrastructure types
print("\nDownloading from OpenStreetMap...")
print("="*70)

results = {}
total = 0

for infra_type in INFRASTRUCTURE_TYPES:
    count = download_infrastructure(infra_type, CALIFORNIA_BBOX)
    results[infra_type['description']] = count
    total += count
    sleep(3)  # Be nice to OSM servers

print("\n" + "="*70)
print("DOWNLOAD SUMMARY")
print("="*70)

for name, count in results.items():
    if count > 0:
        print(f"{name:<30} {count:>6,} locations")

print("-"*70)
print(f"{'TOTAL':<30} {total:>6,} infrastructure assets")

# Create summary
summary = {
    'total_assets': total,
    'datasets': results,
    'coverage': 'California statewide',
    'source': 'OpenStreetMap via Overpass API',
    'bbox': CALIFORNIA_BBOX
}

with open('data/infrastructure/osm_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("✅ All infrastructure downloaded from OpenStreetMap")
print("   Saved to: data/infrastructure/")
print("   Summary: data/infrastructure/osm_summary.json")
print("="*70)