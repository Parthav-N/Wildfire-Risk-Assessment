"""
Download hospital data from OpenStreetMap for California
"""
import requests
import json
import os

def download_osm_hospitals():
    """
    Query Overpass API for hospitals in Camp Fire region
    """
    
    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Query for hospitals in northern California (around Camp Fire area)
    # Bounding box: [south, west, north, east]
    query = """
    [out:json][timeout:60];
    (
      node["amenity"="hospital"](38.5,-122.5,41.0,-120.0);
      way["amenity"="hospital"](38.5,-122.5,41.0,-120.0);
      relation["amenity"="hospital"](38.5,-122.5,41.0,-120.0);
    );
    out center;
    """
    
    output_file = "data/infrastructure/hospitals.geojson"
    os.makedirs("data/infrastructure", exist_ok=True)
    
    print("Downloading hospitals from OpenStreetMap...")
    print("Region: Northern California")
    print("="*60)
    
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=120)
        response.raise_for_status()
        
        osm_data = response.json()
        
        # Convert OSM format to GeoJSON
        features = []
        for element in osm_data['elements']:
            if element['type'] == 'node':
                lat, lon = element['lat'], element['lon']
            elif 'center' in element:
                lat, lon = element['center']['lat'], element['center']['lon']
            else:
                continue
            
            feature = {
                "type": "Feature",
                "properties": {
                    "NAME": element.get('tags', {}).get('name', 'Unknown Hospital'),
                    "CITY": element.get('tags', {}).get('addr:city', 'N/A'),
                    "STATE": "CA",
                    "TYPE": element.get('tags', {}).get('healthcare', 'hospital'),
                    "OSM_ID": element['id']
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, indent=2)
        
        file_size = os.path.getsize(output_file) / 1024
        
        print(f"SUCCESS!")
        print(f"  File: {output_file}")
        print(f"  Hospitals found: {len(features)}")
        print(f"  Size: {file_size:.2f} KB")
        print("="*60)
        
        # Show sample
        if features:
            print("\nSample hospitals:")
            for i, f in enumerate(features[:5]):
                name = f['properties']['NAME']
                city = f['properties']['CITY']
                coords = f['geometry']['coordinates']
                print(f"  {i+1}. {name} - {city} ({coords[1]:.4f}, {coords[0]:.4f})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    download_osm_hospitals()