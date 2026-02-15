"""
Visualize Camp Fire and hospitals on a map
"""
import pandas as pd
import json
import folium
from datetime import datetime

# File paths
FIRE_DATA = "data/fires/fire_archive_SV-C2_716423.csv"
HOSPITALS_DATA = "data/infrastructure/hospitals.geojson"

def load_fire_data():
    """Load Camp Fire CSV data"""
    print("Loading Camp Fire data...")
    df = pd.read_csv(FIRE_DATA)
    
    # Filter for high confidence and significant fires
    df_filtered = df[
        (df['confidence'] == 'h') |  # High confidence
        (df['frp'] > 10)  # OR Fire Radiative Power > 10 MW
    ]
    
    print(f"  Total detections: {len(df)}")
    print(f"  High confidence/intensity: {len(df_filtered)}")
    
    return df_filtered

def load_hospitals():
    """Load hospital GeoJSON data"""
    print("Loading hospitals...")
    with open(HOSPITALS_DATA, 'r') as f:
        data = json.load(f)
    
    hospitals = []
    for feature in data['features']:
        hospitals.append({
            'name': feature['properties']['NAME'],
            'city': feature['properties']['CITY'],
            'lat': feature['geometry']['coordinates'][1],
            'lon': feature['geometry']['coordinates'][0]
        })
    
    print(f"  Total hospitals: {len(hospitals)}")
    return hospitals

def create_map(fires_df, hospitals):
    """Create interactive map with fires and hospitals"""
    print("\nCreating map...")
    
    # Center map on Camp Fire area (Paradise, CA)
    center_lat = 39.76
    center_lon = -121.62
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Add fire points (color by intensity)
    print("  Adding fire detections...")
    for idx, row in fires_df.iterrows():
        # Color based on FRP (fire intensity)
        if row['frp'] > 50:
            color = 'darkred'
            radius = 8
        elif row['frp'] > 20:
            color = 'red'
            radius = 6
        else:
            color = 'orange'
            radius = 4
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=f"Fire Detection<br>Date: {row['acq_date']}<br>Time: {row['acq_time']}<br>FRP: {row['frp']:.1f} MW<br>Confidence: {row['confidence']}"
        ).add_to(m)
    
    # Add hospitals
    print("  Adding hospitals...")
    for hospital in hospitals:
        folium.Marker(
            location=[hospital['lat'], hospital['lon']],
            popup=f"<b>{hospital['name']}</b><br>{hospital['city']}, CA",
            icon=folium.Icon(color='blue', icon='plus-sign', prefix='glyphicon'),
            tooltip=hospital['name']
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Legend</b></p>
    <p><span style="color: darkred;">●</span> Very High Intensity Fire (>50 MW)</p>
    <p><span style="color: red;">●</span> High Intensity Fire (>20 MW)</p>
    <p><span style="color: orange;">●</span> Moderate Fire</p>
    <p><span style="color: blue;">📍</span> Hospital</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_file = "camp_fire_map.html"
    m.save(output_file)
    print(f"\n✓ Map saved to: {output_file}")
    print("  Open this file in your browser to view the interactive map")
    
    return m

def main():
    print("="*60)
    print("Camp Fire + Hospitals Visualization")
    print("="*60)
    
    # Load data
    fires_df = load_fire_data()
    hospitals = load_hospitals()
    
    # Create map
    create_map(fires_df, hospitals)
    
    print("="*60)

if __name__ == "__main__":
    main()