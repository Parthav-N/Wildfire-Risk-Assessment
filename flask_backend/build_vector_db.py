"""
Build vector database from historical data - fully configurable
"""
import pandas as pd
import numpy as np
import pickle
import yaml
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_datasets(config: dict) -> tuple:
    """Load datasets with error handling"""
    print("📂 Loading datasets...")
    
    try:
        fires_df = pd.read_csv(config['data']['fires'])
        print(f"  ✓ Fires: {len(fires_df):,} records")
    except Exception as e:
        print(f"  ✗ Failed to load fires: {e}")
        fires_df = pd.DataFrame()
    
    try:
        infra_df = pd.read_csv(config['data']['infrastructure'])
        print(f"  ✓ Infrastructure: {len(infra_df):,} records")
    except Exception as e:
        print(f"  ✗ Failed to load infrastructure: {e}")
        infra_df = pd.DataFrame()
    
    try:
        training_df = pd.read_csv(config['data']['training'])
        print(f"  ✓ Training: {len(training_df):,} records")
    except Exception as e:
        print(f"  ✗ Failed to load training: {e}")
        training_df = pd.DataFrame()
    
    return fires_df, infra_df, training_df

def process_fire_events(fires_df: pd.DataFrame, config: dict) -> List[Dict]:
    """Generate fire event summaries"""
    
    if not config['processing']['fire_events']['enabled'] or fires_df.empty:
        return []
    
    print("📊 Processing fire events...")
    
    cols = config['columns']['fires']
    group_by = config['processing']['fire_events']['group_by']
    
    documents = []
    
    fires_by_date = fires_df.groupby(cols['date']).agg({
        cols['lat']: 'mean',
        cols['lon']: 'mean',
        cols['frp']: ['max', 'mean', 'count'],
        cols['brightness']: 'max'
    }).reset_index()
    
    for _, row in fires_by_date.iterrows():
        date = row[cols['date']]
        lat = row[(cols['lat'], 'mean')]
        lon = row[(cols['lon'], 'mean')]
        max_frp = row[(cols['frp'], 'max')]
        avg_frp = row[(cols['frp'], 'mean')]
        count = row[(cols['frp'], 'count')]
        brightness = row[(cols['brightness'], 'max')]
        
        text = f"On {date}, {count} fires were detected in California near coordinates ({lat:.2f}, {lon:.2f}). Maximum fire radiative power was {max_frp:.1f} MW with average intensity {avg_frp:.1f} MW. Peak brightness: {brightness:.0f}K."
        
        documents.append({
            'text': text,
            'type': 'fire_event',
            'date': str(date),
            'metadata': {
                'lat': float(lat),
                'lon': float(lon),
                'fire_count': int(count),
                'max_intensity': float(max_frp),
                'avg_intensity': float(avg_frp)
            }
        })
    
    print(f"  ✓ {len(documents)} fire event summaries")
    return documents

def process_infrastructure(infra_df: pd.DataFrame, config: dict) -> List[Dict]:
    """Generate infrastructure summaries"""
    
    if not config['processing']['infrastructure']['enabled'] or infra_df.empty:
        return []
    
    print("🏥 Processing infrastructure data...")
    
    cols = config['columns']['infrastructure']
    min_count = config['processing']['infrastructure']['min_city_count']
    
    documents = []
    
    # Overall counts by type
    infra_by_type = infra_df.groupby(cols['type']).agg({
        cols['id']: 'count',
        cols['lat']: 'mean',
        cols['lon']: 'mean'
    }).reset_index()
    
    for _, row in infra_by_type.iterrows():
        infra_type = row[cols['type']]
        count = row[cols['id']]
        
        text = f"California has {count} {infra_type} facilities in the infrastructure database, distributed across the state."
        
        documents.append({
            'text': text,
            'type': 'infrastructure_summary',
            'metadata': {
                'asset_type': str(infra_type),
                'count': int(count)
            }
        })
    
    # City-specific counts
    if cols['city'] in infra_df.columns:
        infra_by_city = infra_df.groupby([cols['city'], cols['type']]).size().reset_index(name='count')
        infra_by_city = infra_by_city[infra_by_city['count'] >= min_count]
        
        for _, row in infra_by_city.iterrows():
            city = row[cols['city']]
            infra_type = row[cols['type']]
            count = row['count']
            
            text = f"In {city}, there are {count} {infra_type} that could be at risk during wildfire events."
            
            documents.append({
                'text': text,
                'type': 'infrastructure_city',
                'metadata': {
                    'city': str(city),
                    'asset_type': str(infra_type),
                    'count': int(count)
                }
            })
    
    print(f"  ✓ {len(documents)} infrastructure summaries")
    return documents

def process_risk_scenarios(training_df: pd.DataFrame, config: dict) -> List[Dict]:
    """Generate risk scenario descriptions"""
    
    if not config['processing']['risk_scenarios']['enabled'] or training_df.empty:
        return []
    
    print("⚠️  Processing high-risk scenarios...")
    
    cols = config['columns']['training']
    min_risk = config['processing']['risk_scenarios']['min_risk_score']
    max_samples = config['processing']['risk_scenarios']['max_samples']
    
    documents = []
    
    high_risk = training_df[training_df[cols['risk_score']] > min_risk].copy()
    
    if len(high_risk) > max_samples:
        high_risk = high_risk.sample(max_samples, random_state=42)
    
    for _, row in high_risk.iterrows():
        text = f"High risk scenario documented: {int(row[cols['num_fires']])} fires within 30km radius, wind speed {row[cols['wind_speed']]:.1f} km/h, minimum distance to fire {row[cols['min_distance']]:.1f} km, maximum fire intensity {row[cols['max_frp']]:.1f} MW. This resulted in a risk score of {row[cols['risk_score']]:.0f}%. Temperature: {row[cols['temperature']]:.1f}°C, Humidity: {row[cols['humidity']]:.0f}%, Wind-fire alignment: {row[cols['wind_alignment']]:.2f}."
        
        documents.append({
            'text': text,
            'type': 'risk_scenario',
            'metadata': {
                'num_fires': int(row[cols['num_fires']]),
                'wind_speed': float(row[cols['wind_speed']]),
                'min_distance': float(row[cols['min_distance']]),
                'risk_score': float(row[cols['risk_score']])
            }
        })
    
    print(f"  ✓ {len(documents)} risk scenarios")
    return documents

def add_major_fires(config: dict) -> List[Dict]:
    """Add curated major fire events"""
    
    if not config['processing']['major_fires']['enabled']:
        return []
    
    print("🔥 Adding major fire events...")
    
    major_fires = [
        {
            'name': 'Camp Fire',
            'date': '2018-11-08',
            'location': 'Paradise, Butte County, CA',
            'casualties': 85,
            'structures_destroyed': 18804,
            'acres': 153336,
            'facts': 'Deadliest and most destructive wildfire in California history. Started by PG&E transmission lines. Destroyed the town of Paradise. Adventist Health Feather River Hospital was destroyed. Extreme winds of 87 km/h (54 mph) drove rapid spread.'
        },
        {
            'name': 'Thomas Fire',
            'date': '2017-12-04',
            'location': 'Ventura and Santa Barbara Counties, CA',
            'casualties': 2,
            'structures_destroyed': 1063,
            'acres': 281893,
            'facts': 'Largest fire in California history at the time. Burned for over a month. Strong Santa Ana winds accelerated spread. Multiple hospitals evacuated.'
        },
        {
            'name': 'Tubbs Fire',
            'date': '2017-10-08',
            'location': 'Napa and Sonoma Counties, CA',
            'casualties': 22,
            'structures_destroyed': 5636,
            'acres': 36807,
            'facts': 'Part of the October 2017 Northern California firestorm. Destroyed sections of Santa Rosa. Kaiser Permanente hospital evacuated. Wind gusts exceeded 79 km/h.'
        },
        {
            'name': 'Dixie Fire',
            'date': '2021-07-13',
            'location': 'Butte, Plumas, Lassen, Shasta, and Tehama Counties, CA',
            'casualties': 1,
            'structures_destroyed': 1329,
            'acres': 963309,
            'facts': 'Second-largest wildfire in California history. Burned for nearly 3 months. Destroyed the town of Greenville. Multiple power substations damaged.'
        },
        {
            'name': 'Woolsey Fire',
            'date': '2018-11-08',
            'location': 'Los Angeles and Ventura Counties, CA',
            'casualties': 3,
            'structures_destroyed': 1643,
            'acres': 96949,
            'facts': 'Started same day as Camp Fire. Reached Pacific Ocean. Malibu evacuated. Strong Santa Ana winds of 80+ km/h.'
        }
    ]
    
    documents = []
    
    for fire in major_fires:
        text = f"The {fire['name']} occurred on {fire['date']} in {fire['location']}. {fire['facts']} Casualties: {fire['casualties']} deaths. Structures destroyed: {fire['structures_destroyed']}. Total area burned: {fire['acres']} acres."
        
        documents.append({
            'text': text,
            'type': 'major_fire',
            'metadata': fire
        })
    
    print(f"  ✓ {len(documents)} major fire events")
    return documents

def process_weather_analysis(training_df: pd.DataFrame, config: dict) -> List[Dict]:
    """Analyze weather patterns"""
    
    if not config['processing']['weather_analysis']['enabled'] or training_df.empty:
        return []
    
    print("🌤️  Processing weather patterns...")
    
    cols = config['columns']['training']
    thresholds = config['processing']['weather_analysis']
    
    documents = []
    
    # Extreme wind
    extreme_wind = training_df[training_df[cols['wind_speed']] > thresholds['extreme_wind_threshold']]
    if len(extreme_wind) > 0:
        avg_risk = extreme_wind[cols['risk_score']].mean()
        text = f"Analysis of {len(extreme_wind)} events with extreme wind conditions (>{thresholds['extreme_wind_threshold']} km/h) shows average risk scores of {avg_risk:.1f}%. High winds significantly increase fire spread rates and infrastructure threat."
        
        documents.append({
            'text': text,
            'type': 'weather_analysis',
            'metadata': {'condition': 'extreme_wind', 'avg_risk': float(avg_risk)}
        })
    
    # Extreme temperature
    extreme_temp = training_df[training_df[cols['temperature']] > thresholds['extreme_temp_threshold']]
    if len(extreme_temp) > 0:
        avg_risk = extreme_temp[cols['risk_score']].mean()
        text = f"Analysis of {len(extreme_temp)} events with extreme temperatures (>{thresholds['extreme_temp_threshold']}°C) shows average risk scores of {avg_risk:.1f}%. High temperatures increase fuel dryness and fire intensity."
        
        documents.append({
            'text': text,
            'type': 'weather_analysis',
            'metadata': {'condition': 'extreme_temp', 'avg_risk': float(avg_risk)}
        })
    
    # Low humidity
    low_humidity = training_df[training_df[cols['humidity']] < thresholds['low_humidity_threshold']]
    if len(low_humidity) > 0:
        avg_risk = low_humidity[cols['risk_score']].mean()
        text = f"Analysis of {len(low_humidity)} events with low humidity (<{thresholds['low_humidity_threshold']}%) shows average risk scores of {avg_risk:.1f}%. Low humidity creates ideal conditions for rapid fire spread."
        
        documents.append({
            'text': text,
            'type': 'weather_analysis',
            'metadata': {'condition': 'low_humidity', 'avg_risk': float(avg_risk)}
        })
    
    print(f"  ✓ {len(documents)} weather analyses")
    return documents

def main():
    """Main execution"""
    
    # Load config
    config = load_config()
    
    print("🔧 Loading embedding model...")
    embedder = SentenceTransformer(config['embedding']['model_name'])
    
    # Load datasets
    fires_df, infra_df, training_df = load_datasets(config)
    
    # Generate documents
    documents = []
    documents.extend(process_fire_events(fires_df, config))
    documents.extend(process_infrastructure(infra_df, config))
    documents.extend(process_risk_scenarios(training_df, config))
    documents.extend(add_major_fires(config))
    documents.extend(process_weather_analysis(training_df, config))
    
    if not documents:
        print("❌ No documents generated. Check your data paths and config.")
        return
    
    # Generate embeddings
    print(f"\n🧠 Generating embeddings for {len(documents)} documents...")
    texts = [doc['text'] for doc in documents]
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=config['embedding']['batch_size']
    )
    
    # Create vector database
    vector_db = {
        'documents': documents,
        'embeddings': embeddings,
        'model_name': config['embedding']['model_name'],
        'created_at': pd.Timestamp.now().isoformat(),
        'config': config,
        'stats': {
            'total_docs': len(documents),
            'fire_events': len([d for d in documents if d['type'] == 'fire_event']),
            'infrastructure': len([d for d in documents if d['type'].startswith('infrastructure')]),
            'risk_scenarios': len([d for d in documents if d['type'] == 'risk_scenario']),
            'major_fires': len([d for d in documents if d['type'] == 'major_fire']),
            'weather_analyses': len([d for d in documents if d['type'] == 'weather_analysis'])
        }
    }
    
    # Save
    output_path = Path(config['output']['vector_db'])
    with open(output_path, 'wb') as f:
        pickle.dump(vector_db, f)
    
    print(f"\n✅ Vector database saved to: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"\n📊 Database Statistics:")
    for key, value in vector_db['stats'].items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()