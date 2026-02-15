import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

print("="*70)
print("VALIDATING CURRENT BAYESIAN NN ON CAMP FIRE")
print("="*70)

# Load model
model = tf.keras.models.load_model('models/bayesian_risk_model.keras')
scaler = joblib.load('models/feature_scaler.pkl')
print("✓ Model and scaler loaded\n")

# Load Camp Fire
fires_df = pd.read_csv("data/fires/fire_archive_SV-C2_716423.csv")
fires_df['datetime'] = pd.to_datetime(
    fires_df['acq_date'] + ' ' + fires_df['acq_time'].astype(str).str.zfill(4),
    format='%Y-%m-%d %H%M'
)
fires_df = fires_df[fires_df['confidence'] == 'h']

with open("data/infrastructure/hospitals.geojson", 'r') as f:
    hospitals_data = json.load(f)

hospitals = []
for feature in hospitals_data['features']:
    hospitals.append({
        'name': feature['properties']['NAME'],
        'city': feature['properties']['CITY'],
        'lat': feature['geometry']['coordinates'][1],
        'lon': feature['geometry']['coordinates'][0]
    })

print(f"✓ Loaded {len(fires_df)} fire detections")
print(f"✓ Loaded {len(hospitals)} hospitals\n")

def haversine(lat1, lon1, lats, lons):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat = lats - lat1
    dlon = lons - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def predict_risk(hospital, fires, model, scaler):
    """Predict risk with uncertainty using MC Dropout"""
    
    distances = haversine(
        hospital['lat'], hospital['lon'],
        fires['latitude'].values,
        fires['longitude'].values
    )
    
    features = pd.DataFrame([{
        'min_distance_km': distances.min(),
        'mean_distance_km': distances.mean(),
        'num_fires_30km': (distances < 30).sum(),
        'max_frp': fires['frp'].max(),
        'wind_speed_kmh': 65.0,  # Camp Fire extreme winds
        'wind_direction': 45.0,
        'temperature_c': 18.0,
        'humidity': 25.0
    }])
    
    features_scaled = scaler.transform(features)
    
    # MC Dropout: 50 predictions
    predictions = []
    for _ in range(50):
        pred = model(features_scaled, training=True).numpy()[0, 0]
        predictions.append(pred)
    
    preds = np.array(predictions)
    
    return {
        'risk_score': preds.mean(),
        'uncertainty': preds.std(),
        'confidence': 100 * (1 - min(preds.std() / 30, 1.0)),
        'distance_km': distances.min(),
        'fires_nearby': (distances < 30).sum()
    }

# Test on Day 1 of Camp Fire
day1_fires = fires_df[fires_df['acq_date'] == '2018-11-08']

print("="*70)
print(f"CAMP FIRE - DAY 1 ({len(day1_fires)} fire hotspots detected)")
print("="*70)
print()

results = []
for hospital in hospitals:
    risk_data = predict_risk(hospital, day1_fires, model, scaler)
    risk_data['name'] = hospital['name']
    risk_data['city'] = hospital['city']
    results.append(risk_data)

# Sort by risk
results.sort(key=lambda x: x['risk_score'], reverse=True)

print(f"{'Rank':<6}{'Hospital':<40}{'City':<18}{'Risk':<12}{'Conf':<10}{'Status'}")
print("-"*95)

for i, r in enumerate(results[:25], 1):
    risk = r['risk_score']
    conf = r['confidence']
    
    if risk > 70:
        status = "🔴 EVACUATE"
    elif risk > 40:
        status = "🟠 PREPARE"
    elif risk > 20:
        status = "🟡 MONITOR"
    else:
        status = "🟢 LOW"
    
    print(f"{i:<6}{r['name'][:38]:<40}{r['city'][:16]:<18}{risk:>5.1f}%    {conf:>5.1f}%   {status}")

print("\n" + "="*70)
print("KEY VALIDATION - PARADISE HOSPITAL")
print("="*70)

# Find Paradise Hospital (actual destroyed hospital)
paradise = next((r for r in results if 'Feather River' in r['name'] or 'Paradise' in r['city']), None)

if paradise:
    print(f"Hospital: {paradise['name']}")
    print(f"City: {paradise['city']}")
    print(f"Risk Score: {paradise['risk_score']:.1f}%")
    print(f"Uncertainty: ±{paradise['uncertainty']:.1f}")
    print(f"Confidence: {paradise['confidence']:.1f}%")
    print(f"Distance from fire: {paradise['distance_km']:.1f} km")
    print(f"Fires within 30km: {paradise['fires_nearby']}")
    print()
    
    if paradise['risk_score'] > 70:
        print("✅ SUCCESS: Model correctly flagged as CRITICAL RISK")
        print("   Historical fact: This hospital WAS evacuated/destroyed")
    else:
        print("❌ FAILURE: Model missed critical threat")
else:
    print("⚠️  Paradise Hospital not found in dataset")

print("="*70)