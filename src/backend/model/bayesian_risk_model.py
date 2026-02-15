"""
Bayesian NN with MC Dropout - Lazy loading with proper architecture
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"

# Global cache
_model = None
_scaler = None

def build_model():
    """
    Rebuild exact architecture from training
    MUST match what was trained!
    """
    import tensorflow as tf
    
    inputs = tf.keras.Input(shape=(9,))  # 9 features
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)  # Single output
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def get_model():
    """Lazy load model on first use"""
    global _model, _scaler
    
    if _model is not None:
        return _model, _scaler
    
    try:
        print("Loading Bayesian NN...")
        
        # Build architecture
        _model = build_model()
        
        # Load weights
        _model.load_weights(str(MODEL_DIR / "bayesian_risk_model_final.weights.h5"))
        
        # Load scaler
        _scaler = joblib.load(str(MODEL_DIR / "feature_scaler_final.pkl"))
        
        print(f"✓ Bayesian NN loaded from {MODEL_DIR}")
        
    except Exception as e:
        print(f"✗ Model load failed: {e}")
        print("  Will use fallback heuristic")
        _model = "FAILED"
        _scaler = None
    
    return _model, _scaler

def haversine_vectorized(lat1, lon1, lats2, lons2):
    R = 6371
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lats2, lons2 = np.radians(lats2), np.radians(lons2)
    dlat = lats2 - lat1
    dlon = lons2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def calculate_wind_alignment(fire_lat, fire_lon, asset_lat, asset_lon, wind_dir):
    lat1, lon1 = np.radians(fire_lat), np.radians(fire_lon)
    lat2, lon2 = np.radians(asset_lat), np.radians(asset_lon)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    bearing = (bearing + 360) % 360
    wind_toward = (wind_dir + 180) % 360
    angle_diff = abs(wind_toward - bearing)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    return np.cos(np.radians(angle_diff))

def fallback_heuristic(min_dist, num_fires, wind_speed, wind_align):
    """Physics-based fallback if ML fails"""
    if min_dist < 1:
        base = 95
    elif min_dist < 5:
        base = 75
    elif min_dist < 15:
        base = 45
    elif min_dist < 30:
        base = 25
    else:
        base = 10
    
    wind_factor = 1.0 + max(0, wind_speed - 20) / 40
    if wind_align > 0.5:
        wind_factor *= 1.2
    
    return min(base * wind_factor * (1 + min(num_fires/30, 0.3)), 100)

def compute_asset_risk(asset: Dict, fires: List[Dict], weather: Dict) -> Dict:
    """Compute risk with Bayesian NN or fallback"""
    
    if not fires:
        return {
            "asset_id": asset["id"],
            "risk_score": 0.0,
            "confidence": 0.0,
            "risk_bucket": "low",
            "features": {}
        }
    
    # Calculate features
    fire_lats = np.array([f["lat"] for f in fires])
    fire_lons = np.array([f["lon"] for f in fires])
    distances = haversine_vectorized(asset["lat"], asset["lon"], fire_lats, fire_lons)
    
    min_dist = distances.min()
    mean_dist = distances.mean()
    num_nearby = (distances < 30).sum()
    max_frp = max([f.get("frp", f.get("brightness", 50)) for f in fires])
    
    closest_fire = fires[distances.argmin()]
    wind_align = calculate_wind_alignment(
        closest_fire["lat"], closest_fire["lon"],
        asset["lat"], asset["lon"],
        weather.get("wind_direction_deg", 180)
    )
    
    # Try ML model
    model, scaler = get_model()
    
    if model != "FAILED" and model is not None:
        try:
            import tensorflow as tf
            
            features = pd.DataFrame([{
                'min_distance_km': min_dist,
                'mean_distance_km': mean_dist,
                'num_fires_30km': int(num_nearby),
                'max_frp': max_frp,
                'wind_speed_kmh': weather.get('wind_speed_kmh', 15),
                'wind_direction': weather.get('wind_direction_deg', 180),
                'temperature_c': weather.get('temperature_c', 20),
                'humidity': weather.get('humidity_pct', 50),
                'wind_fire_alignment': wind_align
            }])
            
            feat_scaled = scaler.transform(features)
            
            preds = [model(feat_scaled, training=True).numpy()[0, 0] for _ in range(50)]
            preds = np.array(preds)
            
            risk_raw = np.clip(preds.mean(), 0, 100)
            unc = preds.std()
            
            return {
                "asset_id": asset["id"],
                "risk_score": float(risk_raw / 100),
                "confidence": float(1.0 - min(unc / 30, 1.0)),
                "uncertainty": float(unc),
                "risk_bucket": _bucket(risk_raw / 100),
                "model_used": "bayesian_nn",
                "features": {
                    "min_dist_to_fire_km": round(float(min_dist), 2),
                    "wind_alignment": round(float(wind_align), 2),
                    "wind_speed_kmh": weather.get('wind_speed_kmh', 15),
                    "num_fires_nearby": int(num_nearby)
                }
            }
        except Exception as e:
            print(f"Prediction failed: {e}")
    
    # Fallback
    risk_raw = fallback_heuristic(min_dist, num_nearby, weather.get('wind_speed_kmh', 15), wind_align)
    
    return {
        "asset_id": asset["id"],
        "risk_score": float(risk_raw / 100),
        "confidence": 0.6,
        "risk_bucket": _bucket(risk_raw / 100),
        "model_used": "heuristic",
        "features": {
            "min_dist_to_fire_km": round(float(min_dist), 2),
            "num_fires_nearby": int(num_nearby)
        }
    }

def _bucket(score):
    return "high" if score >= 0.7 else "medium" if score >= 0.4 else "low"