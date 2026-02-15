from backend.model.bayesian_risk_model import compute_asset_risk

# Test with fake data
asset = {"id": "test1", "lat": 39.76, "lon": -121.62}
fires = [{"lat": 39.77, "lon": -121.63, "brightness": 350}]
weather = {
    "wind_speed_kmh": 65,
    "wind_direction_deg": 45,
    "temperature_c": 18,
    "humidity_pct": 25
}

result = compute_asset_risk(asset, fires, weather)

print("Test Result:")
print(f"  Risk: {result['risk_score']*100:.1f}%")
print(f"  Confidence: {result['confidence']*100:.1f}%")
print(f"  Uncertainty: ±{result['uncertainty']:.1f}")
print(f"  Bucket: {result['risk_bucket']}")