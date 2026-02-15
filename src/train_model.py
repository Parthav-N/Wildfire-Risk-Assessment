import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

print("="*70)
print("TRAINING FINAL BAYESIAN NEURAL NETWORK")
print("="*70)

# Load training data
print("\n1. Loading training dataset...")
df = pd.read_csv('data/training_dataset_final.csv')
print(f"   ✓ {len(df):,} samples")

# Features (exclude infrastructure_type - it's categorical metadata)
feature_cols = ['min_distance_km', 'mean_distance_km', 'num_fires_30km', 'max_frp',
                'wind_speed_kmh', 'wind_direction', 'temperature_c', 'humidity',
                'wind_fire_alignment']

X = df[feature_cols].values
y = df['risk_score'].values

print(f"\n2. Features ({len(feature_cols)}):")
for col in feature_cols:
    print(f"   - {col}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n3. Data split:")
print(f"   Training: {len(X_train):,}")
print(f"   Testing:  {len(X_test):,}")
print(f"   Risk range: {y_train.min():.1f} - {y_train.max():.1f}")

# Build model
print(f"\n4. Building Bayesian Neural Network...")

inputs = tf.keras.Input(shape=(len(feature_cols),))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae']
)

print(f"   ✓ Model built ({model.count_params():,} parameters)")

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train
print(f"\n5. Training...")
print(f"   Max epochs: 100")
print(f"   Early stopping patience: 15")
print("-"*70)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

print("-"*70)
print(f"   ✓ Training stopped at epoch {len(history.history['loss'])}")
print(f"   ✓ Best validation loss: {min(history.history['val_loss']):.4f}")

# Evaluate with uncertainty
print(f"\n6. Evaluating with uncertainty quantification...")

def predict_with_uncertainty(model, X, n_samples=100):
    predictions = []
    for _ in range(n_samples):
        pred = model(X, training=True).numpy().flatten()
        predictions.append(pred)
    
    predictions = np.array(predictions)
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    
    return mean, std

y_pred_mean, y_pred_std = predict_with_uncertainty(model, X_test, n_samples=100)

mae = np.mean(np.abs(y_test - y_pred_mean))
rmse = np.sqrt(np.mean((y_test - y_pred_mean)**2))

print(f"\n{'='*70}")
print("MODEL PERFORMANCE")
print(f"{'='*70}")
print(f"MAE:  {mae:.2f} risk points")
print(f"RMSE: {rmse:.2f}")
print(f"Avg Uncertainty: {y_pred_std.mean():.2f}")
print(f"Uncertainty range: {y_pred_std.min():.2f} - {y_pred_std.max():.2f}")

# Sample predictions
print(f"\n7. Sample predictions:")
print(f"{'Actual':<10}{'Predicted':<15}{'Uncertainty':<12}{'Confidence':<12}{'Status'}")
print("-"*70)

for i in range(min(20, len(y_test))):
    actual = y_test[i]
    pred = y_pred_mean[i]
    unc = y_pred_std[i]
    conf = 100 * (1 - min(unc / 30, 1.0))
    
    if pred > 70:
        status = "🔴 CRITICAL"
    elif pred > 40:
        status = "🟠 HIGH"
    elif pred > 20:
        status = "🟡 MODERATE"
    else:
        status = "🟢 LOW"
    
    print(f"{actual:<10.1f}{pred:>6.1f} ± {unc:>4.1f}   {unc:>6.1f}      {conf:>5.1f}%     {status}")

# Save model
os.makedirs('models', exist_ok=True)
model.save('models/bayesian_risk_model_final.keras')
joblib.dump(scaler, 'models/feature_scaler_final.pkl')

metadata = {
    'features': feature_cols,
    'mae': float(mae),
    'rmse': float(rmse),
    'avg_uncertainty': float(y_pred_std.mean()),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'epochs_trained': len(history.history['loss']),
    'final_val_loss': float(min(history.history['val_loss']))
}

with open('models/model_metadata_final.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n{'='*70}")
print("✅ MODEL SAVED")
print(f"{'='*70}")
print(f"Model: models/bayesian_risk_model_final.keras")
print(f"Scaler: models/feature_scaler_final.pkl")
print(f"Metadata: models/model_metadata_final.json")
print(f"{'='*70}")