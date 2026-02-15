import tensorflow as tf
import joblib

# Load the model
model = tf.keras.models.load_model('models/bayesian_risk_model_final.keras')

# Save in H5 format (more compatible)
model.save('models/bayesian_risk_model_final.weights.h5')

print("✅ Model saved as H5 format")