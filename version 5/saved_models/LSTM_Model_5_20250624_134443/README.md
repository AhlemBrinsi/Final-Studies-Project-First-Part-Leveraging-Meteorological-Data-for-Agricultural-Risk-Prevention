# LSTM Model 5

Saved on: 2025-06-24 13:44:44

## Model Details:
- Model Type: LSTM Weather Forecasting Model
- Sequence Length: 15 days
- Forecast Horizon: 3 days
- Number of Classes: 9
- Weather Conditions: Clear Sky ☀️, Cloudy ☁️, Heavy Drizzle 🌧, Heavy Rain 🌧, Light Drizzle 🌦, Light Rain 🌦, Moderate Drizzle 🌧, Moderate Rain 🌧, Partly Clear 🌤/⛅
- Temp/Humidity Features: temperature_2m_max (°C), temperature_2m_min (°C), relative_humidity_2m_max (%), relative_humidity_2m_min (%)

## Files:
- keras_model.keras: TensorFlow/Keras model architecture and weights
- label_encoder.pkl: Label encoder for weather conditions
- scaler.pkl: Feature scaler (if used)
- temp_humidity_scaler.pkl: Scaler for temperature/humidity features (if used)
- model_config.json: Model configuration and metadata
- full_model.pkl: Complete model object (without Keras model)
- README.md: This file

## Loading Instructions:
python
# Load using the load_lstm_model function
model = load_lstm_model('saved_models/LSTM_Model_5_20250624_134443')

# Or load components manually:
from tensorflow.keras.models import load_model
import pickle

# Load Keras model
keras_model = load_model('saved_models/LSTM_Model_5_20250624_134443/keras_model.keras')

# Load label encoder
with open('saved_models/LSTM_Model_5_20250624_134443/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load scalers if needed
