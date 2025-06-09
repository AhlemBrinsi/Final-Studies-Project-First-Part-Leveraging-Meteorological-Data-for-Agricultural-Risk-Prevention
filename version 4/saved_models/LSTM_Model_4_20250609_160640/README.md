# LSTM Model 4

Saved on: 2025-06-09 16:06:40

## Model Details:
- Model Type: LSTM Weather Forecasting Model
- Sequence Length: 10 days
- Forecast Horizon: 4 days
- Number of Classes: 10
- Weather Conditions: Clear Sky ☀️, Cloudy ☁️, Heavy Drizzle 🌧, Heavy Rain 🌧, Light Drizzle 🌦, Light Rain 🌦, Mainly Clear 🌤, Moderate Drizzle 🌧, Moderate Rain 🌧, Partly Cloudy ⛅

## Files:
- `keras_model.keras`: TensorFlow/Keras model architecture and weights
- `label_encoder.pkl`: Label encoder for weather conditions
- `scaler.pkl`: Feature scaler (if used)
- `model_config.json`: Model configuration and metadata
- `full_model.pkl`: Complete model object (without Keras model)
- `README.md`: This file

## Loading Instructions:
```python
# Load using the load_lstm_model function
model = load_lstm_model('saved_models/LSTM_Model_4_20250609_160640')

# Or load components manually:
from tensorflow.keras.models import load_model
import pickle

# Load Keras model
keras_model = load_model('saved_models/LSTM_Model_4_20250609_160640/keras_model.keras')

# Load label encoder
with open('saved_models/LSTM_Model_4_20250609_160640/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

## Model Performance:
Add your performance metrics here after evaluation.
