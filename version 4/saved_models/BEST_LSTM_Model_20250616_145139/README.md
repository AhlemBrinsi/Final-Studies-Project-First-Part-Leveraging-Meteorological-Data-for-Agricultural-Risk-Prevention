# BEST LSTM Model

Saved on: 2025-06-16 14:51:40

## Model Details:
- Model Type: LSTM Weather Forecasting Model
- Sequence Length: 10 days
- Forecast Horizon: 4 days
- Number of Classes: 8
- Weather Conditions: Clear Sky ☀️, Cloudy ☁️, Heavy Drizzle 🌧, Light Drizzle 🌦, Light Rain 🌦, Moderate Drizzle 🌧, Moderate/Heavy Rain 🌧, Partly Clear 🌤/⛅

## Files:
- `keras_model.keras`: TensorFlow/Keras model architecture and weights
- `label_encoder.pkl`: Label encoder for weather conditions
- `scaler.pkl`: Feature scaler (if used)
- `model_config.json`: Model configuration and metadata
- `essential_attributes.pkl`: Essential model attributes
- `README.md`: This file

## Loading Instructions:

### Method 1: Using the load_lstm_model function
```python
# Load using the improved load function
model_components = load_lstm_model('saved_models/BEST_LSTM_Model_20250616_145139')

# Reconstruct your model class
model = YourModelClass(
    sequence_length=model_components['config']['sequence_length'],
    forecast_horizon=model_components['config']['forecast_horizon']
)

# Assign loaded components
model.model = model_components['keras_model']
model.label_encoder = model_components['label_encoder']
model.scaler = model_components['scaler']

# Restore other attributes
for attr, value in model_components['essential_attributes'].items():
    setattr(model, attr, value)
```

### Method 2: Manual loading
```python
from tensorflow.keras.models import load_model
import pickle
import json

# Load components individually
keras_model = load_model('saved_models/BEST_LSTM_Model_20250616_145139/keras_model.keras')

with open('saved_models/BEST_LSTM_Model_20250616_145139/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('saved_models/BEST_LSTM_Model_20250616_145139/model_config.json', 'r') as f:
    config = json.load(f)

# Create your model instance and assign components
model = YourModelClass(
    sequence_length=config['sequence_length'],
    forecast_horizon=config['forecast_horizon']
)
model.model = keras_model
model.label_encoder = label_encoder
```

## Troubleshooting:
If you encounter pickle errors, this usually means your model class contains:
- Lambda functions
- Nested functions
- References to local objects
- Non-serializable objects

The solution is to reconstruct the model class and only load the essential components.

## Model Performance:
Add your performance metrics here after evaluation.
