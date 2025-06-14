# Weather Transformer v1

Saved on: 2025-06-14 16:51:09

## Model Details:
- Model Type: Time Series Transformer Weather Forecasting Model
- Sequence Length: 10 days
- Forecast Horizon: 4 days
- Number of Classes: 8
- Weather Conditions: Clear Sky ☀️, Cloudy ☁️, Heavy Drizzle 🌧, Light Drizzle 🌦, Light Rain 🌦, Moderate Drizzle 🌧, Moderate/Heavy Rain 🌧, Partly Clear 🌤/⛅
- Embedding Dimension: 64
- Number of Attention Heads: 8
- Feed Forward Dimension: 256

## Files:
- `transformer_model.keras`: TensorFlow/Keras Transformer model architecture and weights
- `label_encoder.pkl`: Label encoder for weather conditions
- `scaler.pkl`: Feature scaler (if used)
- `model_config.json`: Model configuration and metadata
- `full_model.pkl`: Complete model object (without Keras model)
- `training_data.pkl`: Training data for potential retraining
- `README.md`: This file

## Loading Instructions:
```python
# Load using the load_transformer_model function
model = load_transformer_model('saved_models/Weather_Transformer_v1_20250614_165054')

# Or load components manually:
from tensorflow.keras.models import load_model
import pickle

# Load Transformer model
transformer_model = load_model('saved_models/Weather_Transformer_v1_20250614_165054/transformer_model.keras')

# Load label encoder
with open('saved_models/Weather_Transformer_v1_20250614_165054/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
```

## Model Performance:
Add your performance metrics here after evaluation.
