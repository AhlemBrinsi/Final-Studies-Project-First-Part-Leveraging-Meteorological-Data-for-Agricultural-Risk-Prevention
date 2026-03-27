# Final Studies Project First Part Leveraging Meteorological Data for Agricultural Risk Prevention

## Section One: Data Analysis and Model Development

### Introduction
This project addresses the critical need for accurate short-term weather forecasting in the agricultural sector by developing advanced deep learning models. This first section focuses on **data preparation, exploratory analysis, feature engineering, model architecture development** while the second section focus on **Model Evaluation and Results**.

---

## Data Preparation and Exploration

### Data Presentation

#### Data Description
The dataset was obtained from the OpenMeteo Historical Weather API and downloaded in CSV format (also available in XLSX and JSON). Historical data is derived from reanalysis datasets, combining observations from weather stations, aircraft, buoys, radar, and satellites to reconstruct past atmospheric conditions.

Daily historical weather data for Monastir was collected from January 2000 to April 2025, containing **9,252 entries and 27 features**.

#### Dataset Columns Summary

| Column Name | Description | Data Type |
|-------------|-------------|-----------|
| date | Date | object |
| weather_code | Coded weather condition (clear, cloudy, rainy, etc.) | int64 |
| temperature_2m_max | Max temperature at 2m height | float64 |
| temperature_2m_min | Min temperature at 2m height | float64 |
| daylight_duration | Total daylight duration (s) | float64 |
| sunshine_duration | Total sunshine duration (s) | float64 |
| rain_sum | Daily accumulated rainfall (mm) | float64 |
| precipitation_hours | Number of hours with precipitation | float64 |
| wind_speed_10m_max | Max wind speed at 10m height | float64 |
| wind_gusts_10m_max | Max wind gusts at 10m height | float64 |
| cloud_cover_max | Max cloud cover (%) | int64 |
| cloud_cover_min | Min cloud cover (%) | int64 |
| dew_point_2m_max | Max dew point at 2m height | float64 |
| dew_point_2m_min | Min dew point at 2m height | float64 |
| relative_humidity_2m_max | Max relative humidity at 2m | float64 |
| relative_humidity_2m_min | Min relative humidity at 2m | float64 |
| pressure_msl_max | Max mean sea-level pressure | float64 |
| pressure_msl_min | Min mean sea-level pressure | float64 |
| surface_pressure_max | Max surface pressure | float64 |
| surface_pressure_min | Min surface pressure | float64 |
| wind_gusts_10m_min | Min wind gusts at 10m height | float64 |
| wind_speed_10m_min | Min wind speed at 10m height | float64 |
| wet_bulb_temperature_2m_min | Min wet-bulb temperature at 2m | float64 |
| wet_bulb_temperature_2m_max | Max wet-bulb temperature at 2m | float64 |
| soil_moisture_0_to_100cm_mean | Average soil moisture (0-100 cm) | float64 |
| soil_temperature_0_to_100cm_mean | Average soil temperature (0-100 cm) | float64 |
| et0_fao_evapotranspiration_sum | Reference evapotranspiration (FAO) | float64 |

---

### Problem Formulation
The project involves **two predictive tasks**:

1. **Classification Task** – Predict the weather condition class for the next three days (multi-class). Classes include Cloudy, Partly Cloudy, Mainly Clear, Clear Sky, Light Drizzle, Moderate Drizzle, Heavy Drizzle, Light Rain, Moderate Rain, Heavy Rain.  
2. **Regression Task** – Forecast four continuous variables: Max Temperature, Min Temperature, Max Humidity, Min Humidity.  
   - **MISO**: Each variable predicted by an independent LSTM.  
   - **MIMO**: Single LSTM predicts all four variables simultaneously.  

---

### Data Preparation

#### Preprocessing Steps
- Convert date to datetime format for temporal feature extraction.  
- Handle missing values and map `weather_code` to weather conditions.  
- Normalize continuous features to [0,1] range.  
- Segment time series into sliding windows of 10 days.  
- Chronologically split dataset into training, validation, and test sets.  
- Reshape inputs to 3D arrays (samples, timesteps, features) for LSTM/Transformer.

#### Handling Imbalanced Classes (Classification)
- Group classes by frequency: Majority (>20%), Balanced (8–20%), Minority (<8%).  
- Oversample minority classes and downsample majority classes.  
- Generated 9,494 synthetic sequences for underrepresented classes.

#### Regression Task Preparation
- Exclude `weather_condition` and `date`.  
- Prepare sliding windows for MISO and MIMO models.  
- Outputs kept unnormalized for linear output layer.

---

### Exploratory Data Analysis (EDA)

#### Feature Distributions
- **Histograms** – Distribution of numerical features.  
- **Scatter Plots** – Relationships between temperature, rainfall, and humidity.  
- **Bar Plots** – Monthly averages of max/min temperatures.  
- **Heatmap** – Correlation analysis for feature selection.

#### Classification Target Distribution
- Cloudy dominates, Heavy Rain is rare.

#### Seasonal Trends
- Temperature shows clear yearly trends; humidity lacks clear seasonality.

#### Final Decisions
- Drop highly correlated/redundant features: dew_point_max, wind_speed_max, surface pressure features, wet-bulb temperature features, soil temperature.

---

### Feature Engineering

#### Initial Features
- **Temporal:** day, month, year, day_of_week, is_weekend  
- **Cyclic:** month_sin, month_cos, dayofyear_sin, dayofyear_cos  
- **Temperature:** temp_range  
- **Wind:** wind_gust_range, avg_wind_speed, wind_variability  
- **Humidity/Dew point:** humidity_range, dew_point_range  
- **Solar radiation:** sunshine_ratio, daylight_to_sunshine_ratio  
- **Precipitation:** rain_today  
- **Pressure:** pressure_range  

#### Feature Importance Analysis
- Techniques: Random Forest, ExtraTrees, F-statistic, Mutual Information  
- Top features selected for derived features.

#### Iterative Feature Engineering
- Tier 1: Core weather features → lagged values, rolling statistics, trends, EWMA  
- Tier 2: Physical variables → selected lagged values, rolling means/std, trends  
- Interaction and seasonal anomaly features added  
- Low-importance non-temporal features removed

---

## Model Architecture Development

### Model Design
- **LSTM** – Captures long-term dependencies in sequential weather data.  
- **Transformer Encoder** – Self-attention highlights relevant historical patterns.

#### LSTM Classification Model
- Task: Multi-step classification  
- Architecture: 3 LSTM layers (dropout + batch norm), 3 dense layers, softmax output  
- Loss: Categorical Focal Loss  
- Optimizer: Adam with gradient clipping

#### Transformer Encoder Classification Model
- Task: Multi-step classification  
- Architecture: Input embedding → positional encoding → 4 encoder layers → dense softmax output  
- Loss: Categorical Focal Loss  
- Optimizer: Adam with gradient clipping

#### MIMO LSTM Regression Model
- Task: Multi-step, multi-target regression  
- Architecture: 3 LSTM layers → 3 dense layers → linear output  
- Loss: Mean Squared Error (MSE)  
- Optimizer: Adam

#### MISO LSTM Regression Model
- Task: Multi-step, single-target regression  
- Architecture: 3 LSTM layers → 3 dense layers → linear output  
- Loss: Mean Squared Error (MSE)  
- Optimizer: Adam

---

### Summary Tables

#### Classification Models

| Component | LSTM | Transformer Encoder |
|-----------|------|-------------------|
| Core Layers | 3 LSTM | 4 Transformer encoders |
| Recurrent Dropout | Yes | N/A |
| Batch Normalization | After LSTM | Layer norm in encoder |
| Dense Layers | 3 | 1 |
| Dropout | Yes | Yes |
| Output Activation | Softmax | Softmax |
| Loss Function | Categorical Focal Loss | Categorical Focal Loss |
| Optimizer | Adam | Adam |
| Forecast Type | Multi-step | Multi-step |

#### Regression Models

| Component | MISO LSTM | MIMO LSTM |
|-----------|-----------|-----------|
| Core Layers | 3 LSTM | 3 LSTM |
| Recurrent Dropout | Yes | Yes |
| Batch Normalization | After LSTM | After LSTM |
| Dense Layers | 3 | 3 |
| Dropout | Yes (0.3) | Yes (0.3) |
| Output Activation | Linear | Linear |
| Loss Function | MSE | MSE |
| Optimizer | Adam | Adam |
| Forecast Type | Single-output | Multi-output |

---

## Section Two: Model Evaluation and Results

### Classification Model Evaluation

#### Metrics
- Accuracy, Precision, Recall, F1-Score

#### Hyperparameter Optimization
- Using **Optuna**, tuning layers, hidden units, dropout, learning rate, batch size.

#### Results Summary
- Optimized LSTM achieved higher precision, recall, and F1-scores, especially for imbalanced classes.  
- Transformer Encoder reached comparable performance.  
- Sliding window adjustments could further improve results for majority/balanced classes.

---

### Regression Model Evaluation

#### Metrics
- Mean Squared Error (MSE), R² Score

#### Comparative Analysis: MISO vs MIMO

| Target Variable | MISO LSTM | MIMO LSTM | Observation |
|-----------------|-----------|-----------|-------------|
| Max Temperature | Lower MSE | Slightly higher MSE | MISO generalizes better |
| Min Temperature | Lower MSE | Slightly higher MSE | MISO preferred |
| Max Humidity    | Lower MSE | Higher MSE | MISO outperforms MIMO |
| Min Humidity    | Lower MSE | Higher MSE | MISO significantly better |

- **Insight:** MISO consistently outperforms MIMO, especially for humidity.  

---

### Key Observations
1. Multi-layer LSTM architectures with MISO strategy are highly effective.  
2. Transformer Encoder shows promise for classification tasks.  
3. Sliding window length and class balancing significantly affect predictive performance.

---

### Conclusion
- **LSTM models** with MISO are highly effective for 3-day weather forecasts.  
- **Hyperparameter tuning and preprocessing adjustments** improved performance for both classification and regression.  
- **Transformer Encoder** offers a promising alternative for classification and potential regression tasks.  
