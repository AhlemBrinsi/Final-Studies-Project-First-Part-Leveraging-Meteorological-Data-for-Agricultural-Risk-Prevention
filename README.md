# Final Studies Project – First Part: Leveraging Meteorological Data for Agricultural Risk Prevention

## Table of Contents
1. [Introduction](#introduction)
2. [Methodology & Quality Framework](#methodology--quality-framework)
3. [Data Preparation and Exploration](#data-preparation-and-exploration)
4. [Model Architecture Development](#model-architecture-development)
5. [Model Evaluation and Results](#model-evaluation-and-results)
6. [Tools & Technologies](#tools--technologies)
7. [Conclusion](#conclusion)

---

## Introduction
This project addresses the critical need for accurate short-term weather forecasting in the agricultural sector by developing advanced deep learning models. This first part of the project focuses on **data preparation, exploratory analysis, feature engineering, model architecture development, and model evaluation**.

The project was structured using the **DMAIC (Define–Measure–Analyze–Improve–Control)** Six Sigma framework to ensure a rigorous, data-driven, and quality-focused development process.

---

## Methodology & Quality Framework

This project follows the **DMAIC Six Sigma methodology**, ensuring structured problem-solving and continuous quality improvement at every stage.

### Define
- Identified weather-related agricultural risk as the core problem requiring predictive intervention.
- Scoped **two predictive tasks**:
  1. **Classification** – Predict weather condition class for the next 3 days (10 classes: Clear Sky → Heavy Rain).
  2. **Regression** – Forecast 4 continuous variables: Max/Min Temperature and Max/Min Humidity.
- Sourced **9,252 daily entries (Jan 2000 – Apr 2025)** across **27 meteorological features** from the OpenMeteo Historical Weather API.

### Measure
- Established a data quality baseline through preprocessing: datetime conversion, missing value handling, normalization to [0,1], and sliding window segmentation (10-day windows).
- Applied **data stratification** by class frequency (Majority >20%, Balanced 8–20%, Minority <8%) to quantify class imbalance severity.
- Generated **9,494 synthetic sequences** via oversampling to address underrepresented weather classes.
- Used correlation heatmaps and multi-method feature importance analysis (Random Forest, ExtraTrees, F-statistic, Mutual Information) to measure feature relevance.

### Analyze
- Performed comprehensive EDA (histograms, scatter plots, bar plots, heatmaps) to identify seasonal trends and feature relationships.
- Applied a **Pareto-style importance ranking** to identify the top predictors across 20+ engineered features (temporal, cyclic, wind, humidity, solar, precipitation dimensions).
- Dropped highly correlated/redundant features: dew_point_max, wind_speed_max, surface pressure features, wet-bulb temperature features, soil temperature.
- Conducted **root cause analysis** on class imbalance — confirmed that Cloudy class dominates while Heavy Rain is rare, directly impacting model training strategy.
- Compared MISO vs. MIMO regression strategies and confirmed MISO consistently outperforms MIMO across all 4 targets.

### Improve
- Iteratively engineered features in two tiers:
  - **Tier 1:** Core weather features → lagged values, rolling statistics, trends, EWMA
  - **Tier 2:** Physical variables → selected lagged values, rolling means/std, seasonal anomaly features
- Developed and tuned **4 deep learning architectures** using a **PDCA (Plan–Do–Check–Act)** iterative cycle:
  - LSTM Classifier (3 LSTM layers + Categorical Focal Loss)
  - Transformer Encoder Classifier (4 encoder layers + positional encoding)
  - MISO LSTM Regressor (independent model per target, MSE loss)
  - MIMO LSTM Regressor (single multi-output model, MSE loss)
- Applied **Optuna hyperparameter optimization** (layers, hidden units, dropout, learning rate, batch size) to maximize model performance.

### Control
- Delivered a 3-tier full-stack dashboard (React/Vite.js + Node.js/Express + Flask ML API + MongoDB Atlas) to monitor forecast outputs in real time.
- Integrated real-time 3-day predictions, personalized farming recommendations, admin analytics, and activity logging.
- Dashboard serves as a proof-of-concept prototype for the **WeeFarm** platform, ensuring sustained model-driven decision-making for end-users.

---

## Data Preparation and Exploration

### Data Description
The dataset was obtained from the **OpenMeteo Historical Weather API** (CSV format). Historical data is derived from reanalysis datasets, combining observations from weather stations, aircraft, buoys, radar, and satellites.

Daily historical weather data for **Monastir** was collected from **January 2000 to April 2025**, containing **9,252 entries and 27 features**.

### Dataset Columns Summary

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
| soil_moisture_0_to_100cm_mean | Average soil moisture (0–100 cm) | float64 |
| soil_temperature_0_to_100cm_mean | Average soil temperature (0–100 cm) | float64 |
| et0_fao_evapotranspiration_sum | Reference evapotranspiration (FAO) | float64 |

### Preprocessing Steps
- Convert `date` to datetime format for temporal feature extraction.
- Handle missing values and map `weather_code` to weather conditions.
- Normalize continuous features to the [0,1] range.
- Segment time series into sliding windows of 10 days.
- Chronologically split dataset into training, validation, and test sets.
- Reshape inputs to 3D arrays `(samples, timesteps, features)` for LSTM/Transformer models.

### Handling Imbalanced Classes
- Grouped classes by frequency: Majority (>20%), Balanced (8–20%), Minority (<8%).
- Oversampled minority classes and downsampled majority classes.
- Generated **9,494 synthetic sequences** for underrepresented classes.

### Feature Engineering

#### Initial Features
| Category | Features |
|----------|----------|
| Temporal | day, month, year, day_of_week, is_weekend |
| Cyclic | month_sin, month_cos, dayofyear_sin, dayofyear_cos |
| Temperature | temp_range |
| Wind | wind_gust_range, avg_wind_speed, wind_variability |
| Humidity/Dew Point | humidity_range, dew_point_range |
| Solar Radiation | sunshine_ratio, daylight_to_sunshine_ratio |
| Precipitation | rain_today |
| Pressure | pressure_range |

#### Iterative Feature Engineering (PDCA-driven)
- **Tier 1:** Core weather features → lagged values, rolling statistics, trends, EWMA
- **Tier 2:** Physical variables → selected lagged values, rolling means/std, trends
- Interaction and seasonal anomaly features added
- Low-importance non-temporal features removed after **Pareto-style importance ranking**

---

## Model Architecture Development

### Model Design
- **LSTM** – Captures long-term dependencies in sequential weather data.
- **Transformer Encoder** – Self-attention highlights relevant historical patterns.

### Classification Models

| Component | LSTM | Transformer Encoder |
|-----------|------|-------------------|
| Core Layers | 3 LSTM | 4 Transformer encoders |
| Recurrent Dropout | Yes | N/A |
| Batch Normalization | After LSTM | Layer norm in encoder |
| Dense Layers | 3 | 1 |
| Dropout | Yes | Yes |
| Output Activation | Softmax | Softmax |
| Loss Function | Categorical Focal Loss | Categorical Focal Loss |
| Optimizer | Adam (gradient clipping) | Adam (gradient clipping) |
| Forecast Type | Multi-step | Multi-step |

### Regression Models

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

## Model Evaluation and Results

### Classification Results
- Metrics: Accuracy, Precision, Recall, F1-Score
- Hyperparameter optimization via **Optuna** (layers, hidden units, dropout, learning rate, batch size)
- Optimized LSTM achieved higher precision, recall, and F1-scores, especially for imbalanced classes
- Transformer Encoder reached comparable performance

### Regression Results: MISO vs MIMO Comparative Analysis

| Target Variable | MISO LSTM | MIMO LSTM | Observation |
|-----------------|-----------|-----------|-------------|
| Max Temperature | Lower MSE | Slightly higher MSE | MISO generalizes better |
| Min Temperature | Lower MSE | Slightly higher MSE | MISO preferred |
| Max Humidity | Lower MSE | Higher MSE | MISO outperforms MIMO |
| Min Humidity | Lower MSE | Higher MSE | MISO significantly better |

> **Key Insight:** MISO consistently outperforms MIMO across all 4 regression targets, especially for humidity forecasting.

### Key Observations
1. Multi-layer LSTM architectures with MISO strategy are highly effective for 3-day weather forecasting.
2. Transformer Encoder shows strong promise for multi-class weather classification.
3. Sliding window length and class balancing (5S-standardized evaluation pipeline) significantly affect predictive performance.

---

## Tools & Technologies

| Category | Tools |
|----------|-------|
| Core Language | Python |
| Data & ML | NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch |
| Hyperparameter Tuning | Optuna |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebooks |
| Quality Framework | DMAIC (Six Sigma), PDCA (Lean), 5S, Pareto Analysis |

---

## Conclusion

This project demonstrates that a **DMAIC-structured, data-driven approach** to weather forecasting can deliver reliable short-term agricultural risk predictions:

- **LSTM models with MISO strategy** are highly effective for 3-day forecasts across all regression targets.
- **Hyperparameter tuning (Optuna) and iterative PDCA-driven preprocessing** significantly improved both classification and regression performance.
- **Transformer Encoder** offers a strong alternative for weather classification tasks.
- The **WeeFarm dashboard prototype** closes the loop by translating model outputs into actionable farming decisions, sustaining the improvements identified through the DMAIC Control phase.

> Future work: Explore deep learning extensions (attention-based regression), cloud deployment for scalability, and A/B testing of recommendation strategies.
