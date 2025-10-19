# MLOps Lab 3 - Energy Consumption Feature Engineering with TensorFlow Transform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Overview

This project demonstrates scalable feature engineering for **time series forecasting** using **TensorFlow Transform (TFT)** and **Apache Beam**. We build a preprocessing pipeline for energy consumption data that transforms raw measurements into ML-ready features suitable for sequence models (LSTM, GRU, Transformers).

**Key Focus**: Production-grade feature engineering with TF Transform for time series data, ensuring consistent preprocessing between training and serving.

---

## Learning Objectives

- Implement TensorFlow Transform pipelines with Apache Beam
- Engineer temporal features with cyclical encoding (sin/cos)
- Handle time series data preprocessing at scale
- Create windowed datasets for sequence forecasting
- Understand train/test splitting for temporal data
- Analyze feature correlations and patterns

---

## Dataset

### Synthetic Energy Consumption Time Series

**Specifications:**
- **Total Records**: 10,000 hourly observations
- **Time Period**: January 1, 2020 → February 20, 2021 (~13.5 months)
- **Sampling Rate**: Hourly (24 records/day, 168 records/week)
- **Train/Test Split**: 80/20 temporal split (8,000 / 2,000 records)

### Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `Power_kW` | Target | 0-120 kW | Energy consumption (our prediction target) |
| `Temperature_C` | Continuous | 5-30°C | Ambient temperature |
| `Humidity_pct` | Continuous | 20-95% | Relative humidity |
| `Wind_Speed_ms` | Continuous | 0-12 m/s | Wind speed |
| `Solar_Radiation_Wm2` | Continuous | 0-1000 W/m² | Solar radiation intensity |
| `Day_of_Week` | Categorical | 0-6 | Day of week (0=Monday, 6=Sunday) |
| `Timestamp` | Temporal | - | Date and time of measurement |

### Data Patterns

The synthetic dataset includes realistic patterns:
- 📈 **Daily cycles**: Higher consumption during daytime hours
- 📅 **Weekly patterns**: ~20% reduction on weekends
- 🌡️ **Seasonal variations**: Temperature-driven consumption changes
- ☀️ **Solar correlation**: Inverse relationship with artificial lighting needs
- ⚡ **Anomalies**: Random power outages (~0.1% of data)

---

## Feature Engineering

### Transformation Pipeline

**Input**: 7 raw features → **Output**: 15 engineered features

### 1. Temporal Features (8 features)

**Cyclical Encoding** for periodic patterns:

```python
# Hour of day (0-23)
Hour_sin = sin(2π * hour / 24)
Hour_cos = cos(2π * hour / 24)

# Day of week (0-6)
DayOfWeek_sin = sin(2π * day / 7)
DayOfWeek_cos = cos(2π * day / 7)

# Month of year
Month_sin = sin(2π * day_of_year / 365.25)
Month_cos = cos(2π * day_of_year / 365.25)

# Weekend indicator
Is_Weekend = (day_of_week >= 5) ? 1 : 0
```

**Why sin/cos?** Preserves periodicity (hour 23 and 0 are mathematically close).

### 2. Environmental Features (5 features)

```python
Temperature_C              # Scaled [0, 1]
Humidity_pct              # Scaled [0, 1]
Wind_Speed_ms             # Scaled [0, 1]
Solar_Radiation_Wm2       # Scaled [0, 1]
Temp_Deviation            # |temp - 20°C| scaled
```

### 3. Behavioral Indicators (2 features)

```python
Needs_Heating = (temp < 18°C) ? 1 : 0
Needs_Cooling = (temp > 25°C) ? 1 : 0
```

### 4. Normalization

All continuous features scaled to [0, 1] using `tft.scale_to_0_1()` for stable training.

---

## Windowed Dataset Configuration

For sequence-to-point forecasting:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **History Window** | 168 hours (7 days) | Input sequence length |
| **Prediction Horizon** | 24 hours (1 day) | How far ahead to predict |
| **Stride** | 1 hour | Use every hourly record |
| **Window Shift** | 24 hours | Shift between training examples |
| **Batch Size** | 32 | Examples per training batch |

**Tensor Shapes:**
- **Input**: `(batch_size, 168, 15)` = (32, 168 timesteps, 15 features)
- **Output**: `(batch_size,)` = (32 scalar predictions)

**Training Examples:**
- Train set: ~333 windows
- Test set: ~83 windows

---

## Project Structure

```
MLOps-Lab3/
│
├── Lab3-TFT-YashKhare.ipynb
│   └── Main notebook with complete pipeline
│
├── data/
│   └── energy_consumption.csv
│       └── Generated synthetic dataset (10,000 records)
│
├── transform_dir_energy/
│   ├── transform_fn/
│   │   ├── saved_model.pb          # Transform graph
│   │   └── assets/                  # Transform assets
│   │
│   ├── transform_train-*            # Training TFRecords
│   ├── transform_test-*             # Test TFRecords
│   └── transformed_metadata/        # Output schema
│
├── README.md
└── .gitignore
```

---

## Installation & Usage

### Option 1: Google Colab (Recommended)

1. **Open notebook in Colab**:
   ```
   File → Open notebook → GitHub
   Enter: https://github.com/YashKhare20/MLOps-Lab3
   ```

2. **Install Python 3.10** (required for TFX compatibility):
   ```python
   !wget https://github.com/korakot/kora/releases/download/v0.10/py310.sh
   !bash ./py310.sh -b -f -p /usr/local
   ```
   
   **⚠️ IMPORTANT**: Restart runtime after Python 3.10 installation

3. **Install dependencies**:
  ```python
  !uv pip install tensorflow \
                  tensorflow-transform \
                  tfx-bsl \
                  apache-beam \
                  pyarrow \
                  pandas \
                  numpy \
                  matplotlib \
                  seaborn \
                  cryptography \
                  pyOpenSSL
  ```

4. **Run all cells sequentially**

### Option 2: Local Environment

**Prerequisites**: Python 3.10

```bash
# Clone repository
git clone https://github.com/YashKhare20/MLOps-Lab3.git
cd MLOps-Lab3

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch Jupyter
jupyter notebook Lab3-TFT-YashKhare.ipynb
```

**Python Version**: 3.10 (TFX/TFT not compatible with 3.11+)

---

## Results & Outputs

### 1. Exploratory Data Analysis

- **Time series visualizations**: 14-day patterns for all features
- **Hourly consumption profile**: Peak at noon (~110 kW), low at 3 AM (~55 kW)
- **Weekly patterns**: 20% reduction on weekends
- **Correlation heatmap**: Temperature (0.28), Humidity (-0.24) correlations identified

### 2. Transform Pipeline

Successfully executed Apache Beam pipeline:
- Processed 10,000 records
- Generated 15 engineered features from 7 raw features
- Created train/test TFRecords
- Saved reusable transform graph

### 3. Windowed Datasets

Created batched datasets ready for model training:
- Train: 333 windows across ~11 months
- Test: 83 windows across ~2.7 months
- Each window: 168-hour history → 24-hour prediction

### 4. Key Insights

- **Temperature deviation** is stronger predictor than raw temperature
- **Cyclical encoding** captures overnight patterns correctly
- **Weekend indicator** significantly improves predictions
- **Solar radiation** inversely correlated with power (-0.13)

---

## Machine Learning Pipeline

### Next Steps (Not Implemented in Lab)

This lab focuses on feature engineering. To complete the ML pipeline:

1. **Build Model**:
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.LSTM(128, return_sequences=True),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.LSTM(64),
       tf.keras.layers.Dense(32, activation='relu'),
       tf.keras.layers.Dense(1)
   ])
   ```

2. **Train**: Use windowed datasets with early stopping

3. **Evaluate**: RMSE, MAE, MAPE on test set

4. **Deploy**: Export model + transform graph for production

---

## Key Concepts Demonstrated

### 1. TensorFlow Transform (TFT)

**Benefits**:
- Consistent preprocessing (training & serving)
- Scalable with Apache Beam
- Reusable transform graphs
- Integrates with TFX pipelines

### 2. Temporal Data Handling

- Proper train/test splitting (no data leakage)
- Cyclical encoding for periodic features
- Windowing for sequence models

### 3. Feature Engineering Best Practices

- Domain knowledge (HVAC indicators)
- Interaction features (temperature deviation)
- Appropriate scaling methods

---

## Contributing

This is a lab assignment, but suggestions are welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## Author

**Yash Khare**
- GitHub: [@YashKhare20](https://github.com/YashKhare20)
- Course: MLOps (IE7374)
- Lab: Feature Engineering with TensorFlow Transform
