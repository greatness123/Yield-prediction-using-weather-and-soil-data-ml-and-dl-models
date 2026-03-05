# 🌾 XGBoost Crop Yield Prediction Model

A production-ready machine learning model for predicting crop yields based on environmental and soil conditions using XGBoost gradient boosting algorithm.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Outputs](#outputs)
- [Feature Importance](#feature-importance)
- [Example Predictions](#example-predictions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This model uses **XGBoost** (eXtreme Gradient Boosting) to predict crop yields (in tons per hectare) based on seven environmental and soil parameters:

- **N**: Nitrogen content (kg/ha)
- **P**: Phosphorus content (kg/ha)
- **K**: Potassium content (kg/ha)
- **Temperature**: Ambient temperature (°C)
- **Humidity**: Relative humidity (%)
- **pH**: Soil pH level
- **Rainfall**: Annual rainfall (mm)

The model achieves high accuracy through XGBoost's advanced features including built-in regularization, parallel processing, and intelligent tree pruning.

## ✨ Features

- **True XGBoost Implementation**: Uses the actual `xgboost` library with DMatrix optimization
- **Automatic Fallback**: Falls back to sklearn's GradientBoostingRegressor if XGBoost is unavailable
- **Comprehensive Evaluation**: Includes R², RMSE, MAE, and cross-validation metrics
- **Rich Visualizations**: 8 different plots for model analysis
- **Feature Importance Analysis**: Identifies key factors affecting crop yield
- **Production-Ready**: Includes prediction function for real-world deployment
- **Early Stopping**: Prevents overfitting through intelligent training termination

## 📦 Requirements

### Core Dependencies

```
python >= 3.7
pandas >= 1.3.0
numpy >= 1.21.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
xgboost >= 1.5.0  # Recommended
```

### Optional

- **xgboost**: For optimal performance (highly recommended)
- **jupyter**: For interactive exploration

## 🔧 Installation

### 1. Install XGBoost (Recommended)

**Using pip:**
```bash
pip install xgboost
```

**Using conda:**
```bash
conda install -c conda-forge xgboost
```

**Using apt (Ubuntu/Debian):**
```bash
apt-get install python3-xgboost
```

### 2. Install Other Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Clone or Download

```bash
git clone <your-repository-url>
cd crop-yield-prediction
```

### 4. Verify Installation

```python
import xgboost as xgb
print(xgb.__version__)
```

## 📊 Dataset

The model uses the **Crop Recommendation Dataset** with the following characteristics:

- **Source**: `Crop_recommendation.csv`
- **Samples**: 2,200 observations
- **Features**: 7 environmental/soil parameters
- **Target**: Synthetic yield variable (2-10 tons/hectare)

### Synthetic Yield Generation

The yield is created using a realistic formula combining:
- Linear relationships with NPK nutrients
- Non-linear interactions (N×P, Temperature×Humidity)
- Environmental factors (temperature, humidity, rainfall, pH)
- Random variation to simulate real-world uncertainty

## 🏗️ Model Architecture

### XGBoost Configuration

```python
Parameters:
- objective: 'reg:squarederror'
- max_depth: 5
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- num_boost_round: 500
- early_stopping_rounds: 50
```

### Training Process

1. **Data Splitting**: 80% training, 20% testing
2. **Feature Scaling**: StandardScaler normalization
3. **Model Training**: XGBoost with early stopping
4. **Cross-Validation**: 5-fold CV for robustness
5. **Evaluation**: Multiple metrics on test set

## 🚀 Usage

### Basic Usage

```python
# Run the complete pipeline
python xgboost_crop_yield_model.py
```

### Making Predictions

```python
# Use the built-in prediction function
predicted_yield = predict_yield_xgboost(
    N=95,           # Nitrogen (kg/ha)
    P=55,           # Phosphorus (kg/ha)
    K=50,           # Potassium (kg/ha)
    temperature=25, # Temperature (°C)
    humidity=82,    # Humidity (%)
    ph=6.8,         # Soil pH
    rainfall=230    # Rainfall (mm)
)

print(f"Predicted Yield: {predicted_yield:.2f} tons/hectare")
```

### Custom Training

```python
from xgboost import XGBRegressor

# Create custom model
model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300
)

# Train
model.fit(X_train_scaled, y_train)

# Predict
predictions = model.predict(X_test_scaled)
```

## 📈 Performance Metrics

### Expected Results

| Metric | Training Set | Testing Set |
|--------|-------------|-------------|
| **R² Score** | ~0.98 | ~0.97 |
| **RMSE** | ~0.30 | ~0.35 |
| **MAE** | ~0.24 | ~0.28 |
| **CV R² (5-fold)** | - | ~0.97 ±0.01 |

### Interpretation

- **R² Score**: 97% of yield variance explained by the model
- **RMSE**: Average prediction error of ±0.35 tons/hectare
- **MAE**: Typical absolute error of 0.28 tons/hectare
- **CV Score**: Consistent performance across different data splits

## 📁 Outputs

The script generates the following files:

### Predictions and Results

- **`true_xgboost_predictions.csv`**: Test set predictions with errors
- **`true_xgboost_model_info.csv`**: Model performance summary
- **`true_xgboost_feature_importance.csv`**: Feature importance scores

### Visualizations

- **`true_xgboost_analysis.png`**: 8-panel comprehensive analysis
  - Actual vs Predicted scatter plot
  - Residual plot
  - Feature importance chart
  - Error distribution histogram
  - Correlation heatmap
  - Learning curve
  - Performance metrics comparison
  - Accuracy by yield range

- **`xgboost_native_importance.png`**: XGBoost-native importance plot
- **`xgboost_tree_plot.png`**: Visualization of first decision tree

## 🔍 Feature Importance

Typical feature importance ranking:

1. **Temperature** (~25-30%): Most influential factor
2. **Rainfall** (~20-25%): Critical for crop growth
3. **Humidity** (~15-20%): Affects plant transpiration
4. **pH** (~10-15%): Soil nutrient availability
5. **N, P, K** (~5-10% each): Essential nutrients

## 💡 Example Predictions

### Scenario 1: Optimal Conditions
```python
Input: N=95, P=55, K=50, Temp=25°C, Humidity=82%, pH=6.8, Rainfall=230mm
Predicted Yield: 7.45 tons/hectare
```

### Scenario 2: Low Nutrient Stress
```python
Input: N=30, P=25, K=20, Temp=22°C, Humidity=70%, pH=5.8, Rainfall=160mm
Predicted Yield: 4.82 tons/hectare
```

### Scenario 3: Heat/Drought Stress
```python
Input: N=70, P=40, K=35, Temp=38°C, Humidity=45%, pH=6.5, Rainfall=80mm
Predicted Yield: 3.91 tons/hectare
```

## 🔧 Troubleshooting

### XGBoost Import Error

**Problem**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**:
```bash
pip install xgboost
# or
conda install -c conda-forge xgboost
```

### Memory Issues

**Problem**: Out of memory during training

**Solution**: Reduce `num_boost_round` or use smaller dataset
```python
params['num_boost_round'] = 200  # Instead of 500
```

### Poor Performance

**Problem**: Low R² score or high RMSE

**Solutions**:
1. Tune hyperparameters (max_depth, learning_rate)
2. Increase training data
3. Check for data quality issues
4. Add feature engineering

### Visualization Errors

**Problem**: Plots not displaying

**Solution**:
```bash
# For Jupyter notebooks
%matplotlib inline

# For scripts
import matplotlib
matplotlib.use('Agg')
```

## 🛠️ Customization

### Adjusting Model Parameters

```python
# More complex model (slower, potentially more accurate)
params = {
    'max_depth': 7,           # Deeper trees
    'learning_rate': 0.05,    # Smaller steps
    'n_estimators': 1000,     # More trees
    'subsample': 0.9,
    'colsample_bytree': 0.9
}

# Faster model (quicker, potentially less accurate)
params = {
    'max_depth': 3,
    'learning_rate': 0.2,
    'n_estimators': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}
```

### Adding New Features

```python
# Add interaction terms
df['N_P_ratio'] = df['N'] / (df['P'] + 1)
df['temp_humidity_interaction'] = df['temperature'] * df['humidity']

# Update feature list
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
            'N_P_ratio', 'temp_humidity_interaction']
```

## 📝 Model Deployment

### Save the Model

```python
import pickle

# Save model
with open('xgboost_yield_model.pkl', 'wb') as f:
    pickle.dump(sklearn_model, f)

# Save scaler
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### Load and Use

```python
import pickle

# Load model
with open('xgboost_yield_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
new_data = [[95, 55, 50, 25, 82, 6.8, 230]]
scaled_data = scaler.transform(new_data)
prediction = model.predict(scaled_data)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Crop Recommendation Dataset
- XGBoost: Tianqi Chen and Carlos Guestrin
- Scikit-learn: Pedregosa et al.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with 🌱 for sustainable agriculture**
