# Crop Yield Prediction using 1D CNN

This project implements a 1D Convolutional Neural Network for predicting crop yields based on weather and soil data from Indian agriculture (1997-2020).

## 📋 Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Files Description](#files-description)
- [Results](#results)

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Required Packages

```bash
# Install TensorFlow and other dependencies
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy

# Or install specific versions
pip install tensorflow==2.15.0 scikit-learn==1.3.0 pandas numpy matplotlib seaborn scipy
```

### Alternative: Using Conda
```bash
conda create -n crop_yield python=3.9
conda activate crop_yield
conda install tensorflow scikit-learn pandas numpy matplotlib seaborn scipy
```

## 📊 Dataset

**Source**: Indian Agriculture Dataset (1997-2020)

**Features**:
- **Weather Data**: Annual Rainfall
- **Soil/Agricultural Inputs**: Fertilizer usage, Pesticide usage, Area
- **Categorical Variables**: Crop type, Season, State
- **Temporal**: Crop Year
- **Target**: Crop Yield

**Dataset Statistics**:
- Total Samples: ~19,689
- Features: 10 columns
- Crops: 55 different types
- States: 30 Indian states
- Time Range: 1997-2020

## 🏗️ Model Architecture

### 1D CNN Architecture

```
Input Layer (8 features × 1 channel)
    ↓
Conv1D Block 1 (64 filters, kernel=3)
    → BatchNormalization
    → MaxPooling1D
    → Dropout(0.3)
    ↓
Conv1D Block 2 (128 filters, kernel=3)
    → BatchNormalization
    → MaxPooling1D
    → Dropout(0.3)
    ↓
Conv1D Block 3 (256 filters, kernel=2)
    → BatchNormalization
    → GlobalMaxPooling1D
    → Dropout(0.4)
    ↓
Dense Layer 1 (128 units)
    → BatchNormalization
    → Dropout(0.4)
    ↓
Dense Layer 2 (64 units)
    → BatchNormalization
    → Dropout(0.3)
    ↓
Dense Layer 3 (32 units)
    → Dropout(0.2)
    ↓
Output Layer (1 unit, linear activation)
```

**Key Features**:
- **Loss Function**: Huber Loss (robust to outliers)
- **Optimizer**: Adam (learning rate = 0.001)
- **Regularization**: Batch Normalization + Dropout
- **Callbacks**: Early Stopping, Learning Rate Reduction

## 🚀 Usage

### Training the Model

```bash
# Run the main training script
python crop_yield_1d_cnn.py
```

This will:
1. Load and preprocess the dataset
2. Split data into train/validation/test sets (70%/15%/15%)
3. Train the 1D CNN model
4. Generate performance visualizations
5. Save the trained model and results

### Using the Trained Model

```python
import pickle
import numpy as np
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('crop_yield_1d_cnn_model.h5')

# Load the scaler and label encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Prepare new data
new_data = np.array([[
    2020,           # Crop_Year
    50000,          # Area
    2000,           # Annual_Rainfall
    1000000,        # Fertilizer
    5000,           # Pesticide
    10,             # Crop_encoded (use label_encoders['Crop'])
    1,              # Season_encoded (use label_encoders['Season'])
    5               # State_encoded (use label_encoders['State'])
]])

# Scale and reshape
new_data_scaled = scaler.transform(new_data)
new_data_cnn = new_data_scaled.reshape(new_data_scaled.shape[0], 
                                        new_data_scaled.shape[1], 1)

# Predict
prediction = model.predict(new_data_cnn)
print(f"Predicted Yield: {prediction[0][0]:.2f}")
```

## 📁 Files Description

| File | Description |
|------|-------------|
| `crop_yield_1d_cnn.py` | Main training script with complete pipeline |
| `crop_yield_alternative_models.py` | Alternative ML models (Random Forest, XGBoost, etc.) |
| `prediction_example.py` | Example script for making predictions |
| `crop_yield_1d_cnn_model.h5` | Saved trained model |
| `scaler.pkl` | Fitted StandardScaler for feature normalization |
| `label_encoders.pkl` | Label encoders for categorical variables |
| `model_performance.png` | Comprehensive performance visualizations |
| `feature_importance.png` | Feature importance analysis |
| `model_results_summary.txt` | Detailed results and metrics |

## 📈 Results

### Expected Performance Metrics

Based on the dataset characteristics:

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **RMSE** | ~10-15 | ~12-18 | ~12-18 |
| **MAE** | ~5-8 | ~7-10 | ~7-10 |
| **R²** | ~0.85-0.92 | ~0.80-0.88 | ~0.80-0.88 |

*Note: Actual values may vary depending on data preprocessing and hyperparameters*

### Visualizations Generated

1. **Training History**: Loss and MAE curves over epochs
2. **Predictions vs Actual**: Scatter plot with perfect prediction line
3. **Residuals Distribution**: Histogram of prediction errors
4. **Residual Plot**: Residuals vs predicted values
5. **Performance Comparison**: Bar chart of metrics across datasets
6. **Absolute Error Distribution**: Distribution of prediction errors
7. **Sample Predictions**: Comparison of actual vs predicted for random samples
8. **Q-Q Plot**: Normality check for residuals
9. **Feature Importance**: Permutation-based importance scores

### Key Insights

**Most Important Features** (typically):
1. Fertilizer usage
2. Area cultivated
3. Annual Rainfall
4. Crop type
5. State

**Model Strengths**:
- Handles non-linear relationships between features
- Robust to outliers (Huber loss)
- Good generalization (dropout + batch normalization)
- Captures local patterns in feature space

## 🔍 Model Interpretation

### Why 1D CNN for Yield Prediction?

1. **Feature Interactions**: CNNs can capture local dependencies between adjacent features
2. **Pattern Recognition**: Identifies patterns in the sequence of features
3. **Parameter Efficiency**: Fewer parameters than fully connected networks
4. **Regularization**: Built-in regularization through pooling and dropout

### Comparison with Traditional Methods

| Method | Advantages | Disadvantages |
|--------|-----------|---------------|
| **1D CNN** | Captures feature interactions, robust, scalable | Requires more data, longer training |
| **Random Forest** | Interpretable, handles non-linearity | May overfit, less scalable |
| **Linear Regression** | Fast, interpretable | Cannot capture non-linear patterns |
| **XGBoost** | High accuracy, handles missing data | Less interpretable, hyperparameter tuning |

## 🛠️ Customization

### Hyperparameter Tuning

Modify these parameters in `crop_yield_1d_cnn.py`:

```python
# Model architecture
filters = [64, 128, 256]  # Number of filters in Conv1D layers
kernel_sizes = [3, 3, 2]   # Kernel sizes
dropout_rates = [0.3, 0.3, 0.4]  # Dropout rates

# Training
learning_rate = 0.001
batch_size = 64
epochs = 150

# Callbacks
early_stopping_patience = 20
reduce_lr_patience = 10
```

### Adding More Features

To include additional features (temperature, humidity, pH):

```python
# Add to feature_columns list
feature_columns = [
    'Crop_Year',
    'Area',
    'Annual_Rainfall',
    'Temperature',  # New feature
    'Humidity',      # New feature
    'Soil_pH',       # New feature
    'Fertilizer',
    'Pesticide',
    'Crop_encoded',
    'Season_encoded',
    'State_encoded'
]
```

## 📝 Citation

If you use this code or dataset, please cite:

```
@misc{crop_yield_cnn_2024,
  title={1D CNN for Crop Yield Prediction using Weather and Soil Data},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/crop-yield-prediction}
}
```

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: Indian Agriculture Dataset (1997-2020)
- TensorFlow/Keras for deep learning framework
- Scikit-learn for preprocessing and metrics
