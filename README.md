# Predictive Maintenance System - RUL Prediction

A machine learning-based predictive maintenance system for forecasting Remaining Useful Life (RUL) of turbofan engines using the NASA turbofan engine dataset.

## Overview

This project implements an end-to-end pipeline for:
- **Data Preprocessing**: Loading and cleaning sensor data from turbofan engines
- **Feature Engineering**: Creating meaningful features from raw sensor signals
- **Model Training**: Training RUL prediction models using XGBoost and ensemble techniques
- **Health Score Calculation**: Converting RUL predictions into intuitive health scores
- **Predictions**: Generating predictions with health status indicators

## Project Structure

```
├── main.py                          # Main execution script and pipeline orchestrator
├── data_preprocessing.py            # Data loading and preprocessing utilities
├── data_preparation.py              # Data preparation and splitting logic
├── feature_engineering.py           # Feature extraction and engineering
├── model_training.py                # RUL model training (XGBoost, Ensemble)
├── prediction_engine.py             # End-to-end prediction pipeline
├── health_score_calculator.py       # RUL to health score conversion
├── evaluation_metrics.py            # Model evaluation and metrics
├── train_FD001.txt                  # Training dataset (NASA FD001)
├── test_FD001.txt                   # Test dataset (NASA FD001)
├── RUL_FD001.txt                    # Ground truth RUL values
├── predictions.csv                  # Generated predictions
└── __pycache__/                     # Python cache directory
```

## Features

### Data Processing
- Loads NASA turbofan engine sensor data (FD001 subset)
- Preprocesses operational cycles and sensor readings
- Handles multiple engines and operating conditions

### Feature Engineering
- Generates statistical features from sensor data
- Creates time-based and degradation-based features
- Supports both simple and advanced feature sets

### Model Training
- XGBoost models for RUL regression
- Ensemble methods for improved predictions
- Hyperparameter tuning capabilities

### Health Score System
- **Green (Healthy)**: Health > 70% - Normal operation
- **Yellow (Warning)**: 30% < Health ≤ 70% - Maintenance soon
- **Red (Critical)**: Health ≤ 30% - Immediate maintenance required

## Usage

### Training a Model
```bash
python main.py --mode train --train_file train_FD001.txt
```

### Making Predictions
```bash
python main.py --mode predict --test_file test_FD001.txt --engine_id 5
```

### Using the Prediction Engine
```python
from prediction_engine import PredictionEngine

# Initialize and load trained model
engine = PredictionEngine(model_path='rul_model.pkl', scaler_path='scaler.pkl')
engine.load_artifacts()

# Make predictions for new engine data
rul_pred = engine.predict_single_engine(engine_id=5)
```

## Requirements

- pandas
- numpy
- scikit-learn
- xgboost

## Dataset

Uses the NASA Prognostics Center of Excellence (PCoE) turbofan engine degradation dataset:
- **FD001**: Single operating condition
- **Features**: 21 sensor readings per operational cycle
- **Target**: Remaining Useful Life (cycles)

## Model Performance

Models are evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Output

Predictions are saved to `predictions.csv` with columns:
- Engine ID
- Predicted RUL
- Health Score (%)
- Health Status (Green/Yellow/Red)
- Confidence metrics

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `DataLoader` | Loads and parses NASA turbofan engine data |
| `FeatureEngineer` | Extracts features from raw sensor data |
| `DataPreparation` | Prepares data for model training |
| `RULModel` | Individual RUL prediction models |
| `EnsembleModel` | Combines multiple models for better predictions |
| `PredictionEngine` | Complete prediction pipeline |
| `HealthScoreCalculator` | Converts RUL to health metrics |
| `ModelEvaluator` | Evaluates and compares model performance |

