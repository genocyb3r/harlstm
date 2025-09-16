# Hybrid AR-LSTM Model for Gas Price Forecasting

## Overview
This project implements a **Hybrid Auto-Regressive + LSTM (HAR-LSTM)** model for gasoline price prediction.  
The model combines the strengths of **deep learning (LSTM)** for capturing temporal dependencies and **auto-regressive dense features** (trend, volatility, mean) for statistical interpretability.  

The system is designed as an **engineering prototype** focused on building, training, and evaluating a forecasting framework, with emphasis on accuracy and robustness rather than traditional hypothesis testing.

---

## Features
- Hybrid architecture: LSTM + Auto-Regressive Dense layers  
- Support for StandardScaler or MinMaxScaler  
- Gaussian noise injection for generalization  
- Dropout and BatchNormalization for stability  
- Train/Validation/Test split with performance evaluation (RMSE, MAE, R², MAPE)  
- Multi-year forecasting with ensemble predictions and uncertainty estimation  
- Rich visualizations: forecast plots, error metrics, performance comparisons  

---

## Requirements
Install dependencies before running the script:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
