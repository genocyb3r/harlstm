# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# Randomize Seeding
np.random.seed(42)
tf.random.set_seed(42)


# Constants
LSTM_LOOKBACK = 24
AR_LOOKBACK = 12
LSTM_UNITS = 100
DENSE_UNITS = 64
LEARNING_RATE = 0.008
EPOCHS = 200
BATCH_SIZE = 16

# Data splitting
TEST_SIZE = 0.15 # 15%
VALIDATION_SPLIT = 0.1 #10%

PREDICTION_YEARS = 5
PERIODS_PER_YEAR = 12
FUTURE_PERIODS = PREDICTION_YEARS * PERIODS_PER_YEAR

MODEL_PATH = "model.keras"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "gasoline.csv"

RETRAIN_MODEL = False

USE_STANDARDSCALER = True
ADD_TREND_FEATURES = True
USE_ENSEMBLE = True
NOISE_FACTOR = 0.001 # Gaussian noise

print(f"Configuration:")
print(f"  Prediction horizon: {PREDICTION_YEARS} years ({FUTURE_PERIODS} periods)")
print(f"  LSTM lookback: {LSTM_LOOKBACK}")
print(f"  AR lookback: {AR_LOOKBACK}")
print(f"  Model path: {MODEL_PATH}")
print(f"  Retrain model: {RETRAIN_MODEL}")
print(f"  Use StandardScaler: {USE_STANDARDSCALER}")
print(f"  Add trend features: {ADD_TREND_FEATURES}")

df = pd.read_csv(DATA_PATH)
data = df["Price"].values
year_values = df["Year"].values
print(f"Data length: {len(data)}")
print(f"Data range: {data.min():.2f} - {data.max():.2f}")
print(f"Data std: {data.std():.2f}")


def prepare_hybrid_data(data, lstm_lookback=LSTM_LOOKBACK, ar_lookback=AR_LOOKBACK, test_size=TEST_SIZE):
    if USE_STANDARDSCALER:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    if NOISE_FACTOR > 0:
        noise = np.random.normal(0, NOISE_FACTOR, scaled_data.shape)
        scaled_data_noisy = scaled_data + noise
    else:
        scaled_data_noisy = scaled_data

    split_idx = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data_noisy[:split_idx]
    test_data = scaled_data[split_idx:]

    def create_lstm_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def create_ar_features(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):

            ar_features = data[i-lookback:i].tolist()

            if ADD_TREND_FEATURES:
                recent = data[i-lookback:i]
                trend = recent[-1] - recent[0]
                volatility = np.std(recent)
                mean_val = np.mean(recent)
                ar_features.extend([trend, volatility, mean_val])

            X.append(ar_features)
            y.append(data[i])
        return np.array(X), np.array(y)

    X_lstm_train, y_train = create_lstm_sequences(train_data, lstm_lookback)
    X_ar_train, _ = create_ar_features(train_data, ar_lookback)

    max_lookback = max(lstm_lookback, ar_lookback)
    start_idx = max_lookback - min(lstm_lookback, ar_lookback)

    if lstm_lookback > ar_lookback:
        X_ar_train = X_ar_train[start_idx:]
    else:
        X_lstm_train = X_lstm_train[start_idx:]
        y_train = y_train[start_idx:]

    full_test_data = np.concatenate([train_data[-max_lookback:], test_data])

    X_lstm_test, y_test = create_lstm_sequences(full_test_data, lstm_lookback)
    X_ar_test, _ = create_ar_features(full_test_data, ar_lookback)

    test_samples = len(test_data)
    X_lstm_test = X_lstm_test[-test_samples:]
    X_ar_test = X_ar_test[-test_samples:]
    y_test = y_test[-test_samples:]

    return (X_lstm_train, X_ar_train, y_train), (X_lstm_test, X_ar_test, y_test), scaler


def build_hybrid_model(lstm_lookback=LSTM_LOOKBACK, ar_lookback=AR_LOOKBACK,
                               lstm_units=LSTM_UNITS, dense_units=DENSE_UNITS):
    # LSTM layers
    lstm_input = Input(shape=(lstm_lookback, 1), name='lstm_input')
    lstm_out = LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_input)
    lstm_out = LSTM(lstm_units//2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    # AR Layers
    ar_features = ar_lookback + (3 if ADD_TREND_FEATURES else 0)
    ar_input = Input(shape=(ar_features,), name='ar_input')
    ar_out = Dense(dense_units, activation='relu')(ar_input)
    ar_out = BatchNormalization()(ar_out)
    ar_out = Dropout(0.3)(ar_out)
    ar_out = Dense(dense_units//2, activation='relu')(ar_out)
    ar_out = Dropout(0.2)(ar_out)
    # Hybrid
    combined = Concatenate()([lstm_out, ar_out])
    combined = Dense(dense_units, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(dense_units//2, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    combined = Dense(dense_units//4, activation='relu')(combined)
    output = Dense(1, activation='linear')(combined)

    model = Model(inputs=[lstm_input, ar_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                 loss='mse', metrics=['mae', 'mape'])

    return model
def save_model_and_scaler(model, scaler, validation_indices, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    model.save(model_path)

    model_artifacts = {
        'scaler': scaler,
        'val_split_idx': validation_indices['val_split_idx'],
        'total_train_samples': validation_indices['total_train_samples']
    }

    with open(scaler_path, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"Model saved to: {model_path}")
    print(f"Model artifacts saved to: {scaler_path}")

def load_model_and_scaler(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)

        with open(scaler_path, 'rb') as f:
            model_artifacts = pickle.load(f)


        if isinstance(model_artifacts, dict) and 'validation_indices' in model_artifacts:
            scaler = model_artifacts['scaler']
            validation_indices = model_artifacts['validation_indices']
            print(f"Loaded validation split: {validation_indices['train_only_samples']} train, {validation_indices['val_samples']} val")
        else:
            scaler = model_artifacts.get('scaler', model_artifacts) if isinstance(model_artifacts, dict) else model_artifacts
            validation_indices = None
            print("Warning: No validation split info found in saved model. Will need to re-split data.")

        print(f"Model loaded from: {model_path}")
        print(f"Scaler loaded from: {scaler_path}")
        return model, scaler, validation_indices
    else:
        return None, None, None


(X_lstm_train, X_ar_train, y_train), (X_lstm_test, X_ar_test, y_test), scaler = prepare_hybrid_data(data)

print(f"Enhanced Training shapes:")
print(f"  LSTM input: {X_lstm_train.shape}")
print(f"  AR input: {X_ar_train.shape}")
print(f"  Target: {y_train.shape}")
print(f"Enhanced Test shapes:")
print(f"  LSTM input: {X_lstm_test.shape}")
print(f"  AR input: {X_ar_test.shape}")
print(f"  Target: {y_test.shape}")


X_lstm_train = X_lstm_train.reshape(X_lstm_train.shape[0], X_lstm_train.shape[1], 1)
X_lstm_test = X_lstm_test.reshape(X_lstm_test.shape[0], X_lstm_test.shape[1], 1)

model, loaded_scaler, validation_info= load_model_and_scaler()

if model is None or RETRAIN_MODEL:
    print("\n=== Training Enhanced Model ===")
    model = build_hybrid_model()
    print("Enhanced Model Architecture:")
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    val_split_idx = int(len(X_lstm_train) * (1 - VALIDATION_SPLIT))
    X_lstm_val = X_lstm_train[val_split_idx:]
    X_ar_val = X_ar_train[val_split_idx:]
    y_val = y_train[val_split_idx:]

    X_lstm_train_only = X_lstm_train[:val_split_idx]
    X_ar_train_only = X_ar_train[:val_split_idx]
    y_train_only = y_train[:val_split_idx]

    validation_indices = {
    'val_split_idx': val_split_idx,
    'total_train_samples': len(X_lstm_train),
    'train_only_samples': len(X_lstm_train_only),
    'val_samples': len(X_lstm_val),
    'test_samples': len(X_lstm_test),
    'validation_split': VALIDATION_SPLIT
}


    validation_data = {
        'X_lstm': X_lstm_val,
        'X_ar': X_ar_val,
        'y': y_val
    }
    training_only_data = {
        'X_lstm': X_lstm_train_only,
        'X_ar': X_ar_train_only,
        'y': y_train_only
    }

    print("Training enhanced model...")
    history = model.fit(
        [X_lstm_train_only, X_ar_train_only], y_train_only,
        validation_data=([X_lstm_val, X_ar_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        callbacks=[early_stopping, reduce_lr]
    )

    save_model_and_scaler(model, scaler, validation_indices)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Training History - MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()

    plt.tight_layout()
    plt.show()

else:
    print("\n=== Using Loaded Enhanced Model - Reconstructing validation split ===")
    scaler = loaded_scaler

    if validation_info and 'val_split_idx' in validation_info:
        val_split_idx = validation_info['val_split_idx']
        print(f"Using saved validation split index: {val_split_idx}")
    else:
        val_split_idx = int(len(X_lstm_train) * (1 - VALIDATION_SPLIT))
        print(f"Recalculating validation split index: {val_split_idx}")

    X_lstm_val = X_lstm_train[val_split_idx:]
    X_ar_val = X_ar_train[val_split_idx:]
    y_val = y_train[val_split_idx:]

    X_lstm_train_only = X_lstm_train[:val_split_idx]
    X_ar_train_only = X_ar_train[:val_split_idx]
    y_train_only = y_train[:val_split_idx]

    validation_data = {
        'X_lstm': X_lstm_val,
        'X_ar': X_ar_val,
        'y': y_val
    }
    training_only_data = {
        'X_lstm': X_lstm_train_only,
        'X_ar': X_ar_train_only,
        'y': y_train_only
    }



train_only_pred = model.predict([training_only_data['X_lstm'], training_only_data['X_ar']])
val_pred = model.predict([validation_data['X_lstm'], validation_data['X_ar']])
test_pred = model.predict([X_lstm_test, X_ar_test])

train_only_pred_orig = scaler.inverse_transform(train_only_pred.reshape(-1, 1)).flatten()
val_pred_orig = scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
test_pred_orig = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

y_train_only_orig = scaler.inverse_transform(training_only_data['y'].reshape(-1, 1)).flatten()
y_val_orig = scaler.inverse_transform(validation_data['y'].reshape(-1, 1)).flatten()
y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

train_rmse = np.sqrt(mean_squared_error(y_train_only_orig, train_only_pred_orig))
val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))

train_mae = mean_absolute_error(y_train_only_orig, train_only_pred_orig)
val_mae = mean_absolute_error(y_val_orig, val_pred_orig)
test_mae = mean_absolute_error(y_test_orig, test_pred_orig)

train_r2 = r2_score(y_train_only_orig, train_only_pred_orig)
val_r2 = r2_score(y_val_orig, val_pred_orig)
test_r2 = r2_score(y_test_orig, test_pred_orig)


def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


train_mape = calculate_mape(y_train_only_orig, train_only_pred_orig)
val_mape = calculate_mape(y_val_orig, val_pred_orig)
test_mape = calculate_mape(y_test_orig, test_pred_orig)


print(f"\n=== Model Performance (Fixed) ===")
print(f"Training Set (76.5% of data):")
print(f"  RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}, MAPE: {train_mape:.2f}%")
print(f"Validation Set (8.5% of data):")
print(f"  RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}, R²: {val_r2:.3f}, MAPE: {val_mape:.2f}%")
print(f"Test Set (15% of data):")
print(f"  RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}, MAPE: {test_mape:.2f}%")

def predict_future_enhanced(model, data, scaler, n_periods=FUTURE_PERIODS,
                           lstm_lookback=LSTM_LOOKBACK, ar_lookback=AR_LOOKBACK,
                           use_ensemble=USE_ENSEMBLE, n_ensemble=5):

    if hasattr(data, 'values'):
        data = data.values
    data = np.array(data)

    scaled_data = scaler.transform(data.reshape(-1, 1)).flatten()

    if use_ensemble:
        print(f"Generating {n_periods} future predictions using {n_ensemble}-model ensemble...")
        all_predictions = []

        for ensemble_idx in range(n_ensemble):
            predictions = []
            current_data = scaled_data.copy()

            if ensemble_idx > 0:
                noise = np.random.normal(0, 0.01, len(current_data))
                current_data += noise

            for i in range(n_periods):
                lstm_seq = current_data[-lstm_lookback:].reshape(1, lstm_lookback, 1)

                ar_features = current_data[-ar_lookback:].tolist()
                if ADD_TREND_FEATURES:
                    recent = current_data[-ar_lookback:]
                    trend = recent[-1] - recent[0]
                    volatility = np.std(recent)
                    mean_val = np.mean(recent)
                    ar_features.extend([trend, volatility, mean_val])

                ar_seq = np.array(ar_features).reshape(1, -1)

                next_pred = model.predict([lstm_seq, ar_seq], verbose=0)[0, 0]
                predictions.append(next_pred)

                current_data = np.append(current_data, next_pred)

            all_predictions.append(predictions)
            print(f"  Completed ensemble {ensemble_idx + 1}/{n_ensemble}")

        all_predictions = np.array(all_predictions)
        predictions_mean = np.mean(all_predictions, axis=0)
        predictions_std = np.std(all_predictions, axis=0)

        predictions_orig = scaler.inverse_transform(predictions_mean.reshape(-1, 1)).flatten()
        predictions_std_orig = predictions_std * scaler.scale_[0]

        return predictions_orig, predictions_std_orig

    else:
        print(f"Generating {n_periods} future predictions...")
        predictions = []
        current_data = scaled_data.copy()

        for i in range(n_periods):
            lstm_seq = current_data[-lstm_lookback:].reshape(1, lstm_lookback, 1)

            ar_features = current_data[-ar_lookback:].tolist()
            if ADD_TREND_FEATURES:
                recent = current_data[-ar_lookback:]
                trend = recent[-1] - recent[0]
                volatility = np.std(recent)
                mean_val = np.mean(recent)
                ar_features.extend([trend, volatility, mean_val])

            ar_seq = np.array(ar_features).reshape(1, -1)

            next_pred = model.predict([lstm_seq, ar_seq], verbose=0)[0, 0]
            predictions.append(next_pred)

            current_data = np.append(current_data, next_pred)

            if (i + 1) % 12 == 0:
                year = (i + 1) // 12
                print(f"  Completed year {year}/{PREDICTION_YEARS}")

        predictions_orig = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        return predictions_orig, None

print(f"\n=== {PREDICTION_YEARS}-Year Future Predictions ===")
if USE_ENSEMBLE:
    future_pred, future_std = predict_future_enhanced(model, data, scaler, n_periods=FUTURE_PERIODS)
else:
    future_pred, future_std = predict_future_enhanced(model, data, scaler, n_periods=FUTURE_PERIODS)

print(f"\n Yearly forecast summaries:")
for year in range(PREDICTION_YEARS):
    start_idx = year * PERIODS_PER_YEAR
    end_idx = start_idx + PERIODS_PER_YEAR
    year_data = future_pred[start_idx:end_idx]

    if future_std is not None:
        year_std = future_std[start_idx:end_idx]
        print(f"Year {year + 1}: Avg: {year_data.mean():.2f} (±{year_std.mean():.2f}), "
              f"Min: {year_data.min():.2f}, Max: {year_data.max():.2f}")
    else:
        print(f"Year {year + 1}: Avg: {year_data.mean():.2f}, Min: {year_data.min():.2f}, Max: {year_data.max():.2f}")


plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)

historical_dates = pd.date_range(start=f"{year_values[0]}-01", periods=len(data), freq="M")

future_dates = pd.date_range(start=historical_dates[-1] + pd.offsets.MonthBegin(1),
                             periods=FUTURE_PERIODS, freq="M")

plt.plot(historical_dates, data, label='Historical Data', linewidth=2, color='blue')
plt.plot(future_dates, future_pred, label=f'{PREDICTION_YEARS}-Year Forecast',
         linewidth=2, linestyle='--', color='red', alpha=0.8)

if future_std is not None:
    plt.fill_between(future_dates,
                     future_pred - 1.96 * future_std,
                     future_pred + 1.96 * future_std,
                     alpha=0.2, color='red', label='95% Confidence Interval')

plt.axvline(x=historical_dates[-1], color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
plt.title(f'Gas Price {PREDICTION_YEARS}-Year Forecast')
plt.xlabel('Year')
plt.ylabel('Gas Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.subplot(2, 2, 2)

total_data_length = len(data)
start_year = year_values[0]
end_year = year_values[-1]

full_historical_dates = pd.date_range(start=f"{start_year}-01",
                                     periods=total_data_length,
                                     freq="M")


test_split_point = int(total_data_length * (1 - TEST_SIZE))
train_val_data_length = len(y_train_only_orig) + len(y_val_orig)


train_dates = full_historical_dates[:len(y_train_only_orig)]
val_dates = full_historical_dates[len(y_train_only_orig):train_val_data_length]
test_dates = full_historical_dates[test_split_point:test_split_point + len(y_test_orig)]


plt.plot(train_dates, y_train_only_orig, label='Actual Train', alpha=0.8, color='blue', linewidth=1.5)
plt.plot(train_dates, train_only_pred_orig, label='Predicted Train', alpha=0.8, color='lightblue', linewidth=1.5)
plt.plot(val_dates, y_val_orig, label='Actual Val', alpha=0.8, color='orange', linewidth=1.5)
plt.plot(val_dates, val_pred_orig, label='Predicted Val', alpha=0.8, color='gold', linewidth=1.5)
plt.plot(test_dates, y_test_orig, label='Actual Test', alpha=0.8, color='red', linewidth=1.5)
plt.plot(test_dates, test_pred_orig, label='Predicted Test', alpha=0.8, color='pink', linewidth=1.5)


plt.axvline(x=train_dates[-1], color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Train/Val Split')
plt.axvline(x=val_dates[-1], color='black', linestyle='--', alpha=0.7, linewidth=2, label='Val/Test Split')


plt.title('Train/Validation/Test Predictions (1990-2025)', fontsize=12, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Gas Price ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.xlim(train_dates[0], test_dates[-1])


y_min = min(y_train_only_orig.min(), y_val_orig.min(), y_test_orig.min()) * 0.95
y_max = max(y_train_only_orig.max(), y_val_orig.max(), y_test_orig.max()) * 1.05
plt.ylim(y_min, y_max)


plt.text(train_dates[len(train_dates)//2], y_max*0.9, 'Training',
         ha='center', va='center', fontsize=9, alpha=0.7,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
plt.text(val_dates[len(val_dates)//2], y_max*0.9, 'Validation',
         ha='center', va='center', fontsize=9, alpha=0.7,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.5))
plt.text(test_dates[len(test_dates)//2], y_max*0.9, 'Test',
         ha='center', va='center', fontsize=9, alpha=0.7,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.5))

plt.subplot(2, 2, 3)
years = []
yearly_avg = []
yearly_min = []
yearly_max = []
yearly_std = []

for year in range(PREDICTION_YEARS):
    start_idx = year * PERIODS_PER_YEAR
    end_idx = start_idx + PERIODS_PER_YEAR
    year_data = future_pred[start_idx:end_idx]
    years.append(f'Y{year + 1}')
    yearly_avg.append(year_data.mean())
    yearly_min.append(year_data.min())
    yearly_max.append(year_data.max())

    if future_std is not None:
        yearly_std.append(future_std[start_idx:end_idx].mean())

x_pos = np.arange(len(years))
bars = plt.bar(x_pos, yearly_avg, alpha=0.7, label='Average Price')

if future_std is not None:
    plt.errorbar(x_pos, yearly_avg, yerr=yearly_std,
                fmt='none', ecolor='black', capsize=3, label='Uncertainty')

plt.title('Yearly Forecast Summary')
plt.xlabel('Forecast Year')
plt.ylabel('Average Gas Price')
plt.xticks(x_pos, years)
plt.legend()
plt.grid(True, alpha=0.3)

for i, (bar, avg) in enumerate(zip(bars, yearly_avg)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{avg:.2f}', ha='center', va='bottom', fontsize=9)

plt.subplot(2, 2, 4)
metrics = ['RMSE', 'MAE', 'R²', 'MAPE (%)']
train_values = [train_rmse, train_mae, train_r2, train_mape]
val_values = [val_rmse, val_mae, val_r2, val_mape]
test_values = [test_rmse, test_mae, test_r2, test_mape]

x_pos = np.arange(len(metrics))
width = 0.25

plt.bar(x_pos - width, train_values, width, label='Train', alpha=0.7)
plt.bar(x_pos, val_values, width, label='Validation', alpha=0.7)
plt.bar(x_pos + width, test_values, width, label='Test', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Performance: Train/Val/Test')
plt.xticks(x_pos, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
    plt.text(i - width/2, train_val + max(train_values) * 0.01,
             f'{train_val:.3f}', ha='center', va='bottom', fontsize=8)
    plt.text(i + width/2, test_val + max(test_values) * 0.01,
             f'{test_val:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)


recent_data = data[-24:]
recent_dates = pd.date_range(start="2023-01", periods=24, freq="M")

future_dates = pd.date_range(start="2025-01", periods=24, freq="M")

plt.plot(recent_dates, recent_data, label='Recent Historical', linewidth=2, color='blue')
plt.plot(future_dates, future_pred[:24], label='Near-term Forecast',
         linewidth=2, linestyle='--', color='red')

plt.axvline(x=pd.to_datetime("2024-12-31"), color='gray', linestyle=':', alpha=0.7)

plt.title('Recent vs Near-term Forecast')
plt.xlabel('Year')
plt.ylabel('Gas Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)


plt.subplot(1, 3, 2)
price_changes = np.diff(future_pred)
plt.plot(price_changes, color='green', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title('Predicted Month-to-Month Changes')
plt.xlabel('Future Period')
plt.ylabel('Price Change')
plt.grid(True, alpha=0.3)



plt.figure(figsize=(15, 6))
val_start_date = pd.date_range(start="2017-01", periods=len(y_val_orig), freq="M")
test_start_date = pd.date_range(start=val_start_date[-1] + pd.offsets.MonthBegin(1),
                               periods=len(y_test_orig), freq="M")

plt.plot(val_start_date, y_val_orig, 'o-', label='Actual Validation',
         color='orange', linewidth=2, markersize=4, alpha=0.8)
plt.plot(val_start_date, val_pred_orig, 's-', label='Predicted Validation',
         color='gold', linewidth=2, markersize=4, alpha=0.8)

plt.plot(test_start_date, y_test_orig, 'o-', label='Actual Test',
         color='red', linewidth=2, markersize=4, alpha=0.8)
plt.plot(test_start_date, test_pred_orig, '^-', label='Predicted Test',
         color='pink', linewidth=2, markersize=4, alpha=0.8)

plt.axvline(x=val_start_date[-1], color='black', linestyle='--',
           alpha=0.7, linewidth=2, label='Val/Test Boundary')

val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred_orig))
test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
val_mape = calculate_mape(y_val_orig, val_pred_orig)
test_mape = calculate_mape(y_test_orig, test_pred_orig)

plt.text(0.02, 0.98, f'Validation RMSE: {val_rmse:.2f}\nValidation MAPE: {val_mape:.2f}%',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

plt.text(0.98, 0.98, f'Test RMSE: {test_rmse:.2f}\nTest MAPE: {test_mape:.2f}%',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='pink', alpha=0.3))

plt.title('Validation vs Test Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Gas Price')
plt.legend(loc='center right')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.subplot(1, 3, 3)
if future_std is not None:
    plt.plot(future_std, color='orange', linewidth=2)
    plt.title('Prediction Uncertainty Over Time')
    plt.xlabel('Future Period')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
else:
    window = 12
    rolling_vol = pd.Series(future_pred).rolling(window=window).std()
    plt.plot(rolling_vol, color='orange', linewidth=2)
    plt.title(f'Rolling {window}-Period Volatility')
    plt.xlabel('Future Period')
    plt.ylabel('Rolling Std')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n=== Final Summary ===")
print(f"Model: HARLSTM")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"Historical data range: {data.min():.2f} - {data.max():.2f}")
print(f"5-year forecast range: {future_pred.min():.2f} - {future_pred.max():.2f}")
print(f"Model saved: {MODEL_PATH}")
print(f"Scaler saved: {SCALER_PATH}")
print(f"Set RETRAIN_MODEL = True to retrain the model next time")

