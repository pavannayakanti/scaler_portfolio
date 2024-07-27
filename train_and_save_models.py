import pandas as pd
from src.data_processing import load_data, clean_data
from src.feature_engineering import add_time_features, preprocess_data
from src.model_training import train_baseline_model, train_rf_model, train_xgb_model, create_sequences, train_lstm_model
from src.model_evaluation import evaluate_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer  # Import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pickle
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and clean data
data = load_data('data/TRAIN.csv')
logging.info(f"{datetime.datetime.now()} - Columns after loading data: {data.columns}")
data = clean_data(data)
logging.info(f"{datetime.datetime.now()} - Columns after cleaning data: {data.columns}")
# Feature engineering
data = add_time_features(data)
logging.info(f"{datetime.datetime.now()} - Columns after feature engineering: {data.columns}")

# Define features and target
numerical_features = ['#Order', 'Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Sales_Last_Week']
categorical_features = [] 
X = data.drop(columns=['Sales', 'Date', 'ID', 'Store_Type','Location_Type', 'Region_Code','Discount','Store_Location']) 
y = data['Sales']

# Verify that 'Sales' column is dropped correctly
logging.info(f"{datetime.datetime.now()} - Columns after dropping 'Sales' and 'Date': {X.columns}")

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Initialize lists to store evaluation metrics
baseline_mae_scores = []
rf_mae_scores = []
xgb_mae_scores = []
lstm_mae_scores = []

baseline_rmse_scores = []
rf_rmse_scores = []
xgb_rmse_scores = []
lstm_rmse_scores = []

baseline_mape_scores = []
rf_mape_scores = []
xgb_mape_scores = []
lstm_mape_scores = []

# Loop through each fold in the time series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Handle missing values (Imputation)
    imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with your preferred strategy
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Convert NumPy arrays back to DataFrames
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)

    # Preprocess data
    # Pass X_train_imputed and data to preprocess_data
    X_train_preprocessed, preprocessor = preprocess_data(X_train_imputed, data, numerical_features, categorical_features)  # Pass data
    X_test_preprocessed = preprocessor.transform(X_test_imputed)

    # Train models
    baseline_model = train_baseline_model(X_train_preprocessed, y_train)
    rf_model = train_rf_model(X_train_preprocessed, y_train)
    xgb_model = train_xgb_model(X_train_preprocessed, y_train)

    # LSTM Model
    seq_length = 7
    X_train_lstm, y_train_lstm = create_sequences(y_train.values, seq_length)
    X_test_lstm, y_test_lstm = create_sequences(y_test.values, seq_length)
    lstm_model = train_lstm_model(X_train_lstm, y_train_lstm, seq_length)

    # Evaluate models
    baseline_mae, baseline_rmse = evaluate_model(baseline_model, X_test_preprocessed, y_test)
    rf_mae, rf_rmse = evaluate_model(rf_model, X_test_preprocessed, y_test)
    xgb_mae, xgb_rmse = evaluate_model(xgb_model, X_test_preprocessed, y_test)
    lstm_mae, lstm_rmse = evaluate_model(lstm_model, X_test_lstm, y_test_lstm)

    # Calculate MAPE
    baseline_mape = mean_absolute_percentage_error(y_test, baseline_model.predict(X_test_preprocessed))
    rf_mape = mean_absolute_percentage_error(y_test, rf_model.predict(X_test_preprocessed))
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_model.predict(X_test_preprocessed))
    lstm_mape = mean_absolute_percentage_error(y_test_lstm, lstm_model.predict(X_test_lstm))

    # Append scores to lists
    baseline_mae_scores.append(baseline_mae)
    rf_mae_scores.append(rf_mae)
    xgb_mae_scores.append(xgb_mae)
    lstm_mae_scores.append(lstm_mae)

    baseline_rmse_scores.append(baseline_rmse)
    rf_rmse_scores.append(rf_rmse)
    xgb_rmse_scores.append(xgb_rmse)
    lstm_rmse_scores.append(lstm_rmse)

    baseline_mape_scores.append(baseline_mape)
    rf_mape_scores.append(rf_mape)
    xgb_mape_scores.append(xgb_mape)
    lstm_mape_scores.append(lstm_mape)

# Print average scores across folds
print(f'Baseline Model - MAE: {np.mean(baseline_mae_scores)}, RMSE: {np.mean(baseline_rmse_scores)}, MAPE: {np.mean(baseline_mape_scores)}')
print(f'Random Forest Model - MAE: {np.mean(rf_mae_scores)}, RMSE: {np.mean(rf_rmse_scores)}, MAPE: {np.mean(rf_mape_scores)}')
print(f'XGBoost Model - MAE: {np.mean(xgb_mae_scores)}, RMSE: {np.mean(xgb_rmse_scores)}, MAPE: {np.mean(xgb_mape_scores)}')
print(f'LSTM Model - MAE: {np.mean(lstm_mae_scores)}, RMSE: {np.mean(lstm_rmse_scores)}, MAPE: {np.mean(lstm_mape_scores)}')

# Residual Analysis (Example for Baseline Model)
y_pred_baseline = baseline_model.predict(X_test_preprocessed)
residuals_baseline = y_test - y_pred_baseline

# Save the trained models
try:
    with open('models/baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline_model, f)
    logging.info(f"{datetime.datetime.now()}: Baseline model saved successfully.")

    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    logging.info(f"{datetime.datetime.now()}:  Random Forest model saved successfully.")

    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    logging.info(f"{datetime.datetime.now()}: XGBoost model saved successfully.")

    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    logging.info(f"{datetime.datetime.now()}: Preprocessor saved successfully.")

except Exception as e:
    logging.error(f"{datetime.datetime.now()} - Error saving models: {e}")


# Plot residuals
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals_baseline)
plt.xlabel('Actual Sales')
plt.ylabel('Residuals')
plt.title('Residual Analysis for Baseline Model')
plt.show()
