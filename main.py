import pandas as pd
from src.data_processing import load_data, clean_data
from src.feature_engineering import add_time_features, preprocess_data
from src.model_training import train_baseline_model, train_rf_model, train_xgb_model, create_sequences
from src.model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split

# Load and clean data
data = load_data('data/TRAIN.csv')
data = clean_data(data)

# Feature engineering
data = add_time_features(data)

# Define features and target
numerical_features = ['Sales', '#Order', 'Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Sales_Last_Week']
categorical_features = ['Store_Type', 'Location_Type', 'Region_Code', 'Store_Location']
X = data.drop(columns=['Sales', 'Date'])
y = data['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train_preprocessed, preprocessor = preprocess_data(X_train, numerical_features, categorical_features)
X_test_preprocessed = preprocessor.transform(X_test)

# Train models
baseline_model = train_baseline_model(X_train_preprocessed, y_train)
rf_model = train_rf_model(X_train_preprocessed, y_train)
xgb_model = train_xgb_model(X_train_preprocessed, y_train)

# Evaluate models
baseline_mae, baseline_rmse = evaluate_model(baseline_model, X_test_preprocessed, y_test)
rf_mae, rf_rmse = evaluate_model(rf_model, X_test_preprocessed, y_test)
xgb_mae, xgb_rmse = evaluate_model(xgb_model, X_test_preprocessed, y_test)

print(f'Baseline Model - MAE: {baseline_mae}, RMSE: {baseline_rmse}')
print(f'Random Forest Model - MAE: {rf_mae}, RMSE: {rf_rmse}')
print(f'XGBoost Model - MAE: {xgb_mae}, RMSE: {xgb_rmse}')
