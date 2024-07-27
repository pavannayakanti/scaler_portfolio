import logging
import datetime

def add_time_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Sales_Last_Week'] = data['Sales'].shift(7).rolling(7).sum()
    data['Store_Location'] = data['Store_Type'] + '_' + data['Location_Type']
    return data

def preprocess_data(data, data_original, numerical_features, categorical_features):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Ensure that the features are in the DataFrame
    logging.info(f"{datetime.datetime.now()}: Numerical features in data: {data_original[numerical_features].columns}")
    logging.info(f"{datetime.datetime.now()}: Categorical features in data: {data_original[categorical_features].columns}")
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    data_preprocessed = preprocessor.fit_transform(data)
    return data_preprocessed, preprocessor

