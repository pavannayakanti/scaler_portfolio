def add_time_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Sales_Last_Week'] = data['Sales'].shift(7).rolling(7).sum()
    data['Store_Location'] = data['Store_Type'] + '_' + data['Location_Type']
    return data

def preprocess_data(data, numerical_features, categorical_features):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    data_preprocessed = preprocessor.fit_transform(data)
    return data_preprocessed, preprocessor
