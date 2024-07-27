import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def clean_data(data):
    data = data.drop_duplicates()
    data['Sales'] = data['Sales'].fillna(data['Sales'].median())
    data['#Order'] = data['#Order'].fillna(data['#Order'].mean())
    data['Store_Type'] = data['Store_Type'].fillna(data['Store_Type'].mode()[0])
    data['Location_Type'] = data['Location_Type'].fillna(data['Location_Type'].mode()[0])
    data['Region_Code'] = data['Region_Code'].fillna(data['Region_Code'].mode()[0])
    return data
