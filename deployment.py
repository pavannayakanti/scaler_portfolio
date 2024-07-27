import pickle
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Global variables to store the models
baseline_model = None
rf_model = None
xgb_model = None
preprocessor = None

# Load the model before handling the first request
@app.before_request
def load_model():
    global baseline_model, rf_model, xgb_model, preprocessor
    if baseline_model is None or rf_model is None or xgb_model is None or preprocessor is None:
        baseline_model = pickle.load(open('models/baseline_model.pkl', 'rb'))
        rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
        xgb_model = pickle.load(open('models/xgb_model.pkl', 'rb'))
        preprocessor = pickle.load(open('models/preprocessor.pkl', 'rb'))

# API Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Assuming the input data is in the same format as your features
    input_df = pd.DataFrame([data])
    input_df = preprocessor.transform(input_df)
    
    # Make predictions using the chosen model
    prediction = baseline_model.predict(input_df)[0]
    
    return jsonify({'prediction': prediction})

# Route for the root URL
@app.route('/')
def index():
    return 'Welcome to the Sales Forecasting API!'

if __name__ == '__main__':
    app.run(debug=True)


### Testing the API
# (.venv) pavanay@Nayakantis-MacBook-Air sales_forecasting % curl -X POST -H "Content-Type: application/json" -d '{"Store_Type": "A", "Location_Type": "X", "Region_Code": "1", "#Order": 10, "Year": 2022, "Month": 7, "Week": 28, "Day": 3, "DayOfWeek": 2, "Sales_Last_Week": 500}' http://127.0.0.1:5000/predict

# {
#   "prediction": -1460.3498379738303
# }

####