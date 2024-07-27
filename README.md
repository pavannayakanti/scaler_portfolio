# Sales Forecasting Project
## Project Structure
sales_forecasting/
|-- data/
| |-- TRAIN.csv
|-- notebooks/
| |-- EDA.ipynb
|-- src/
| |-- init.py
| |-- data_processing.py
| |-- feature_engineering.py
| |-- model_training.py
| |-- model_evaluation.py
|-- main.py
|-- requirements.txt
|-- README.md
└── models/
    └── baseline_model.pkl
    └── rf_model.pkl
    └── xgb_model.pkl
    └── preprocessor.pkl

## Setup

1. **Clone the repository:**
   ```sh
   git clone <repository_url>
   cd sales_forecasting

2. **Create a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt

## Training and Saving Models
1. **To run the project, execute the following command:**
   ```sh
   python train_and_save_models.py
   ```
   This script will train the models (baseline, random forest, XGBoost, and LSTM) and save them to the models directory as pickle files.

## Hosting the Models
1. **run the deployment.py script:**
   ```sh
   python deployment.py
   ```
   This script will start a Flask API server that loads the saved models and serves predictions. And successful deployment shows the message "Welcome to the Sales Forecasting API!" on the screen.

## Validating the Models
1. **Test the API:**
+ Send a POST request to http://127.0.0.1:5000/predict with JSON data containing the features required for prediction.
+ Example request (using curl):
   ```sh
      curl -X POST -H "Content-Type: application/json" -d '{"#Order": 10, "Year": 2023, "Month": 12, "Week": 50, "Day": 24, "DayOfWeek": 6, "Sales_Last_Week": 1500}' http://127.0.0.1:5000/predict
   ```
+ Example response:
   ```sh
   {
   "prediction": 1234.56
   }
   ```
+ This response will contain the predicted sales value based on the input features.








