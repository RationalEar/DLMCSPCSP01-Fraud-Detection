import datetime
from flask import Flask, request, jsonify
from data_simulator.generator import generate_dataset_and_save, transform_dataset_and_save
from model_training import models
from model_training.models import load_data
from real_time_detector.predictor import test_predictions

DATE_FORMAT = "%Y-%m-%d"
START_DATE = '2024-01-01'
LOAD_START_DATE = '2024-03-11'
LOAD_END_DATE = '2024-06-14'
TRAINING_START_DATE = '2024-04-25'
NUMBER_OF_DAYS = 183
NUMBER_OF_CUSTOMERS = 5000
NUMBER_OF_TERMINALS = 10000
DIR_RAW_DATA = "files/simulated-data-raw/"
DIR_TRANSFORMED_DATA = "files/simulated-data-transformed/"
MODELS_DIR = "files/models/"
MODELS = ('decision_tree', 'logistic_regression', 'random_forest', 'xgboost')

app = Flask(__name__)
app.config['REQUEST_TIMEOUT'] = 600

@app.route('/generate-data', methods=['GET'])
def generate_data(overwrite=False):
    end_date_obj = datetime.datetime.strptime(START_DATE, DATE_FORMAT) + datetime.timedelta(days=NUMBER_OF_DAYS)
    end_date = end_date_obj.strftime(DATE_FORMAT)
    print(f"Start date: {START_DATE}, End date: {end_date}, Number of days: {NUMBER_OF_DAYS}")
    
    print('Generating data...')
    generate_dataset_and_save(NUMBER_OF_CUSTOMERS, NUMBER_OF_TERMINALS, NUMBER_OF_DAYS, START_DATE, DIR_RAW_DATA, overwrite)
    
    print('Transforming data...')
    transform_dataset_and_save(START_DATE, end_date, DIR_RAW_DATA, DIR_TRANSFORMED_DATA, overwrite)
    return 'Data generated and transformed'


@app.route('/train', methods=['GET'])
def train(overwrite=False):
    print('Training models...')
    transaction_df = load_data(DIR_TRANSFORMED_DATA, LOAD_START_DATE, LOAD_END_DATE)
    print(f"Number of transactions for training: {len(transaction_df)}")
    
    performances = models.train_models(transaction_df, TRAINING_START_DATE, MODELS_DIR, MODELS, overwrite)
    print(performances)
    return 'Models trained'


@app.route('/predict', methods=['GET'])
def predict():
    start_date_obj = datetime.datetime.strptime(START_DATE, DATE_FORMAT) + datetime.timedelta(days=NUMBER_OF_DAYS-8)
    start_date = start_date_obj.strftime(DATE_FORMAT)
    end_date_obj = datetime.datetime.strptime(START_DATE, DATE_FORMAT) + datetime.timedelta(days=NUMBER_OF_DAYS)
    end_date = end_date_obj.strftime(DATE_FORMAT)
    predictions = test_predictions(start_date, end_date, DIR_TRANSFORMED_DATA, MODELS_DIR, MODELS)
    prediction = 'Fraudulent' if predictions == 1 else 'Non-fraudulent'
    return jsonify({'prediction': prediction})
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(debug=True)
    

