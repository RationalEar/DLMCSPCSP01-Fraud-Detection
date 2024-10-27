import time

import joblib
import pandas as pd

from helpers.common_functions import read_from_files

INPUT_FEATURES = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']

def get_random_transaction(start_date, end_date, input_dir):
    # Load the data
    transactions_df = read_from_files(input_dir, start_date, end_date)
    # Get Fraudulent transactions
    fraudulent_transactions = transactions_df[transactions_df['TX_FRAUD'] == 1]
    # Get random but equal number of non-fraudulent transactions
    non_fraudulent_transactions = transactions_df[transactions_df['TX_FRAUD'] == 0].sample(
        n=len(fraudulent_transactions))
    # Combine the two dataframes
    transactions = pd.concat([fraudulent_transactions, non_fraudulent_transactions])
    # Get random transaction
    return transactions.sample()
    

def predict_fraud_status(transaction, model):
    # Load the model
    loaded_model = joblib.load(model)
    # Make prediction
    prediction = loaded_model.predict(transaction)
    return prediction


def test_predictions(start_date, end_date, data_input_dir, models_dir, models):
    # Get a random transaction
    original_transaction = get_random_transaction(start_date, end_date, data_input_dir)
    # get fraud status
    fraud_status = original_transaction['TX_FRAUD'].values[0]
    # select input features
    transaction = original_transaction[INPUT_FEATURES]
    
    # Make prediction using each model
    predictions = []
    model_predictions = {}
    start_time = time.time()
    for model_name in models:
        model = models_dir + model_name + '.pkl'
        prediction = predict_fraud_status(transaction, model)
        predictions.append(prediction[0])
        model_predictions[model_name] = int(prediction[0])
        print(f"Model: {model_name}, Predicted Fraud Status: {prediction[0]}")
    
    print("Time to make prediction: {0:.2}s".format(time.time() - start_time))
    # set predicted status = True if at least one of models predict fraud
    predicted_status = 1 if sum(predictions) > 0 else 0
    
    print(f"Actual Fraud Status: {fraud_status}, \nPredicted Fraud Status: {predicted_status}")
    results = {'actual_status': int(fraud_status), 'predicted_status': predicted_status,
               'prediction_time': time.time() - start_time, 'model_predictions': model_predictions,
               'transaction': original_transaction.to_dict('records')}
    return results

