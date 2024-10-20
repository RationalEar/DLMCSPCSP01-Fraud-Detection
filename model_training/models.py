import datetime
import os
import time

import imblearn
import joblib
import pandas as pd
import sklearn
import xgboost

from helpers.common_functions import read_from_files
from model_training.model_helpers import card_precision_top_k_custom, model_selection_wrapper, \
    model_selection_wrapper_with_sampler

OUTPUT_FEATURE = "TX_FRAUD"

INPUT_FEATURES = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']


def load_data(dir_input, begin_date, end_date):
    print("Load  files")
    transactions_df = read_from_files(dir_input, begin_date, end_date)
    print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),
                                                                                   transactions_df.TX_FRAUD.sum()))
    
    return transactions_df


def setup_training_validation_parameters(transactions_df, training_date):
    # Number of folds for the prequential validation
    n_folds = 4
    
    # Set the starting day for the training period, and the deltas
    start_date_training = datetime.datetime.strptime(training_date, "%Y-%m-%d")
    delta_train = delta_delay = delta_test = delta_valid = delta_assessment = 7
    
    start_date_training_for_valid = start_date_training + datetime.timedelta(days=-(delta_delay + delta_valid))
    start_date_training_for_test = start_date_training + datetime.timedelta(days=(n_folds - 1) * delta_test)
    
    # Only keep columns that are needed as argument to the custom scoring function
    # (in order to reduce the serialization time of transaction dataset)
    transactions_df_scorer = transactions_df[['CUSTOMER_ID', 'TX_FRAUD', 'TX_TIME_DAYS']]
    
    card_precision_top_100 = sklearn.metrics.make_scorer(card_precision_top_k_custom,
                                                         response_method='predict',
                                                         top_k=100,
                                                         transactions_df=transactions_df_scorer)
    
    performance_metrics_list_grid = ['roc_auc', 'average_precision', 'card_precision@100']
    performance_metrics_list = ['AUC ROC', 'Average precision', 'Card Precision@100']
    
    scoring = {'roc_auc': 'roc_auc',
               'average_precision': 'average_precision',
               'card_precision@100': card_precision_top_100,
               }
    
    return (n_folds, start_date_training, delta_train, delta_delay, delta_test, delta_valid,
            delta_assessment, start_date_training_for_valid, start_date_training_for_test,
            scoring, performance_metrics_list, performance_metrics_list_grid)


def train_model(transactions_df, training_date, classifier, parameters):
    (n_folds, start_date_training, delta_train, delta_delay, delta_test, delta_valid,
     delta_assessment, start_date_training_for_valid, start_date_training_for_test,
     scoring, performance_metrics_list, performance_metrics_list_grid) = setup_training_validation_parameters(
        transactions_df, training_date)
    
    start_time = time.time()
    
    # Fit models and assess performances for all parameters
    performances_df, trained_model = model_selection_wrapper(transactions_df, classifier,
                                                             INPUT_FEATURES, OUTPUT_FEATURE,
                                                             parameters, scoring,
                                                             start_date_training_for_valid,
                                                             start_date_training_for_test,
                                                             n_folds=n_folds,
                                                             delta_train=delta_train,
                                                             delta_delay=delta_delay,
                                                             delta_assessment=delta_assessment,
                                                             performance_metrics_list_grid=performance_metrics_list_grid,
                                                             performance_metrics_list=performance_metrics_list,
                                                             n_jobs=-1)
    
    execution_time = time.time() - start_time
    return performances_df, execution_time, trained_model


def train_model_with_sampler(transactions_df, training_date, classifier, parameters, sampler_list):
    (n_folds, start_date_training, delta_train, delta_delay, delta_test, delta_valid,
     delta_assessment, start_date_training_for_valid, start_date_training_for_test,
     scoring, performance_metrics_list, performance_metrics_list_grid) = setup_training_validation_parameters(
        transactions_df, training_date)
    
    start_time = time.time()
    
    # Fit models and assess performances for all parameters
    performances_df, trained_model = model_selection_wrapper_with_sampler(transactions_df, classifier, sampler_list,
                                                                          INPUT_FEATURES, OUTPUT_FEATURE,
                                                                          parameters, scoring,
                                                                          start_date_training_for_valid,
                                                                          start_date_training_for_test,
                                                                          n_folds=n_folds,
                                                                          delta_train=delta_train,
                                                                          delta_delay=delta_delay,
                                                                          delta_assessment=delta_assessment,
                                                                          performance_metrics_list_grid=performance_metrics_list_grid,
                                                                          performance_metrics_list=performance_metrics_list,
                                                                          n_jobs=1)
    
    execution_time = time.time() - start_time
    return performances_df, execution_time, trained_model


def train_models(transactions_df, training_date, models_dir, models=('decision_tree',), overwrite=False):
    performances = []
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for model in models:
        print(f"Training model {model}")
        model_function_name = "train_" + model
        model_file = models_dir + model + ".pkl"
        performance_file = models_dir + model + "_perf.pkl"
        execution_time_file = models_dir + model + "_exec_time.pkl"
        if model_function_name not in globals():
            print(f"Function {model_function_name} not found")
            continue
        
        model_function = globals()[model_function_name]
        if not overwrite and os.path.exists(model_file) and os.path.exists(performance_file) and os.path.exists(
                execution_time_file):
            print(f"Model {model} already exists. Skipping training.")
            performance_df = joblib.load(performance_file)
            execution_time = joblib.load(execution_time_file)
        else:
            performance_df, execution_time, trained_model = model_function(transactions_df, training_date)
            print(f"Model {model} trained in {execution_time:.2f}s")
            joblib.dump(trained_model, model_file)
            joblib.dump(performance_df, performance_file)
            joblib.dump(execution_time, execution_time_file)
        
        model_training = {"model": model, "performance": performance_df, "execution_time": execution_time}
        performances.append(model_training)
    
    return performances


def train_decision_tree(transactions_df, training_date):
    # Define classifier
    classifier = sklearn.tree.DecisionTreeClassifier()
    
    # Set of parameters for which to assess model performances
    parameters = {'clf__max_depth': [5], 'clf__random_state': [0],
                  'clf__class_weight': [{0: w} for w in [0.01, 0.05, 0.1, 0.5, 1]]}
    
    return train_model(transactions_df, training_date, classifier, parameters)


def train_logistic_regression(transactions_df, training_date):
    # Define classifier
    classifier = sklearn.linear_model.LogisticRegression()
    
    # Set of parameters for which to assess model performances
    parameters = {'clf__C': [1], 'clf__random_state': [0],
                  'clf__class_weight': [{0: w} for w in [0.01, 0.05, 0.1, 0.5, 1]]}
    
    return train_model(transactions_df, training_date, classifier, parameters)


def train_decision_tree_with_smote(transactions_df, training_date):
    # Define classifier
    classifier = sklearn.tree.DecisionTreeClassifier()
    
    # Define sampling strategy
    sampler_list = [('sampler', imblearn.over_sampling.SMOTE(random_state=0))]
    
    # Set of parameters for which to assess model performances
    parameters = {'clf__max_depth': [5], 'clf__random_state': [0],
                  'sampler__sampling_strategy': [0.01, 0.05, 0.1, 0.5, 1], 'sampler__random_state': [0]}
    
    return train_model_with_sampler(transactions_df, training_date, classifier, parameters, sampler_list)


def train_random_forest(transactions_df, training_date):
    classifier = sklearn.ensemble.RandomForestClassifier()
    
    parameters = {'clf__max_depth': [20],
                  'clf__n_estimators': [100],
                  'clf__random_state': [0],
                  'clf__n_jobs': [-1]}
    
    return train_model(transactions_df, training_date, classifier, parameters)


def train_xgboost(transactions_df, training_date):
    classifier = xgboost.XGBClassifier()
    
    parameters = {'clf__max_depth': [3],
                  'clf__n_estimators': [50],
                  'clf__learning_rate': [0.3],
                  'clf__random_state': [0],
                  'clf__n_jobs': [-1]}
    
    return train_model(transactions_df, training_date, classifier, parameters)


def train_balanced_random_forest(transactions_df, training_date):
    classifier = imblearn.ensemble.BalancedRandomForestClassifier()
    
    parameters = {'clf__max_depth': [20],
                  'clf__n_estimators': [100],
                  'clf__sampling_strategy': [0.01, 0.05, 0.1, 0.5, 1],
                  'clf__random_state': [0],
                  'clf__n_jobs': [-1]}
    
    return train_model(transactions_df, training_date, classifier, parameters)


def train_weighted_xgboost(transactions_df, training_date):
    classifier = xgboost.XGBClassifier()
    
    parameters = {'clf__max_depth': [3],
                  'clf__n_estimators': [50],
                  'clf__learning_rate': [0.3],
                  'clf__scale_pos_weight': [1, 5, 10, 50, 100],
                  'clf__random_state': [0],
                  'clf__n_jobs': [-1]}
    
    return train_model(transactions_df, training_date, classifier, parameters)
