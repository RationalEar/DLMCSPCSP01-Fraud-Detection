import datetime

import imblearn
import numpy as np
import pandas as pd
import sklearn


def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7, delta_delay=7, delta_test=7,
                       sampling_ratio=1.0,
                       random_state=0):
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                               (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                   days=delta_train))]
    
    # Get the test set data
    test_df = []
    
    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed
    
    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)
    
    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()
    
    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                      delta_train + delta_delay +
                                      day]
        
        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded
        # customers
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                   delta_train +
                                                   day - 1]
        
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        
        test_df.append(test_df_day)
    
    test_df = pd.concat(test_df)
    
    # If subsample
    if sampling_ratio < 1:
        train_df_frauds = train_df[train_df.TX_FRAUD == 1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine = train_df[train_df.TX_FRAUD == 0].sample(frac=sampling_ratio, random_state=random_state)
        train_df = pd.concat([train_df_frauds, train_df_genuine])
    
    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')
    
    return train_df, test_df


def prequential_split(transactions_df,
                      start_date_training,
                      n_folds=4,
                      delta_train=7,
                      delta_delay=7,
                      delta_assessment=7):
    prequential_split_indices = []
    
    # For each fold
    for fold in range(n_folds):
        # Shift back start date for training by the fold index times the assessment period (delta_assessment)
        # (See Fig. 5)
        start_date_training_fold = start_date_training - datetime.timedelta(days=fold * delta_assessment)
        
        # Get the training and test (assessment) sets
        (train_df, test_df) = get_train_test_set(transactions_df,
                                                 start_date_training=start_date_training_fold,
                                                 delta_train=delta_train, delta_delay=delta_delay,
                                                 delta_test=delta_assessment)
        
        # Get the indices from the two sets, and add them to the list of prequential splits
        indices_train = list(train_df.index)
        indices_test = list(test_df.index)
        
        prequential_split_indices.append((indices_train, indices_test))
    
    return prequential_split_indices


def prequential_grid_search(transactions_df,
                            classifier,
                            input_features, output_feature,
                            parameters, scoring,
                            start_date_training,
                            n_folds=4,
                            expe_type='Test',
                            delta_train=7,
                            delta_delay=7,
                            delta_assessment=7,
                            performance_metrics_list_grid=('roc_auc',),
                            performance_metrics_list=('AUC ROC',),
                            n_jobs=-1):
    estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', classifier)]
    pipe = sklearn.pipeline.Pipeline(estimators)
    
    prequential_split_indices = prequential_split(transactions_df,
                                                  start_date_training=start_date_training,
                                                  n_folds=n_folds,
                                                  delta_train=delta_train,
                                                  delta_delay=delta_delay,
                                                  delta_assessment=delta_assessment)
    
    grid_search = sklearn.model_selection.GridSearchCV(pipe, parameters, scoring=scoring, cv=prequential_split_indices,
                                                       refit='card_precision@100', n_jobs=n_jobs)
    
    X = transactions_df[input_features]
    y = transactions_df[output_feature]
    
    grid_search.fit(X, y)
    
    performances_df = pd.DataFrame()
    
    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i] + ' ' + expe_type] = grid_search.cv_results_[
            'mean_test_' + performance_metrics_list_grid[i]]
        performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = grid_search.cv_results_[
            'std_test_' + performance_metrics_list_grid[i]]
    
    performances_df['Parameters'] = grid_search.cv_results_['params']
    performances_df['Execution time'] = grid_search.cv_results_['mean_fit_time']
    
    return performances_df, grid_search.best_estimator_


def model_selection_wrapper(transactions_df,
                            classifier,
                            input_features, output_feature,
                            parameters,
                            scoring,
                            start_date_training_for_valid,
                            start_date_training_for_test,
                            n_folds=4,
                            delta_train=7,
                            delta_delay=7,
                            delta_assessment=7,
                            performance_metrics_list_grid=('roc_auc',),
                            performance_metrics_list=('AUC ROC',),
                            n_jobs=-1):
    # Get performances on the validation set using prequential validation
    performances_df_validation, trained_model = prequential_grid_search(transactions_df, classifier,
                                                                        input_features, output_feature,
                                                                        parameters, scoring,
                                                                        start_date_training=start_date_training_for_valid,
                                                                        n_folds=n_folds,
                                                                        expe_type='Validation',
                                                                        delta_train=delta_train,
                                                                        delta_delay=delta_delay,
                                                                        delta_assessment=delta_assessment,
                                                                        performance_metrics_list_grid=performance_metrics_list_grid,
                                                                        performance_metrics_list=performance_metrics_list,
                                                                        n_jobs=n_jobs)
    
    # Get performances on the test set using prequential validation
    performances_df_test, tm2 = prequential_grid_search(transactions_df, classifier,
                                                        input_features, output_feature,
                                                        parameters, scoring,
                                                        start_date_training=start_date_training_for_test,
                                                        n_folds=n_folds,
                                                        expe_type='Test',
                                                        delta_train=delta_train,
                                                        delta_delay=delta_delay,
                                                        delta_assessment=delta_assessment,
                                                        performance_metrics_list_grid=performance_metrics_list_grid,
                                                        performance_metrics_list=performance_metrics_list,
                                                        n_jobs=n_jobs)
    
    # Bind the two resulting DataFrames
    performances_df_validation.drop(columns=['Parameters', 'Execution time'], inplace=True)
    performances_df = pd.concat([performances_df_test, performances_df_validation], axis=1)
    
    # And return as a single DataFrame
    return performances_df, trained_model


def model_selection_wrapper_with_sampler(transactions_df,
                                         classifier,
                                         sampler_list,
                                         input_features, output_feature,
                                         parameters,
                                         scoring,
                                         start_date_training_for_valid,
                                         start_date_training_for_test,
                                         n_folds=4,
                                         delta_train=7,
                                         delta_delay=7,
                                         delta_assessment=7,
                                         performance_metrics_list_grid=('roc_auc',),
                                         performance_metrics_list=('AUC ROC',),
                                         n_jobs=-1):
    # Get performances on the validation set using prequential validation
    performances_df_validation, trained_model = prequential_grid_search_with_sampler(transactions_df, classifier,
                                                                                     sampler_list,
                                                                                     input_features, output_feature,
                                                                                     parameters, scoring,
                                                                                     start_date_training=start_date_training_for_valid,
                                                                                     n_folds=n_folds,
                                                                                     expe_type='Validation',
                                                                                     delta_train=delta_train,
                                                                                     delta_delay=delta_delay,
                                                                                     delta_assessment=delta_assessment,
                                                                                     performance_metrics_list_grid=performance_metrics_list_grid,
                                                                                     performance_metrics_list=performance_metrics_list,
                                                                                     n_jobs=n_jobs)
    
    # Get performances on the test set using prequential validation
    performances_df_test, tm = prequential_grid_search_with_sampler(transactions_df, classifier, sampler_list,
                                                                    input_features, output_feature,
                                                                    parameters, scoring,
                                                                    start_date_training=start_date_training_for_test,
                                                                    n_folds=n_folds,
                                                                    expe_type='Test',
                                                                    delta_train=delta_train,
                                                                    delta_delay=delta_delay,
                                                                    delta_assessment=delta_assessment,
                                                                    performance_metrics_list_grid=performance_metrics_list_grid,
                                                                    performance_metrics_list=performance_metrics_list,
                                                                    n_jobs=n_jobs)
    
    # Bind the two resulting DataFrames
    performances_df_validation.drop(columns=['Parameters', 'Execution time'], inplace=True)
    performances_df = pd.concat([performances_df_test, performances_df_validation], axis=1)
    
    # And return as a single DataFrame
    return performances_df, trained_model


def prequential_grid_search_with_sampler(transactions_df,
                                         classifier, sampler_list,
                                         input_features, output_feature,
                                         parameters, scoring,
                                         start_date_training,
                                         n_folds=4,
                                         expe_type='Test',
                                         delta_train=7,
                                         delta_delay=7,
                                         delta_assessment=7,
                                         performance_metrics_list_grid=('roc_auc',),
                                         performance_metrics_list=('AUC ROC',),
                                         n_jobs=-1):
    estimators = sampler_list.copy()
    estimators.extend([('clf', classifier)])
    
    pipe = imblearn.pipeline.Pipeline(estimators)
    
    prequential_split_indices = prequential_split(transactions_df,
                                                  start_date_training=start_date_training,
                                                  n_folds=n_folds,
                                                  delta_train=delta_train,
                                                  delta_delay=delta_delay,
                                                  delta_assessment=delta_assessment)
    
    grid_search = sklearn.model_selection.GridSearchCV(pipe, parameters, scoring=scoring, cv=prequential_split_indices,
                                                       refit='average_precision', n_jobs=n_jobs)
    
    X = transactions_df[input_features]
    y = transactions_df[output_feature]
    
    grid_search.fit(X, y)
    
    performances_df = pd.DataFrame()
    
    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i] + ' ' + expe_type] = grid_search.cv_results_[
            'mean_test_' + performance_metrics_list_grid[i]]
        performances_df[performance_metrics_list[i] + ' ' + expe_type + ' Std'] = grid_search.cv_results_[
            'std_test_' + performance_metrics_list_grid[i]]
    
    performances_df['Parameters'] = grid_search.cv_results_['params']
    performances_df['Execution time'] = grid_search.cv_results_['mean_fit_time']
    
    return performances_df, grid_search.best_estimator_


def card_precision_top_k_day(df_day, top_k):
    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)
    
    # Get the top k most suspicious cards
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)
    
    # Compute precision top k
    _card_precision_top_k = len(list_detected_compromised_cards) / top_k
    
    return list_detected_compromised_cards, _card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    # Sort days by increasing order
    list_days = list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()
    
    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []
    
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []
    
    # For each day, compute precision top k
    for day in list_days:
        
        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]
        
        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False]
        
        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))
        
        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day, top_k)
        
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        
        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
    
    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    
    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k


def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):
    # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
    # (indices of the y_true vector)
    predictions_df = transactions_df.iloc[y_true.index.values].copy()
    predictions_df['predictions'] = y_pred
    
    # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
    nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k = card_precision_top_k(
        predictions_df, top_k)
    
    # Return the mean_card_precision_top_k
    return mean_card_precision_top_k
