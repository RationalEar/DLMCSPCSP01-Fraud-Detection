# Load a set of pickle files, put them together in a single DataFrame, and order them by time
# It takes as input the folder DIR_INPUT where the files are stored, and the BEGIN_DATE and END_DATE
import datetime
import os

import pandas as pd


def read_from_files(dir_input, begin_date, end_date):
    files = [os.path.join(dir_input, f) for f in os.listdir(dir_input) if
             begin_date + '.pkl' <= f <= end_date + '.pkl']
    
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    
    df_final = df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True, inplace=True)
    df_final = df_final.replace([-1], 0)
    
    return df_final


def get_stats(transactions_df):
    # Number of transactions per day
    nb_tx_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
    # Number of fraudulent transactions per day
    nb_fraud_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
    # Number of fraudulent cards per day
    nb_fraudcard_per_day = transactions_df[transactions_df['TX_FRAUD'] > 0].groupby(
        ['TX_TIME_DAYS']).CUSTOMER_ID.nunique()
    
    return nb_tx_per_day, nb_fraud_per_day, nb_fraudcard_per_day


# Compute the number of transactions per day, fraudulent transactions per day and fraudulent cards per day

def get_tx_stats(transactions_df, start_date_df):
    # Number of transactions per day
    nb_tx_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['CUSTOMER_ID'].count()
    # Number of fraudulent transactions per day
    nb_fraudulent_transactions_per_day = transactions_df.groupby(['TX_TIME_DAYS'])['TX_FRAUD'].sum()
    # Number of compromised cards per day
    nb_compromised_cards_per_day = transactions_df[transactions_df['TX_FRAUD'] == 1].groupby(
        ['TX_TIME_DAYS']).CUSTOMER_ID.nunique()
    
    tx_stats = pd.DataFrame({"nb_tx_per_day": nb_tx_per_day,
                             "nb_fraudulent_transactions_per_day": nb_fraudulent_transactions_per_day,
                             "nb_compromised_cards_per_day": nb_compromised_cards_per_day})
    
    tx_stats = tx_stats.reset_index()
    
    start_date = datetime.datetime.strptime(start_date_df, "%Y-%m-%d")
    tx_date = start_date + tx_stats['TX_TIME_DAYS'].apply(datetime.timedelta)
    
    tx_stats['tx_date'] = tx_date
    
    return tx_stats

