import datetime
import os

from data_simulator.generators import *
from helpers.common_functions import read_from_files

DATE_FORMAT = "%Y-%m-%d"


def generate_dataset_and_save(n_customers, n_terminals, nb_days, start_date, output_dir, overwrite=False):
    start_date_obj = datetime.datetime.strptime(start_date, DATE_FORMAT)
    filename_output = start_date_obj.strftime(DATE_FORMAT) + '.pkl'
    
    if not overwrite and os.path.exists(output_dir + filename_output):
        print('Data already exists. Skipping generation.')
        return
    
    (customer_profiles_table, terminal_profiles_table, transactions_df) = generate_dataset(
        n_customers=n_customers,
        n_terminals=n_terminals,
        nb_days=nb_days,
        start_date=start_date,
        r=5)
    print(f"Generated {len(transactions_df)} transactions")
    transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        
        date = start_date_obj + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d") + '.pkl'
        
        # Protocol=4 required for Google Colab
        transactions_day.to_pickle(output_dir + filename_output, protocol=4)
    
    print("Generated data saved in", os.path.abspath(output_dir))


def transform_dataset_and_save(start_date, end_date, input_dir, output_dir, overwrite=False):
    start_date_obj = datetime.datetime.strptime(start_date, DATE_FORMAT)
    filename_output = start_date_obj.strftime(DATE_FORMAT) + '.pkl'
    
    if not overwrite and os.path.exists(output_dir + filename_output):
        print('Transformed data already exists. Skipping transformation.')
        return
    
    transactions_df = read_from_files(input_dir, start_date, end_date)
    # Apply date and time transformations
    start_time = time.time()
    transactions_df['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
    transactions_df['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
    print("Time to apply date and time transformations: {0:.2}s".format(time.time() - start_time))
    
    # Customer ID transformation
    start_time = time.time()
    transactions_df = transactions_df.groupby('CUSTOMER_ID').apply(
        lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1, 7, 30]))
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    print("Time to apply Customer ID transformations: {0:.2}s".format(time.time() - start_time))
    
    # Terminal ID transformation
    start_time = time.time()
    transactions_df = transactions_df.groupby('TERMINAL_ID').apply(
        lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1, 7, 30],
                                                feature="TERMINAL_ID"))
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    print("Time to apply Terminal ID transformations: {0:.2}s".format(time.time() - start_time))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        
        date = start_date_obj + datetime.timedelta(days=day)
        filename_output = date.strftime(DATE_FORMAT) + '.pkl'
        
        transactions_day.to_pickle(output_dir + filename_output, protocol=4)
    
    print("Transformed data saved in", os.path.abspath(output_dir))
    
