## NTSB Data Cleaning

# This script cleans the training data in the following ways:
# - Drops rows with unknown injury counts
# - Drops sparse columns (> 20% missing)
# - Drops columns that have been processed and are no longer needed
# - Fills missing values
# - Reduces categories
#     - Manually combine categories for select variables
#     - Remove categories with < 1% frequency
# - Creates dummy variables
# - Feature engineering
#     - Calculate the number of people onboard and proportions of each injury category

import pandas as pd
import numpy as np


def drop_rows_missing_injuries_or_location(data):
    '''Drop rows of planes with missing injury data or latitude/longitude'''

    # Drop rows with unknown event injury totals (~20 rows in the training data)
    data = data[~data['ev_inj_tot_t'].isna()]

    # Drop planes in multi-aircraft accidents with unknown injury counts
    acft_missing_counts_keys = data.loc[data['acft_total_person_count'].isna(), 'event_key']
    for ev in acft_missing_counts_keys:
        key = ev[:-1]+'2' # suffix 2 indicates a multi-aircraft accident
        if key in data['event_key'].values:
            # Drop event with >= 1 aircraft with unknowable injury counts
            data = data[~(data['event_key'].str.startswith(ev[:-1]))] 

    # Drop rows with unknown latitude / longitude
    data = data[~data['latitude'].isna()]

    return data


def drop_sparse_columns(data, threshold):
    '''
    Drops columns from data where the train data does not contain at least a given proportion of non-empty entries
    
    Inputs
        data: dictionary {'train':data_tr, 'val':data_val, 'test':data_te} with train/validation/test datasets
        threshold: float in [0,1], all columns with less than this proportion of non-empty entries are dropped
    Outputs
        data
    '''
    tr = 'train'
    for col in data[tr].columns:
        # proportion of na entries in the column, in the training set
        prop_na = data[tr][col].isna().sum() / len(data[tr])
        
        # drop col from each dataset if the column is too sparse in the training set
        if prop_na > 1 - threshold:
            for key in data.keys():
                data[key].drop(columns=col, inplace=True)

    return data
    
    
def remove_already_processed_and_almost_all_same_columns(data):
    # Already processed
    data.drop(columns=[#'Aircraft_Key', 
                    'ev_id', 
                    #'finding_description'    # no longer in data
                    # 'total_seats' # No longer need
                    ], inplace=True)

    #(Almost) all rows have same value
    data.drop(columns=['certs_held', 'unmanned'], inplace=True)
    return data


def compute_injury_counts(data):
    '''Calculate missing aircraft-level injury data from event-level injury data'''

    count_cols = {
        'acft_fatal_count': 'ev_inj_tot_f',
        'acft_minor_count': 'ev_inj_tot_m',
        'acft_none_count': 'ev_inj_tot_n',
        'acft_serious_count': 'ev_inj_tot_s'
    }

    for missing_col, present_col in count_cols.items():
        data.loc[data[missing_col].isna(), missing_col] = data.loc[data[missing_col].isna(), present_col]

    data['acft_total_person_count'] = data[count_cols.keys()].sum(axis=1)
    data['acft_injured_person_count'] = data['acft_total_person_count'] - data['acft_none_count']

    return data


def compute_injury_proportions(data):
    '''Calculate the proportion of people onboard in each injury category'''

    data['acft_prop_inj_n'] =  data['acft_none_count'] / data['acft_total_person_count']
    data['acft_prop_inj_m'] =  data['acft_minor_count'] / data['acft_total_person_count']
    data['acft_prop_inj_s'] =  data['acft_serious_count'] / data['acft_total_person_count']
    data['acft_prop_inj_f'] =  data['acft_fatal_count'] / data['acft_total_person_count']
    return data


def impute_engines(data):
    '''Impute value 1 into 'num_eng' if there are at most 15 people onboard, impute 2 otherwise'''
    data.loc[(data['num_eng'].isna())& (data['acft_total_person_count']<=15), 'num_eng'] = 1
    data.loc[(data['num_eng'].isna())& (data['acft_total_person_count']>15), 'num_eng'] = 2

    return data


def combine_phase_categories(data):
    '''Combine categories for BroadPhaseofFlight into Ground/Takeoff/Air/Landing'''

    phase_dict = {
        'Landing': 'Landing',
        'Enroute': 'Air',
        'Maneuvering': 'Air',
        'Takeoff': 'Takeoff',
        'Approach': 'Landing', # or 'Air'
        'Initial Climb': 'Takeoff',
        'Taxi': 'Ground',
        'Standing': 'Ground',
        'Emergency Descent': 'Air', # or 'Landing'
        'Uncontrolled Descent': 'Air', # or 'Landing'
        'Pushback/Tow': 'Ground',
        'Post-Impact': 'Ground',
        'Unknown': 'Unknown',
        'other/unknown': 'Unknown',
        np.nan: 'Unknown'
    }

    data['BroadPhaseofFlight'] = data['BroadPhaseofFlight'].apply(lambda x: phase_dict[x])
    return data


def format_aircraft_make_spelling(data):
    '''Combine varied spellings of the same aircraft make'''

    data['acft_make'] = data['acft_make'].fillna('other/unknown')
    data['acft_make'] = data['acft_make'].str.lower()

    # list of makes representing more than 1% of the data 
    # (except 'bell', which needs to be treated separately below)
    common_makes = ['cessna','piper','boeing','mooney','robinson helicopter']

    # anything starting with the name of the make is equivalent
    for make in common_makes:
        data.loc[data['acft_make'].str.startswith(make),'acft_make'] = make

    # ad-hoc processing of other common makes that don't work with the for loop above
    data.loc[data['acft_make']=='robinson', 'acft_make'] = 'robinson helicopter'
    data.loc[data['acft_make'].str.startswith('bell helicopter'), 'acft_make'] = 'bell'

    return data


def reduce_categories_fill_na(data, columns, threshold):
    '''
    For each of the specified columns, find the values that occur with frequency lower than the threshold,
    and replace these values and missing values by 'other/unknown'.
    This is only intended for categorical variables
    
    Inputs
        data: dictionary {'train':data_tr, 'val':data_val, 'test':data_te} with train/validation/test datasets
        columns: list of column names to simplify
        threshold: float in [0,1], frequency threshold for removing 
    Outputs
        data
    '''
    tr='train'
    freq_thresh = threshold * len(data[tr])

    for col in columns:
        # Find frequent values in col of training data
        counts = data[tr][col].value_counts()
        freq_vals = [str(i) for i in counts.index if counts[i] >= freq_thresh]
        
        # Replace infrequent values
        for key in data.keys():
            data[key].loc[~data[key][col].isin(freq_vals), col] = 'other/unknown'    

    return data


def compute_days_since_last_inspection(data):
    '''
    Compute the number of days since the last aircraft inspection.
    Converts date columns to datetime and calculates the difference.

    Outputs:
        data: pandas DataFrame with new column 'days_since_insp' 
              and old columns 'date_last_insp' and 'ev_date' dropped
    '''

    # Use proper datetime format to avoid parsing warnings
    datetime_format = '%m/%d/%y %H:%M:%S'

    data['insp_date'] = pd.to_datetime(
        data['date_last_insp'].replace('other/unknown', np.nan),
        format=datetime_format, errors='coerce'
    )

    data['event_date'] = pd.to_datetime(
        data['ev_date'],
        format=datetime_format, errors='coerce'
    )

    # Calculate days between inspection and event
    data['days_since_insp'] = (data['event_date'] - data['insp_date']).dt.days

    # Replace negative values (future inspections) with 0
    data.loc[data['days_since_insp'] < 0, 'days_since_insp'] = 0

    # Drop original date columns
    data = data.drop(columns=['date_last_insp', 'ev_date'])

    return data
    
    
#########################

if __name__ == '__main__':
    ### Create dictionary to store train/validation/test data
    # This will be useful for applying the same function to each dataset in a loop
    tr = 'train'
    val = 'val'
    te = 'test'
    data = {tr:None, val:None, te:None}

    ### Read the three data sets
    for key in data.keys():
        data[key] = pd.read_csv('data/ntsb_processed/master_'+key+'.csv')

    ### Select categorical features to clean
    categorical_features = ['light_cond', 'BroadPhaseofFlight', 'eng_type', 'far_part', 
                            'acft_make', 'acft_category','homebuilt', 'fixed_retractable', 
                            'second_pilot']
    # Note: intentionally omitted 'ntsb_no', 'ev_highest_injury', 'Aircraft_ID', 'event_key', 
    #       'damage', 'ev_id'
    # TODO: update this if needed

    ### Clean data
    # This loop applies the same functions to each of the training, validation, and test sets
    for key in data.keys():
        data[key] = drop_rows_missing_injuries_or_location(data[key])
        data[key] = remove_already_processed_and_almost_all_same_columns(data[key])
        data[key] = compute_injury_counts(data[key])
        data[key] = compute_injury_proportions(data[key])
        data[key] = impute_engines(data[key])
        data[key] = combine_phase_categories(data[key])
        data[key] = format_aircraft_make_spelling(data[key])
        data[key] = compute_days_since_last_inspection(data[key])

    ### Data cleaning that depends on frequency of NAs / values
    # All three sets are cleaned together because the modifications must rely only on the training data
    # and not on the validation or test data
    data = drop_sparse_columns(data, 0.8)
    data = reduce_categories_fill_na(data, categorical_features, 0.01)

    ### Create dummies + write to file
    for key in data.keys():
        data[key] = pd.get_dummies(data[key], columns=categorical_features)
        data[key].to_csv('data/ntsb_processed/ntsb_'+key+'_cleaned.csv', index=False)