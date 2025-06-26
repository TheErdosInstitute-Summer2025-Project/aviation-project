## NTSB Table Join

# This script takes separate tables from the NTSB data (2008-2025) and 
# joins them together while respecting the aircraft-level unit analysis 
# found in the supplemental tables. To do this, we construct a "event_key" 
# variable which can be understood as a measure of a single aircraft in an
# event of some nature, rather than a measure at the event-level itself.  

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def read_engines(filename):
    '''Read engines data'''
    engines = pd.read_csv(filename,usecols=['ev_id',
                                            'Aircraft_Key',
                                            'eng_type',
                                            'eng_no'] # Extract relevant columns
                          , dtype = {'ev_id': str} # Avoid dtype warning
                          )

    engines = engines[engines['eng_no']==1].drop(columns='eng_no') # Record information about first engine only
    engines['event_key'] = engines['ev_id'].astype(str) + '_' + engines['Aircraft_Key'].astype(str) # Create event_key

    return engines


def read_aircraft(filename):
    '''Read aircraft data'''
    aircraft = pd.read_csv(filename,usecols=['ev_id',
                                            'Aircraft_Key',
                                            'far_part',
                                            'damage',
                                            'acft_make',
                                            'acft_category',
                                            'homebuilt',
                                            'total_seats',
                                            'num_eng',
                                            'fixed_retractable',
                                            'date_last_insp',
                                            'owner_acft',
                                            'certs_held',    
                                            'oprtng_cert',
                                            'oper_cert',
                                            'second_pilot',
                                            'evacuation',
                                            'rwy_len',
                                            'rwy_width',
                                            'acft_year',
                                            'fuel_on_board',
                                            'unmanned']     # Extract Relevant Columns
                                            , dtype = {'ev_id': str}) # Avoid dtype warning
    aircraft['event_key'] = aircraft['ev_id'].astype(str) + '_' + aircraft['Aircraft_Key'].astype(str) # Create event key 
    
    return aircraft


def read_injuries(filename):
    '''Read injury data
    The .csv file lists separate injury counts for passengers, crew, etc. so this requires some processing 
    '''
    inj_df = pd.read_csv(filename
                         , dtype = {'ev_id': str}) # Avoid dtype warning

    ## Remove NaN values to ensure valid summation when grouping by 'event_key'
    inj_df = inj_df[~inj_df['inj_person_count'].isnull()]

    ## Create event keys
    inj_df['event_key'] = inj_df['ev_id'].astype(str) + '_' + inj_df['Aircraft_Key'].astype(str)

    ## All possible injury levels
    injury_levels = ['NONE', 'MINR', 'FATL', 'SERS']

    ## Count the number of people at each injury level
    injuries = (
        inj_df[inj_df['injury_level'].isin(injury_levels)].pivot_table(index='event_key',
                                                                       columns='injury_level',
                                                                       values='inj_person_count',
                                                                       aggfunc='sum',
                                                                       fill_value=0).reset_index()
    )

    ## Rename columns
    injuries.columns.name = None
    injuries = injuries.rename(columns={
        'FATL': 'acft_fatal_count',
        'SERS': 'acft_serious_count',
        'MINR': 'acft_minor_count',
        'NONE': 'acft_none_count'
    })

    ## Add total and injured counts
    injuries['acft_total_person_count'] = (
        injuries[['acft_fatal_count', 'acft_serious_count', 'acft_minor_count', 'acft_none_count']].sum(axis=1)
    )
    injuries['acft_injured_person_count'] = (
        injuries[['acft_fatal_count', 'acft_serious_count', 'acft_minor_count']].sum(axis=1)
    )

    # Reconstruct ev_id and Aircraft_key
    injuries['ev_id'] = injuries['event_key'].str.split('_').str[0]
    injuries['Aircraft_Key'] = injuries['event_key'].str.split('_').str[1].astype(int)

    return injuries


def read_from_carol(filename):
    '''Read phase, longitude, and latitude data from CAROL, NTSB query tool.'''
    carol_data = pd.read_csv(filename, usecols=['NtsbNo',
                                           'BroadPhaseofFlight', 
                                           'Latitude',
                                           'Longitude'])

    # Rename to match corresponding column name in events
    carol_data.rename({'NtsbNo':'ntsb_no', 'Latitude':'latitude', 'Longitude':'longitude'}, axis=1, inplace=True)
    return carol_data


def read_events(events_filename):
    '''
    Read event table.
    '''
    events = pd.read_csv(events_filename,
                         usecols=['ev_id',
                                'ntsb_no',
                                'ev_country',
                                'ev_type', 
                                'ev_year',
                                'ev_month',
                                'ev_date',
                                # 'latitude',
                                # 'longitude', ## latitude and longitude are obtained from CAROL NTSB query
                                'apt_dist',
                                'altimeter',
                                'ev_time',
                                'light_cond',
                                'wind_vel_kts',
                                'gust_kts',
                                'ev_highest_injury',
                                'inj_tot_f',
                                'inj_tot_m',
                                'inj_tot_n',
                                'inj_tot_s',
                                'inj_tot_t',
                                'on_ground_collision',
                                'inj_f_grnd',
                                'inj_m_grnd',
                                'inj_s_grnd',
                                ] # Extract relevant columns
                         , dtype = {'ev_id': str}) # Avoid dtype warning

    # Rename injury columns to distinguish event-level injury counts from aircraft-level injury counts
    new_names = {
        'inj_tot_f': 'ev_inj_tot_f',
        'inj_tot_m': 'ev_inj_tot_m',
        'inj_tot_n': 'ev_inj_tot_n',
        'inj_tot_s': 'ev_inj_tot_s',
        'inj_tot_t': 'ev_inj_tot_t',
    }
    events.rename(columns=new_names, inplace=True)
    return events.copy()


def merge_data(engines, aircraft, injuries, events, carol_data):
    """
    Merge NTSB aircraft-level tables (engines, aircraft, injuries) and data from CAROL NTSB query (Latitude, Longitude, phase) with event-level data into a unified DataFrame.
    """

    tables = pd.merge(engines,aircraft,on=['event_key','ev_id','Aircraft_Key'],how='left') # Join on all 3 to avoid _x and _y duplicate columns.  Also ensures correct specification.
    tables = pd.merge(tables,injuries,on=['event_key','ev_id','Aircraft_Key'],how='left') # Join "injuries" data as well.
    tables['Aircraft_ID'] = tables.groupby('ev_id').cumcount() + 1 # Some events start at 2 - this line creates a new Aircraft Key that is uniform across all coding schemes.
    
    tables['event_key'] = tables['ev_id'].astype(str) + '_' + tables['Aircraft_ID'].astype(str) # Resets the "event_key" variable to match our adjusted aircraft ID

    aircraft_counts = pd.DataFrame(tables.groupby('ev_id')['Aircraft_ID'].count()).reset_index() # Counts how many unique values of "Aircraft_ID" per event - need this to tell "events" set where to duplicate
    aircraft_counts.rename(columns={'Aircraft_ID':'aircraft_count'},inplace=True)
    aircraft_counts['aircraft_count'] = aircraft_counts['aircraft_count'].fillna(1) # Fill in missing - if no aircraft info, assume 1 (true for most)

    data = pd.merge(events, carol_data, how='left', on=['ntsb_no'])

    df = pd.merge(data,aircraft_counts,on='ev_id',how='left') # Concatenates "aircraft count" var to dataset, will indicate how many replicate rows to generate
    df['aircraft_count'] = df['aircraft_count'].fillna(1) # Some events were not in any of the 3 supplemental sets - set aircraft count as 1 for these

    df_repeated = df.loc[df.index.repeat(df['aircraft_count'])].copy() # Creates repeated rows based on # of aircraft (indicated in column we created)

    df_repeated['Aircraft_ID'] = df_repeated.groupby('ev_id').cumcount() + 1 # Re-creates aircraft ID in the event data so we can join the individual aircraft from the tables dataset
    df_repeated['event_key'] = df_repeated['ev_id'].astype(str) + '_' + df_repeated['Aircraft_ID'].astype(str) # Re-creates "event_key" for same reason.
    

    merged = df_repeated.merge(tables, on=['event_key','ev_id','Aircraft_ID'], how='left') # Joins "event" dataset with supplemental sets.

    return merged


def stratified_group_split(data, strat_col, group_col):
    """
        Perform a stratified group split to divide data into training, validation, and test sets. The distribution of the stratification variable (e.g. strat_col='damage') is preserved across splits.
        
        The function uses `StratifiedGroupKFold` with 5 folds, treating the first fold as the test set and the second fold as the validation set. The resulting split consists of 60% training, 20% validation, and 20% test data.
        
    Parameters:
    -----
    data : pd.DataFrame
    The input dataset to be split.

    strat_col : str
        Column name on which to stratify the data (e.g., 'damage').

    group_col : str
        Column name used to group data points (e.g., 'ev_id') so that all rows
        within the same group are kept together in a single split.
    -----
    """
    # Validation and test will both be 20% of data, hence 5 splits
    strat_kfold = StratifiedGroupKFold(n_splits = 5, shuffle=True, random_state=412)  

    # Grouped by 'ev_id', stratified by 'damage'
    splits = strat_kfold.split(data, data[strat_col], data[group_col].astype(str))   

    # data_test  <- test set from the first fold (20% of data)
    # data_val   <- test set from the second fold (20% of data)
    # data_train <- remaining data (60%)
    for i, (train_index, test_index) in enumerate(splits):
        if i==0:
            data_test = data.iloc[test_index]
            data_train = data.iloc[train_index]
        elif i==1:
            data_val = data.iloc[test_index]
            data_train.drop(test_index)
            break

    return data_train, data_val, data_test


if __name__ == '__main__':
    ### 1. Read data
    engines = read_engines('data/ntsb_raw/ntsb_engines.csv')
    aircraft = read_aircraft('data/ntsb_raw/ntsb_aircraft.csv')
    injuries = read_injuries('data/ntsb_raw/ntsb_injuries.csv')
    carol_data = read_from_carol('data/ntsb_raw/ntsb_from_carol.csv')
    events = read_events('data/ntsb_raw/ntsb_events.csv')
    
    ### 2. Merge data into a single DataFrame
    merged = merge_data(engines, aircraft, injuries, events,carol_data)
    merged.to_csv('data/ntsb_processed/master.csv', index=False)

    ### 3. Restrict data to the scope of our problem:
    #      - Accidents
    #      - In the USA
    merged = merged[(merged['ev_country']=="USA") & (merged['ev_type']=="ACC")]
    merged.loc[merged['damage'].isna(),'damage'] = 'UNK'

    # Drop recent data with many missing entries
    data = merged[merged['ev_year'] < 2022]

    # Drop data with missing / corrupted values for crucial variable we aren't imputing
    data = data[~data['date_last_insp'].isna()]

    data = data.reset_index().drop(columns=['index'])

    ### 4. Train / validation / test split
    #       - Stratified by damage level
    #       - Grouped so that multiple aircraft in a single crash are placed in the same split
    data_train, data_val, data_test = stratified_group_split(data, strat_col='damage', group_col='ev_id')

    ### 5. Write data to files 
    data_train.to_csv('data/ntsb_processed/master_train.csv', index=False)
    data_val.to_csv('data/ntsb_processed/master_val.csv', index=False)
    data_test.to_csv('data/ntsb_processed/master_test.csv', index=False)
