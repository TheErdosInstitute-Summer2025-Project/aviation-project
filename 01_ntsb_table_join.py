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
                                            'eng_no']) # Extract relevant columns

    engines = engines[engines['eng_no']==1].drop(columns='eng_no') # Record information about first engine only
    engines['event_key'] = engines['ev_id'].astype(str) + '_' + engines['Aircraft_Key'].astype(str) # Create event_key

    return engines


def read_aircraft(filename):
    '''Read aircraft data'''
    aircraft = pd.read_csv(filename, usecols=['ev_id',
                                              'Aircraft_Key',
                                              'acft_year',
                                              'acft_make',
                                              'acft_category',
                                              'homebuilt',
                                              'num_eng',
                                              'fixed_retractable',
                                              'date_last_insp',
                                              'fuel_on_board',
                                              'far_part',
                                              'unmanned',
                                              'second_pilot',
                                              'certs_held',  
                                              'oprtng_cert',
                                              'oper_cert',
                                              'rwy_len',
                                              'rwy_width',
                                              'damage',
                                              'evacuation']) # Extract relevant columns
    aircraft['event_key'] = aircraft['ev_id'].astype(str) + '_' + aircraft['Aircraft_Key'].astype(str) # Create event key 
    
    return aircraft


def read_injuries(filename):
    '''Read injury data
    The .csv file lists separate injury counts for passengers, crew, etc. so this requires some processing 
    '''
    inj_df = pd.read_csv(filename)

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


def read_phase(filename):
    '''Read phase data'''
    phase = pd.read_csv(filename, usecols=['NtsbNo','BroadPhaseofFlight'])

    # Rename to match corresponding column name in events
    phase.rename({'NtsbNo':'ntsb_no'}, axis=1, inplace=True)

    return phase


def read_events(events_filename, phase_filename):
    events = pd.read_csv(events_filename,usecols=['ev_id',
                                                            'ntsb_no',
                                                            'ev_country',
                                                            'ev_type',
                                                            'ev_year',
                                                            'ev_month',
                                                            'ev_date',
                                                            'latitude',
                                                            'longitude',
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
                                                            ]) # Extract relevant columns

    # Rename injury columns to distinguish event-level injury counts from aircraft-level injury counts
    new_names = {
        'inj_tot_f': 'ev_inj_tot_f',
        'inj_tot_m': 'ev_inj_tot_m',
        'inj_tot_n': 'ev_inj_tot_n',
        'inj_tot_s': 'ev_inj_tot_s',
        'inj_tot_t': 'ev_inj_tot_t',
    }
    events.rename(columns=new_names, inplace=True)

    return events


def merge_data(engines, aircraft, injuries, events, phase):
    #TODO finish cleaning this up
    '''DOCSTRING'''


    ### 1. Merge engine, aircraft, and injuries data
    # Joining on 'event_key','ev_id', and'Aircraft_Key' avoids _x and _y duplicate columns 
    # and ensures correct specification.
    tables = pd.merge(engines,aircraft,on=['event_key','ev_id','Aircraft_Key'],how='left') 
    tables = pd.merge(tables,injuries,on=['event_key','ev_id','Aircraft_Key'],how='left')

    # Creates a new Aircraft Key that is uniform across all coding schemes. (Some events start at 2)
    tables['Aircraft_ID'] = tables.groupby('ev_id').cumcount() + 1 
    
    # Resets the "event_key" variable to match our adjusted aircraft ID
    tables['event_key'] = tables['ev_id'].astype(str) + '_' + tables['Aircraft_ID'].astype(str) 

    # Counts how many unique values of "Aircraft_ID" per event - need this to tell "events" set where to duplicate
    aircraft_counts = pd.DataFrame(tables.groupby('ev_id')['Aircraft_ID'].count()).reset_index() 
    aircraft_counts.rename(columns={'Aircraft_ID':'aircraft_count'},inplace=True)

    # Fill in missing - if no aircraft info, assume 1 (true for most)
    aircraft_counts['aircraft_count'] = aircraft_counts['aircraft_count'].fillna(1) 

    ### 2. Merge events and phase data
    data= pd.merge(events, phase, how='left', on=['ntsb_no']) # Move this into merge_data

    # Concatenates "aircraft count" var to dataset, will indicate how many replicate rows to generate
    df = pd.merge(data,aircraft_counts,on='ev_id',how='left')

    # Some events were not in any of the 3 supplemental sets - set aircraft count as 1 for these 
    df['aircraft_count'] = df['aircraft_count'].fillna(1) 

    # Creates repeated rows based on # of aircraft (indicated in column we created)
    df_repeated = df.loc[df.index.repeat(df['aircraft_count'])].copy() 

    # Re-creates Aircraft_ID and event_key in the event data so we can join the individual aircraft from the tables dataset
    df_repeated['Aircraft_ID'] = df_repeated.groupby('ev_id').cumcount() + 1 
    df_repeated['event_key'] = df_repeated['ev_id'].astype(str) + '_' + df_repeated['Aircraft_ID'].astype(str)

    # Joins "event" dataset with supplemental sets.
    merged = df_repeated.merge(tables, on=['event_key','ev_id','Aircraft_ID'], how='left') 

    return merged


def stratified_group_split(data, strat_col, group_col):

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
    injuries = read_injuries('data/ntsb_raw/ntsb_aircraft.csv')
    phase = read_phase('data/ntsb_raw/ntsb_from_carol.csv')
    events = read_events('data/ntsb_raw/ntsb_aircraft.csv')

    ### 2. Merge data into a single DataFrame
    merged = merge_data(engines, aircraft, injuries, phase, events)
    
    ### 3. Restrict data to the scope of our problem:
    #      - Accidents
    #      - In the USA
    #      - Before 2022  #TODO change to June 2022 if it's not too big a pain
    merged = merged[(merged['ev_country']=="USA") & (merged['ev_type']=="ACC")]
    merged.loc[merged['damage'].isna(),'damage'] = 'UNK'

    data = merged[merged['ev_year'] < 2022]
    data = data.reset_index().drop(columns=['index'])

    # Create post_covid dataset, needed for timeseries analysis
    post_covid = merged[merged['ev_year'] >= 2020]

    ### 4. Train / validation / test split
    #       - Stratified by damage level
    #       - Grouped so that multiple aircraft in a single crash are placed in the same split
    data_train, data_val, data_test = stratified_group_split(data, strat_col='damage', group_col='ev_id')

    ### 5. Write data to files 
    data_train.to_csv('../data/ntsb_processed/master_train.csv', index=False)
    data_val.to_csv('../data/ntsb_processed/master_val.csv', index=False)
    data_test.to_csv('../data/ntsb_processed/master_test.csv', index=False)
    post_covid.to_csv('../data/ntsb_processed/master_post_covid.csv', index=False)
