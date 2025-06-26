## Modeling Injury Proportions, Part 2

# This script examines the performances of the models from Part 1,
# extracts the models with the best performance per target, (serious and fatal proprtions),
# fits it to the training set one last time, and then predicts the proportions in the 
# test set.  Finally, performance of test predictions and feature importances are 
# displayed or saved.
# Note: These modeling scripts are separated in order to avoid re-running
# the arduous regression script unnecessarily.

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor

############################################################
##             Checking Model Performances                ##
############################################################

performances = pd.read_csv('data/regression_performances.csv')
f_performances = performances[performances['target']=='prop_inj_fatal']
s_performances = performances[performances['target']=='prop_inj_serious']

print(f_performances[f_performances['val_mse'] == f_performances['val_mse'].min()]) # Best "fatal" model
print(s_performances[s_performances['val_mse'] == s_performances['val_mse'].min()]) # Best "serious" model

############################################################
##                 Fitting to Test Set                    ##
############################################################

# Read in Training and Test Data, Create Relevant Objects
train = pd.read_csv('data/ntsb_processed/ntsb_train_cleaned.csv').dropna()

test = pd.read_csv('data/ntsb_processed/ntsb_test_cleaned.csv').dropna()

target_f   = ['acft_prop_inj_f']
target_s   = ['acft_prop_inj_s']
features = ['latitude','longitude','apt_dist','gust_kts','altimeter','aircraft_count',
            'num_eng','days_since_insp','light_cond_DAYL','light_cond_DUSK','light_cond_NDRK',
            'light_cond_NITE','light_cond_other/unknown','BroadPhaseofFlight_Air',
            'BroadPhaseofFlight_Ground','BroadPhaseofFlight_Landing','BroadPhaseofFlight_Takeoff',
            'BroadPhaseofFlight_other/unknown','eng_type_REC','eng_type_TF','eng_type_TP','eng_type_TS',
            'eng_type_other/unknown','far_part_091','far_part_121','far_part_135','far_part_137','far_part_PUBU',
            'far_part_other/unknown','acft_make_beech','acft_make_bell','acft_make_boeing','acft_make_cessna',
            'acft_make_mooney','acft_make_other/unknown','acft_make_piper','acft_make_robinson helicopter',
            'acft_category_AIR','acft_category_HELI','acft_category_other/unknown','homebuilt_N','homebuilt_Y',
            'fixed_retractable_FIXD','fixed_retractable_RETR',
            'second_pilot_N','second_pilot_Y','second_pilot_other/unknown']


X_train = train[features]
y_train_f = np.ravel(train[target_f]) # Fatal Injuries proportion target
y_train_s = np.ravel(train[target_s]) # Serious Injuries proportion target

X_test = test[features]
y_test_f = np.ravel(test[target_f])
y_test_s = np.ravel(test[target_s])



## Insert ## Modeling Here

# Fatal Proportions
# model.fit(X_train, y_train_f)


############################################################
##           Model Performance & Important Features       ##
############################################################

