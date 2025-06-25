## Modeling: Injury Proportions

# This script imports the NTSB training and validation sets,
# and iteratively tests the performance of various regression-based learners
# while grid searching over a range of hyperparameters for each, comparing
# the performance of all learners to each other and to a naive prediction
# which takes the mean value of each proprtion. After establishing the 
# best performing learner, feature importances are printed and the 
# model is applied to the test dataset.

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor


# Read in data, prepare for evaluation
## NOTE: The train/validation split has already been conducted at an earlier stage.

train = pd.read_csv('data/ntsb_processed/ntsb_train_cleaned.csv')
validation = pd.read_csv('data/ntsb_processed/ntsb_val_cleaned.csv')

target_f   = ['acft_prop_inj_f']
target_s   = ['acft_prop_inj_f']
features = ['latitude','longitude','apt_dist','gust_kts','altimeter','aircraft_count',
            'num_eng','days_since_insp','light_cond_DAYL','light_cond_DUSK','light_cond_NDRK',
            'light_cond_NITE','light_cond_other/unknown','BroadPhaseofFlight_Air',
            'BroadPhaseofFlight_Ground','BroadPhaseofFlight_Landing','BroadPhaseofFlight_Takeoff',
            'BroadPhaseofFlight_other/unknown','eng_type_REC','eng_type_TF','eng_type_TP','eng_type_TS',
            'eng_type_other/unknown','far_part_091','far_part_121','far_part_135','far_part_137','far_part_PUBU',
            'far_part_other/unknown','acft_make_beech','acft_make_bell','acft_make_boeing','acft_make_cessna',
            'acft_make_mooney','acft_make_other/unknown','acft_make_piper','acft_make_robinson helicopter',
            'acft_category_AIR','acft_category_HELI','acft_category_other/unknown','homebuilt_N','homebuilt_Y',
            'homebuilt_other/unknown','fixed_retractable_FIXD','fixed_retractable_RETR','fixed_retractable_other/unknown',
            'second_pilot_N','second_pilot_Y','second_pilot_other/unknown']

X_train = train[features]
y_train_f = train[target_f] # Fatal Injuries proportion target
y_train_s = train[target_s] # Serious Injuries proportion target

X_val = validation[features]
y_val_f = validation[target_f]
y_val_s = validation[target_s]


# Initialize performance dataframe
performances = pd.DataFrame(columns=['learner','hyperparams','target',
                                     'train_mse','train_mae','val_mse','val'])



# Modeling: FATAL INJURIES

## Random Forest Regressor Grid Search

rf = RandomForestRegressor()
param_grid = {
    'n_estimators':[10,500,1000],
    'min_samples_leaf':[2,5,10, 20],
    'max_samples':[100,500,1000]
}
grid = GridSearchCV(
    rf,
    param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid.fit(X_train,y_train_f)
model = grid.best_estimator_ # Select best hyperparameter values for training set 

# Testing validation performance


