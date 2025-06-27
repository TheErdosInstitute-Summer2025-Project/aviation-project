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
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor
seed = 51841519

############################################################
##             Checking Model Performances                ##
############################################################

performances = pd.read_csv('data/model_performance/regression_performances.csv')
f_performances = performances[performances['target']=='prop_inj_fatal']
s_performances = performances[performances['target']=='prop_inj_serious']

print(f_performances[f_performances['val_mse'] == f_performances['val_mse'].min()]) # Best "fatal" model
print(s_performances[s_performances['val_mse'] == s_performances['val_mse'].min()]) # Best "serious" model

# In both cases, the Histogram Gradient Boosting Regressor performs best.
# For "fatal" proportions, hyperparameter 

############################################################
##                 Predicting Test Set                    ##
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


# Best Fatal Injury Proportions: 
## Histgrad with params: {'learning_rate': 0.1, 'max_iter': 100, 'max_leaf_nodes': 6}
# Best Serious Injury Proportions: 
## Histgrad with params: {'learning_rate': 0.01, 'max_iter': 500, 'max_leaf_nodes': 3}


# Re-Fit Histogram Gradient Boosting Regressors with Appropriate Hyperparameters to Training

## Fatal Proportion Model 
histgrad_f = HistGradientBoostingRegressor(random_state=seed,
                                         learning_rate=0.1,
                                         max_iter=100,
                                         max_leaf_nodes=6)
histgrad_f.fit(X_train,y_train_f)

## Serious Proportion Model
histgrad_s = HistGradientBoostingRegressor(random_state=seed,
                                         learning_rate=0.01,
                                         max_iter=500,
                                         max_leaf_nodes=3)
histgrad_s.fit(X_train,y_train_s)


# Predict Test Proportion Data With "Best" Models
y_pred_f = histgrad_f.predict(X_test)
y_pred_s = histgrad_s.predict(X_test)

# Naive "Mean" Predictor
naive_pred_f = [y_train_f.mean()] * len(y_test_f)
naive_pred_s = [y_train_s.mean()] * len(y_test_s)



############################################################
##           Model Performance & Important Features       ##
############################################################

# Fatal Injuries - Compare Model and Naive Performance
print(f"HistGrad Performance - MAE:{mean_absolute_error(y_test_f,y_pred_f)},  MSE:{mean_squared_error(y_test_f,y_pred_f)}")
print(f"Naive Performance - MAE:{mean_absolute_error(y_test_f, naive_pred_f)},  MSE:{mean_squared_error(y_test_f, naive_pred_f)}")


# Serious Injuries - Compare Model and Naive Performance
print(f"HistGrad Performance - MAE:{mean_absolute_error(y_test_s,y_pred_s)},  MSE:{mean_squared_error(y_test_s,y_pred_s)}")
print(f"Naive Performance - MAE:{mean_absolute_error(y_test_s, naive_pred_s)},  MSE:{mean_squared_error(y_test_s, naive_pred_s)}")


############################################################

# Exporting Permutation Importance Plots

## Fatal Injury Proportions
# Get permutation importance for TRAIN set

perm_train = permutation_importance(histgrad_f, X_train, y_train_f, n_repeats=25, random_state=seed)
train_importances = pd.DataFrame({
    'Feature': features,
    'Importance': perm_train.importances_mean
}).sort_values(by='Importance', ascending=True)

# Get permutation importance for TEST set
perm_test = permutation_importance(histgrad_f, X_test, y_test_f, n_repeats=25, random_state=seed)
test_importances = pd.DataFrame({
    'Feature': features,
    'Importance': perm_test.importances_mean
}).sort_values(by='Importance', ascending=True)


# Plotting both sets side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

axes[0].barh(test_importances['Feature'], test_importances['Importance'], color='steelblue')
axes[0].set_title('Permutation Importances (Test)')
axes[0].invert_yaxis()

axes[1].barh(train_importances['Feature'], train_importances['Importance'], color='seagreen')
axes[1].set_title('Permutation Importances (Train)')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("img/fatal_permutation_importances.png", dpi=300, bbox_inches='tight')


## Serious Injury Proportions
# Get permutation importance for TRAIN set

perm_train = permutation_importance(histgrad_s, X_train, y_train_s, n_repeats=25, random_state=seed)
train_importances = pd.DataFrame({
    'Feature': features,
    'Importance': perm_train.importances_mean
}).sort_values(by='Importance', ascending=True)

# Get permutation importance for TEST set
perm_test = permutation_importance(histgrad_s, X_test, y_test_s, n_repeats=25, random_state=seed)
test_importances = pd.DataFrame({
    'Feature': features,
    'Importance': perm_test.importances_mean
}).sort_values(by='Importance', ascending=True)


# Plotting both sets side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

axes[0].barh(test_importances['Feature'], test_importances['Importance'], color='steelblue')
axes[0].set_title('Permutation Importances (Test)')
axes[0].invert_yaxis()

axes[1].barh(train_importances['Feature'], train_importances['Importance'], color='seagreen')
axes[1].set_title('Permutation Importances (Train)')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("img/serious_permutation_importances.png", dpi=300, bbox_inches='tight')