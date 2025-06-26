## Modeling Injury Proportions, Part 1

# This script imports the NTSB training and validation sets,
# and iteratively tests the performance of various regression-based learners
# while grid searching over a range of hyperparameters for each, comparing
# the performance of all learners to each other and to a naive prediction
# which takes the mean value of each proprtion. After establishing the 
# best performing learner, the dataset containing the performances of each fitted model
# is exported to then be applied to the test set.

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor


# Read in data, prepare for evaluation
## NOTE: The train/validation split has already been conducted at an earlier stage.

train = pd.read_csv('data/ntsb_processed/ntsb_train_cleaned.csv').dropna()
validation = pd.read_csv('data/ntsb_processed/ntsb_val_cleaned.csv').dropna()

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

X_val = validation[features]
y_val_f = np.ravel(validation[target_f])
y_val_s = np.ravel(validation[target_s])


# Initialize performance dataframe
performances = pd.DataFrame(columns=['learner','hyperparams','target',
                                     'train_mse','train_mae','val_mse','val_mae'])

# Initialize Grid Search & Performance Append Function
def fatal_grid_search(model, param_grid, label):
    "Iteratively runs grid search on specified parameter grid and appends performance info to master dataset."
    
    target='prop_inj_fatal' # Declares label for target variable in master dataset

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid.fit(X_train,y_train_f)
    best_mod = grid.best_estimator_
    best_params = grid.best_params_

    # Train and validation prediction objects 
    y_train_pred = best_mod.predict(X_train)
    y_val_pred = best_mod.predict(X_val)

    # Performance metrics
    train_mse = mean_squared_error(y_train_f,y_train_pred)
    val_mse = mean_squared_error(y_val_f,y_val_pred)
    train_mae = mean_absolute_error(y_train_f, y_train_pred)
    val_mae = mean_absolute_error(y_val_f, y_val_pred)

    # Append to performance dataframe
    performances.loc[len(performances)] = [
        label,
        str(best_params),
        target,
        train_mse,
        train_mae,
        val_mse,
        val_mae
    ]
    return best_mod


############################################################
##             Modeling: FATAL INJURIES                   ##
############################################################
# Run iterative grid searches of hyperparameter values per model
# Save best performing model & its metrics

## Random Forest Regressor Grid Search
rf = RandomForestRegressor()
rf_param_grid = {
    'n_estimators':[10,500,1000],
    'min_samples_leaf':[2,5,10, 20],
    'max_samples':[100,500,1000]
}
rf_mod_f = fatal_grid_search(rf,rf_param_grid,"randomforest")


## Histogram Gradient Boosting Regressor Search
histgrad = HistGradientBoostingRegressor()
hg_param_grid = {
    'learning_rate': [0.01,0.05,0.1,0.5,1],
    'max_iter': [100,200,500],
    'max_leaf_nodes': [3,6,9]
}
hg_mod_f = fatal_grid_search(histgrad, hg_param_grid, "histgrad")



## Extra Trees Regressor
extrees = ExtraTreesRegressor()
et_param_grid = {
    'max_depth': [2,10,50,100,1000],
    'n_estimators': [10,100,500,1000],
    'max_leaf_nodes': [3,6,36,90]
}
et_mod_f = fatal_grid_search(extrees,et_param_grid,"extrees")



## Bagging Regressor
baggingreg = BaggingRegressor()
bg_param_grid = {
    'n_estimators': [10,100,500,1000],
}
bag_mod_f = fatal_grid_search(baggingreg,bg_param_grid,"bagging")


## XGBoost
xgb = XGBRegressor()
xgb_param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
xgb_mod_f = fatal_grid_search(xgb,xgb_param_grid,"xgboost")

## "Naive" Prediction - Average Proportion of Fatal Injuries
naive_train_pred = [y_train_f.mean()] * len(y_train_f)
naive_val_pred = [y_train_f.mean()] * len(y_val_f)

train_mse = mean_squared_error(y_train_f,naive_train_pred)
val_mse = mean_squared_error(y_val_f, naive_val_pred)
train_mae = mean_absolute_error(y_train_f, naive_train_pred)
val_mae = mean_absolute_error(y_val_f, naive_val_pred)

performances.loc[len(performances)] = [
        'naivemean', '{n/a}', 'prop_inj_fatal',
        train_mse, train_mae,
        val_mse, val_mae]



############################################################
# Initializing Serious Accidents Grid Search Function      #
############################################################


# Initialize Grid Search & Performance Append Function
def serious_grid_search(model, param_grid, label):
    "Iteratively runs grid search on specified parameter grid and appends performance info to master dataset."
    
    target='prop_inj_serious' # Declares label for target variable in master dataset

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid.fit(X_train,y_train_s)
    best_mod = grid.best_estimator_
    best_params = grid.best_params_

    # Train and validation prediction objects 
    y_train_pred = best_mod.predict(X_train)
    y_val_pred = best_mod.predict(X_val)

    # Performance metrics
    train_mse = mean_squared_error(y_train_s,y_train_pred)
    val_mse = mean_squared_error(y_val_s,y_val_pred)
    train_mae = mean_absolute_error(y_train_s, y_train_pred)
    val_mae = mean_absolute_error(y_val_s, y_val_pred)

    # Append to performance dataframe
    performances.loc[len(performances)] = [
        label,
        str(best_params),
        target,
        train_mse,
        train_mae,
        val_mse,
        val_mae
    ]
    
    return best_mod



############################################################
##             Modeling: SERIOUS INJURIES                 ##
############################################################
# Run iterative grid searches of hyperparameter values per model
# Save best performing model & its metrics
# Since most objects for this script were created in previous section, all that must be done is to 
# run the new function on the same grid search & learner objects.

## Random Forest Regressor Grid Search
rf_mod_s = serious_grid_search(rf,rf_param_grid,"randomforest")

## Histogram Gradient Boosting Regressor Search
hg_mod_s = serious_grid_search(histgrad, hg_param_grid, "histgrad")

## Extra Trees Regressor
et_mod_s = serious_grid_search(extrees,et_param_grid,"extrees")

## Bagging Regressor
bag_mod_s = serious_grid_search(baggingreg,bg_param_grid,"bagging")

## XGBoost Regressor
xgb_mod_s = serious_grid_search(xgb,xgb_param_grid,"xgboost")

## "Naive" Prediction - Average Proportion of Serious Injuries
naive_train_pred = [y_train_s.mean()] * len(y_train_s)
naive_val_pred = [y_train_s.mean()] * len(y_val_s)

train_mse = mean_squared_error(y_train_s,naive_train_pred)
val_mse = mean_squared_error(y_val_s, naive_val_pred)
train_mae = mean_absolute_error(y_train_s, naive_train_pred)
val_mae = mean_absolute_error(y_val_s, naive_val_pred)

performances.loc[len(performances)] = [
        'naivemean', '{n/a}', 'prop_inj_serious',
        train_mse, train_mae,
        val_mse, val_mae]

############################################################
##             Performance Data Export                    ##
############################################################

performances.to_csv('data/regression_performances.csv',index=False)


############################################################
##             Model Comparison (w/ Naive Learner)        ##
############################################################

print(performances)

# Filter only for MSE columns
df_filtered = performances.copy()

# Melt MSE columns
mse_plot_df = df_filtered.melt(
    id_vars=['learner', 'target'],
    value_vars=['train_mse', 'val_mse'],
    var_name='metric',
    value_name='score'
)
# Extract 'Train' and "Validation"
mse_plot_df['dataset'] = mse_plot_df['metric'].apply(lambda x: 'Train' if 'train' in x else 'Validation')

# Sort learners by average score across all targets/datasets
sorted_order = (
    mse_plot_df.groupby('learner')['score']
    .mean()
    .sort_values()
    .index
    .tolist()
)

# Plot
g = sns.catplot(
    data=mse_plot_df,
    kind='bar',
    y='learner',
    x='score',
    hue='dataset',
    col='target',
    palette='Blues_d',
    height=5,
    aspect=1.4,
    sharey=True,
    order=sorted_order
)

g.set_titles("Target: {col_name}")
g.set_axis_labels("MSE", "Model")
g.fig.suptitle("Training vs Validation MSE by Target Variable", y=1.08)

plt.tight_layout()
g.savefig('img/validation_learners_scores.png')



############################################################
##                 Fitting to Test Set                    ##
############################################################

# Read in Test Data
test = pd.read_csv('data/ntsb_processed/ntsb_test_cleaned.csv').dropna()

y_test_f = np.ravel(test[target_f])
y_test_s = np.ravel(test[target_s])
X_test = test[features]

print(performances[performances['val_mse'] == performances['val_mse'].min()])

