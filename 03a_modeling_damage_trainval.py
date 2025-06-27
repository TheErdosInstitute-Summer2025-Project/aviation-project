## Modeling Damage, Part 1

# This script imports the NTSB training and validation sets,
# and iteratively tests the performance of various regression-based learners
# while grid searching over a range of hyperparameters for each, comparing
# the performance of all learners to each other and to a naive prediction
# which takes all 'SUBS'. After establishing the 
# best performing learner, classification and confusion matrix are printed and the 
# model is applied to the test dataset in part 2.

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import argparse
import json
import os

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
seed = 51841519 # Set random seed - when re-fitting in 3b, ensures reproducibility

# Read in data, prepare for evaluation
## NOTE: The train/validation split has already been conducted at an earlier stage.

def load_data():
    train = pd.read_csv('data/ntsb_processed/ntsb_train_cleaned.csv')
    validation = pd.read_csv('data/ntsb_processed/ntsb_val_cleaned.csv')
    target = 'damage'
    train = train[train[target] != 'UNK']
    validation = validation[validation[target] != 'UNK']
    return train, validation

    



models = {
        "randomforest": (RandomForestClassifier(random_state=seed), {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}),
        "histgrad": (HistGradientBoostingClassifier(random_state=seed), {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_iter': [100, 200, 500],
    'max_leaf_nodes': [3,9,15,31],
    'min_samples_leaf': [10, 20, 30]
}),
        "extrees": (ExtraTreesClassifier(random_state=seed), {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}),
        "xgboost": (XGBClassifier(random_state=seed), {
    'n_estimators': [20, 100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
})
    }

PRESET_PARAMS = {k: v[1] for k, v in models.items()}






# Initialize Grid Search & Performance Append Function
def grid_search(model, label, 
                X_train, X_val, 
                y_train_encoded, y_val_encoded, 
                param_grid=None):
    if USE_PRESET_PARAMS and param_grid is None:
        param_grid = PRESET_PARAMS[label]

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='f1_macro',
        cv=5,
        verbose=1
    )
    grid.fit(X_train, y_train_encoded)
    best_mod = grid.best_estimator_
    best_params = grid.best_params_

    y_train_pred = best_mod.predict(X_train)
    y_val_pred = best_mod.predict(X_val)

    
    f1_train = f1_score(y_train_encoded, y_train_pred, average='macro')
    f1_val = f1_score(y_val_encoded, y_val_pred, average='macro')

    performances.loc[len(performances)] = [
        label,
        str(best_params),
        'damage',
        f1_train,
        f1_val,
        grid.best_score_
    ]
    classification_report_dict[label] = {
        "report_dict": classification_report(y_val_encoded, y_val_pred, output_dict=True),
        "report_str": classification_report(y_val_encoded, y_val_pred)
    }
    confusion_matrix_dict[label] = confusion_matrix(y_val_encoded, y_val_pred)

    return best_mod


############################################################
##             Modeling: DAMAGE                           ##
############################################################
# Run iterative grid searches of hyperparameter values per model
# Save best performing model & its metrics
def main(use_preset=True, user_params={}):
    global USE_PRESET_PARAMS
    USE_PRESET_PARAMS = use_preset
    global USER_PARAMS
    USER_PARAMS = user_params
    train, validation = load_data()

    features = ['latitude','longitude','apt_dist','gust_kts','aircraft_count',
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
    target = 'damage'

    X_train = train[features]
    X_val = validation[features]
    y_train = train[target]
    y_val = validation[target]

    label_map = {'MINR': 0, 'SUBS': 1, 'DEST': 2}
    y_train_encoded = y_train.map(label_map)
    y_val_encoded = y_val.map(label_map)
    
    global performances, classification_report_dict, confusion_matrix_dict
    performances = pd.DataFrame(columns=['learner', 'hyperparams', 'target', 'train_f1_macro', 'val_f1_macro','cv_f1_macro'])
    classification_report_dict = {}
    confusion_matrix_dict = {}

    
    for label, (model, _) in models.items():
        grid_search(model, label, X_train, X_val, y_train_encoded, y_val_encoded,
                    param_grid = None if USE_PRESET_PARAMS else user_params)

    y_dumb_val = np.full_like(y_val_encoded, 1)
    dumb_score = f1_score(y_val_encoded, y_dumb_val, average='macro')
    performances.loc[len(performances)] = ["Naive", 'None', target,
                                           dumb_score,
                                           dumb_score,
                                           dumb_score]

    ############################################################
    ##             Model Comparison (w/ Naive Learner)        ##
    ############################################################

    print(performances)

    for label, cm in confusion_matrix_dict.items():
        print(f"\nClassification Report of {label}:")
        print(classification_report_dict[label])
        print(f"\nConfusion matrix of {label}:")
        print(label, cm)

if __name__ == "__main__":


#     parser = argparse.ArgumentParser(description="Run damage model with or without grid search")
#     parser.add_argument('--grid', action='store_true', help="Run full grid search (default is preset params)")
#     parser.add_argument('--params', type=str, default=None,
#                         help='''Optional JSON string of user-defined hyperparameters.
# Example:
# --params '{"randomforest": {"n_estimators": [100, 200], "max_depth": [30]},
#            "xgboost": {"n_estimators": [100], "learning_rate": [0.1]},
#            "histgrad": {"n_estimators": [100], "max_iter": [200]},
#            "extrees": {"n_estimators": [100], "max_depth": [10]}}'
# ''')

#     args = parser.parse_args()

#     if args.params:
#         user_params = json.loads(args.params)
#     else:
#         user_params = {}

    main()
    

############################################################
##             Performance and results Export             ##
############################################################

    performances.to_csv('data/model_performance/classification_performances_trainval.csv',index=False)
    with open('data/model_performance/classification_report_dict_val.pkl', 'wb') as f:
        pkl.dump(classification_report_dict, f)
    with open('data/model_performance/classification_confusion_matrix_dict_val.pkl', 'wb') as f:
        pkl.dump(confusion_matrix_dict, f)
