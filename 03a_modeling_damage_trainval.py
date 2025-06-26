## Modeling: Damage

# This script imports the NTSB training and validation sets,
# and iteratively tests the performance of various regression-based learners
# while grid searching over a range of hyperparameters for each, comparing
# the performance of all learners to each other and to a naive prediction
# which takes all 'SUBS'. After establishing the 
# best performing learner, feature importances are printed and the 
# model is applied to the test dataset.

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

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
        "randomforest": (RandomForestClassifier(), {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}),
        "histgrad": (HistGradientBoostingClassifier(), {
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'max_iter': [100, 200, 500],
    'max_leaf_nodes': [3,9,15,31],
    'min_samples_leaf': [10, 20, 30]
}),
        "extrees": (ExtraTreesClassifier(), {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}),
        "xgboost": (XGBClassifier(), {
    'n_estimators': [20, 100, 200, 500],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
})
    }

PRESET_PARAMS = {k: v[1] for k, v in models.items()}


# Initialize performance dataframe
performances = pd.DataFrame(columns=['learner','hyperparams','target',
                                     'train_f1_macro','val_f1_macro'])

# Initialize feature importance dataframe dict
feature_importance_df_dict = {}

# Initialize classification reports dict
classification_report_dict = {}

# Initialize confusion matrix dict
confusion_matrix_dict = {}



# Initialize Grid Search & Performance Append Function
def grid_search(model, label, 
                X_train, X_val, 
                y_train_encoded, y_val_encoded, 
                features, param_grid=None):
    if USE_PRESET_PARAMS and param_grid is None:
        param_grid = PRESET_PARAMS[label]

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='f1_macro',
        cv=5
    )
    grid.fit(X_train, y_train_encoded)
    best_mod = grid.best_estimator_
    best_params = grid.best_params_

    y_train_pred = best_mod.predict(X_train)
    y_val_pred = best_mod.predict(X_val)

    if label != 'xgboost' and label!='histgrad':
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': best_mod.feature_importances_
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        feature_importance_df_dict[label] = feature_importance_df

    f1_train = f1_score(y_train_encoded, y_train_pred, average='macro')
    f1_val = f1_score(y_val_encoded, y_val_pred, average='macro')

    performances.loc[len(performances)] = [
        label,
        str(best_params),
        'damage',
        f1_train,
        f1_val
    ]

    classification_report_dict[label] = classification_report(y_val_encoded, y_val_pred)
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

    global performances, feature_importance_df_dict, classification_report_dict, confusion_matrix_dict
    performances = pd.DataFrame(columns=['learner', 'hyperparams', 'target', 'train_f1_macro', 'val_f1_macro'])
    feature_importance_df_dict = {}
    classification_report_dict = {}
    confusion_matrix_dict = {}

    
    for label, (model, _) in models.items():
        grid_search(model, label, X_train, X_val, y_train_encoded, y_val_encoded, features,
                    param_grid=None if USE_PRESET_PARAMS else user_params)

    y_dumb_val = np.full_like(y_val_encoded, 1)
    performances.loc[len(performances)] = ["Naive", 'None', target,
                                           f1_score(y_train_encoded, np.full_like(y_train_encoded, 1), average='macro'),
                                           f1_score(y_val_encoded, y_dumb_val, average='macro')]

    ############################################################
    ##             Model Comparison (w/ Naive Learner)        ##
    ############################################################

    print(performances)

    for label, cm in confusion_matrix_dict.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.keys()))
        disp.plot(cmap='Blues', xticks_rotation=45)
        disp.ax_.set_title(f"Confusion Matrix of {label}")
        disp.figure_.tight_layout()
        if label in feature_importance_df_dict:
            print(f"\nFeature importances of {label}:")
            print(feature_importance_df_dict[label].head(10))
        print(f"\nClassification Report of {label}:")
        print(classification_report_dict[label])
        print(label, cm)

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run damage model with or without grid search")
    parser.add_argument('--grid', action='store_true', help="Run full grid search (default is preset params)")
    parser.add_argument('--params', type=str, default=None,
                        help='''Optional JSON string of user-defined hyperparameters.
Example:
--params '{"randomforest": {"n_estimators": [100, 200], "max_depth": [30]},
           "xgboost": {"n_estimators": [100], "learning_rate": [0.1]},
           "histgrad": {"n_estimators": [100], "max_iter": [200]},
           "extrees": {"n_estimators": [100], "max_depth": [10]}}'
''')

    args = parser.parse_args()

    if args.params:
        user_params = json.loads(args.params)
    else:
        user_params = {}

    main(use_preset=not args.grid, user_params=user_params)
    
## The following full parameter grids were used to determine the default settings:

# # Random Forest
# rf_param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [None, 5, 10, 20, 30],
#     'min_samples_split': [2, 5, 10, 20],
#     'max_features': ['sqrt', 'log2', None]
# }

# # Histogram Gradient Boosting
# hg_param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.5],
#     'max_iter': [100, 200, 500],
#     'max_leaf_nodes': [15, 31, 63],
#     'min_samples_leaf': [20, 50, 100]
# }

# # Extra Trees
# et_param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [None, 5, 10, 20, 30],
#     'min_samples_split': [2, 5, 10, 20],
#     'max_features': ['sqrt', 'log2', None]
# }

# # XGBoost
# xgb_param_grid = {
#     'n_estimators': [20, 100, 200, 500, 1000],
#     'max_depth': [3, 6, 10],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0]
# }


############################################################
##             Performance Data Export                    ##
############################################################

performances.to_csv('data/classification_performances.csv',index=False)

# Filter only for MSE columns
df_filtered = performances.copy()

f1_plot_df = df_filtered.melt(
    id_vars=['learner', 'target'],
    value_vars=['train_f1_macro', 'val_f1_macro'],
    var_name='metric',
    value_name='score'
)


# Extract 'Train' and "Validation"
f1_plot_df['dataset'] = f1_plot_df['metric'].apply(lambda x: 'Train' if 'train' in x else 'Validation')

# Sort learners by average score across all targets/datasets
sorted_order = (
    f1_plot_df.groupby('learner')['score']
    .mean()
    .sort_values()
    .index
    .tolist()
)

# Plot
g = sns.catplot(
    data=f1_plot_df,
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
g.set_axis_labels("f1_score", "Model")
g.fig.suptitle("Training vs Validation f1_score by Target Variable", y=1.08)


plt.tight_layout()
g.savefig('img/damage_learners_scores.png')
