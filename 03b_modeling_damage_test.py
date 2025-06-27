## Modeling Damage, Part 2

# This script examines the performances of the models from Part 1,
# extracts the models with the best performance,
# fits it to the training set one last time, and then predicts the proportions in the 
# test set.  Finally, performance of test predictions, classification report, confusion matrix and feature importances are 
# displayed or saved.

# Note: These modeling scripts are separated in order to avoid re-running

# Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import ast

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
seed = 51841519 # Set random seed - when re-fitting, ensures reproducibility
label_map = {'MINR':0, 'SUBS' :1, 'DEST' :2}

############################################################
##             Checking Model Performances                ##
############################################################

## load performance data obtained from the part 1
performances = pd.read_csv('data/model_performance/classification_performances_trainval.csv')
with open('data/model_performance/classification_report_dict_val.pkl', 'rb') as f:
    classification_report_dict = pkl.load(f)
with open('data/model_performance/classification_confusion_matrix_dict_val.pkl', 'rb') as f:
    classification_confusion_matrix_dict = pkl.load(f)

df = performances.copy()
best_row = performances[performances['val_f1_macro'] == performances['val_f1_macro'].max()]
print('best model performance:\n',best_row) # Best model


learner = best_row['learner'].values[0] # string format of classifier type



# Performance Plot



f1_plot_df = df.melt(
    id_vars=['learner', 'target'],
    value_vars=['train_f1_macro', 'val_f1_macro'],
    var_name='metric',
    value_name='score'
)


# Extract 'Train' and "Validation"
f1_plot_df['dataset'] = f1_plot_df['metric'].apply(lambda x: 'Train' if 'train' in x else 'Validation')

# Sort learners by average score across all datasets
sorted_order = (
    f1_plot_df.groupby('learner')['score']
    .mean()
    .sort_values(ascending=False)
    .index
    .tolist()
)

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
    sharey=False,
    order=sorted_order
)

g.set_titles("Target: {col_name}")
g.set_axis_labels("f1_score", "Model")
g.fig.suptitle("Training vs Validation f1_score by Target Variable", y=1.08)
g.set_xticklabels(rotation=45)
g.set(xlim=(0, 1.0))
plt.tight_layout()
g.savefig('img/validation_damage_learners_scores_trainval.png', bbox_inches='tight')
    
# Classification Report (includes precision, recall, f1 per class)
print(f"\nClassification Report of {learner} on validation dataset:")
print(classification_report_dict[learner]["report_str"])

## Confusion Matrix
cm = classification_confusion_matrix_dict[learner]
# label_map = {'MINR':0, 'SUBS' :1, 'DEST' :2}
labels_map = {v: k for k, v in label_map.items()}
labels = [labels_map[i] for i in sorted(labels_map)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)

## Export confusion matrix
disp.ax_.set_title(f"Confusion Matrix of {learner} on validation dataset")
disp.ax_.set_xlabel("Predicted Label")
disp.ax_.set_ylabel("True Label")
disp.figure_.tight_layout()
plt.savefig('img/classification_confusion_matrix_val.png', dpi=300, bbox_inches='tight')  # Save as PNG





############################################################
##                 Predicting Test Set                    ##
############################################################


train = pd.read_csv('data/ntsb_processed/ntsb_train_cleaned.csv')
test = pd.read_csv('data/ntsb_processed/ntsb_test_cleaned.csv')
target = 'damage'
train = train[train[target] != 'UNK']
test = test[test[target] != 'UNK']

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

X_train = train[features]
y_train = train[target] 

X_test = test[features]
y_test = test[target]

# label_map = {'MINR': 0, 'SUBS': 1, 'DEST': 2}
y_train_encoded = y_train.map(label_map)
y_test_encoded = y_test.map(label_map)

hyper_params = ast.literal_eval(best_row['hyperparams'].values[0]) # best parameters found in part a for the best classifier

if learner == "randomforest":
    model = RandomForestClassifier( **hyper_params,random_state=seed)
elif learner == "histgrad":
    model = HistGradientBoostingClassifier( **hyper_params,random_state=seed)
elif learner == "extrees":
    model = ExtraTreesClassifier(**hyper_params,random_state=seed)
else:
    model = XGBClassifier(**hyper_params,random_state=seed)


model.fit(X_train, y_train_encoded)

############################################################
##    Get Performance, confusion matrix,                  ##
##    classification record, and feature importance       ##
############################################################

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


f1_train = f1_score(y_train_encoded, y_train_pred, average='macro')
f1_test = f1_score(y_test_encoded, y_test_pred, average='macro')

report_str = classification_report(y_test_encoded, y_test_pred)

cm = confusion_matrix(y_test_encoded, y_test_pred)

feature_names = X_train.columns
# Feature importance DataFrame
marker = False
if hasattr(model, 'feature_importances_'):
    # RandomForest, ExtraTrees, HistGradientBoosting
    importances = model.feature_importances_
    marker = True
elif hasattr(model, 'get_booster'):
    # XGBClassifier
    booster = model.get_booster()
    score_dict = booster.get_score(importance_type='gain')
    
    # Convert from dict to aligned list with all features
    importances = [score_dict.get(f'f{i}', 0) for i in range(len(feature_names))]

    marker = True

## The following new_feature_importance_df should be the same as the feature_importance_df
if marker:
    new_feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)


    # top 10 important features
    plt.figure(figsize=(10, 6))
    sns.barplot(data=new_feature_importance_df.head(10), x='Importance', y='Feature', palette='Blues_r')
    plt.title(f"Top 10 Feature Importances ({learner})")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("img/classification_feature_importance.png", dpi=300)
    plt.show()
    plt.close()
    
# Classification Report 
print("\nClassification Report on Testing dataset:")
print(report_str)

## Confusion Matrix
labels_map = {v: k for k, v in label_map.items()}
labels = [labels_map[i] for i in sorted(labels_map)]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
disp.ax_.set_title("Confusion Matrix for testing dataset")
disp.ax_.set_xlabel("Predicted Label")
disp.ax_.set_ylabel("True Label")
disp.figure_.tight_layout()
plt.savefig('img/classification_confusion_matrix_test.png', dpi=300, bbox_inches='tight')  # Save as PNG