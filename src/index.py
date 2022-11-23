import pandas as pd
import matplotlib.pyplot as plt

import data
import models

MODELS_FROM_FILE = 1
THRESHOLDS_FROM_FILE = 1
BETA = 3
THRESHOLD_POSITIVE_EXAMPLES = 150

data_processed_dir = "data/processed/"
models_dir = "models/"

full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

# Get list of column names with "Diag: " prefix, where number of 
# positive examples is > threshold
diag_cols = [x for x in full_dataset.columns if x.startswith("Diag: ") and 
             full_dataset[x].sum() > THRESHOLD_POSITIVE_EXAMPLES] 

# Create datasets for each diagnosis (different input and output columns)
datasets = data.create_datasets(full_dataset, diag_cols)

# Either load models from file or find best models from scratch
if MODELS_FROM_FILE == 0:
    # Find best models for each diagnosis
    best_classifiers = models.find_best_classifiers(datasets, diag_cols)

    # Save best classifiers and thresholds 
    from joblib import dump
    dump(best_classifiers, models_dir+'best-classifiers.joblib', compress=1)
    
else:
    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')

if THRESHOLDS_FROM_FILE == 0:
    # Find best probability thresholds for each diagnosis
    best_thresholds = models.find_best_thresholds(
            beta=BETA, 
            best_classifiers=best_classifiers, 
            datasets=datasets, 
            diag_cols=diag_cols
    )
    from joblib import dump
    dump(best_thresholds, models_dir+'best-thresholds.joblib', compress=1)
else:
    from joblib import load
    best_thresholds = load(models_dir+'best-thresholds.joblib')

# Print performances of models on validation set
performance_table = models.check_performance(best_classifiers, datasets, best_thresholds, beta = BETA, use_test_set=False)
print(performance_table[['Diag','Recall (Sensitivity)','TNR (Specificity)','ROC AUC Mean CV']].sort_values("ROC AUC Mean CV"))

# Filter only well performing diagnoses
# ROC AUC reference: https://gpsych.bmj.com/content/gpsych/30/3/207.full.pdf
well_performing_diags = models.find_well_performing_diags(performance_table, min_roc_auc_cv=0.8)
print(well_performing_diags)

print(set(diag_cols) - set(well_performing_diags))

