import pandas as pd

import data
import models

MODELS_FROM_FILE = 1
THRESHOLDS_FROM_FILE = 1
BETA = 3

data_processed_dir = "data/processed/"
models_dir = "models/"

full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

# Get list of column names with "Diag: " prefix, where number of 
# positive examples is > threshold
threshold_positive_examples = 70
diag_cols = [x for x in full_dataset.columns if x.startswith("Diag: ") and 
             full_dataset[x].sum() > threshold_positive_examples] 

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
print(performance_table)