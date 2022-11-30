import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import data
import models

def main(performance_margin = 0.02, models_from_file = 1):
    models_from_file = int(models_from_file)
    performance_margin = float(performance_margin)

    models_dir = "models/"
    reports_dir = "reports/"
    data_processed_dir = "data/processed/"

    full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

    # Get list of column names with "Diag: " prefix, where number of 
    # positive examples is > threshold
    threshold_positive_examples = 10
    diag_cols = [x for x in full_dataset.columns if x.startswith("Diag: ") and 
                full_dataset[x].sum() > threshold_positive_examples] 

    # Create datasets for each diagnosis (different input and output columns)
    datasets = data.create_datasets(full_dataset, diag_cols, split_percentage = 0.3)

    # Either load models from file or find best models from scratch
    if models_from_file == 0:
        # Find best models for each diagnosis
        best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers = models.find_best_classifiers_and_scores(datasets, diag_cols, performance_margin)

        # Save best classifiers and thresholds 
        from joblib import dump
        dump(best_classifiers, models_dir+'best-classifiers.joblib', compress=1)
        dump(scores_of_best_classifiers, models_dir+'scores-of-best-classifiers.joblib', compress=1)
        dump(sds_of_scores_of_best_classifiers, models_dir+'sds-of-scores-of-best-classifiers.joblib', compress=1)
        
    else:
        from joblib import load
        best_classifiers = load(models_dir+'best-classifiers.joblib')
        scores_of_best_classifiers = load(models_dir+'scores-of-best-classifiers.joblib')
        sds_of_scores_of_best_classifiers = load(models_dir+'sds-of-scores-of-best-classifiers.joblib')

    df_of_best_classifiers_and_their_score_sds = models.build_df_of_best_classifiers_and_their_score_sds(best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers, full_dataset)
    df_of_best_classifiers_and_their_score_sds.to_csv(reports_dir + "df_of_best_classifiers_and_their_scores.csv")
    df_of_best_classifiers_and_their_score_sds.plot.scatter(x='Number of positive examples', y = 'SD of best score')
 
    plt.savefig(reports_dir+'figures/cv_roc_std_over_nb_pos_examples.png')
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

