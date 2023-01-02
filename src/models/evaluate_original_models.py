import os, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import pandas as pd
import numpy as np
import sys

from sklearn.metrics import roc_auc_score

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def set_up_directories():
    input_data_dir = "data/train_models/"
    models_dir = "models/train_models/"
    input_reports_dir = "reports/train_models/"

    output_reports_dir = "reports/evaluate_original_models/"
    util.create_dir_if_not_exists(output_reports_dir)

    util.clean_dirs([output_reports_dir]) # Remove old reports

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def add_number_of_positive_examples(results, datasets):
    for diag in datasets:
        full_dataset_y = datasets[diag]["y_train"].append(datasets[diag]["y_test"]) # Reconstruct full dataset from train and test
        results.loc[results["Diag"] == diag, "# of Positive Examples"] = full_dataset_y.sum()
    return results

def get_roc_auc(X, y, classifier):
    y_pred_prob = classifier.predict_proba(X)
    roc_auc = roc_auc_score(y, y_pred_prob[:,1])
    return roc_auc

def get_aucs_on_test_set(best_classifiers, datasets, use_test_set, diag_cols):
    aucs = {}
    for diag in diag_cols:
        print(diag)
        print(list(best_classifiers[diag].named_steps.keys())[-1])
        classifier = best_classifiers[diag]
        
        if use_test_set == 1:
            X, y = datasets[diag]["X_test"], datasets[diag]["y_test"]
        else:
            X, y = datasets[diag]["X_val"], datasets[diag]["y_val"]

        roc_auc = get_roc_auc(X, y, classifier)
        aucs[diag] = roc_auc

    results = pd.DataFrame.from_dict(aucs, columns=["Diag", "ROC AUC"], orient="index").sort_values("ROC AUC", ascending=False).reset_index()
    results = add_number_of_positive_examples(results, datasets)

    return results.sort_values(by="ROC AUC", ascending=False)

def get_aucs_cv_from_grid_search(reports_dir, diag_cols):
    auc_cv_from_grid_search = pd.read_csv(reports_dir + "df_of_best_classifiers_and_their_scores.csv")
    auc_cv_from_grid_search = auc_cv_from_grid_search[auc_cv_from_grid_search["Diag"].isin(diag_cols)][["Diag", "Best score", "SD of best score", "Score - SD"]]
    auc_cv_from_grid_search.columns = ["Diag", "ROC AUC Mean CV", "ROC AUC SD CV", "ROC AUC Mean CV - SD"]
    return auc_cv_from_grid_search

def get_roc_aucs(best_classifiers, datasets, use_test_set, diag_cols, input_reports_dir):
    roc_aucs_cv_from_grid_search = get_aucs_cv_from_grid_search(input_reports_dir, diag_cols)
    roc_aucs_on_test_set = get_aucs_on_test_set(best_classifiers, datasets, use_test_set=use_test_set, diag_cols=diag_cols)
    roc_aucs = roc_aucs_cv_from_grid_search.merge(roc_aucs_on_test_set, on="Diag")
    return roc_aucs

def main(use_test_set=1):
    use_test_set = int(use_test_set)

    dirs = set_up_directories()

    from joblib import load
    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    
    diag_cols = best_classifiers.keys()

    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    # Print performances of models on validation set
    roc_aucs = get_roc_aucs(best_classifiers, datasets, use_test_set=use_test_set, diag_cols=diag_cols, input_reports_dir=dirs["input_reports_dir"])

    if use_test_set == 1:
        roc_aucs.to_csv(dirs["output_reports_dir"]+"performance_table_all_features.csv", index=False)    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])