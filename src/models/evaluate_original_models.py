import os, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import pandas as pd
import numpy as np
import sys
import argparse
from joblib import load

from sklearn.metrics import roc_auc_score

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def build_output_dir_name(params_from_train_models, params_from_evaluate_original_models):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + util.build_param_string_for_dir_name(params_from_train_models) + "___" +\
        util.build_param_string_for_dir_name(params_from_evaluate_original_models)

def set_up_directories(use_test_set, args_to_read_data):

    data_dir = "../diagnosis_predictor_data_archive/"

    # Input dirs
    input_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/", args_to_read_data)
    models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/", args_to_read_data)
    input_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir+ "reports/train_models/", args_to_read_data)

    # Output dirs

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from train_models.py]
    #    - use test set
    params_from_train_models = util.get_params_from_current_data_dir_name(models_dir)
    params_from_current_file = {"use_test_set": use_test_set}
    current_output_dir_name = build_output_dir_name(params_from_train_models, params_from_current_file)

    output_reports_dir = data_dir + "reports/evaluate_original_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def add_number_of_positive_examples(results, datasets):
    for diag in datasets:
        full_dataset_y = pd.concat([datasets[diag]["y_train"], datasets[diag]["y_test"]]) # Reconstruct full dataset from train and test
        results.loc[results["Diag"] == diag, "# of Positive Examples"] = full_dataset_y.sum()
    return results

def get_roc_auc(X, y, estimator):
    y_pred_prob = estimator.predict_proba(X)
    roc_auc = roc_auc_score(y, y_pred_prob[:,1])
    return roc_auc

def get_aucs_on_test_set(best_estimators, datasets, use_test_set, diag_cols):
    aucs = {}
    
    for diag in diag_cols:
        print(diag)
        print(util.get_base_model_name_from_pipeline(best_estimators[diag]))
        estimator = best_estimators[diag]
    
        if use_test_set == 1:
            X, y = datasets[diag]["X_test"], datasets[diag]["y_test"] 
            X_hc, y_hc = datasets[diag]["X_test_only_healthy_controls"], datasets[diag]["y_test_only_healthy_controls"]
        else:
            X, y = datasets[diag]["X_val"], datasets[diag]["y_val"]
            X_hc, y_hc = datasets[diag]["X_val_only_healthy_controls"], datasets[diag]["y_val_only_healthy_controls"]

        roc_auc = get_roc_auc(X, y, estimator)

        # Only calculate ROC AUC on healthy controls if diag is not Diag.No Diagnosis Given
        roc_auc_hc = get_roc_auc(X_hc, y_hc, estimator) if diag != "Diag.No Diagnosis Given" else np.nan
        
        aucs[diag] = [roc_auc, roc_auc_hc]

    # Example of aucs: {'Diag1': [0.5, 0.4], 'Diag2': [0.6, 0.5], 'Diag3': [0.7, 0.6]}
    # Convert to a dataframe
    results = pd.DataFrame.from_dict(aucs, columns=["ROC AUC", "ROC AUC Healthy Controls"], orient="index").sort_values("ROC AUC", ascending=False).reset_index().rename(columns={'index': 'Diag'})
    print(results)
    results = add_number_of_positive_examples(results, datasets)

    return results.sort_values(by="ROC AUC", ascending=False)

def get_aucs_cv_from_grid_search(reports_dir, diag_cols):
    auc_cv_from_grid_search = pd.read_csv(reports_dir + "df_of_best_estimators_and_their_scores.csv")
    auc_cv_from_grid_search = auc_cv_from_grid_search[auc_cv_from_grid_search["Diag"].isin(diag_cols)][["Diag", "Best score", "SD of best score", "Score - SD"]]
    auc_cv_from_grid_search.columns = ["Diag", "ROC AUC Mean CV", "ROC AUC SD CV", "ROC AUC Mean CV - SD"]
    return auc_cv_from_grid_search

def get_roc_aucs(best_estimators, datasets, use_test_set, diag_cols, input_reports_dir):
    roc_aucs_cv_from_grid_search = get_aucs_cv_from_grid_search(input_reports_dir, diag_cols)
    roc_aucs_on_test_set = get_aucs_on_test_set(best_estimators, datasets, use_test_set=use_test_set, diag_cols=diag_cols)
    roc_aucs = roc_aucs_cv_from_grid_search.merge(roc_aucs_on_test_set, on="Diag").sort_values(by="ROC AUC Mean CV - SD", ascending=False)
    return roc_aucs

def main():
    parser = argparse.ArgumentParser()
    # New args
    parser.add_argument("--val-set", action='store_true', help="Use the validation set instead of the test set")
    
    # Args to read data from previous step
    parser.add_argument('--distrib-only', action='store_true', help='Only generate assessment distribution, do not create datasets')
    parser.add_argument('--parent-only', action='store_true', help='Only use parent-report assessments')
    parser.add_argument('--use-other-diags', action='store_true', help='Use other diagnoses as input')
    parser.add_argument('--free-only', action='store_true', help='Only use free assessments')
    parser.add_argument('--learning', action='store_true', help='Use additional assessments like C3SR (reduces # of examples)')
    parser.add_argument('--nih', action='store_true', help='Use NIH toolbox scores')
    parser.add_argument('--fix-n-all', action='store_true', help='Fix number of training examples when using less assessments')
    parser.add_argument('--fix-n-learning', action='store_true', help='Fix number of training examples when using less assessments')

    use_test_set = 0 if parser.parse_args().val_set else 1
    args_to_read_data = {
        "only_parent_report": parser.parse_args().parent_only,
        "use_other_diags_as_input": parser.parse_args().use_other_diags,
        "only_free_assessments": parser.parse_args().free_only,
        "learning?": parser.parse_args().learning,
        "NIH?": parser.parse_args().nih,
        "fix_n_all": parser.parse_args().fix_n_all, 
        "fix_n_learning": parser.parse_args().fix_n_learning
    }
    
    dirs = set_up_directories(use_test_set, args_to_read_data)

    best_estimators = load(dirs["models_dir"]+'best-estimators.joblib')
    
    diag_cols = best_estimators.keys()

    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    # Print performances of models on validation set
    roc_aucs = get_roc_aucs(best_estimators, datasets, use_test_set=use_test_set, 
                            diag_cols=diag_cols, input_reports_dir=dirs["input_reports_dir"])
    
    if use_test_set == 1:
        roc_aucs.to_csv(dirs["output_reports_dir"]+"performance_table_all_features_test_set.csv", float_format='%.3f', index=False)    
    else:
        roc_aucs.to_csv(dirs["output_reports_dir"]+"performance_table_all_features_val_set.csv", float_format='%.3f', index=False)

if __name__ == "__main__":
    main()