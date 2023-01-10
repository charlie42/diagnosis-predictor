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
import util, models

def build_output_dir_name(params_from_train_models, params_from_evaluate_original_models):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + models.build_param_string_for_dir_name(params_from_train_models) + "___" + models.build_param_string_for_dir_name(params_from_evaluate_original_models)

def set_up_directories(use_test_set):

    data_dir = "../diagnosis_predictor_data/"

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/train_models/")
    models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/train_models/")

    # Output dirs

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from train_models.py]
    #    - use test set
    params_from_train_models = models.get_params_from_current_data_dir_name(input_data_dir)
    params_from_current_file = {"use_test_set": use_test_set}
    current_output_dir_name = build_output_dir_name(params_from_train_models, params_from_current_file)

    output_reports_dir = data_dir + "reports/evaluate_original_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

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

    # Example of aucs: {'Diag1': 0.5, 'Diag2': 0.6, 'Diag3': 0.7}
    # Convert to a dataframe
    results = pd.DataFrame.from_dict(aucs, columns=["ROC AUC"], orient="index").sort_values("ROC AUC", ascending=False).reset_index().rename(columns={'index': 'Diag'})
    print(results)
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
    roc_aucs = roc_aucs_cv_from_grid_search.merge(roc_aucs_on_test_set, on="Diag").sort_values(by="ROC AUC Mean CV - SD", ascending=False)
    return roc_aucs

def main(use_test_set=1):
    use_test_set = int(use_test_set)

    dirs = set_up_directories(use_test_set)

    from joblib import load
    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    
    diag_cols = best_classifiers.keys()

    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    # Print performances of models on validation set
    roc_aucs = get_roc_aucs(best_classifiers, datasets, use_test_set=use_test_set, diag_cols=diag_cols, input_reports_dir=dirs["input_reports_dir"])

    if use_test_set == 1:
        roc_aucs.to_csv(dirs["output_reports_dir"]+"performance_table_all_features.csv", index=False)    

if __name__ == "__main__":
    main(sys.argv[1])