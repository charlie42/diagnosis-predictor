import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import sys, inspect

from joblib import load, dump

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, models, util

DEBUG_MODE = True

def build_output_dir_name(params_from_create_datasets):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params_part = util.build_param_string_for_dir_name(params_from_create_datasets) + "___" +\
                  util.build_param_string_for_dir_name({"debug_mode": DEBUG_MODE})
    
    return datetime_part + "___" + params_part

def set_up_directories():

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from create_datasets.py]
    #    - use other diags as input
    #    - debug mode
    params_from_create_datasets = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_create_datasets)

    models_dir = data_dir + "models/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(models_dir)

    reports_dir = data_dir + "reports/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(reports_dir) 

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "reports_dir": reports_dir}

def set_up_load_directories():
    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.

    data_dir = "../diagnosis_predictor_data/"
    
    load_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "reports/train_models/")
    
    return {"load_data_dir": load_data_dir, "load_models_dir": load_models_dir, "load_reports_dir": load_reports_dir}
    
def get_base_models_and_param_grids():
    
    # Define base models
    rf = RandomForestClassifier(n_estimators=200 if DEBUG_MODE else 400)
    svc = svm.SVC()
    lr = LogisticRegression(solver="saga")
    
    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Standardize data
    scaler = StandardScaler()

    # Make pipelines
    rf_pipe = make_pipeline(imputer, scaler, rf)
    svc_pipe = make_pipeline(imputer, scaler, svc)
    lr_pipe = make_pipeline(imputer, scaler, lr)
    
    # Define parameter grids to search for each pipe
    from scipy.stats import loguniform, uniform
    rf_param_grid = {
        'randomforestclassifier__max_depth' : np.random.randint(5, 150, 30),
        'randomforestclassifier__min_samples_split': np.random.randint(2, 50, 30),
        'randomforestclassifier__min_samples_leaf': np.random.randint(1, 20, 30),
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75, 1.0],
        'randomforestclassifier__criterion': ['gini', 'entropy'],
        'randomforestclassifier__class_weight':["balanced", "balanced_subsample", None],
        "randomforestclassifier__class_weight": ['balanced', None]
    }
    svc_param_grid = {
        'svc__C': loguniform(1e-1, 1e3),
        'svc__gamma': loguniform(1e-04, 1e+01),
        'svc__degree': uniform(2, 5),
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        "svc__class_weight": ['balanced', None]
    }
    lr_param_grid = {
        'logisticregression__C': loguniform(1e-5, 1e4),
        'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
        'logisticregression__class_weight': ['balanced', None],
        'logisticregression__l1_ratio': uniform(0, 1)
    }
    
    base_models_and_param_grids = [
        (rf_pipe, rf_param_grid),
        (svc_pipe, svc_param_grid),
        (lr_pipe, lr_param_grid),
    ]
    if DEBUG_MODE:
        base_models_and_param_grids = [base_models_and_param_grids[-1]] # Only do LR in debug mode
        #base_models_and_param_grids = [base_models_and_param_grids[-1], base_models_and_param_grids[0]] # Only do LR and RF in debug mode
        pass
    
    return base_models_and_param_grids


# Find best estimator
def find_best_estimators_and_scores(datasets, diag_cols, performance_margin):
    best_estimators = {}
    scores_of_best_estimators = {}
    sds_of_scores_of_best_estimators = {}

    base_models_and_param_grids = get_base_models_and_param_grids()
    base_model, param_grid = base_models_and_param_grids[0] # Test only on LR for now        

    for i, diag in enumerate(diag_cols):
        print(diag, f'{i+1}/{len(diag_cols)}')

        X_train = datasets[diag]["X_train"]
        y_train = datasets[diag]["y_train"]

        # Run outer CV to find best estimator
        inner_cv = StratifiedKFold(n_splits=3 if DEBUG_MODE else 8, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=6 if DEBUG_MODE else 8, shuffle=True, random_state=i)
        
        # Nested CV with parameter optimization
        rs = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, cv=inner_cv, scoring="roc_auc", n_iter=50 if DEBUG_MODE else 200, n_jobs = -1, verbose=1)
        nested_score = cross_val_score(rs, X=X_train, y=y_train, cv=outer_cv, verbose=1, n_jobs = -1)
        print("Nested CV score:", nested_score.mean(), "+/-", nested_score.std())
            
    return best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators

def dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators):
    print(dirs["models_dir"])
    dump(best_estimators, dirs["models_dir"]+'best-estimators.joblib', compress=1)
    dump(scores_of_best_estimators, dirs["reports_dir"]+'scores-of-best-estimators.joblib', compress=1)
    dump(sds_of_scores_of_best_estimators, dirs["reports_dir"]+'sds-of-scores-of-best-estimators.joblib', compress=1)

def save_coefficients_of_lr_models(best_estimators, datasets, diag_cols, output_dir):
    for diag in diag_cols:
        best_estimator = best_estimators[diag]
        if util.get_base_model_name_from_pipeline(best_estimator) == "logisticregression":
            X_train = datasets[diag]["X_train_train"]
            models.save_coefficients_from_lr(diag, best_estimator, X_train, output_dir)

def main(performance_margin = 0.02, models_from_file = 1):
    models_from_file = int(models_from_file)
    performance_margin = float(performance_margin) # Margin of error for ROC AUC (for prefering logistic regression over other models)

    dirs = set_up_directories()
    load_dirs = set_up_load_directories()

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        #diag_cols = diag_cols[0:1]
        #diag_cols = ["Diag.Processing Speed Deficit (test)"]
        pass

    if models_from_file == 1:
        
        best_estimators = load(load_dirs["load_models_dir"]+'best-estimators.joblib')
        scores_of_best_estimators = load(load_dirs["load_reports_dir"]+'scores-of-best-estimators.joblib')
        sds_of_scores_of_best_estimators = load(load_dirs["load_reports_dir"]+'sds-of-scores-of-best-estimators.joblib')

        dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
    else: 
        # Find best models for each diagnosis
        best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators = find_best_estimators_and_scores(datasets, diag_cols, performance_margin)
        
        # Save best estimators and thresholds 
        dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
       
    # Build and save dataframe of best estimators and their scores
    df_of_best_estimators_and_their_score_sds = build_df_of_best_estimators_and_their_score_sds(best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators)
    df_of_best_estimators_and_their_score_sds.to_csv(dirs["reports_dir"] + "df_of_best_estimators_and_their_scores.csv", float_format='%.3f')
    print(df_of_best_estimators_and_their_score_sds)

    # Save feature coefficients for logistic regression models
    save_coefficients_of_lr_models(best_estimators, datasets, diag_cols, dirs["reports_dir"])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])