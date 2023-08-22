import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from scipy.stats import loguniform, uniform

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import cross_val_score

import sys, inspect
from joblib import load, dump
import time

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

def get_performance_of_models(datasets, diag_cols, pipeline, cv_perf):


class MyPipeline(Pipeline): # Needed to expose feature importances for RFE
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

def main(models_from_file = 1):
    start_time = time.time() # Start timer for measuring total time of script

    models_from_file = int(models_from_file)

    clinical_config = util.read_config("clinical")
    technical_config = util.read_config("technical")
    number_of_features_to_check = clinical_config["max items in screener"]
    performance_margin = technical_config["performance margin"] # Margin of error for ROC AUC (for prefering logistic regression over other models)

    dirs = set_up_directories()
    load_dirs = set_up_load_directories()

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        diag_cols = diag_cols[0:1]
        #diag_cols = ["Diag.Processing Speed Deficit (test)"]
        pass

    
    ###########
    # Model
    lr = LogisticRegression(solver="saga")

    # Parameters
    lr_param_grid = {
        #'C': loguniform(1e-5, 1e4), 
        'sequentialfeatureselector__estimator__penalty': ['l1', 'l2', 'elasticnet'], 
        #'class_weight': ['balanced', None], 
        #'l1_ratio': uniform(0, 1) 
    }

    # Pipeline
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("lr",lr)])

    # Feature selection
    fs = SFS(estimator=pipeline, k_features=number_of_features_to_check, forward=True, floating=False if DEBUG_MODE else True, scoring='roc_auc', n_jobs=-1, verbose=2)

    n_splits = 2 if DEBUG_MODE else 10
    cv_rs = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    cv_perf = StratifiedKFold(n_splits, shuffle=True, random_state=0)

    # Search
    rs = RandomizedSearchCV(
        estimator=fs,
        param_distributions=lr_param_grid,
        n_iter=10 if DEBUG_MODE else 100,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv_rs,
        refit=True,
        random_state=0
    )

    # Get cross_val_score at each number of features for each diagnosis
    cv_perf_scores = {}
    for diag in diag_cols:
        print(f"Training {diag}")
        cv_perf_scores = []
        for i in range(1, number_of_features_to_check+1):
            print(f"Checking {i} features")
            fs.k_features = i
            cv_perf_scores.append(np.mean(cross_val_score(rs, datasets[diag]["X_train_train"], datasets[diag]["y_train_train"], cv=cv_perf, scoring="roc_auc", n_jobs=-1)))
        print(cv_perf_scores)
        cv_perf_scores[diag] = cv_perf_scores

    # Get optimal # features for each diagnosis -- where performance reaches 95% of max performance among all # features
    optimal_number_of_features = {}
    for diag in diag_cols:
        optimal_number_of_features[diag] = np.argmax(cv_perf_scores[diag] >= np.max(cv_perf_scores[diag]) * performance_margin) + 1

    # Save models, optimal # features, and cross_val_score at each # features
    dump(rs, dirs["models_dir"]+'models.joblib')
    dump(optimal_number_of_features, dirs["models_dir"]+'optimal_number_of_features.joblib')
    dump(cv_perf_scores, dirs["models_dir"]+'cv_perf_scores.joblib')
    ###########


    util.print_and_save_string(time.time() - start_time, dirs["reports_dir"], "execution-time.txt")

if __name__ == "__main__":
    main(sys.argv[1])