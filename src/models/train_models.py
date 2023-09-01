import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

import sys, inspect
import argparse

from joblib import load, dump

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, models, util

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
    data_dir = "../diagnosis_predictor_data_archive/"
    util.create_dir_if_not_exists(data_dir)

    # Input dirs
    input_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from create_datasets.py]
    #    - use other diags as input
    #    - debug mode
    params_from_create_datasets = util.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_create_datasets)

    models_dir = data_dir + "models/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(models_dir)

    reports_dir = data_dir + "reports/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(reports_dir) 

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "reports_dir": reports_dir}

def set_up_load_directories(models_from_file):
    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.
    # When the script is run on a new location for the first time, there won't be any non-empty directories. 
    #   We only take non-empthy directries when we load existing models (script arguemnt 'models_from_file')

    data_dir = "../diagnosis_predictor_data_archive/"
    
    load_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/") if models_from_file == 1 else None
    load_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "reports/train_models/") if models_from_file == 1 else None
    
    return {"load_data_dir": load_data_dir, "load_models_dir": load_models_dir, "load_reports_dir": load_reports_dir}
    

def get_base_models_and_param_grids():
    
    # Define base models
    rf = RandomForestClassifier(n_estimators=200 if DEBUG_MODE else 400)
    svc = svm.SVC()
    lr = LogisticRegression(solver="saga")
    lgbm = HistGradientBoostingClassifier()
    
    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Standardize data
    scaler = StandardScaler()

    # Make pipelines
    rf_pipe = make_pipeline(imputer, scaler, rf)
    svc_pipe = make_pipeline(imputer, scaler, svc)
    lr_pipe = make_pipeline(imputer, scaler, lr)
    lgbm_pipe = make_pipeline(scaler, lgbm) # LGBM can handle missing values
    
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
    # lr_param_grid = {
    #     'logisticregression__C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],  # Logarithmic values from 1e-5 to 1e4
    #     'logisticregression__penalty': ['l1', 'l2', 'elasticnet'],
    #     'logisticregression__class_weight': ['balanced', None],
    #     'logisticregression__l1_ratio': [i / 100 for i in range(101)]  # Linear values from 0.0 to 1.0 with a step of 0.01
    # }
    lgbm_param_grid = {
        'histgradientboostingclassifier__learning_rate': np.logspace(-3, 0, num=100),  # Learning rate values from 0.001 to 1
        'histgradientboostingclassifier__max_depth': [3, 5, 7, None],  # Maximum depth of the trees
        'histgradientboostingclassifier__max_iter': np.arange(100, 1001, 100),  # Number of boosting iterations
        'histgradientboostingclassifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'histgradientboostingclassifier__l2_regularization': np.logspace(-3, 3, num=100)  # L2 regularization values from 0.001 to 1000
    }
    # lgbm_param_grid = {
    #     'histgradientboostingclassifier__learning_rate': [10 ** (-i) for i in range(4)],  # Logarithmic values from 0.001 to 1
    #     'histgradientboostingclassifier__max_depth': [3, 5, 7, None],
    #     'histgradientboostingclassifier__max_iter': list(range(100, 1001, 100)),  # Linear values from 100 to 1000 with a step of 100
    #     'histgradientboostingclassifier__min_samples_leaf': [1, 2, 4],
    #     'histgradientboostingclassifier__l2_regularization': [10 ** i for i in range(-3, 4)]  # Logarithmic values from 0.001 to 1000
    # }
    
    base_models_and_param_grids = [
        (rf_pipe, rf_param_grid),
        (svc_pipe, svc_param_grid),
        (lr_pipe, lr_param_grid),
        (lgbm_pipe, lgbm_param_grid)
    ]
    if DEBUG_MODE:
        base_models_and_param_grids = [base_models_and_param_grids[-2]] # Only do LR in debug mode
        #base_models_and_param_grids = [base_models_and_param_grids[-1], base_models_and_param_grids[-2]] # Only do LR and LGBM in debug mode
        pass
    
    return base_models_and_param_grids

def get_best_estimator(base_model, grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=3 if DEBUG_MODE else 8)
    rs = RandomizedSearchCV(estimator=base_model, param_distributions=grid, cv=cv, scoring="roc_auc", n_iter=50 if DEBUG_MODE else 200, n_jobs = -1, verbose=1)
    #rs = HalvingRandomSearchCV(estimator=base_model, param_distributions=grid, cv=cv, scoring="roc_auc",
    #                            random_state=0,
    #                            max_resources=100,
    #                            n_jobs = -1, verbose=1)
    
    print("Fitting", base_model, "...")
    rs.fit(X_train, y_train) 
    
    best_estimator = rs.best_estimator_
    best_score = rs.best_score_
    sd_of_score_of_best_estimator = rs.cv_results_['std_test_score'][rs.best_index_]

    # If chosen model is SVM add a predict_proba parameter (not needed for grid search, and slows it down significantly)
    if 'svc' in best_estimator.named_steps.keys():
        best_estimator.set_params(svc__probability=True)

    return (best_estimator, best_score, sd_of_score_of_best_estimator)

def find_best_estimator_for_diag_and_its_score(X_train, y_train, performance_margin):
    base_models_and_param_grids = get_base_models_and_param_grids()
    best_estimators_and_scores = []
    
    for (base_model, grid) in base_models_and_param_grids:
        best_estimator_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model = get_best_estimator(base_model, grid, X_train, y_train)
        model_type = list(base_model.named_steps.keys())[-1]
        best_estimators_and_scores.append([model_type, best_estimator_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model])
    
    best_estimators_and_scores = pd.DataFrame(best_estimators_and_scores, columns = ["Model type", "Best estimator", "Best score", "SD of best score"])
    print(best_estimators_and_scores)
    best_estimator = best_estimators_and_scores.sort_values("Best score", ascending=False)["Best estimator"].iloc[0]
    best_score = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Best score"].iloc[0]
    sd_of_score_of_best_estimator = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["SD of best score"].iloc[0]
    
    # If LogisticRegression is not much worse than the best model, prefer LogisticRegression (much faster than rest)
    best_base_model = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Model type"].iloc[0]
    if best_base_model != "logisticregression":
        lr_score = best_estimators_and_scores[best_estimators_and_scores["Model type"] == "logisticregression"]["Best score"].iloc[0]
        print("lr_score: ", lr_score, "; best_score: ", best_score)
        if best_score - lr_score <= performance_margin:
            best_estimator = best_estimators_and_scores[best_estimators_and_scores["Model type"] == "logisticregression"]["Best estimator"].iloc[0]
            best_score = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["Best score"].iloc[0]
            sd_of_score_of_best_estimator = best_estimators_and_scores[best_estimators_and_scores["Best estimator"] == best_estimator]["SD of best score"].iloc[0]
        
    print("best estimator:")
    print(best_estimator)
    
    return best_estimator, best_score, sd_of_score_of_best_estimator

# Find best estimator
def find_best_estimators_and_scores(datasets, diag_cols, performance_margin):
    best_estimators = {}
    scores_of_best_estimators = {}
    sds_of_scores_of_best_estimators = {}
    for i, diag in enumerate(diag_cols):
        print(diag, f'{i+1}/{len(diag_cols)}')

        X_train = datasets[diag]["X_train_train"]
        y_train = datasets[diag]["y_train_train"]
        
        best_estimator_for_diag, best_score_for_diag, sd_of_score_of_best_estimator_for_diag = find_best_estimator_for_diag_and_its_score(X_train, y_train, performance_margin)
        best_estimators[diag] = best_estimator_for_diag
        sds_of_scores_of_best_estimators[diag] = sd_of_score_of_best_estimator_for_diag
        scores_of_best_estimators[diag] = best_score_for_diag

        if DEBUG_MODE and util.get_base_model_name_from_pipeline(best_estimators[diag]) == "logisticregression":
            # In debug mode print top features from LR
            models.print_top_features_from_lr(best_estimators[diag], X_train, 10)
            
    return best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators

def build_df_of_best_estimators_and_their_score_sds(best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators):
    best_estimators_and_score_sds = []
    for diag in best_estimators.keys():
        best_estimator = best_estimators[diag]
        score_of_best_estimator = scores_of_best_estimators[diag]
        sd_of_score_of_best_estimator = sds_of_scores_of_best_estimators[diag]
        model_type = util.get_base_model_name_from_pipeline(best_estimator)
        best_estimators_and_score_sds.append([diag, model_type, best_estimator, score_of_best_estimator, sd_of_score_of_best_estimator])
    best_estimators_and_score_sds = pd.DataFrame(best_estimators_and_score_sds, columns = ["Diag", "Model type", "Best estimator", "Best score", "SD of best score"])
    best_estimators_and_score_sds["Score - SD"] = best_estimators_and_score_sds['Best score'] - best_estimators_and_score_sds['SD of best score'] 
    return best_estimators_and_score_sds

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance-margin", type=float, default=0.02, help="Margin of error for ROC AUC (for prefering logistic regression over other models)")
    parser.add_argument("--from-file", action='store_true', help="Load existing models from file instead of training new models")

    models_from_file = parser.parse_args().from_file
    performance_margin = parser.parse_args().performance_margin

    dirs = set_up_directories()
    load_dirs = set_up_load_directories(models_from_file)

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        diag_cols = diag_cols[0:1]
        #diag_cols = ["Diag.Processing Speed Deficit (test)"]
        pass

    if models_from_file is True:
        
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
    main()