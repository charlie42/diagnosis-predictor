import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

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

def make_feature_selectors(model, number_of_features_to_check):
    cv = StratifiedKFold(n_splits=2 if DEBUG_MODE else 8)
    rfe = RFE(model, 
              n_features_to_select=number_of_features_to_check, 
              step=1,
              verbose=1)
    sfs = SFS(model, 
        k_features=number_of_features_to_check, 
        forward=True, 
        scoring='roc_auc',
        cv=cv,
        floating=True, 
        verbose=0,
        n_jobs=-1)
    #return [rfe, sfs]
    return [rfe]

def make_models():
    # Define base models

    models = [
        RandomForestClassifier(n_estimators=200 if DEBUG_MODE else 400),
        svm.SVC(),
        LogisticRegression(solver="saga")
    ]

    return models

def make_pipeline(imputer, scaler, feature_selector1, model): # feature_selector2,
    # Make pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('featureselector1', feature_selector1),
        #('featureselector2', feature_selector2),
        ('model', model)
    ])
    return pipeline

def make_pipelines(number_of_features_to_check):
    
    base_models = make_models()
    feature_selectors = []
    for model in base_models:
        feature_selectors.append(make_feature_selectors(model, number_of_features_to_check))

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Standardize data
    scaler = StandardScaler()

    # Make pipelines
    pipelines = []
    for selectors_for_model, model in zip(feature_selectors, base_models):
        pipeline = make_pipeline(imputer, scaler, *selectors_for_model, model)
        pipelines.append(pipeline)
        print(pipeline)
        
    return pipelines
    
def make_params():
 
    # Define parameter grids to search for each pipe
    from scipy.stats import loguniform, uniform

    grids = []
    
    rf_param_grid = {
        'max_depth' : np.random.randint(5, 150, 30),
        'min_samples_split': np.random.randint(2, 50, 30),
        'min_samples_leaf': np.random.randint(1, 20, 30),
        'max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75, 1.0],
        'criterion': ['gini', 'entropy'],
        'class_weight':["balanced", "balanced_subsample", None],
        "lass_weight": ['balanced', None]
    }
     # Append "sequentialfeatureselector__estimator" to each key (the full name of the step in the pipeline):
    rf_pipe_param_grid = {}
    for key in rf_param_grid.keys():
        #rf_pipe_param_grid["sequentialfeatureselector__estimator__"+key] = rf_param_grid[key]
        rf_pipe_param_grid["model__"+key] = rf_param_grid[key]
    grids.append(rf_pipe_param_grid)

    svc_param_grid = {
        'C': loguniform(1e-1, 1e3),
        'gamma': loguniform(1e-04, 1e+01),
        'degree': uniform(2, 5),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        "class_weight": ['balanced', None]
    }
    svc_pipe_param_grid = {}
    for key in svc_param_grid.keys():
        #svc_pipe_param_grid["sequentialfeatureselector__estimator__"+key] = svc_param_grid[key]
        svc_pipe_param_grid["model__"+key] = svc_param_grid[key]
    grids.append(svc_pipe_param_grid)

    lr_param_grid = {
        'C': loguniform(1e-5, 1e4), 
        'penalty': ['l1', 'l2', 'elasticnet'], 
        'class_weight': ['balanced', None], 
        'l1_ratio': uniform(0, 1) 
    }
    lr_pipe_param_grid = {}
    for key in lr_param_grid.keys():
        #lr_pipe_param_grid["sequentialfeatureselector__estimator__"+key] = lr_param_grid[key]
        lr_pipe_param_grid["model__"+key] = lr_param_grid[key]
    grids.append(lr_pipe_param_grid)

    return grids

def make_pipes_and_params(number_of_features_to_check):
    # Make pipelines
    pipes = make_pipelines(number_of_features_to_check)

    # Make params
    params = make_params()

    # Just to get names of base models later
    base_models = [x.__class__.__name__.lower() for x in make_models()]

    pipes_and_param_grids = list(zip(pipes, params, base_models)) # List of tuples of (pipe, param_grid, base_model)
    
    if DEBUG_MODE:
        pipes_and_param_grids = [pipes_and_param_grids[-1]] # Only do LR in debug mode
        #base_models_and_param_grids = [base_models_and_param_grids[-1], base_models_and_param_grids[0]] # Only do LR and RF in debug mode
        pass

    return pipes_and_param_grids

def fit_param_search(pipe, grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=3 if DEBUG_MODE else 8)
    rs = RandomizedSearchCV(estimator=pipe, param_distributions=grid, cv=cv, scoring="roc_auc", n_iter=50 if DEBUG_MODE else 200, n_jobs = -1, verbose=2, error_score="raise")
    
    print(f"Fitting {pipe} with {grid}")
    rs.fit(X_train, y_train) 
    
    return rs

def add_extra_param(best_estimator_and_performance, param, value):
    best_estimator_and_performance.set_params(**{param: value})

def get_fit_param_search_objects_per_diag(X_train, y_train, number_of_features_to_check):
    pipes_and_param_grids = make_pipes_and_params(number_of_features_to_check)
    fit_param_search_objects = {}
    
    for (pipe, grid, base_model) in pipes_and_param_grids:
        rs = fit_param_search(pipe, grid, X_train, y_train)

        # If chosen model is SVM add a predict_proba parameter (not needed for grid search, and slows it down significantly)
        if 'svm'in base_model:
            rs = add_extra_param(rs, param= "model__probability", value=True)

        fit_param_search_objects[base_model] = rs

    return fit_param_search_objects

def get_fit_param_search_objects_dict(datasets, diag_cols, number_of_features_to_check):
    fit_param_search_objects = {}
    for i, diag in enumerate(diag_cols):
        print(diag, f'{i+1}/{len(diag_cols)}')

        X_train = datasets[diag]["X_train_train"]
        y_train = datasets[diag]["y_train_train"]
        
        fit_param_search_objects_diag = get_fit_param_search_objects_per_diag(X_train, y_train, number_of_features_to_check)
        fit_param_search_objects[diag] = fit_param_search_objects_diag

    return fit_param_search_objects

def build_df_with_best_estimators_and_performances_for_diags(fit_param_search_objects):
    best_estimators_and_performances_list = []
    for diag in fit_param_search_objects.keys():
        for base_model in fit_param_search_objects[diag].keys():

            fit_param_search_object_for_model = fit_param_search_objects[diag][base_model]
            
            best_estimator_for_model = fit_param_search_object_for_model.best_estimator_
            best_score_for_model = fit_param_search_object_for_model.best_score_
            sd_of_best_score_for_model = fit_param_search_object_for_model.cv_results_['std_test_score'][fit_param_search_object_for_model.best_index_]
            
            best_estimators_and_performances_list.append([diag, base_model, best_estimator_for_model, best_score_for_model, sd_of_best_score_for_model])

    best_estimators_and_performances_df = pd.DataFrame(best_estimators_and_performances_list, columns = ["Diag", "Model type", "Best estimator", "Best score", "SD of best score"])
    return best_estimators_and_performances_df

def choose_best_base_models_for_diags(df, performance_margin):

    best_base_estimators_and_performances = []

    for diag in df["Diag"].unique():

        # Choose best model for diag
        best_score = df[df["Diag"] == diag]["Best score"].max()
        best_estimator = df[df["Diag"] == diag][df["Best score"] == best_score]["Best estimator"].iloc[0]
        best_model_type = df[df["Diag"] == diag][df["Best score"] == best_score]["Model type"].iloc[0]
        sd_of_best_score = df[df["Diag"] == diag][df["Best score"] == best_score]["SD of best score"].iloc[0]

        # If LogisticRegression is not much worse than the best model, prefer LogisticRegression (much faster than rest)
        if best_model_type != "logisticregression":    
            lr_score = df[df["Diag"] == diag][df["Model type"] == "logisticregression"]["Best score"].iloc[0]
            if best_score - lr_score <= performance_margin:
                best_estimator = df[df["Diag"] == diag][df["Model type"] == "logisticregression"]["Best estimator"].iloc[0]
                best_model_type = df[df["Diag"] == diag][df["Model type"] == "logisticregression"]["Model type"].iloc[0]
                sd_of_best_score = df[df["Diag"] == diag][df["Model type"] == "logisticregression"]["SD of best score"].iloc[0]
        
        best_base_estimators_and_performances.append([diag, best_model_type, best_estimator, best_score, sd_of_best_score])

    restult_df = pd.DataFrame(best_base_estimators_and_performances, columns = ["Diag", "Model type", "Best estimator", "Best score", "SD of best score"]).sort_values(by="Best score", ascending=False).set_index("Diag")
    
    return restult_df

def get_best_estimators_dict(df):
    best_estimators = {}
    for diag in df["Diag"].unique():
        best_estimator = df[df["Diag"] == diag]["Best estimator"].iloc[0]
        best_estimators[diag] = best_estimator
    return best_estimators

def dump_estimators_and_performances(dirs, best_estimators, scores_of_best_estimators, sds_of_scores_of_best_estimators):
    dump(best_estimators, dirs["models_dir"]+'best-estimators.joblib', compress=1)
    dump(scores_of_best_estimators, dirs["reports_dir"]+'scores-of-best-estimators.joblib', compress=1)
    dump(sds_of_scores_of_best_estimators, dirs["reports_dir"]+'sds-of-scores-of-best-estimators.joblib', compress=1)

def save_coefficients_of_lr_models(fit_param_search_objects_dict, datasets, diag_cols, output_dir):
    for diag in diag_cols:
        lr_fit_param_search_object = fit_param_search_objects_dict[diag]["logisticregression"]
        lr_estimator = lr_fit_param_search_object.best_estimator_
        X_train = datasets[diag]["X_train_train"]
        models.save_coefficients_from_lr(diag, lr_estimator, X_train, output_dir)

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

    if models_from_file == 1:
        
        fit_param_search_objects_dict = load(load_dirs["load_models_dir"]+'fit-param-search-objects-dict.joblib')

        dump(fit_param_search_objects_dict, dirs["models_dir"]+'fit-param-search-objects-dict.joblib', compress=1)
    else: 
        # Find best models for each diagnosis
        fit_param_search_objects_dict = get_fit_param_search_objects_dict(datasets, diag_cols, number_of_features_to_check)

        dump(fit_param_search_objects_dict, dirs["models_dir"]+'fit-param-search-objects-dict.joblib', compress=1)
       
    # Save feature coefficients for logistic regression models
    save_coefficients_of_lr_models(fit_param_search_objects_dict, datasets, diag_cols, dirs["reports_dir"])
    
    # Save restuls as a df
    df = build_df_with_best_estimators_and_performances_for_diags(fit_param_search_objects_dict)
    print(df)
    
    # Choose best base model for each diagnosis
    df_best = choose_best_base_models_for_diags(df, performance_margin)
    print(df_best)
    
    # Save best estimators and thresholds 
    dump(df_best, dirs["models_dir"]+'best-estimators-and-performances-df.joblib', compress=1)
    dump(df, dirs["models_dir"]+'estimators-and-performances-df.joblib', compress=1)
    df_best.to_csv(dirs["reports_dir"] + "df_of_best_estimators_and_their_scores.csv", float_format='%.3f')
    df.to_csv(dirs["reports_dir"] + "df_of_estimators_and_their_scores.csv", float_format='%.3f')

    util.print_and_save_string(time.time() - start_time, dirs["reports_dir"], "execution-time.txt")

if __name__ == "__main__":
    main(sys.argv[1])