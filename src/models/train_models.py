import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from scipy.stats import loguniform, uniform

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.base import clone
import multiprocessing
from copy import deepcopy
import sys, inspect
from joblib import load, dump
import time, math

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, models, util

DEBUG_MODE = True # Run with less folds and iterations to debug code
DEV_MODE = True # The fastest, with the crudest models, can run on local machine
N_FEATURES_TO_CHECK = 2 if DEV_MODE else 27 # Max n features in final assessment

def build_output_dir_name(params_from_create_datasets):
    # Build filename to save output with timestamp and dataset params 

    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params_part = util.build_param_string_for_dir_name(params_from_create_datasets) + "___" +\
                  util.build_param_string_for_dir_name({"debug_mode": DEBUG_MODE})
    
    return datetime_part + "___" + params_part

def set_up_directories():
    # Set up directories to save the output data

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
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
    # Set up where to read the data from

    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.
    # When the script is run on a new location for the first time, there won't be any non-empty directories. 
    #   We only take non-empthy directries when we load existing models (script arguemnt 'models_from_file')

    data_dir = "../diagnosis_predictor_data/"
    
    load_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/") if models_from_file == 1 else None
    load_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "reports/train_models/") if models_from_file == 1 else None
    
    return {"load_data_dir": load_data_dir, "load_models_dir": load_models_dir, "load_reports_dir": load_reports_dir}

def get_opt_n(sfs, performance_fraction):
    '''
    Get optimal number of features from sfs performances object 

    Format of sfs.get_metric_dict() (dict): {
        1: {
            'feature_idx': (306,), 
            'cv_scores': array([0.79714058, 0.737824  ]), 
            'avg_score': 0.7674822865269321, 
            'feature_names': ('306',), 
            'ci_bound': 0.12760932885928136, 
            'std_dev': 0.029658291493562583, 
            'std_err': 0.02965829149356258
            }, 
        2: {
            'feature_idx': (306, 313), 
            ...
        }
    '''

    avg_performances = [sfs.get_metric_dict()[n]["avg_score"] for n in sfs.get_metric_dict().keys()]
    max_performance = np.max(avg_performances)
    optimal_performance = max_performance * performance_fraction
    opt_n = np.argmax(avg_performances >= optimal_performance) + 1
    
    return opt_n

def get_sens_spec_every_thresh(y_test, y_pred_proba):
    # Make a df with sens and spec on each thereshold

    thresholds = np.arange(0, 1.01, 0.005)
    rows = []
    dict = {}
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(bool) 
        sens = recall_score(y_test, y_pred_binary)
        spec = recall_score(y_test, y_pred_binary, pos_label=0)

        rows.append([threshold, sens, spec])
        dict[threshold] = [sens, spec]

    df = pd.DataFrame(rows, columns=["threshold","sens", "spec"])
    
    df = df.drop_duplicates(subset=["sens"], keep="last") # Keep the last duplicate (the one with the highest spec)
    
    return df, dict

def get_opt_thresh(df, opt_sens):
    '''
    Find the threshold where sensitivity the closest to opt_sens and sensitivity is > specificity 
    If there are multiple thresholds with the same sensitivity, return the one with the highest specificity
    If at the threshold where sensitivity is the closest to opt_sens, sensitivity is not > specificity, 
    return the threshold with min sensisitivity where sensitivity is > specificity
    opt_sens is a float between 0 and 1
    '''

    sens_closest_to_opt_sens = 2 # sensitivity closest to optimal (search will start with the highest sensitivity)
    
    df = df.set_index("threshold")

    print("Looking for sensitivity closest to", opt_sens)
    for threshold in df.index:
        sens = df.loc[threshold,"sens"].item()
        # If sens is closer to opt_sens than the current closest
        if abs(sens - opt_sens) <= abs(sens_closest_to_opt_sens - opt_sens):
            sens_closest_to_opt_sens = sens
    print("Sensitivity closest to", opt_sens, "is", sens_closest_to_opt_sens)

    # Find the threshold with the highest specificity where sensitivity is the closest to opt_sens
    print("Looking for highest specificity where sensitivity is closest to", sens_closest_to_opt_sens)
    spec_highest = 0
    for threshold in df.index:
        sens = df.loc[threshold,"sens"].item()
        spec = df.loc[threshold,"spec"].item()
        if sens == sens_closest_to_opt_sens and spec >= spec_highest:
            spec_highest = spec
    print("Highest specificity where sensitivity is closest to", sens_closest_to_opt_sens, "is", spec_highest)

    # If found spec is higher than sens, return the threshold with the highest sensitivity where sensitivity is > specificity and sensitivity > opt_sens
    if spec_highest > sens_closest_to_opt_sens:
        print(f"Looks like specicify is higher than sensitivity: sens {sens_closest_to_opt_sens}, spec {spec_highest}. Looking for highest sensitivity where sensitivity is > specificity and sensitivity is > opt_sens")
        for threshold in df.index:
            sens = df.loc[threshold,"sens"].item()
            spec = df.loc[threshold,"spec"].item()
            if sens > spec and sens >= opt_sens:
                sens_closest_to_opt_sens = sens
                spec_highest = spec
    print("Highest sensitivity where sensitivity is > specificity and sensitivity is > opt_sens is", sens_closest_to_opt_sens)

    # If multiple threshld with the same optimal sens and spec, take the middle one (to better extrapolate to a slightly different test set)
    df_with_opt_sens_spec = df[(df["sens"] == sens_closest_to_opt_sens) & (df["spec"] == spec_highest)]
    print("DEBUG INDEX THRESHOLDS", df_with_opt_sens_spec)
    if len(df_with_opt_sens_spec) == 1:
        opt_thresh = df_with_opt_sens_spec.index[0]
    else:
        opt_thresh = np.median(df_with_opt_sens_spec.index) # Median is always in the set

    return opt_thresh

def parallel_grid_search(args):
    ''' 
    Main function, executed in parallel for each diagnosis. 
    Includes nested CV, with hyperparameter search and feature selection in the 
    inner CV, and performance evaluation on each feature subset in outer.
    Also identify optimal threshold in the outer loop.
    Test set is used to check sensetivity and specificity on the identified threshold.
    Model is retrained on the whole dataset.
    '''

    dataset, output_name = args

    # Model
    lr = LogisticRegression(
        solver="saga", # Need for elasticnet pentalty
        class_weight="balanced", # Adjust probability threshold to prevalence
        penalty="elasticnet",
    )

    # Parameters
    lr_param_grid = {
        'model__C': loguniform(1e-5, 1e4),
        'model__l1_ratio': uniform(0, 1) 
    } 

    model = lr
    grid = lr_param_grid

    n_splits = 2 if DEV_MODE else 4 if DEBUG_MODE else 10 
    test_size = 0.2
    cv_rs = StratifiedShuffleSplit(n_splits, test_size=test_size, random_state=0)
    cv_fs = StratifiedShuffleSplit(n_splits, test_size=test_size, random_state=0)
    cv_perf = StratifiedShuffleSplit(n_splits, test_size=test_size, random_state=0)

    # Pipeline 
    pipeline_for_rs = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ('model', model)])

    # Hyperparameter Search
    rs = RandomizedSearchCV(
        estimator=pipeline_for_rs,
        param_distributions=grid,
        n_iter=20 if DEV_MODE else 200, 
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv_rs,
        refit=True,
        error_score='raise',
        random_state=0,
        verbose=1
    )
    
    # Result object to store performances
    cv_perf_scores = {
        "hp_search_best_score": [],
        "models": [],
        "auc_all_features": [],
        "opt_ns": [],
        "rfe_features": [],
        "n_positives": [], # DEBUG
        "perf_on_features": {x:{
            "auc":[], 
            "auc_sum_score":[], 
            "opt_thresh":[], 
            "features":[], 
            "coefs":[],
            "spec_sens_dict":[], #DEBUG
            } for x in range(1, N_FEATURES_TO_CHECK+1)}
    }
    # Result object to store fitted search objects
    cv_rs_objects = []
    # Result object to store trained models at each number of features
    fitted_models = {x:[] for x in range(1, N_FEATURES_TO_CHECK+1)}

    # Outer CV
    for train_index, test_index in cv_perf.split(dataset["X_train"], dataset["y_train"]):

        X_train, y_train = dataset["X_train"].iloc[train_index], dataset["y_train"].iloc[train_index]
        X_test, y_test = dataset["X_train"].iloc[test_index], dataset["y_train"].iloc[test_index]

        # Log number of positive cases in the train set of the fold
        cv_perf_scores["n_positives"].append({"y_train": y_train.sum(), "y_test": y_test.sum()})
        
        # Fit rs to get best model and feature subsets (inner CV)
        rs.fit(X_train, y_train)
        cv_perf_scores["hp_search_best_score"].append(rs.best_score_)

        cv_rs_objects.append(deepcopy(rs)) # Deepcopy saved trained object
    
        # Get perforamnce on all features for this fold

        # Get model with optimal hyperparameters from the pipeline 
        model = clone(rs.best_estimator_.named_steps["model"]) # Unfitted, with same hyperparams

        # Save model string with optimal parameters
        cv_perf_scores["models"].append(str(rs.best_estimator_.named_steps["model"]))

        # Create pipeline using the model with optimized hyperparams, for feature selection
        pipe_with_best_model = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("model", model)])

        # Create RFE object in a pipeline and fit
        rfe = RFE(
            estimator=pipe_with_best_model,
            importance_getter="named_steps.model.coef_",
            step=1, 
            n_features_to_select=845 if DEV_MODE else 1, # all features, sorted
            verbose=1
        )
        rfe_pipe = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="median")),
            ("scale",StandardScaler()),
            ("rfe", rfe)])
        rfe_pipe.fit(X_train, y_train)

        # Get top N features from RFE to proceed with SFS
        rfe = rfe_pipe.named_steps["rfe"]
        feature_rankings = pd.DataFrame(rfe.ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)
        features = feature_rankings[feature_rankings["Rank"] <= N_FEATURES_TO_CHECK].index.tolist()
        cv_perf_scores["rfe_features"].append(features)
        
        # Fit SFS to data with only the top features identified by RFE
        sfs = SFS(
            estimator=pipe_with_best_model,
            k_features=N_FEATURES_TO_CHECK,
            cv=cv_fs,
            forward=True, 
            floating=True, 
            scoring='roc_auc', 
            n_jobs=-1, 
            verbose=1 if DEV_MODE else 2
        )        
        sfs.fit(X_train[features], y_train)
        
        # Get optimal N features for each diagnosis -- where performance reaches 95% of max performance among all # features
        opt_n = get_opt_n(sfs, 0.95)
        cv_perf_scores["opt_ns"].append(opt_n)

        # Get performance on each subset
        for subset in range(1, N_FEATURES_TO_CHECK+1):
            
            # Get features from SFS object
            features = list(sfs.get_metric_dict()[subset]["feature_names"])
            
            # Fit the model to the subset
            model = clone(rs.best_estimator_.named_steps["model"]) # Unfitted, with same hyperparams
            pipe_with_best_model = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model", model)])
            print(f"Fitting new model to {subset} features, dataset:", X_train[features])
            pipe_with_best_model.fit(X_train[features], y_train)
            fitted_models[subset].append(deepcopy(pipe_with_best_model))

            # Get perforamnce of the trained model on the test set of the fold
            y_pred = pipe_with_best_model.predict_proba(
                X_test[features]
            )[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            cv_perf_scores["perf_on_features"][subset]["auc"].append(auc)

            # Get performance using sum-scores (for scoring without ML)
            sum_score_calculator = models.SumScoreCalculator(
                X_test[features], 
                pipe_with_best_model
            )
            y_pred_sum_score = sum_score_calculator.calculate_sum_score()
            sum_score_auc = roc_auc_score(y_test, y_pred_sum_score)
            cv_perf_scores["perf_on_features"][subset]["auc_sum_score"].append(sum_score_auc)

            # Find optimal threshold
            all_sens_and_specs_df, all_sens_and_specs_dict = get_sens_spec_every_thresh(y_test, y_pred)
            cv_perf_scores["perf_on_features"][subset]["spec_sens_dict"].append(all_sens_and_specs_dict)
            opt_thresh = get_opt_thresh(all_sens_and_specs_df, opt_sens=0.8)
            cv_perf_scores["perf_on_features"][subset]["opt_thresh"].append(opt_thresh)
            
            # Save model coefficients
            if isinstance(model, LogisticRegression):
                coefs = pipe_with_best_model.named_steps["model"].coef_[0]
                cv_perf_scores["perf_on_features"][subset]["coefs"].append(coefs)
                
            cv_perf_scores["perf_on_features"][subset]["features"].append(features)

    # Get sensitivity and specificity on test set with average optimal threshold
    #   and average optiman n features
    avg_opt_n = math.ceil(np.mean(cv_perf_scores["opt_ns"]))
    avg_opt_thresh = np.median(cv_perf_scores["perf_on_features"][avg_opt_n]["opt_thresh"])

    opt_model = fitted_models[avg_opt_n][0] # Fitted model from any fold would work, taking first
    features = cv_perf_scores["perf_on_features"][avg_opt_n]["features"][0]
    y_pred = opt_model.predict_proba(dataset["X_test"][features])[:, 1]
    sens_spec_df, sens_spec_dict = get_sens_spec_every_thresh(dataset["y_test"], y_pred)

    auc = roc_auc_score(dataset["y_test"], y_pred)
    sens = recall_score(dataset["y_test"], y_pred >= avg_opt_thresh)
    spec = recall_score(dataset["y_test"], y_pred >= avg_opt_thresh, pos_label=0)

    # Save test-set performances
    cv_perf_scores["auc_test_set"] = auc
    cv_perf_scores["sens_test_set"] = sens
    cv_perf_scores["spec_test_set"] = spec
    cv_perf_scores["sens_spec_test_set"] = sens_spec_dict

    # Re-train the full pipeline to get the final trained model and feature sets
    X_train, y_train = dataset["X_full"], dataset["y_full"]
    rs.fit(X_train, y_train)
    final_trained_model = deepcopy(rs.best_estimator_.named_steps["model"])
    model = clone(rs.best_estimator_.named_steps["model"]) # Unfitted, with same params
    pipe_with_best_model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model", model)])
    rfe = RFE(
        estimator=pipe_with_best_model,
        importance_getter="named_steps.model.coef_",
        step=1, 
        n_features_to_select=845 if DEV_MODE else 1, # all features sorted (DEV 845)
        verbose=1
    )
    rfe_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("rfe", rfe)])
    rfe_pipe.fit(X_train, y_train)
    rfe = rfe_pipe.named_steps["rfe"]
    feature_rankings = pd.DataFrame(rfe.ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)
    features = feature_rankings[feature_rankings["Rank"] <= N_FEATURES_TO_CHECK].index.tolist()
    sfs = SFS(
        estimator=pipe_with_best_model,
        k_features=N_FEATURES_TO_CHECK,
        cv=cv_fs,
        forward=True, 
        floating=True, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1 if DEV_MODE else 2
    )        
    sfs.fit(X_train[features], y_train)
    
    # Save final feature set
    final_feature_sets = {}
    for subset in range(1, N_FEATURES_TO_CHECK+1):
        features = list(sfs.get_metric_dict()[subset]["feature_names"])
        final_feature_sets[subset] = features

    cv_perf_scores["final_trained_model"] = final_trained_model
    cv_perf_scores["final_feature_sets"] = final_feature_sets

    return output_name, cv_perf_scores, cv_rs_objects, deepcopy(rs)


def main():
    start_time = time.time() # Start timer for measuring total time of script

    # Set up input and output directories
    dirs = set_up_directories()
    load_dirs = set_up_load_directories(models_from_file = 0)

    # Load datasets created in previous step
    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())

    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE: # Only run 3 diiagnoses in debug mode
        diag_cols = ["Diag.Autism Spectrum Disorder", 
                    "Diag.ADHD-Combined Type",
                    "Diag.Specific Learning Disorder with Impairment in Reading (test)"]

    if DEV_MODE: # Only run 1 diagnosis in dev mode
        diag_cols = diag_cols[0:1]

    # Create lists of arguments for the multiprocessing function: diagnosis name, and dataset for the diagnosis
    args_list = [(dataset, output_name) for dataset, output_name in zip([datasets[diag] for diag in diag_cols], diag_cols)]

    # Start multiprocessing pool for each diagnosis
    with multiprocessing.Pool() as pool:
        results = pool.map(parallel_grid_search, args_list)

    # Aggregate the results
    scores_dict = {}
    cv_rs_dict = {}
    final_rs_dict = {}

    for result in results:
        # Recevice otuput_name, scores, rs object
        output_name, scores, cv_rs_objects, final_rs_object = result

        scores_dict[output_name] = scores  # Add scores to dict
        cv_rs_dict[output_name] = cv_rs_objects  # Add cv objects to list of objects
        final_rs_dict[output_name] = final_rs_object  # Add rs object to list of objects

    # Save as .joblib files
    dump(scores_dict, dirs["models_dir"]+f'scores_objects.joblib')
    dump(cv_rs_dict, dirs["models_dir"]+f'cv_rs_objects.joblib')
    dump(final_rs_dict, dirs["models_dir"]+f'final_rs_objects.joblib')

    print("\n\nExecution time:", time.time() - start_time)

if __name__ == "__main__":
    main()
