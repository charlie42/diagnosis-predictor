import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import numpy as np
import pandas as pd

from scipy.stats import loguniform, uniform

from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, recall_score, make_scorer
from sklearn.base import clone
import mlxtend
import multiprocessing

import sys, inspect
from joblib import load, dump
import time

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, models, util

DEBUG_MODE = True
DEV_MODE = True
N_FEATURES_TO_CHECK = 3 if DEV_MODE else 27

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


    data_dir = "../diagnosis_predictor_data/"
    
    load_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/") if models_from_file == 1 else None
    load_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "reports/train_models/") if models_from_file == 1 else None
    
    return {"load_data_dir": load_data_dir, "load_models_dir": load_models_dir, "load_reports_dir": load_reports_dir}

class MyPipeline(Pipeline): # Needed to expose feature importances for RFE
    @property
    def coef_(self):
        return self._final_estimator.coef_
    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

class CSequentialFeatureSelector(mlxtend.feature_selection.SequentialFeatureSelector):
    def predict(self, X):
        X = self.transform(X)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        X = self.transform(X)
        return self.estimator.predict_proba(X)

    #def fit(self, X, y):
     #   self.fit(X, y) # fit helper is the 'old' fit method, which I copied and renamed to fit_helper
     #   self.estimator.fit(self.transform(X), y)
     #   return self

def get_opt_n(sfs, performance_fraction):
    '''
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
    opt_n = np.argmax(avg_performances >= max_performance * performance_fraction) + 1

    return opt_n

def get_opt_thresh(y_test, y_pred_proba, opt_sens):
    ''''
    Find the threshold where sensitivity the closest to opt_sens and sensitivity is > specificity 
    If there are multiple thresholds with the same sensitivity, return the one with the highest specificity
    If at the threshold where sensitivity is the closest to opt_sens, sensitivity is not > specificity, 
    return the threshold with min sensisitivity where sensitivity is > specificity
    opt_sens is a float between 0 and 1
    '''
    
    # Make a df with sens and spec on each thereshold
    thresholds = np.arange(0, 1.01, 0.01)
    df = pd.DataFrame(columns=["sens", "spec"], index=thresholds)
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba >= threshold).astype(bool) 
        sens = recall_score(y_test, y_pred_binary)
        spec = recall_score(y_test, y_pred_binary, pos_label=0)

        df.loc[threshold] = [sens, spec]
        
    df = df.drop_duplicates(subset=["sens"], keep="last") # Keep the last duplicate (the one with the highest spec)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    sens_closest_to_opt_sens = 2 # sensitivity closest to optimal (search will start with the highest sensitivity)
    # Find the threshold where sensitivity is the closest to opt_sens
    print("Looking for sensitivity closest to", opt_sens)
    for threshold in df.index:
        sens = df.loc[threshold,"sens"].item()
        # If sens is closer to opt_sens than the current closest
        if abs(sens - opt_sens) <= abs(sens_closest_to_opt_sens - opt_sens):
            sens_closest_to_opt_sens = sens
    print("Sensitivity closest to", opt_sens, "is", sens_closest_to_opt_sens, "at threshold", df[df["sens"] == sens_closest_to_opt_sens].index[0])

    # Find the threshold with the highest specificity where sensitivity is the closest to opt_sens
    print("Looking for highest specificity where sensitivity is closest to", sens_closest_to_opt_sens)
    spec_highest = 0
    opt_thresh = None
    for threshold in df.index:
        sens = df.loc[threshold,"sens"].item()
        spec = df.loc[threshold,"spec"].item()
        if sens == sens_closest_to_opt_sens and spec >= spec_highest:
            spec_highest = spec
            opt_thresh = threshold
    print("Highest specificity where sensitivity is closest to", sens_closest_to_opt_sens, "is", spec_highest, "at threshold", opt_thresh)

    # If found spec is higher than sens, return the threshold with the highest sensitivity where sensitivity is > specificity and sensitivity > opt_sens
    if spec_highest > sens_closest_to_opt_sens:
        print(f"Looks like specicify is higher than sensitivity: sens {sens_closest_to_opt_sens}, spec {spec_highest}. Looking for highest sensitivity where sensitivity is > specificity and sensitivity is > opt_sens")
        for threshold in df.index:
            sens = df.loc[threshold,"sens"].item()
            spec = df.loc[threshold,"spec"].item()
            if sens > spec and sens >= opt_sens:
                sens_closest_to_opt_sens = sens
                spec_highest = spec
                opt_thresh = threshold
    print("Highest sensitivity where sensitivity is > specificity and sensitivity is > opt_sens is", sens_closest_to_opt_sens, "at threshold", opt_thresh)

    return opt_thresh

def get_subset_at_n(sfs, n):
    '''
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
    features = sfs.get_metric_dict()[n]["feature_idx"]

    return list(features)

def agg_features_and_coefs(feature_lists, coefs_lists):
    '''''
    Use features most folds agreed on
    If no consensus, take one from the first fold
    '''
    print("DEBUG feature_lists", feature_lists)
    print("DEBUG coefs_lists", coefs_lists)
    final_subset = []
    final_coefs = []
    for subset in np.arange(0, len(feature_lists[0]))+1:
        # Get the most common feature at this index from each fold
        features_at_index = [feature_list[subset] for feature_list in feature_lists] # eg AAB
        coefs_at_index = [coefs_list[subset] for coefs_list in coefs_lists] # eg 0.1, 0.2, 0.3 (coefs for A, A, B)
        most_common_feature = max(set(features_at_index), key=features_at_index.count) # eg A
        coef_for_most_common_feature = coefs_at_index[features_at_index.index(most_common_feature)] # eg 0.1

        final_subset.append(most_common_feature)
        final_coefs.append(coef_for_most_common_feature)

    return final_subset, final_coefs

def parallel_grid_search(args):

    dataset, output_name = args

    # Model
    lr = LogisticRegression(solver="saga")
    lgbm = HistGradientBoostingClassifier()

    # Parameters
    lr_param_grid = {'model__C': loguniform(1e-5, 1e4)} if DEV_MODE else {
        'model__C': loguniform(1e-5, 1e4), 
        'model__penalty': ['l1', 'l2', 'elasticnet'], 
        'model__class_weight': ['balanced', None], 
        'model__l1_ratio': uniform(0, 1) 
    }
    lgbm_param_grid = {
        'model__learning_rate': np.logspace(-3, 0, num=100),  # Learning rate values from 0.001 to 1
        'model__max_depth': [3, 5, 7, None],  # Maximum depth of the trees
        'model__max_iter': np.arange(100, 1001, 100),  # Number of boosting iterations
        'model__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'model__l2_regularization': np.logspace(-3, 3, num=100)  # L2 regularization values from 0.001 to 1000
    }

    #for model, grid in zip([lr, lgbm], [lr_param_grid, lgbm_param_grid]):
    # DEBUG
    model = lr
    grid = lr_param_grid

    print("Model", model)

    pipeline_for_fs = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model",model)])
    
    n_splits = 2 if DEV_MODE else 4 if DEBUG_MODE else 10
    cv_rs = StratifiedKFold(n_splits, shuffle=True, random_state=0) #n_splits, shuffle=True, random_state=0)
    cv_fs = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    cv_perf = StratifiedKFold(n_splits, shuffle=True, random_state=0)

    rfe = RFE(
        estimator=pipeline_for_fs,
        importance_getter="named_steps.model.coef_",
        step=1, 
        n_features_to_select=840 if DEV_MODE else 100,
        verbose=1
    )

    # Feature selection
    fs = SFS(
    #fs = CSequentialFeatureSelector(
        estimator=pipeline_for_fs,
        k_features=N_FEATURES_TO_CHECK,
        cv=cv_fs,
        forward=True, 
        floating=True, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=1 if DEV_MODE else 2
    )
    
    # Pipeline
    pipeline_for_rs = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("rfe", rfe),
        ('selector', fs),
        ("model", model)])

    # Search
    rs = RandomizedSearchCV(
        estimator=pipeline_for_rs,
        param_distributions=grid,
        n_iter=2 if DEV_MODE else 50 if DEBUG_MODE else 200, 
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv_rs,
        refit=True,
        error_score='raise',
        random_state=0,
        verbose=1
    )

    # rs = HalvingRandomSearchCV( # Need a lot of folds, otherwise ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
    #     estimator=pipeline_for_rs, 
    #     param_distributions=grid, 
    #     cv=cv_rs, 
    #     scoring="roc_auc",
    #     random_state=0,
    #     max_resources=200, #max_resources=100,
    #     #error_score='raise',
    #     n_jobs = -1, 
    #     verbose=1)

    # Get cross_val_score at each number of features for each diagnosis
    cv_perf_scores = {
        "auc_all_features": [],
        "auc_27": [],
        "auc_27_healthy": [],
        "auc_27_under_8": [],
        "auc_27_8_11": [],
        "auc_27_12_15": [],
        "auc_27_over_15": [],
        "opt_ns": [],
        "avg_features": [],
        "avg_coefs": [],
        "perf_on_features": {x:{"auc":[], "opt_thresh":[], "features":[], "coefs":[]} for x in range(1, N_FEATURES_TO_CHECK+1)}
    }
    rs_objects = []

    # DEBUG
    y_train_only_healthy_controls = dataset["y_train"][dataset["y_train_only_healthy_controls"]]
    y_test_only_healthy_controls = dataset["y_test"][dataset["y_test_only_healthy_controls"]]
    print("DEBUG y_train_only_healthy_controls", len(y_train_only_healthy_controls), sum(y_train_only_healthy_controls))
    print("DEBUG y_test_only_healthy_controls", len(y_test_only_healthy_controls), sum(y_test_only_healthy_controls))
    print("DEBUG y_train", len(dataset["y_train"]), sum(dataset["y_train"]))
    print("DEBUG y_test", len(dataset["y_test"]), sum(dataset["y_test"]))

    # Get cross_val_score at each number of features for each diagnosis
    # If model is logistic regression, get average of coefficients for each feature subset
    for fold in cv_perf.split(dataset["X_train"], dataset["y_train"]):
        X_train, y_train = dataset["X_train"].iloc[fold[0]], dataset["y_train"].iloc[fold[0]]
        X_test, y_test = dataset["X_train"].iloc[fold[1]], dataset["y_train"].iloc[fold[1]]
        # Use dataset["X_train_only_healthy_controls"] mask to get only healthy controls
        X_test_only_healthy_controls = X_test[dataset["X_train_only_healthy_controls"].iloc[fold[1]]]
        y_test_only_healthy_controls = y_test[dataset["X_train_only_healthy_controls"].iloc[fold[1]]]
        # DEBUG :TODO assign X_train, check how many positives there
        X_train_only_healthy_controls = X_train[dataset["X_train_only_healthy_controls"].iloc[fold[0]]]
        y_train_only_healthy_controls = y_train[dataset["X_train_only_healthy_controls"].iloc[fold[0]]]
        print("DEBUG FOLD")
        print("DEBUG y_train_only_healthy_controls", len(y_train_only_healthy_controls), sum(y_train_only_healthy_controls))
        print("DEBUG y_test_only_healthy_controls", len(y_test_only_healthy_controls), sum(y_test_only_healthy_controls))
        print("DEBUG y_train", len(y_train), sum(y_train))
        print("DEBUG y_test", len(y_test), sum(y_test))

        rs_objects.append(clone(rs))

        # Fit rs to get best model and feature subsets
        rs.fit(X_train, y_train)
    
        # Get perforamnce on all features
        pipe_with_best_model = clone(rs.best_estimator_)
        pipe_with_best_model.fit(X_train, y_train)
        y_pred = pipe_with_best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        cv_perf_scores["auc_all_features"].append(auc)
        print("DEBUG auc all features", auc)

        # Get SFS object
        sfs = rs.best_estimator_.named_steps["selector"]

        # Get perforamcne at 27 features
        features = sfs.get_metric_dict()[N_FEATURES_TO_CHECK]["feature_idx"]
        pipe_with_best_model = clone(rs.best_estimator_)
        pipe_with_best_model.fit(X_train.iloc[:, list(features)], y_train)
        y_pred = pipe_with_best_model.predict_proba(X_test.iloc[:, list(features)])[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        cv_perf_scores["auc_27"].append(auc)
        print("DEBUG auc 27", auc)

        # Get performance on only healthy controls on 27 features, only if 
        # more than 20 positive examples in the test set
        if sum(y_test_only_healthy_controls) > 20:
            y_pred = pipe_with_best_model.predict_proba(X_test_only_healthy_controls.iloc[:, list(features)])[:, 1]
            print("DEBUG y_test_only_healthy_controls", y_test_only_healthy_controls, len(y_test_only_healthy_controls), sum(y_test_only_healthy_controls))
            auc = roc_auc_score(y_test_only_healthy_controls, y_pred)
            cv_perf_scores["auc_27_healthy"].append(auc)
            print("DEBUG auc 27 healthy", auc)
        else:
            cv_perf_scores["auc_27_healthy"].append(None)
            print("DEBUG auc 27 healthy", None)

        # Stratified by age
        age_col = "Basic_Demos,Age"
        X_test_under_8, y_test_under_8 = X_test[X_test[age_col] < 8], y_test[X_test[age_col] < 8]
        X_test_8_11, y_test_8_11 = X_test[(X_test[age_col] >= 8) & (X_test[age_col] <= 11)], y_test[(X_test[age_col] >= 8) & (X_test[age_col] <= 11)]
        X_test_12_15, y_test_12_15 = X_test[(X_test[age_col] >= 12) & (X_test[age_col] <= 15)], y_test[(X_test[age_col] >= 12) & (X_test[age_col] <= 15)]
        X_test_over_15, y_test_over_15 = X_test[X_test[age_col] > 15], y_test[X_test[age_col] > 15]
        y_pred = pipe_with_best_model.predict_proba(X_test_under_8.iloc[:, list(features)])[:, 1]
        auc = roc_auc_score(y_test_under_8, y_pred)
        cv_perf_scores["auc_27_under_8"].append(auc)
        print("DEBUG auc 27 under 8", auc)
        y_pred = pipe_with_best_model.predict_proba(X_test_8_11.iloc[:, list(features)])[:, 1]
        auc = roc_auc_score(y_test_8_11, y_pred)
        cv_perf_scores["auc_27_8_11"].append(auc)
        print("DEBUG auc 27 8-11", auc)
        y_pred = pipe_with_best_model.predict_proba(X_test_12_15.iloc[:, list(features)])[:, 1]
        auc = roc_auc_score(y_test_12_15, y_pred)
        cv_perf_scores["auc_27_12_15"].append(auc)
        print("DEBUG auc 27 12-15", auc)
        y_pred = pipe_with_best_model.predict_proba(X_test_over_15.iloc[:, list(features)])[:, 1]
        auc = roc_auc_score(y_test_over_15, y_pred)
        cv_perf_scores["auc_27_over_15"].append(auc)
        print("DEBUG auc 27 over 15", auc)

        # Get optimal # features for each diagnosis -- where performance reaches 95% of max performance among all # features
        # (get aurocs from sfs object)
        opt_n = get_opt_n(sfs, 0.95)
        cv_perf_scores["opt_ns"].append(opt_n)
        print("DEBUG opt n", opt_n)

        # Get performance on each subset
        for subset in range(1, N_FEATURES_TO_CHECK+1):
            features = sfs.get_metric_dict()[subset]["feature_idx"]
            print("DEBUG subset", subset, "features", features, len(features))

            # Fit the model to the subset
            model = clone(rs.best_estimator_.named_steps["model"])
            pipe_with_best_model = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="median")),
                ("scale",StandardScaler()),
                ("model", model)])
            pipe_with_best_model.fit(X_train.iloc[:, list(features)], y_train)

            # Get perforamnce
            y_pred = pipe_with_best_model.predict_proba(X_test.iloc[:, list(features)])[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            cv_perf_scores["perf_on_features"][subset]["auc"].append(auc)
            print("DEBUG auc", subset, auc)

            # Find optimal threshold
            opt_thresh = get_opt_thresh(y_test, y_pred, opt_sens=0.8)
            cv_perf_scores["perf_on_features"][subset]["opt_thresh"].append(opt_thresh)
            print("DEBUG opt thresh", subset, opt_thresh)

            # Get features
            cv_perf_scores["perf_on_features"][subset]["features"].append(features)

            # Get coefficients
            if isinstance(model, LogisticRegression):
                coefs = pipe_with_best_model.named_steps["model"].coef_[0]
                print("DEBUG subset", subset, "coefs", coefs, len(features), len(coefs))
                cv_perf_scores["perf_on_features"][subset]["coefs"].append(coefs)
                print("DEBUG coefs", subset, coefs)

    print(cv_perf_scores)

    average_opt_n = round(np.mean(cv_perf_scores["opt_ns"]))
    print("DEBUG average opt n", average_opt_n)

    # Aggregate features subsets
    cv_perf_scores["avg_features"], cv_perf_scores["avg_coefs"] = agg_features_and_coefs(
        cv_perf_scores["perf_on_features"][average_opt_n]["features"], 
        cv_perf_scores["perf_on_features"][average_opt_n]["coefs"])
    subset_at_opt_n = cv_perf_scores["avg_features"]

    print("DEBUG subset at opt n", subset_at_opt_n)

    cv_perf_scores["avg_threshold"] = np.mean(cv_perf_scores["perf_on_features"][average_opt_n]["opt_thresh"])

    # Get performance using average coefficients for features from the opt n features subset
    if isinstance(model, LogisticRegression):
        coefs = cv_perf_scores["avg_coefs"]
        print("DEBUG coefs", coefs, len(subset_at_opt_n), len(coefs))
        
        # Use coefficients to get performance on test set
        y_pred = np.dot(dataset["X_test"].iloc[:, subset_at_opt_n], coefs)

        # Get auroc, specificity, and sensitivity
        auc = roc_auc_score(dataset["y_test"], y_pred)
        sens = recall_score(dataset["y_test"], y_pred >= cv_perf_scores["avg_threshold"])
        spec = recall_score(dataset["y_test"], y_pred < cv_perf_scores["avg_threshold"], pos_label=0)

        cv_perf_scores["auc_test_set"] = auc
        cv_perf_scores["sens_test_set"] = sens
        cv_perf_scores["spec_test_set"] = spec

    

    
    # cv_scoring = {
    #     'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    # }

    # scores = cross_validate(
    #     rs, 
    #     dataset["X_train"], 
    #     dataset["y_train"], 
    #     cv=cv_perf, 
    #     #scoring="roc_auc", 
    #     scoring=cv_scoring, 
    #     return_estimator=True, 
    #     return_indices=True,
    #     n_jobs=-1, 
    #     verbose=1)

    return output_name, cv_perf_scores, rs_objects


def main():
    start_time = time.time() # Start timer for measuring total time of script

    clinical_config = util.read_config("clinical")
    technical_config = util.read_config("technical")
    number_of_features_to_check = clinical_config["max items in screener"]
    performance_margin = technical_config["performance margin"] # Margin of error for ROC AUC (for prefering logistic regression over other models)

    dirs = set_up_directories()
    load_dirs = set_up_load_directories(models_from_file = 0)

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        #diag_cols = ["Diag.Any Diag"]
        pass

    if DEV_MODE:
        diag_cols = diag_cols[0:2]
        pass

    args_list = [(dataset, output_name) for dataset, output_name in zip([datasets[diag] for diag in diag_cols], diag_cols)]

    with multiprocessing.Pool() as pool:
        results = pool.map(parallel_grid_search, args_list)

    # Aggregate the results into a DataFrame and a list of objects
    #result_df = pd.DataFrame()
    scores_dict = {}
    rs_dict = {}
    for result in results:
        # Recevice otuput_name, scores, rs object
        output_name, scores, rs_objects = result

        scores_dict[output_name] = scores  # Add scores to dict
        rs_dict[output_name] = rs_objects  # Add rs object to list of objects

        
        #mean_auc = pd.Series(scores['test_score']).mean()  # Calculate mean AUC using .mean()
        # mean_auc = pd.Series(scores['test_roc_auc']).mean()  # Calculate mean AUC using .mean()
        # sd_auc = pd.Series(scores['test_roc_auc']).std()  # Calculate sd AUC using .std()

        # result_df = result_df.append(pd.DataFrame({
        #     'output': output_name,
        #     'mean_auc': mean_auc, 
        #     'sd_auc': sd_auc,
        #     }, index=[output_name]))  # Add mean scores to DataFrame

    #result_df = result_df.sort_values(by="mean_auc", ascending=False)
    #print("\n", result_df)

    # Get optimal # features for each diagnosis -- where performance reaches 95% of max performance among all # features
    #optimal_number_of_features = {}
    #for diag in diag_cols:
    #    optimal_number_of_features[diag] = np.argmax(cv_perf_scores[diag] >= np.max(cv_perf_scores[diag]) * performance_margin) + 1

    # Save models, optimal # features, and cross_val_score at each # features
    #dump(rs, dirs["models_dir"]+f'rs_{model}.joblib')
    #dump(optimal_number_of_features, dirs["models_dir"]+f'optimal_number_of_features_{model}.joblib')
    #dump(result_df, dirs["models_dir"]+f'cv_perf_scores_lr_debug.joblib')
    print(scores_dict)
    dump(scores_dict, dirs["models_dir"]+f'scores_objects_lr_debug.joblib')
    dump(rs_dict, dirs["models_dir"]+f'rs_objects_lr_debug.joblib')
    ###########


    print("\n\nExecution time:", time.time() - start_time)

if __name__ == "__main__":
    main()
