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
    df = pd.DataFrame(columns=["threshold", "sens", "spec"])
    thresholds = np.arange(0, 1.01, 0.01)
    for threshold in thresholds:
        y_pred_binary = (y_pred_proba[:,1] >= threshold).astype(bool) 
        sens = recall_score(y_test, y_pred_binary)
        spec = recall_score(y_test, y_pred_binary, pos_label=0)

        df = df.append({"threshold": threshold, "sens": sens, "spec": spec}, ignore_index=True)

    print(df)

    sens_closest_to_opt_sens = 2 # sensitivity closest to optimal (search will start with the highest sensitivity)
    # Find the threshold where sensitivity is the closest to opt_sens
    for threshold in thresholds:
        sens = df[df["threshold"] == threshold]["sens"].values[0]
        # If sens is closer to opt_sens than the current closest
        if abs(sens - opt_sens) < abs(sens_closest_to_opt_sens - opt_sens):
            sens_closest_to_opt_sens = sens

    # Find the threshold with the highest specificity where sensitivity is the closest to opt_sens
    spec_highest = 0
    opt_thresh = None
    for threshold in thresholds:
        sens = df[df["threshold"] == threshold]["sens"].values[0]
        spec = df[df["threshold"] == threshold]["spec"].values[0]
        if sens == sens_closest_to_opt_sens and spec > spec_highest:
            spec_highest = spec
            opt_thresh = threshold

    # If found spec is higher than sens, return the threshold with the highest sensitivity where sensitivity is > specificity and sensitivity > opt_sens
    if spec_highest < sens_closest_to_opt_sens:
        for threshold in thresholds:
            sens = df[df["threshold"] == threshold]["sens"].values[0]
            spec = df[df["threshold"] == threshold]["spec"].values[0]
            if sens > spec and sens > opt_sens:
                sens_closest_to_opt_sens = sens
                spec_highest = spec
                opt_thresh = threshold

    print(opt_thresh, sens_closest_to_opt_sens, spec_highest)

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

    return features


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
        k_features=2 if DEV_MODE else 27,
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
        "auc_27": [],
        "opt_ns": [],
        "perf_on_features": {x:[{"auc":[], "opt_thresh":[]}] for x in range(1, 28)}
    }

    # Get cross_val_score at each number of features for each diagnosis
    for fold in cv_perf.split(dataset["X_train"], dataset["y_train"]):
        X_train, y_train = dataset["X_train"].iloc[fold[0]], dataset["y_train"].iloc[fold[0]]
        X_test, y_test = dataset["X_train"].iloc[fold[1]], dataset["y_train"].iloc[fold[1]]

        # Fit the model to get subsets and performance on 27 features
        rs.fit(X_train, y_train)

        # Get perforamnce
        y_pred = rs.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        cv_perf_scores["auc_27"].append(auc)
        print("DEBUG auc 27", auc)

        # Get optimal # features for each diagnosis -- where performance reaches 95% of max performance among all # features
        # (get aurocs from sfs object)
        sfs = rs.best_estimator_.named_steps["selector"]
        opt_n = get_opt_n(sfs, 0.95)
        cv_perf_scores["opt_ns"].append(opt_n)
        print("DEBUG opt n", opt_n)

        # Get performance on each subset
        for subset in range(1, 28):
            # Fit the model to the subset
            features = sfs.get_metric_dict()[subset]["feature_idx"]
            new_model = clone(rs.best_estimator_.named_steps["model"])
            new_model.fit(X_train.iloc[:, features], y_train)

            # Get perforamnce
            y_pred = new_model.predict_proba(X_test.iloc[:, features])[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            cv_perf_scores["perf_on_features"][subset]["auc"].append(auc)
            print("DEBUG auc", subset, auc)

            # Find optimal threshold
            opt_thresh = get_opt_thresh(y_test, y_pred, opt_sens=0.8)
            cv_perf_scores["perf_on_features"][subset]["opt_thresh"].append(opt_thresh)
            print("DEBUG opt thresh", subset, opt_thresh)

    print(cv_perf_scores)

    average_opt_n = np.mean(cv_perf_scores["opt_ns"])
    print("DEBUG average opt n", average_opt_n)
    # Fit the model to get sensitivity and specificity on optimal n features
    pipe_with_best_model = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("model", rs.best_estimator_.named_steps["model"])
        ])
    
    subset_at_opt_n = get_subset_at_n(sfs)
    print("DEBUG subset at opt n", subset_at_opt_n)
    pipe_with_best_model.fit(dataset["X_train"].iloc[:, subset_at_opt_n], dataset["y_train"])

    # Get perforamnce
    y_pred = pipe_with_best_model.predict_proba(dataset["X_test"].iloc[:, subset_at_opt_n])[:, 1]
    # Get sens, spec at optimal threshold
    average_opt_thresh = np.mean(cv_perf_scores["perf_on_features"][average_opt_n]["opt_thresh"])
    print("DEBUG average opt thresh", average_opt_thresh)
    auc = roc_auc_score(dataset["y_test"], y_pred)
    y_pred_binary = (y_pred >= average_opt_thresh).astype(bool)
    sens = recall_score(dataset["y_test"], y_pred_binary)
    spec = recall_score(dataset["y_test"], y_pred_binary, pos_label=0)
    print("DEBUG test set", auc, sens, spec)

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

    return output_name, cv_perf_scores


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

    ### TEST print value counts when doing 3 nested kfolds

    kf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    kf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    kf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for diag in diag_cols:
        print(diag)
        for train, test in kf1.split(datasets[diag]["X_train"], datasets[diag]["y_train"]):
            #print("kf1")
            #print(datasets[diag]["y_train"].iloc[test].value_counts())
            for train2, test2 in kf2.split(datasets[diag]["X_train"].iloc[train], datasets[diag]["y_train"].iloc[train]):
                #print("kf2")
                #print(datasets[diag]["y_train"].iloc[test2].value_counts())
                for train3, test3 in kf3.split(datasets[diag]["X_train"].iloc[train].iloc[train2], datasets[diag]["y_train"].iloc[train].iloc[train2]):
                    #print("kf3")
                    if datasets[diag]["y_train"].iloc[test3].value_counts()[1] < 10:
                        print(datasets[diag]["y_train"].iloc[test3].value_counts())
    #sys.exit()
    ###
    
    # Get cross_val_score at each number of features for each diagnosis
    #cv_perf_scores = {}
    #for diag in diag_cols:
        # for train, test in cv_perf.split(datasets[diag]["X_train"], datasets[diag]["y_train"]):
        #     rs.fit(datasets[diag]["X_train"].iloc[train], datasets[diag]["y_train"].iloc[train])
        #     print("Best subset", rs.best_estimator_.k_feature_names_)
        #     print("Best score", rs.best_score_)
        #     print("Best params", rs.best_params_)
        #     print("Best estimator", rs.best_estimator_)
    #    cv_perf_scores[diag] = np.mean(cross_val_score(rs, datasets[diag]["X_train"], datasets[diag]["y_train"], cv=cv_perf, scoring="roc_auc", n_jobs=-1, verbose=1))

    #print(cv_perf_scores)

    args_list = [(dataset, output_name) for dataset, output_name in zip([datasets[diag] for diag in diag_cols], diag_cols)]

    with multiprocessing.Pool() as pool:
        results = pool.map(parallel_grid_search, args_list)

    # Aggregate the results into a DataFrame and a list of objects
    #result_df = pd.DataFrame()
    scores_objects = {}
    for result in results:
        # Recevice otuput_name, scores, rs object
        output_name, scores = result

        scores_objects[output_name] = scores  # Add rs object to list of objects
        
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
    dump(scores_objects, dirs["models_dir"]+f'scores_objects_lr_debug.joblib')
    ###########


    print("\n\nExecution time:", time.time() - start_time)

if __name__ == "__main__":
    main()