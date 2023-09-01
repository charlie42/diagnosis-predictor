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
from sklearn.model_selection import cross_val_score
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

def parallel_grid_search(args):

    dataset, output_name = args

    # Model
    lr = LogisticRegression(solver="saga")
    lgbm = HistGradientBoostingClassifier()

    # Parameters
    lr_param_grid = {
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
    
    n_splits = 4 if DEBUG_MODE else 10
    cv_rs = StratifiedKFold(n_splits, shuffle=True, random_state=0) #n_splits, shuffle=True, random_state=0)
    cv_fs = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    cv_perf = StratifiedKFold(n_splits, shuffle=True, random_state=0)

    rfe = RFE(
        estimator=pipeline_for_fs,
        importance_getter="named_steps.model.coef_",
        step=1, 
        n_features_to_select=100,  #n_features_to_select=28, 
        verbose=0
    )

    # Feature selection
    fs = SFS(
    #fs = CSequentialFeatureSelector(
        estimator=pipeline_for_fs,
        k_features=27, #k_features=27, 
        cv=cv_fs,
        forward=True, 
        floating=True, 
        scoring='roc_auc', 
        n_jobs=-1, 
        verbose=2
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
        n_iter=2 if DEBUG_MODE else 200, #n_iter=50 if DEBUG_MODE else 200,
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
    
    scores = cross_val_score(rs, dataset["X_train"], dataset["y_train"], cv=cv_perf, scoring="roc_auc", n_jobs=-1, verbose=1)

    return {output_name: scores}


def main(models_from_file = 1):
    start_time = time.time() # Start timer for measuring total time of script

    models_from_file = int(models_from_file)

    clinical_config = util.read_config("clinical")
    technical_config = util.read_config("technical")
    number_of_features_to_check = clinical_config["max items in screener"]
    performance_margin = technical_config["performance margin"] # Margin of error for ROC AUC (for prefering logistic regression over other models)

    dirs = set_up_directories()
    load_dirs = set_up_load_directories(models_from_file)

    datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
    diag_cols = list(datasets.keys())
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    if DEBUG_MODE:
        #diag_cols = diag_cols[0:1]
        #diag_cols = ["Diag.Any Diag"]
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

    # Aggregate the results into a DataFrame
    result_df = pd.DataFrame()
    for result in results:
        output_name, scores = list(result.items())[0]
        mean_score = pd.Series(scores).mean()  # Calculate mean score using .mean()
        result_df[output_name] = [mean_score]  # Add mean score to DataFrame

    result_df = result_df.T  
    print(result_df)


    # Get optimal # features for each diagnosis -- where performance reaches 95% of max performance among all # features
    #optimal_number_of_features = {}
    #for diag in diag_cols:
    #    optimal_number_of_features[diag] = np.argmax(cv_perf_scores[diag] >= np.max(cv_perf_scores[diag]) * performance_margin) + 1

    # Save models, optimal # features, and cross_val_score at each # features
    #dump(rs, dirs["models_dir"]+f'rs_{model}.joblib')
    #dump(optimal_number_of_features, dirs["models_dir"]+f'optimal_number_of_features_{model}.joblib')
    dump(result_df, dirs["models_dir"]+f'cv_perf_scores_lr_debug.joblib')
    ###########


    print("\n\nExecution time:", time.time() - start_time)

if __name__ == "__main__":
    main(sys.argv[1])