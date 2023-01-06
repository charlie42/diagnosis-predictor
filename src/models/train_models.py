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

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RandomizedSearchCV

import sys, inspect

from joblib import load, dump

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, models, util

DEBUG_MODE = False

def build_params_dict_for_dir_name(other_diags_as_input):

    params_dict = {}
    params_dict["other_diag_as_input"] = other_diags_as_input
    params_dict["debug_mode"] = DEBUG_MODE
    return params_dict

def build_output_dir_name(other_diags_as_input, params_from_make_dataset):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = build_params_dict_for_dir_name(other_diags_as_input)
    params_part = models.build_param_string_for_dir_name(params_from_make_dataset) + "___" +  models.build_param_string_for_dir_name(params) 
    
    return datetime_part + "___" + params_part

def replace_assessment_param(params_from_make_dataset, input_questionnaire):
    del params_from_make_dataset['first_dropped_assessment']
    params_from_make_dataset["single_assessment_used"] = input_questionnaire
    return params_from_make_dataset

def set_up_directories(other_diags_as_input, input_questionnaire):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/make_dataset/")

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from make_dataset.py]
    #    - use other diags as input
    #    - debug mode
    params_from_make_dataset = models.get_params_from_current_data_dir_name(input_data_dir)
    new_params = replace_assessment_param(params_from_make_dataset, input_questionnaire)
    current_output_dir_name = build_output_dir_name(other_diags_as_input, new_params)

    output_data_dir = data_dir + "data/train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_data_dir)

    models_dir = data_dir + "models/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(models_dir)

    reports_dir = data_dir + "reports/" + "train_models/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(reports_dir) 

    return {"input_data_dir": input_data_dir, "output_data_dir": output_data_dir, "models_dir": models_dir, "reports_dir": reports_dir}

def set_up_load_directories():
    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.

    data_dir = "../diagnosis_predictor_data/"
    
    load_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/train_models/")
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
        #(svc_pipe, svc_param_grid),
        (lr_pipe, lr_param_grid),
    ]
    if DEBUG_MODE:
        base_models_and_param_grids = [base_models_and_param_grids[-1]] # Only do LR in debug mode
        #base_models_and_param_grids = [base_models_and_param_grids[-1], base_models_and_param_grids[0]] # Only do LR and RF in debug mode
    
    return base_models_and_param_grids

def get_best_classifier(base_model, grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=3 if DEBUG_MODE else 8)
    rs = RandomizedSearchCV(estimator=base_model, param_distributions=grid, cv=cv, scoring="roc_auc", n_iter=50 if DEBUG_MODE else 200, n_jobs = -1, verbose=1)
    
    print("Fitting", base_model, "...")
    rs.fit(X_train, y_train) 
    
    best_estimator = rs.best_estimator_
    best_score = rs.best_score_
    sd_of_score_of_best_estimator = rs.cv_results_['std_test_score'][rs.best_index_]

    # If chosen model is SVM add a predict_proba parameter (not needed for grid search, and slows it down significantly)
    if 'svc' in best_estimator.named_steps.keys():
        best_estimator.set_params(svc__probability=True)

    return (best_estimator, best_score, sd_of_score_of_best_estimator)

def find_best_classifier_for_diag_and_its_score(X_train, y_train, performance_margin):
    base_models_and_param_grids = get_base_models_and_param_grids()
    best_classifiers_and_scores = []
    
    for (base_model, grid) in base_models_and_param_grids:
        best_classifier_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model = get_best_classifier(base_model, grid, X_train, y_train)
        model_type = list(base_model.named_steps.keys())[-1]
        best_classifiers_and_scores.append([model_type, best_classifier_for_model, best_score_for_model, sd_of_score_of_best_estimator_for_model])
    
    best_classifiers_and_scores = pd.DataFrame(best_classifiers_and_scores, columns = ["Model type", "Best classifier", "Best score", "SD of best score"])
    print(best_classifiers_and_scores)
    best_classifier = best_classifiers_and_scores.sort_values("Best score", ascending=False)["Best classifier"].iloc[0]
    best_score = best_classifiers_and_scores[best_classifiers_and_scores["Best classifier"] == best_classifier]["Best score"].iloc[0]
    sd_of_score_of_best_classifier = best_classifiers_and_scores[best_classifiers_and_scores["Best classifier"] == best_classifier]["SD of best score"].iloc[0]
    
    # If LogisticRegression is not much worse than the best model, prefer LogisticRegression (much faster than rest)
    best_base_model = best_classifiers_and_scores[best_classifiers_and_scores["Best classifier"] == best_classifier]["Model type"].iloc[0]
    if best_base_model != "logisticregression":
        lr_score = best_classifiers_and_scores[best_classifiers_and_scores["Model type"] == "logisticregression"]["Best score"].iloc[0]
        print("lr_score: ", lr_score, "; best_score: ", best_score)
        if best_score - lr_score <= performance_margin:
            best_classifier = best_classifiers_and_scores[best_classifiers_and_scores["Model type"] == "logisticregression"]["Best classifier"].iloc[0]
            best_score = best_classifiers_and_scores[best_classifiers_and_scores["Best classifier"] == best_classifier]["Best score"].iloc[0]
            sd_of_score_of_best_classifier = best_classifiers_and_scores[best_classifiers_and_scores["Best classifier"] == best_classifier]["SD of best score"].iloc[0]
        
    print("best classifier:")
    print(best_classifier)
    
    return best_classifier, best_score, sd_of_score_of_best_classifier

def find_diags_w_enough_positive_examples_in_val_set(full_dataset, all_diags, split_percentage, min_pos_examples_val_set):
    diags_w_enough_positive_examples_in_val_set = []
    for diag in all_diags:
        positive_examples_full_ds = full_dataset[full_dataset[diag] == 1].shape[0]
        # First get # of positive examples in the train set, then from those, get # of positive examples in the validation set 
        # (first we split the dataset into train and test set, then we split the train set into train and validation set)
        positive_examples_val_set = positive_examples_full_ds * (1-split_percentage) * split_percentage 
        if positive_examples_val_set >= min_pos_examples_val_set:
            diags_w_enough_positive_examples_in_val_set.append(diag)
    return diags_w_enough_positive_examples_in_val_set

# Find best classifier
def find_best_classifiers_and_scores(datasets, diag_cols, performance_margin):
    best_classifiers = {}
    scores_of_best_classifiers = {}
    sds_of_scores_of_best_classifiers = {}
    for diag in diag_cols:
        print(diag)

        X_train = datasets[diag]["X_train_train"]
        y_train = datasets[diag]["y_train_train"]
        
        best_classifier_for_diag, best_score_for_diag, sd_of_score_of_best_classifier_for_diag = find_best_classifier_for_diag_and_its_score(X_train, y_train, performance_margin)
        best_classifiers[diag] = best_classifier_for_diag
        sds_of_scores_of_best_classifiers[diag] = sd_of_score_of_best_classifier_for_diag
        scores_of_best_classifiers[diag] = best_score_for_diag

        if DEBUG_MODE and util.get_base_model_name_from_pipeline(best_classifiers[diag]) == "logisticregression":
            # In debug mode print top features from LR
            models.print_top_features_from_lr(best_classifiers[diag], X_train, 10)
            
    return best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers

def build_df_of_best_classifiers_and_their_score_sds(best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers, full_dataset):
    best_classifiers_and_score_sds = []
    for diag in best_classifiers.keys():
        best_classifier = best_classifiers[diag]
        score_of_best_classifier = scores_of_best_classifiers[diag]
        sd_of_score_of_best_classifier = sds_of_scores_of_best_classifiers[diag]
        model_type = util.get_base_model_name_from_pipeline(best_classifier)
        number_of_positive_examples = full_dataset[diag].sum()
        best_classifiers_and_score_sds.append([diag, model_type, best_classifier, score_of_best_classifier, sd_of_score_of_best_classifier, number_of_positive_examples])
    best_classifiers_and_score_sds = pd.DataFrame(best_classifiers_and_score_sds, columns = ["Diag", "Model type", "Best classifier", "Best score", "SD of best score", "Number of positive examples"])
    best_classifiers_and_score_sds["Score - SD"] = best_classifiers_and_score_sds['Best score'] - best_classifiers_and_score_sds['SD of best score'] 
    return best_classifiers_and_score_sds

def dump_classifiers_and_performances(dirs, best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers):
    print(dirs["models_dir"])
    dump(best_classifiers, dirs["models_dir"]+'best-classifiers.joblib', compress=1)
    dump(scores_of_best_classifiers, dirs["reports_dir"]+'scores-of-best-classifiers.joblib', compress=1)
    dump(sds_of_scores_of_best_classifiers, dirs["reports_dir"]+'sds-of-scores-of-best-classifiers.joblib', compress=1)

def main(performance_margin = 0.02, use_other_diags_as_input = 1, models_from_file = 1, input_questionnaire = None):
    models_from_file = int(models_from_file)
    use_other_diags_as_input = int(use_other_diags_as_input)
    performance_margin = float(performance_margin) # Margin of error for ROC AUC (for prefering logistic regression over other models)

    dirs = set_up_directories(use_other_diags_as_input, input_questionnaire)

    full_dataset = pd.read_csv(dirs["input_data_dir"] + "item_lvl_w_impairment.csv")

    # Print dataset shape
    print("Full dataset shape: ", full_dataset.shape)

    # Get list of column names with "Diag." prefix, where number of 
    # positive examples is > threshold
    min_pos_examples_val_set = 20
    split_percentage = 0.2
    all_diags = [x for x in full_dataset.columns if x.startswith("Diag.")]
    diag_cols = find_diags_w_enough_positive_examples_in_val_set(full_dataset, all_diags, split_percentage, min_pos_examples_val_set)
    if DEBUG_MODE: # Only use first two diagnoses for debugging
        print(diag_cols)
        #diag_cols = diag_cols[-1:]
        diag_cols = diag_cols
    print(diag_cols)

    if models_from_file == 1:
        load_dirs = set_up_load_directories()
        datasets = load(load_dirs["load_data_dir"]+'datasets.joblib')
        print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

        best_classifiers = load(load_dirs["load_models_dir"]+'best-classifiers.joblib')
        scores_of_best_classifiers = load(load_dirs["load_reports_dir"]+'scores-of-best-classifiers.joblib')
        sds_of_scores_of_best_classifiers = load(load_dirs["load_reports_dir"]+'sds-of-scores-of-best-classifiers.joblib')

        # Save data, models, and reports to newly created directories
        dump(datasets, dirs["output_data_dir"]+'datasets.joblib', compress=1)
        dump_classifiers_and_performances(dirs, best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers)
    else: 
        # Create datasets for each diagnosis (different input and output columns)
        datasets = data.create_datasets(full_dataset, diag_cols, split_percentage, use_other_diags_as_input, [input_questionnaire])
        print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)
        
        dump(datasets, dirs["output_data_dir"]+'datasets.joblib', compress=1)

        # Find best models for each diagnosis
        best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers = find_best_classifiers_and_scores(datasets, diag_cols, performance_margin)
        
        # Save best classifiers and thresholds 
        dump_classifiers_and_performances(dirs, best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers)
       
    df_of_best_classifiers_and_their_score_sds = build_df_of_best_classifiers_and_their_score_sds(best_classifiers, scores_of_best_classifiers, sds_of_scores_of_best_classifiers, full_dataset)
    df_of_best_classifiers_and_their_score_sds.to_csv(dirs["reports_dir"] + "df_of_best_classifiers_and_their_scores.csv")
    print(df_of_best_classifiers_and_their_score_sds)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])