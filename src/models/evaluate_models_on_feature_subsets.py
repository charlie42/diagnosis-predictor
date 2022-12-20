import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from joblib import dump, load

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

def set_up_directories(keep_old_re_trained_models=0):

    input_data_dir = "data/train_models/"
    input_models_dir = "models/" + "train_models/"
    input_reports_dir = "reports/" + "identify_feature_subsets/"

    output_models_dir = "models/" + "evaluate_models_on_feature_subsets/"
    util.create_dir_if_not_exists(output_models_dir)

    output_reports_dir = "reports/" + "evaluate_models_on_feature_subsets/"
    util.create_dir_if_not_exists(output_reports_dir)

    if keep_old_re_trained_models == 0:
        util.clean_dirs([output_models_dir, output_reports_dir]) # Remove old models and reports

    return {"input_data_dir": input_data_dir,  "input_models_dir": input_models_dir, "output_models_dir": output_models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def get_best_thresholds(best_classifiers, datasets):
    best_thresholds = models.find_best_thresholds(
        best_classifiers=best_classifiers, 
        datasets=datasets
        )
    return best_thresholds

def fit_classifier_on_subset_of_features(best_classifiers, diag, X, y):
    new_classifier_base = clone(best_classifiers[diag][2])
    new_classifier = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), new_classifier_base)
    new_classifier.fit(X, y)
    return new_classifier

def make_performance_table(performances_on_subsets):
    performances_on_subsets = pd.DataFrame.from_dict(performances_on_subsets)
    performances_on_subsets.index = range(1, len(performances_on_subsets)+1)
    performances_on_subsets = performances_on_subsets.rename(columns={"index": "Diagnosis"})
    
    return performances_on_subsets

def get_diags_in_order_of_auc_on_max_features(performance_table):
    roc_table =  performance_table.applymap(lambda x: x[0]).iloc[::-1]
    return roc_table.columns[roc_table.loc[roc_table.first_valid_index()].argsort()[::-1]]

def make_auc_table(performance_table, output_reports_dir):
    roc_table = performance_table.applymap(lambda x: x[0])
    print("Before inverse order of rows:")
    print(roc_table)
    # Inverse order of rows
    print("After inverse order of rows:")
    roc_table = roc_table.iloc[::-1]
    print(roc_table)
    # Sort columns by score on 100 features (first row)
    new_columns = get_diags_in_order_of_auc_on_max_features(performance_table)
    roc_table[new_columns].to_csv(output_reports_dir+'auc_on_subsets.csv')

def make_sens_spec_table(performance_table, output_reports_dir):
    sens_spec_table = performance_table.applymap(lambda x: str(x[1]) + "," + str(x[2]))
    # Inverse order of rows
    sens_spec_table = sens_spec_table.iloc[::-1]
    # Sort columns by score on 100 features (first row)
    new_columns = get_diags_in_order_of_auc_on_max_features(performance_table)
    sens_spec_table[new_columns].to_csv(output_reports_dir+'sens_spec_on_subsets.csv')

def get_top_n_features(feature_subsets, diag, n):
    features_up_top_n = feature_subsets[diag][n]
    return features_up_top_n

def re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}

    for nb_features in feature_subsets[diag].keys():
        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[nb_features] = new_classifier
    
    return classifiers_on_feature_subsets

def re_train_models_on_feature_subsets(feature_subsets, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}
    for diag in feature_subsets.keys():
        print("Re-training models on feature subsets for output: " + diag)
        classifiers_on_feature_subsets[diag] = re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_classifiers)
        
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, datasets):
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

    thresholds_on_feature_subsets = {}

    for nb_features in feature_subsets[diag].keys():
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)

        thresholds_on_feature_subsets[nb_features] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )

    return thresholds_on_feature_subsets

def calculate_thresholds_for_feature_subsets(feature_subsets, classifiers_on_feature_subsets, datasets):
    thresholds_on_feature_subsets = {}
    for diag in feature_subsets.keys():
        thresholds_on_feature_subsets[diag] = calculate_thresholds_for_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, datasets)
    return thresholds_on_feature_subsets

def get_performances_on_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, use_test_set):

    if use_test_set == 1:
        X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
    else:
        X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

    metrics_on_subsets = []
    
    for nb_features in feature_subsets[diag].keys():
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
        new_classifier = classifiers_on_feature_subsets[diag][nb_features]
        new_threshold = thresholds_on_feature_subsets[diag][nb_features]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test)
        relevant_metrics = [
            metrics[-1], # AUC ROC
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]]
        metrics_on_subsets.append(relevant_metrics)

    return metrics_on_subsets

def get_performances_on_feature_subsets(feature_subsets, datasets, best_classifiers, use_test_set):
    classifiers_on_feature_subsets = re_train_models_on_feature_subsets(feature_subsets, datasets, best_classifiers)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets(feature_subsets, classifiers_on_feature_subsets, datasets)

    performances_on_subsets = {}
    
    for diag in feature_subsets.keys():
        performances_on_subsets[diag] = get_performances_on_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, use_test_set)

    return performances_on_subsets


def main(models_from_file = 1):
    models_from_file = int(models_from_file)

    dirs = set_up_directories(models_from_file)

    feature_subsets = load(dirs["input_reports_dir"]+'feature-subsets.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')
    best_classifiers = load(dirs["input_models_dir"]+'best-classifiers.joblib')

    if models_from_file == 1:
        performances_on_feature_subsets = load(dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')    
    else:
        performances_on_feature_subsets = get_performances_on_feature_subsets(feature_subsets, datasets, best_classifiers, use_test_set = 1)
        dump(performances_on_feature_subsets, dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')

    performance_table = make_performance_table(performances_on_feature_subsets)
    print(performance_table)
    performance_table.to_csv(dirs["output_reports_dir"]+'performances-on-feature-subsets.csv')

    make_auc_table(performance_table, dirs["output_reports_dir"])
    make_sens_spec_table(performance_table, dirs["output_reports_dir"])

if __name__ == "__main__":
    main(sys.argv[1])