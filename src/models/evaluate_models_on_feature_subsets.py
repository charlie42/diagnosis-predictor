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

import models

data_processed_dir = "data/processed/"
models_dir = "models/"
reports_dir = "reports/"

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

def re_train_models_on_feature_subsets_per_output(diag, number_of_features_to_check, sfs_objects, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}

    for nb_features in range(1, number_of_features_to_check+1):
        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = models.get_top_n_feaures(nb_features, sfs_objects, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[nb_features] = new_classifier
    
    return classifiers_on_feature_subsets

def re_train_models_on_feature_subsets(sfs_objects, datasets, best_classifiers, number_of_features_to_check):
    classifiers_on_feature_subsets = {}
    for diag in sfs_objects.keys():
        print("Re-training models on feature subsets for output: " + diag)
        classifiers_on_feature_subsets[diag] = re_train_models_on_feature_subsets_per_output(diag, number_of_features_to_check, sfs_objects, datasets, best_classifiers)
        
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets_per_output(diag, sfs_objects, classifiers_on_feature_subsets, datasets, number_of_features_to_check):
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

    thresholds_on_feature_subsets = {}

    for nb_features in range(1, number_of_features_to_check+1):
        top_n_features = models.get_top_n_feaures(nb_features, sfs_objects, diag)

        thresholds_on_feature_subsets[nb_features] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )

    return thresholds_on_feature_subsets

def calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, number_of_features_to_check):
    thresholds_on_feature_subsets = {}
    for diag in sfs_objects.keys():
        thresholds_on_feature_subsets[diag] = calculate_thresholds_for_feature_subsets_per_output(diag, sfs_objects, classifiers_on_feature_subsets, datasets, number_of_features_to_check)
    return thresholds_on_feature_subsets

def get_performances_on_feature_subsets_per_output(diag, sfs_objects, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, number_of_features_to_check, use_test_set):

    if use_test_set == 1:
        X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
    else:
        X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

    metrics_on_sfs_subsets = []
    
    for nb_features in range(1, number_of_features_to_check+1):
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = models.get_top_n_feaures(nb_features, sfs_objects, diag)
        new_classifier = classifiers_on_feature_subsets[diag][nb_features]
        new_threshold = thresholds_on_feature_subsets[diag][nb_features]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test)
        relevant_metrics = [
            metrics[-1], # AUC ROC
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]]
        metrics_on_sfs_subsets.append(relevant_metrics)

    return metrics_on_sfs_subsets

def make_performance_table(performances_on_sfs_subsets, number_of_features_to_check):
    performances_on_sfs_subsets = pd.DataFrame.from_dict(performances_on_sfs_subsets)
    performances_on_sfs_subsets.index = range(1, number_of_features_to_check+1)
    performances_on_sfs_subsets = performances_on_sfs_subsets.rename(columns={"index": "Diagnosis"})
    
    return performances_on_sfs_subsets

def get_performances_on_feature_subsets(sfs_objects, datasets, best_classifiers, number_of_features_to_check, use_test_set):
    
    classifiers_on_feature_subsets = re_train_models_on_feature_subsets(sfs_objects, datasets, best_classifiers, number_of_features_to_check)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, number_of_features_to_check)

    performances_on_sfs_subsets = {}
    
    for diag in sfs_objects.keys():
        performances_on_sfs_subsets[diag] = get_performances_on_feature_subsets_per_output(diag, sfs_objects, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, number_of_features_to_check, use_test_set)

    return performances_on_sfs_subsets

def get_diags_in_order_of_auc_on_max_features(performance_table):
    roc_table =  performance_table.applymap(lambda x: x[0])
    return roc_table.columns[roc_table.loc[roc_table.first_valid_index()].argsort()[::-1]]

def make_auc_table(performance_table):
    roc_table = performance_table.applymap(lambda x: x[0])
    # Inverse order of rows
    roc_table = roc_table.iloc[::-1]
    # Sort columns by score on 100 features (first row)
    new_columns = get_diags_in_order_of_auc_on_max_features(performance_table)
    roc_table[new_columns].to_csv(reports_dir+'auc_on_sfs_subsets.csv')

def make_sens_spec_table(performance_table):
    sens_spec_table = performance_table.applymap(lambda x: str(x[1]) + "," + str(x[2]))
    # Inverse order of rows
    sens_spec_table = sens_spec_table.iloc[::-1]
    # Sort columns by score on 100 features (first row)
    new_columns = get_diags_in_order_of_auc_on_max_features(performance_table)
    sens_spec_table[new_columns].to_csv(reports_dir+'sens_spec_on_sfs_subsets.csv')


###### FROM MODELS #####

def get_top_n_feaures_from_model(n, importances, diag):
    features_up_top_n = importances[diag]["Feature"].iloc[:n]
    return list(features_up_top_n)

def re_train_models_on_feature_subsets_per_output_from_model(diag, number_of_features_to_check, importances, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}

    for nb_features in range(1, number_of_features_to_check+1):
        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_feaures_from_model(nb_features, importances, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[nb_features] = new_classifier
    
    return classifiers_on_feature_subsets

def re_train_models_on_feature_subsets_from_model(importances, datasets, best_classifiers, number_of_features_to_check):
    classifiers_on_feature_subsets = {}
    for diag in importances.keys():
        print("Re-training models on feature subsets for output: " + diag)
        classifiers_on_feature_subsets[diag] = re_train_models_on_feature_subsets_per_output_from_model(diag, number_of_features_to_check, importances, datasets, best_classifiers)
        
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets_per_output_from_model(diag, importances, classifiers_on_feature_subsets, datasets, number_of_features_to_check):
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

    thresholds_on_feature_subsets = {}

    for nb_features in range(1, number_of_features_to_check+1):
        top_n_features = get_top_n_feaures_from_model(nb_features, importances, diag)

        thresholds_on_feature_subsets[nb_features] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )

    return thresholds_on_feature_subsets

def calculate_thresholds_for_feature_subsets_from_model(importances, classifiers_on_feature_subsets, datasets, number_of_features_to_check):
    thresholds_on_feature_subsets = {}
    for diag in importances.keys():
        thresholds_on_feature_subsets[diag] = calculate_thresholds_for_feature_subsets_per_output_from_model(diag, importances, classifiers_on_feature_subsets, datasets, number_of_features_to_check)
    return thresholds_on_feature_subsets

def get_performances_on_feature_subsets_per_output_from_model(diag, importances, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, number_of_features_to_check, use_test_set):

    if use_test_set == 1:
        X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
    else:
        X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

    metrics_on_sfs_subsets = []
    
    for nb_features in range(1, number_of_features_to_check+1):
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_feaures_from_model(nb_features, importances, diag)
        new_classifier = classifiers_on_feature_subsets[diag][nb_features]
        new_threshold = thresholds_on_feature_subsets[diag][nb_features]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test)
        relevant_metrics = [
            metrics[-1], # AUC ROC
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]]
        metrics_on_sfs_subsets.append(relevant_metrics)

    return metrics_on_sfs_subsets

def get_performances_on_feature_subsets_from_model(datasets, best_classifiers, number_of_features_to_check, use_test_set):
    importances = load(models_dir+'fi-from-models.joblib')

    classifiers_on_feature_subsets = re_train_models_on_feature_subsets_from_model(importances, datasets, best_classifiers, number_of_features_to_check)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets_from_model(importances, classifiers_on_feature_subsets, datasets, number_of_features_to_check)

    performances_on_sfs_subsets = {}
    
    for diag in importances.keys():
        performances_on_sfs_subsets[diag] = get_performances_on_feature_subsets_per_output_from_model(diag, importances, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, number_of_features_to_check, use_test_set)

    return performances_on_sfs_subsets

def main(number_of_features_to_check = 100, models_from_file = 1):
    number_of_features_to_check = int(number_of_features_to_check)
    models_from_file = int(models_from_file)

    sfs_objects = load(models_dir+'sfs_importances_objects.joblib')
    datasets = load(models_dir+'datasets.joblib')
    best_classifiers = load(models_dir+'best-classifiers.joblib')

    if models_from_file == 1:
        performances_on_sfs_subsets = load(models_dir+'performances_on_sfs_subsets.joblib')    
    else:
        performances_on_sfs_subsets = get_performances_on_feature_subsets(sfs_objects, datasets, best_classifiers, number_of_features_to_check, use_test_set = 0)
        dump(performances_on_sfs_subsets, models_dir+'performances_on_sfs_subsets.joblib')

    performance_table = make_performance_table(performances_on_sfs_subsets, number_of_features_to_check)
    print("FROM SFS")
    print(performance_table)
    performance_table.to_csv(reports_dir+'performances_on_sfs_subsets.csv')

    make_auc_table(performance_table)
    make_sens_spec_table(performance_table)

    #### FROM MODEL ####
    performances_on_fi_from_model_subsets = get_performances_on_feature_subsets_from_model(datasets, best_classifiers, number_of_features_to_check, use_test_set = 0)
    performance_table = make_performance_table(performances_on_fi_from_model_subsets, number_of_features_to_check)
    print("FROM MODEL")
    print(performance_table)
    performance_table.to_csv(reports_dir+'performances_on_subsets_from_model.csv')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])