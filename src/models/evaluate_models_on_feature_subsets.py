import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from joblib import load

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

def re_train_models_on_feature_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check):
    classifiers_on_feature_subsets = {}
    for diag in sfs_objects.keys():
        optimal_nb_features = optimal_nbs_features[diag]

        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        classifiers_on_feature_subsets[diag] = {}

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = models.get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[diag][optimal_nb_features] = new_classifier
    
        # Get performance for model on top X features checked
        top_n_features = models.get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[diag][number_of_features_to_check] = new_classifier
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, optimal_nbs_features, number_of_features_to_check):
    thresholds_on_feature_subsets = {}
    for diag in sfs_objects.keys():

        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
        X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

        optimal_nb_features = optimal_nbs_features[diag]

        thresholds_on_feature_subsets[diag] = {}

        top_n_features = models.get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        thresholds_on_feature_subsets[diag][optimal_nb_features] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][optimal_nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )
        top_n_features = models.get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        thresholds_on_feature_subsets[diag][number_of_features_to_check] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][number_of_features_to_check], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )
    return thresholds_on_feature_subsets

def get_performances_on_sfs_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check, use_test_set=0):
    
    classifiers_on_feature_subsets = re_train_models_on_feature_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, optimal_nbs_features, number_of_features_to_check)

    performances_on_sfs_subsets = {}

    best_thresholds_all_features = get_best_thresholds(best_classifiers, datasets)
    
    for diag in sfs_objects.keys():
        print(diag)
        performances_on_sfs_subsets[diag] = {}

        optimal_nb_features = optimal_nbs_features[diag]

        if use_test_set == 1:
            X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
        else:
            X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

        metrics_on_sfs_subsets = []
        # Get performance for model on all features
        best_classifier = best_classifiers[diag]
        best_threshold = best_thresholds_all_features[diag]
        metrics, metric_names = models.get_metrics(best_classifier, best_threshold, X_test, y_test)
        metrics_on_sfs_subsets.append([
            len(X_test.columns),
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])

        # Get performance for model on optimal number of features
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = models.get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        new_classifier = classifiers_on_feature_subsets[diag][optimal_nb_features]
        new_threshold = thresholds_on_feature_subsets[diag][optimal_nb_features]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test)
        metrics_on_sfs_subsets.append([
            optimal_nb_features,
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])
        
        # Get performance for model on top X features checked
        top_n_features = models.get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        new_classifier = classifiers_on_feature_subsets[diag][number_of_features_to_check]
        new_threshold = thresholds_on_feature_subsets[diag][number_of_features_to_check]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test)
        metrics_on_sfs_subsets.append([
            number_of_features_to_check,
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])
        
        metrics_on_sfs_subsets_df = pd.DataFrame(metrics_on_sfs_subsets, columns=[
            "Number of features",
            "ROC AUC", 
            "Recall (Sensitivity)",
            "TNR (Specificity)",
        ])

        print(metrics_on_sfs_subsets_df)
        performances_on_sfs_subsets[diag] = metrics_on_sfs_subsets_df

    return performances_on_sfs_subsets

def write_performances_on_sfs_subsets_to_file(performances):
    for diag in performances.keys():
        path = reports_dir + "feature_importances_from_sfs/"
        if not os.path.exists(path):
            os.mkdir(path)
        performances[diag].to_csv(path + diag.replace('/', ' ') + ".csv") # :TODO - remove slashes from diagnosis names in preprocessing

def main(number_of_features_to_check = 100):
    number_of_features_to_check = int(number_of_features_to_check)

    sfs_objects = load(models_dir+'sfs_importances_objects.joblib')
    diag_cols = sfs_objects.keys()

    importances_from_sfs = models.get_importances_from_sfs(sfs_objects)
    optimal_nbs_features = models.get_optimal_nbs_features_from_sfs(importances_from_sfs, diag_cols)
    datasets = load(models_dir+'datasets.joblib')
    best_classifiers = load(models_dir+'best-classifiers.joblib')

    performances_on_sfs_subsets = get_performances_on_sfs_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check, use_test_set=1)
    write_performances_on_sfs_subsets_to_file(performances_on_sfs_subsets)

if __name__ == "__main__":
    main(sys.argv[1])