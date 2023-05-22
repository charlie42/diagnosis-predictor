
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

def fit_estimator_on_subset_of_features(best_estimators, diag, X, y):
    new_estimator_base = clone(best_estimators[diag][2])
    new_estimator = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), new_estimator_base)
    new_estimator.fit(X, y)
    return new_estimator

def get_top_n_features(feature_subsets, diag, n):
    features_up_top_n = feature_subsets[diag][n]
    return features_up_top_n

def re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_estimators):
    estimators_on_feature_subsets = {}

    if diag in datasets.keys():
        for nb_features in feature_subsets[diag].keys():
            X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

            # Create new pipeline with the params of the best estimator (need to re-train the imputer on less features)
            top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
            new_estimator = fit_estimator_on_subset_of_features(best_estimators, diag, X_train[top_n_features], y_train)
            estimators_on_feature_subsets[nb_features] = new_estimator
            
    return estimators_on_feature_subsets

def re_train_models_on_feature_subsets(feature_subsets, datasets, best_estimators):
    estimators_on_feature_subsets = {}
    for i, diag in enumerate(feature_subsets):
        if diag in datasets.keys():
            print("Re-training models on feature subsets for output: " + diag + " (" + str(i+1) + "/" + str(len(feature_subsets)) + ")")
            estimators_on_feature_subsets[diag] = re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_estimators)
            
    return estimators_on_feature_subsets