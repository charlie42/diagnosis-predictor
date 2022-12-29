
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def transform_data_for_rfe(diag, datasets):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    return X_train, y_train

def get_rfe_object(diag, best_classifiers, datasets):
        
    # Impute values outside the pipeline, this can cause a bit of data leakage but we have no choice
    #  until this is fixed https://github.com/scikit-learn/scikit-learn/issues/21743 
    X_train, y_train = transform_data_for_rfe(diag, datasets)
        
    estimator = util.get_estimator_from_pipeline(best_classifiers[diag])
    print(estimator)

    selector = RFE(estimator, step=1, n_features_to_select=1, verbose=1)
    selector = selector.fit(X_train, y_train)
    return selector

def get_feature_ranking_from_rfe(diag, best_classifiers, datasets):
    rfe_object = get_rfe_object(diag, best_classifiers, datasets)
    X_train = datasets[diag]["X_train_train"]
    return pd.DataFrame(rfe_object.ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)

def get_feature_subsets_from_rfe(feature_ranking, number_of_features_to_check):
    # Get dictionary with number of features as keys and list of features as values
    feature_subsets = {}

    # Get a dictionnary with the number of features as keys and the list of features as values
    for n_features in range(1, number_of_features_to_check+1):
        feature_subsets[n_features] = feature_ranking[feature_ranking["Rank"] <= n_features].index.tolist()

    return feature_subsets

def get_feature_subsets_for_linear_models(diag, best_classifiers, datasets, number_of_features_to_check):
    feature_ranking = get_feature_ranking_from_rfe(diag, best_classifiers, datasets) # Get ranking of features from RFE
    feature_subsets = get_feature_subsets_from_rfe(feature_ranking, number_of_features_to_check) # Create subsets of features from RFE ranking
    return feature_subsets