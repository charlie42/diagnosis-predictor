from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

def transform_data_for_rfe(diag, datasets):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    return X_train, y_train

def get_rfe_object(diag, best_estimators, datasets):
        
    # Impute values outside the pipeline, this can cause a bit of data leakage but we have no choice
    #  until this is fixed https://github.com/scikit-learn/scikit-learn/issues/21743 
    X_train, y_train = transform_data_for_rfe(diag, datasets)
        
    estimator = util.get_estimator_from_pipeline(best_estimators[diag])
    print(estimator)

    selector = RFE(estimator, step=1, n_features_to_select=1, verbose=1)
    selector = selector.fit(X_train, y_train)
    return selector

def get_feature_ranking_from_rfe(diag, best_estimators, datasets):
    rfe_object = get_rfe_object(diag, best_estimators, datasets)
    X_train = datasets[diag]["X_train_train"]
    return pd.DataFrame(rfe_object.ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)

def get_first_n_features_from_rfe(diag, best_estimators, datasets, number_of_features_to_check):
    feature_ranking = get_feature_ranking_from_rfe(diag, best_estimators, datasets)
    best_n_features_from_rfe = feature_ranking[feature_ranking["Rank"] <= number_of_features_to_check].index.tolist()
    return best_n_features_from_rfe

def get_feature_subsets_from_rfe_then_sfs(diag, best_estimators, datasets, number_of_features_to_check):
    # Get first n features from RFE
    best_n_features_from_rfe = get_first_n_features_from_rfe(diag, best_estimators, datasets, number_of_features_to_check)

    X_train_top_n_features, y_train = datasets[diag]["X_train_train"][best_n_features_from_rfe], datasets[diag]["y_train_train"]

    # Use SFS to sort first n features
    feature_subsets = models.get_feature_subsets_from_sfs(diag, best_estimators, number_of_features_to_check, X_train_top_n_features, y_train)
    return feature_subsets