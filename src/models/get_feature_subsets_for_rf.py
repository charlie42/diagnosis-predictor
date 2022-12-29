from sklearn.model_selection import StratifiedKFold

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models

def get_first_n_features_from_rfe(diag, best_classifiers, datasets, number_of_features_to_check):
    feature_ranking = models.get_feature_ranking_from_rfe(diag, best_classifiers, datasets)
    best_n_features_from_rfe = feature_ranking[feature_ranking["Rank"] <= number_of_features_to_check].index.tolist()
    return best_n_features_from_rfe

def get_feature_subsets_for_rf(diag, best_classifiers, datasets, number_of_features_to_check):
    # Get first n features from RFE
    best_n_features_from_rfe = get_first_n_features_from_rfe(diag, best_classifiers, datasets, number_of_features_to_check)

    X_train_top_n_features, y_train = datasets[diag]["X_train_train"][best_n_features_from_rfe], datasets[diag]["y_train_train"]

    # Use SFS to sort first n features
    feature_subsets = models.get_feature_subsets_from_sfs(diag, best_classifiers, number_of_features_to_check, X_train_top_n_features, y_train)
    return feature_subsets