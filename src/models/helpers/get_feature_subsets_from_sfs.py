from sklearn.model_selection import StratifiedKFold

DEBUG_MODE = True

def get_sfs_object(diag, best_estimators, number_of_features_to_check, X_train, y_train):
    from mlxtend.feature_selection import SequentialFeatureSelector
    print(diag)
    print("Fitting SFS...")
    diag_estimator = best_estimators[diag]

    cv = StratifiedKFold(n_splits=2 if DEBUG_MODE else 8)
    sfs = SequentialFeatureSelector(diag_estimator, 
        k_features=number_of_features_to_check,
        forward=True, 
        scoring='roc_auc',
        cv=cv,
        floating=True, 
        verbose=1,
        n_jobs=-1)

    sfs = sfs.fit(X_train, y_train)

    return sfs

def get_top_n_feaures_from_sfs_object(n, sfs_object):
    features_up_top_n = sfs_object.subsets_[n]["feature_names"]
    return list(features_up_top_n)

def get_feature_subsets_from_sfs(diag, best_estimators, number_of_features_to_check, X_train, y_train):
    feature_subsets = {}
    sfs_object = get_sfs_object(diag, best_estimators, number_of_features_to_check, X_train, y_train)
    for n in range(1, number_of_features_to_check+1):
        feature_subsets[n] = get_top_n_feaures_from_sfs_object(n, sfs_object)
    return feature_subsets