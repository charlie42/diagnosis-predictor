from sklearn.model_selection import StratifiedKFold

def get_sfs_objects(diag, best_classifiers, datasets, number_of_features_to_check):
    from mlxtend.feature_selection import SequentialFeatureSelector
    print(diag)
    diag_classifier = best_classifiers[diag]

    cv = StratifiedKFold(n_splits=10)
    sfs = SequentialFeatureSelector(diag_classifier, 
        k_features=number_of_features_to_check,
        forward=True, 
        scoring='roc_auc',
        cv=cv,
        floating=False, 
        verbose=1,
        n_jobs=-1)

    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    sfs = sfs.fit(X_train, y_train)

    return sfs

def get_top_n_feaures_from_sfs_object(n, sfs_objects, diag):
    features_up_top_n = sfs_objects[diag].subsets_[n]["feature_names"]
    return list(features_up_top_n)

def get_feature_subsets_from_sfs(diag, best_classifiers, datasets, number_of_features_to_check):
    feature_subsets = {}
    sfs_objects = get_sfs_objects(diag, best_classifiers, datasets, number_of_features_to_check)
    for n in range(1, number_of_features_to_check+1):
        feature_subsets[n] = get_top_n_feaures_from_sfs_object(n, sfs_objects, diag)
    return feature_subsets