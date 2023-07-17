
from sklearn.model_selection import cross_val_score, StratifiedKFold

def get_cv_auc_from_sfs_dict(datasets, best_estimators, feature_subsets, n_folds):
    # Re-train models on the feature subsets, get cross_val_scores
    cv_scores = {}
    for i, diag in enumerate(feature_subsets.keys()):
        print(f"Re-training models on feature subsets for {diag} ({i+1}/{len(feature_subsets.keys())})")
        X_train_train, y_train_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
        cv_scores[diag] = {}
        for n in feature_subsets[diag].keys():
            print(n)
            features = feature_subsets[diag][n]
            cv_scores[diag][n] = cross_val_score(best_estimators[diag], X_train_train[features], y_train_train, cv=StratifiedKFold(n_splits=n_folds), scoring="roc_auc", n_jobs=-1)

    return cv_scores

def get_cv_auc_from_sfs(datasets, best_estimators, feature_subsets, n_folds):
    get_cv_auc_dict = get_cv_auc_from_sfs_dict(datasets, best_estimators, feature_subsets, n_folds)
    
    # Format the cv_scores[diag][n] dict into a df where cols are diags, rows are n
    df = pd.DataFrame.from_dict(get_cv_auc_dict, orient="index")
    df = df.transpose()
    df = df.rename_axis("N features")
    df = df.reset_index()

    return df