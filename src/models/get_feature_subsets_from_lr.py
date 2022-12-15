
import pandas as pd

def get_features_in_importance_order(diag, best_classifiers, datasets, n):
    X_train = datasets[diag]["X_train"]
    importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].coef_
    importances = pd.DataFrame(zip(X_train.columns, abs(importances[0])), columns=["Feature", "Importance"])
    importances = importances[importances["Importance"]>0].sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return list(importances["Feature"])[:n]

def get_feature_subsets_from_lr(diag, best_classifiers, datasets, number_of_features_to_check):
    feature_subsets = {}
    for n in range(1, number_of_features_to_check+1):
        feature_subsets[n] = get_features_in_importance_order(diag, best_classifiers, datasets, n)
    return feature_subsets