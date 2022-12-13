import os, sys
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from joblib import dump, load

data_processed_dir = "data/processed/"
models_dir = "models/"
reports_dir = "reports/"

def get_diags_with_lr(best_classifiers):
    diags = []
    for diag in best_classifiers.keys():
        if list(best_classifiers[diag].named_steps.keys())[-1] == "logisticregression":
            diags.append(diag)
    return diags

def get_features_in_importance_order(best_classifiers, datasets, diag_cols):
    features = {}
    for diag in diag_cols:
        X_train = datasets[diag]["X_train"]
        importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].coef_
        importances = pd.DataFrame(zip(X_train.columns, abs(importances[0])), columns=["Feature", "Importance"])
        importances = importances[importances["Importance"]>0].sort_values(by="Importance", ascending=False).reset_index(drop=True)
        features[diag] = list(importances["Feature"])
    return features

def main():
    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')
    
    # Get diagnoses which use logistic regression as base model
    diag_cols = get_diags_with_lr(best_classifiers)
    print(diag_cols)

    datasets = load(models_dir+'datasets.joblib')

    features_in_importance_order = get_features_in_importance_order(best_classifiers, datasets, diag_cols)
    dump(features_in_importance_order, models_dir+'features-in-importance-order.joblib')

if __name__ == "__main__":
    main()