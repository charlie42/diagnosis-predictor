import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

from joblib import load, dump

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

def set_up_directories(keep_old_importances=0):

    input_data_dir = "data/train_models/"
    models_dir = "models/" + "train_models/"
    input_reports_dir = "reports/" + "train_models/"

    output_reports_dir = "reports/" + "identify_feature_subsets/"
    util.create_dir_if_not_exists(output_reports_dir)

    if keep_old_importances == 0:
        util.clean_dirs([output_reports_dir]) # Remove old reports

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def get_base_model_from_pipeline(pipeline):
    return list(pipeline.named_steps.keys())[-1]

def get_estimator_from_pipeline(pipeline):
    return pipeline.steps[-1][1]

def get_feature_subsets(ignore_non_lr_diags, best_classifiers, datasets, number_of_features_to_check):
    # If base model is LR, get feature subsets from LR coefficients (top n features), otherwise from SFS
    feature_subsets = {}
    #for diag in best_classifiers.keys():
    for diag in list(best_classifiers.keys())[:3]:
        base_model = get_base_model_from_pipeline(best_classifiers[diag])
        print(diag, base_model)

        cv = StratifiedKFold(n_splits=2)

        # Impute values outside the pipeline, this can cause a bit of data leakage but we have no choice
        #  until this is fixed https://github.com/scikit-learn/scikit-learn/issues/21743 
        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        scaler = StandardScaler()
        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
        X_train = imputer.fit_transform(X_train)
        X_train = scaler.fit_transform(X_train)
        estimator = get_estimator_from_pipeline(best_classifiers[diag])
        print(estimator)
        selector = RFECV(estimator, step=1, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=0)
        selector = selector.fit(X_train, y_train)
        feature_subsets[diag] = selector
        print(selector.n_features_)

    #     if base_model == "logisticregression":
    #         feature_subsets[diag] = models.get_feature_subsets_from_lr(diag, best_classifiers, datasets, number_of_features_to_check)
    #     elif base_model != "logisticregression" and ignore_non_lr_diags == 0:
    #         feature_subsets[diag] = models.get_feature_subsets_from_sfs(diag, best_classifiers, datasets, number_of_features_to_check)
    return feature_subsets

def get_feature_subsets_from_rfe(rfe, X_train, number_of_features_to_check):
    # Get dictionary with number of features as keys and list of features as values
    feature_subsets = {}
    for diag in rfe.keys():
        #if diag == "Diag: Oppositional Defiant Disorder":
        print(pd.DataFrame.from_dict(rfe[diag].cv_results_))
        print(len(X_train.columns))
        print(rfe[diag].ranking_)
        print(rfe[diag].n_features_)
        rf_df = pd.DataFrame(rfe[diag].ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)
        print(rf_df.head(30))
        print(rf_df.value_counts())

        feature_subsets[diag] = {}
        # From the df, get a dictionnary with the number of features as keys and the list of features as values
        for n_features in range(1, number_of_features_to_check):
            feature_subsets[diag][n_features] = rf_df[rf_df["Rank"] <= n_features].index.tolist()

        #print(feature_subsets[diag])
        
        #for n_features in rfe[diag].keys():
         #   feature_subsets[diag][n_features] = rfe[diag][n_features].support_.tolist()

    return feature_subsets
    
def write_feature_subsets_to_text_file(feature_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"
    util.write_two_lvl_dict_to_file(feature_subsets, path)
    
def main(number_of_features_to_check = 50, importances_from_file = 0, ignore_non_lr_diags = 0):
    number_of_features_to_check = int(number_of_features_to_check)
    importances_from_file = int(importances_from_file)
    ignore_non_lr_diags = int(ignore_non_lr_diags)

    dirs = set_up_directories(importances_from_file)

    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    if importances_from_file == 1:
        feature_subsets = load(dirs["output_reports_dir"]+'feature-subsets.joblib')
        feature_subsets = get_feature_subsets_from_rfe(feature_subsets, datasets["Diag: Autism Spectrum Disorder"]["X_train_train"], number_of_features_to_check)
    else:
        feature_subsets = get_feature_subsets(ignore_non_lr_diags, best_classifiers, datasets, number_of_features_to_check)
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    
    #write_feature_subsets_to_text_file(feature_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])