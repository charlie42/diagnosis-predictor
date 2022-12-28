import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

from joblib import load, dump

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

def set_up_directories():

    input_data_dir = "data/train_models/"
    models_dir = "models/" + "train_models/"
    input_reports_dir = "reports/" + "train_models/"

    output_reports_dir = "reports/" + "identify_feature_subsets/"
    util.create_dir_if_not_exists(output_reports_dir)
    util.clean_dirs([output_reports_dir]) # Remove old reports

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def get_estimator_from_pipeline(pipeline):
    return pipeline.steps[-1][1]

def transform_data_for_rfe(diag, datasets):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    scaler = StandardScaler()
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)
    return X_train, y_train

def get_rfe_objects(best_classifiers, datasets):
    rfe_objects = {}
    for diag in best_classifiers.keys():
        print(diag)
        
        # Impute values outside the pipeline, this can cause a bit of data leakage but we have no choice
        #  until this is fixed https://github.com/scikit-learn/scikit-learn/issues/21743 
        X_train, y_train = transform_data_for_rfe(diag, datasets)
        
        estimator = get_estimator_from_pipeline(best_classifiers[diag])
        print(estimator)

        selector = RFE(estimator, step=1, n_features_to_select=1, verbose=0)
        selector = selector.fit(X_train, y_train)
        rfe_objects[diag] = selector

    return rfe_objects
       
def get_feature_subsets(best_classifiers, datasets, number_of_features_to_check):

    rfe_objects = get_rfe_objects(best_classifiers, datasets) # Get ranking of features from RFE
    feature_subsets = get_feature_subsets_from_rfe(rfe_objects, datasets, number_of_features_to_check) # Create subsets of features from RFE ranking
    
    return feature_subsets

def get_feature_subsets_from_rfe(rfe_objects, datasets, number_of_features_to_check):
    # Get dictionary with number of features as keys and list of features as values
    feature_subsets = {}
    for diag in rfe_objects.keys():
        feature_subsets[diag] = {}

        X_train = datasets[diag]["X_train_train"]

        # Get a dictionnary with the number of features as keys and the list of features as values
        rf_df = pd.DataFrame(rfe_objects[diag].ranking_, index=X_train.columns, columns=["Rank"]).sort_values(by="Rank", ascending=True)
        for n_features in range(1, number_of_features_to_check+1):
            feature_subsets[diag][n_features] = rf_df[rf_df["Rank"] <= n_features].index.tolist()

    return feature_subsets
    
def write_feature_subsets_to_text_file(feature_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"
    util.write_two_lvl_dict_to_file(feature_subsets, path)
    
def main(number_of_features_to_check = 50):
    number_of_features_to_check = int(number_of_features_to_check)

    dirs = set_up_directories()

    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    feature_subsets = get_feature_subsets(best_classifiers, datasets, number_of_features_to_check)
    dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    
    write_feature_subsets_to_text_file(feature_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1])