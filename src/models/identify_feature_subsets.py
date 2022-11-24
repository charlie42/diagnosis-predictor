import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

import sys, os, inspect
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from joblib import load
import numpy as np
from joblib import dump, load

data_processed_dir = "data/processed/"
models_dir = "models/"
reports_dir = "reports/"

def get_importances_from_models(best_classifiers, datasets, diag_cols):
    importances_from_models = {}
    for diag in diag_cols:
        print(diag)
        print(list(best_classifiers[diag].named_steps.keys())[-1])
        X_train = datasets[diag]["X_train"]
        if list(best_classifiers[diag].named_steps.keys())[-1] == "randomforestclassifier":
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].feature_importances_
            importances = pd.DataFrame(zip(X_train.columns, importances), columns=["Feature", "Importance"])
        else:
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].coef_
            importances = pd.DataFrame(zip(X_train.columns, abs(importances[0])), columns=["Feature", "Importance"])
        importances = importances[importances["Importance"]>0].sort_values(by="Importance", ascending=False).reset_index(drop=True)
        importances_from_models[diag] = importances
        print(importances)
    return importances_from_models

def plot_importances_from_models(importances):
    path = reports_dir + "figures/feature_importances_from_models/"
    if not os.path.exists(path):
        os.mkdir(path)
    for diag in importances.keys():
        importances[diag].plot(y="Importance")
        plt.savefig(path + diag + ".png")

def get_sfs_objects(sfs_importances_from_file, best_classifiers, datasets, diag_cols):
    if sfs_importances_from_file == 1:
        forward_feature_objects = load(models_dir+'sfs_importances_objects.joblib')
    else:
        from mlxtend.feature_selection import SequentialFeatureSelector
        forward_feature_objects = {}
        for diag in diag_cols:
            print(diag)
            diag_classifier = best_classifiers[diag]

            cv = StratifiedKFold(n_splits=3)
            sfs = SequentialFeatureSelector(diag_classifier, 
                k_features=100,
                forward=True, 
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1)

            X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
            sfs = sfs.fit(X_train, y_train)

            forward_feature_objects[diag] = sfs
            
            # Overwrite file after adding each new diagnosis, to not lose progress in case script is interrupted
            dump(forward_feature_objects, models_dir+'sfs_importances_objects.joblib')
    
    return forward_feature_objects

def process_sfs_object(sfs_object):
    subsets = sfs_object.subsets_
    importances_list = []
    for key in subsets:
        importances_list.append(subsets[key]['avg_score'])
    importances_df = pd.DataFrame(importances_list, index=subsets.keys(), columns=["ROC AUC"])
    return importances_df

def plot_importances_from_sfs_per_diag(importances_df, diag, optimal_nb_features):
    plt.figure(figsize=(16,8))
    plt.plot(importances_df)
    plt.xticks(np.arange(1, 100, 3))
    plt.vlines(np.arange(1, 100, 3), ymin=min(importances_df["ROC AUC"]), ymax=max(importances_df["ROC AUC"]), colors='purple', ls=':', lw=1)
    
    # Plot vertical line at the optimal number of features
    plt.vlines(optimal_nb_features, ymin=min(importances_df["ROC AUC"]), ymax=max(importances_df["ROC AUC"]), colors='red', ls=':', lw=1)
    # Print optimal number of features on the plot
    plt.text(optimal_nb_features + 1, max(importances_df["ROC AUC"]), "Optimal number of features: " + str(optimal_nb_features), fontsize=12)

    path = reports_dir + "figures/feature_importances_from_sfs/"
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(path + diag + ".png")

def find_elbow(curve):
    # Find elbow of the curve (draw a line from the first to the last point of the curve and then find the data point 
    # that is farthest away from that line) https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve 
    n_points = len(curve)
    all_coord = np.vstack((range(n_points), curve)).T
    first_point = all_coord[0]
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coord - first_point
    import numpy.matlib
    scalar_product = np.sum(vec_from_first * numpy.matlib.repmat(line_vec_norm, n_points, 1), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    idx_of_best_point = np.argmax(dist_to_line)
    return idx_of_best_point + 1

def get_importances_from_sfs(sfs_objects):
    importances = {}
    for diag in sfs_objects.keys():
        importances_df = process_sfs_object(sfs_objects[diag])
        importances[diag] = importances_df
    return importances

def get_optimal_nb_features_from_sfs(sfs_importances):
    for diag in sfs_importances.keys():
        importances_df = sfs_importances[diag]
        optimal_nb_features = find_elbow(importances_df["ROC AUC"])
        return optimal_nb_features

def print_features_up_to_optimal(sfs_importances, optimal_nb_features, sfs_objects):
    for diag in sfs_importances.keys():
        print(diag)
        print("Optimal number of features: ", optimal_nb_features)
        features_up_to_optimal = sfs_objects[diag][optimal_nb_features]["feature_names"]
        print(features_up_to_optimal)

def plot_importances_from_sfs(sfs_importances, optimal_nb_features):
    for diag in sfs_importances.keys():
        plot_importances_from_sfs_per_diag(sfs_importances[diag], diag, optimal_nb_features)


def main(auc_threshold = 0.8, sfs_importances_from_file = 1):
    auc_threshold = float(auc_threshold)
    sfs_importances_from_file = int(sfs_importances_from_file)

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import data

    full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

    # Get list of column names with "Diag: " prefix, where number of 
    # performance is > threshold (ROC AUC reference: https://gpsych.bmj.com/content/gpsych/30/3/207.full.pdf)
    performance_table = pd.read_csv(reports_dir + "performance_table_all_featuers.csv")
    diag_cols = performance_table[performance_table["ROC AUC Mean CV"] >= auc_threshold]["Diag"].tolist()

    best_classifiers = load(models_dir+'best-classifiers.joblib')

    # Create datasets for each diagnosis (different input and output columns)
    datasets = data.create_datasets(full_dataset, diag_cols)

    importances_from_models = get_importances_from_models(best_classifiers, datasets, diag_cols)
    plot_importances_from_models(importances_from_models)

    sfs_objects = get_sfs_objects(sfs_importances_from_file, best_classifiers, datasets, diag_cols)
    importances_from_sfs = get_importances_from_sfs(sfs_objects)
    optimal_nb_features = get_optimal_nb_features_from_sfs(importances_from_sfs)
    plot_importances_from_sfs(importances_from_sfs, optimal_nb_features)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])