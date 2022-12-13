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

def get_importances_from_models(best_classifiers, datasets, diag_cols):
    importances_from_models = {}
    for diag in diag_cols:
        X_train = datasets[diag]["X_train"]
        if list(best_classifiers[diag].named_steps.keys())[-1] == "randomforestclassifier":
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].feature_importances_
            importances = pd.DataFrame(zip(X_train.columns, importances), columns=["Feature", "Importance"])
        else:
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].coef_
            importances = pd.DataFrame(zip(X_train.columns, abs(importances[0])), columns=["Feature", "Importance"])
        importances = importances[importances["Importance"]>0].sort_values(by="Importance", ascending=False).reset_index(drop=True)
        importances_from_models[diag] = importances
    return importances_from_models

def plot_importances_from_models(importances):
    path = reports_dir + "figures/feature_importances_from_models/"
    if not os.path.exists(path):
        os.mkdir(path)
    for diag in importances.keys():
        importances[diag].plot(y="Importance")
        diag = diag.replace('/', ' ') # :TODO - remove slashes from diagnosis names in preprocessing
        plt.savefig(path + diag + ".png")

def get_sfs_objects(sfs_importances_from_file, best_classifiers, datasets, diag_cols, number_of_features_to_check):
    if sfs_importances_from_file == 1:
        forward_feature_objects = load(models_dir+'sfs_importances_objects.joblib')
    else:
        from mlxtend.feature_selection import SequentialFeatureSelector
        forward_feature_objects = {}
        for diag in diag_cols:
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

def plot_importances_from_sfs_per_diag(importances_df, diag, optimal_nbs_features, number_of_features_to_check):
    plt.figure(figsize=(16,8))
    plt.plot(importances_df)
    plt.xticks(np.arange(1, number_of_features_to_check, 3))
    plt.vlines(np.arange(1, number_of_features_to_check, 3), ymin=min(importances_df["ROC AUC"]), ymax=max(importances_df["ROC AUC"]), colors='purple', ls=':', lw=1)
    
    # Plot vertical line at the optimal number of features
    plt.vlines(optimal_nbs_features[diag], ymin=min(importances_df["ROC AUC"]), ymax=max(importances_df["ROC AUC"]), colors='red', lw=1)
    # Print optimal number of features on the plot
    plt.text(optimal_nbs_features[diag] + 1, max(importances_df["ROC AUC"]), "Optimal number of features: " + str(optimal_nbs_features[diag]), fontsize=12)

    path = reports_dir + "figures/feature_importances_from_sfs/"
    if not os.path.exists(path):
        os.mkdir(path)
    diag = diag.replace('/', ' ') # :TODO - remove slashes from diagnosis names in preprocessing
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

def get_optimal_nbs_features_from_sfs(sfs_importances, diag_cols):
    optimal_nbs_features = {}
    for diag in diag_cols:
        importances_df = sfs_importances[diag]
        optimal_nb_features = find_elbow(importances_df["ROC AUC"])
        optimal_nbs_features[diag] = optimal_nb_features
    return optimal_nbs_features

def get_top_n_feaures(n, sfs_objects, diag):
    features_up_top_n = sfs_objects[diag].subsets_[n]["feature_names"]
    return list(features_up_top_n)

def plot_importances_from_sfs(sfs_importances, optimal_nb_features, number_of_features_to_check):
    for diag in sfs_importances.keys():
        plot_importances_from_sfs_per_diag(sfs_importances[diag], diag, optimal_nb_features, number_of_features_to_check)

def write_top_n_features_to_file(sfs_objects, optimal_nbs_features, number_of_features_to_check):
    for diag in sfs_objects.keys():
        path = reports_dir + "top_n_features/"
        if not os.path.exists(path):
            os.mkdir(path)
        with open(path + diag.replace('/', ' ') + ".txt", "w") as f:
            features_up_top_n = get_top_n_feaures(optimal_nbs_features[diag], sfs_objects, diag)
            f.write("Top " + str(optimal_nbs_features[diag]) + " features: \n" + str(features_up_top_n) + "\n")

            features_up_top_n = get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
            f.write("Top " + str(number_of_features_to_check) + " features: \n" + str(features_up_top_n) + "\n")

def main(auc_threshold = 0.8, number_of_features_to_check = 100, sfs_importances_from_file = 1):
    auc_threshold = float(auc_threshold)
    sfs_importances_from_file = int(sfs_importances_from_file)
    number_of_features_to_check = int(number_of_features_to_check)

    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')
    scores_of_best_classifiers = load(models_dir+'scores-of-best-classifiers.joblib')
    sds_of_scores_of_best_classifiers = load(models_dir+'sds-of-scores-of-best-classifiers.joblib')
    
    # Get list of column names with "Diag: " prefix, where ROC AUC is over threshold and variance under threshold
    # ROC AUC reference: https://gpsych.bmj.com/content/gpsych/30/3/207.full.pd
    diag_cols = [x for x in sds_of_scores_of_best_classifiers.keys() if  
        scores_of_best_classifiers[x] - sds_of_scores_of_best_classifiers[x] >= auc_threshold]
    print("Diagnoses that passed the threshold: ")
    print(diag_cols)

    datasets = load(models_dir+'datasets.joblib')

    importances_from_models = get_importances_from_models(best_classifiers, datasets, diag_cols)
    dump(importances_from_models, models_dir+'fi-from-models.joblib')
    print(importances_from_models)
    plot_importances_from_models(importances_from_models)

    sfs_objects = get_sfs_objects(sfs_importances_from_file, best_classifiers, datasets, diag_cols, number_of_features_to_check)
    importances_from_sfs = get_importances_from_sfs(sfs_objects)
    print(importances_from_sfs)
    optimal_nbs_features = get_optimal_nbs_features_from_sfs(importances_from_sfs, diag_cols)
    plot_importances_from_sfs(importances_from_sfs, optimal_nbs_features, number_of_features_to_check)
    write_top_n_features_to_file(sfs_objects, optimal_nbs_features, number_of_features_to_check)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])