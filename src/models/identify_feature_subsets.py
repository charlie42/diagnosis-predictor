import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from joblib import dump, load

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import data
import models

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

            cv = StratifiedKFold(n_splits=3)
            sfs = SequentialFeatureSelector(diag_classifier, 
                k_features=number_of_features_to_check,
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

def get_best_thresholds(beta, best_classifiers, datasets):
    best_thresholds = models.find_best_thresholds(
        beta=beta, 
        best_classifiers=best_classifiers, 
        datasets=datasets
        )
    return best_thresholds

def fit_classifier_on_subset_of_features(best_classifiers, diag, X, y):
    new_classifier_base = clone(best_classifiers[diag][2])
    new_classifier = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), new_classifier_base)
    new_classifier.fit(X, y)
    return new_classifier

def write_performances_on_sfs_subsets_to_file(performances):
    for diag in performances.keys():
        path = reports_dir + "feature_importances_from_sfs/"
        if not os.path.exists(path):
            os.mkdir(path)
        performances[diag].to_csv(path + diag.replace('/', ' ') + ".csv") # :TODO - remove slashes from diagnosis names in preprocessing

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
    

def re_train_models_on_feature_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check):
    classifiers_on_feature_subsets = {}
    for diag in datasets.keys():
        optimal_nb_features = optimal_nbs_features[diag]

        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        classifiers_on_feature_subsets[diag] = {}

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[diag][optimal_nb_features] = new_classifier
    
        # Get performance for model on top X features checked
        top_n_features = get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[diag][number_of_features_to_check] = new_classifier
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, optimal_nbs_features, number_of_features_to_check, beta):
    thresholds_on_feature_subsets = {}
    for diag in datasets.keys():

        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
        X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

        optimal_nb_features = optimal_nbs_features[diag]

        thresholds_on_feature_subsets[diag] = {}

        top_n_features = get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        thresholds_on_feature_subsets[diag][optimal_nb_features] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][optimal_nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val,
            beta
            )
        top_n_features = get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        thresholds_on_feature_subsets[diag][number_of_features_to_check] = models.calculate_threshold(
            classifiers_on_feature_subsets[diag][number_of_features_to_check], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val,
            beta
            )
    return thresholds_on_feature_subsets

def get_performances_on_sfs_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, beta, number_of_features_to_check, use_test_set=0):
    
    classifiers_on_feature_subsets = re_train_models_on_feature_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, number_of_features_to_check)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets(sfs_objects, classifiers_on_feature_subsets, datasets, optimal_nbs_features, number_of_features_to_check, beta)

    performances_on_sfs_subsets = {}

    best_thresholds_all_features = get_best_thresholds(beta, best_classifiers, datasets)
    
    for diag in datasets.keys():
        print(diag)
        performances_on_sfs_subsets[diag] = {}

        optimal_nb_features = optimal_nbs_features[diag]

        if use_test_set == 1:
            X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
        else:
            X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

        metrics_on_sfs_subsets = []
        # Get performance for model on all features
        best_classifier = best_classifiers[diag]
        best_threshold = best_thresholds_all_features[diag]
        metrics, metric_names = models.get_metrics(best_classifier, best_threshold, X_test, y_test, beta)
        metrics_on_sfs_subsets.append([
            len(X_test.columns),
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])

        # Get performance for model on optimal number of features
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_feaures(optimal_nb_features, sfs_objects, diag)
        new_classifier = classifiers_on_feature_subsets[diag][optimal_nb_features]
        new_threshold = thresholds_on_feature_subsets[diag][optimal_nb_features]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test, beta)
        metrics_on_sfs_subsets.append([
            optimal_nb_features,
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])
        
        # Get performance for model on top X features checked
        top_n_features = get_top_n_feaures(number_of_features_to_check, sfs_objects, diag)
        new_classifier = classifiers_on_feature_subsets[diag][number_of_features_to_check]
        new_threshold = thresholds_on_feature_subsets[diag][number_of_features_to_check]
        metrics, metric_names = models.get_metrics(new_classifier, new_threshold, X_test[top_n_features], y_test, beta)
        metrics_on_sfs_subsets.append([
            number_of_features_to_check,
            metrics[-1], 
            metrics[metric_names.index("Recall (Sensitivity)")],
            metrics[metric_names.index("TNR (Specificity)")]])
        
        metrics_on_sfs_subsets_df = pd.DataFrame(metrics_on_sfs_subsets, columns=[
            "Number of features",
            "ROC AUC", 
            "Recall (Sensitivity)",
            "TNR (Specificity)",
        ])

        print(metrics_on_sfs_subsets_df)
        performances_on_sfs_subsets[diag] = metrics_on_sfs_subsets_df

    return performances_on_sfs_subsets

def main(beta = 3, auc_threshold = 0.8, number_of_features_to_check = 100, performance_margin = 0.03, sfs_importances_from_file = 1):
    beta = float(beta)
    auc_threshold = float(auc_threshold)
    sfs_importances_from_file = int(sfs_importances_from_file)
    number_of_features_to_check = int(number_of_features_to_check)
    performance_margin = float(performance_margin)

    full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')
    scores_of_best_classifiers = load(models_dir+'scores-of-best-classifiers.joblib')
    sds_of_scores_of_best_classifiers = load(models_dir+'sds-of-scores-of-best-classifiers.joblib')
    
    # Get list of column names with "Diag: " prefix, where ROC AUC is over threshold and variance under threshold
    # ROC AUC reference: https://gpsych.bmj.com/content/gpsych/30/3/207.full.pd
    # diag_cols = [x for x in sds_of_scores_of_best_classifiers.keys() if  
    #             sds_of_scores_of_best_classifiers[x] <= performance_margin and 
    #             scores_of_best_classifiers[x] >= auc_threshold]

    diag_cols = [x for x in sds_of_scores_of_best_classifiers.keys() if  
        scores_of_best_classifiers[x] - sds_of_scores_of_best_classifiers[x] >= auc_threshold]
    print("Diagnoses that passed the threshold: ")
    print(diag_cols)

    # Create datasets for each diagnosis (different input and output columns)
    datasets = data.create_datasets(full_dataset, diag_cols)

    importances_from_models = get_importances_from_models(best_classifiers, datasets, diag_cols)
    plot_importances_from_models(importances_from_models)

    sfs_objects = get_sfs_objects(sfs_importances_from_file, best_classifiers, datasets, diag_cols, number_of_features_to_check)
    importances_from_sfs = get_importances_from_sfs(sfs_objects)
    optimal_nbs_features = get_optimal_nbs_features_from_sfs(importances_from_sfs, diag_cols)
    plot_importances_from_sfs(importances_from_sfs, optimal_nbs_features, number_of_features_to_check)
    performances_on_sfs_subsets = get_performances_on_sfs_subsets(sfs_objects, optimal_nbs_features, datasets, best_classifiers, beta, number_of_features_to_check, use_test_set=1)
    write_performances_on_sfs_subsets_to_file(performances_on_sfs_subsets)
    write_top_n_features_to_file(sfs_objects, optimal_nbs_features, number_of_features_to_check)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])