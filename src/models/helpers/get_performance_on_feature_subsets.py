import os, sys, inspect

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import numpy as np
import math

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models

def get_best_thresholds(best_classifiers, datasets):
    best_thresholds = models.find_best_thresholds(
        best_classifiers=best_classifiers, 
        datasets=datasets
        )
    return best_thresholds

def get_matrix_metrics(real_values,pred_values):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]+0.01 # +0.01 To avoid division by 0 errors
    FN = CM[1][0]+0.01
    TP = CM[1][1]+0.01
    FP = CM[0][1]+0.01
    Population = TN+FN+TP+FP
    Prevalence = round( (TP+FN) / Population,2)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    NPV        = round( TN / (TN+FN),4 )
    FDR        = round( FP / (TP+FP),4 )
    FOR        = round( FN / (TN+FN),4 ) 
    check_Pos  = Precision + FDR
    check_Neg  = NPV + FOR
    Recall     = round( TP / (TP+FN),4 )
    FPR        = round( FP / (TN+FP),4 )
    FNR        = round( FN / (TP+FN),4 )
    TNR        = round( TN / (TN+FP),4 ) 
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos      = round( Recall/FPR,4 ) 
    LRNeg      = round( FNR / TNR ,4 )
    #DOR        = round( LRPos/LRNeg)
    DOR        = 1 # FIX, LINE ABOVE
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1   
    Predicted_Positive_Ratio = round( (TP+FP) / Population,2)
    
    mat_met = [TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,MCC,BM,MK,Predicted_Positive_Ratio]
    metric_names = ['TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall (Sensitivity)','FPR','FNR','TNR (Specificity)','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','MCC','BM','MK','Predicted Positive Ratio','ROC AUC']   

    return (mat_met, metric_names)

def get_metrics(classifier, threshold, X, y):
       
    y_pred_prob = classifier.predict_proba(X)
    y_pred = (y_pred_prob[:,1] >= threshold).astype(bool) 

    metrics, metric_names = get_matrix_metrics(y, y_pred)

    roc_auc = models.get_roc_auc(X, y, classifier)
    metrics.append(roc_auc)
    
    return metrics, metric_names

def fit_classifier_on_subset_of_features(best_classifiers, diag, X, y):
    new_classifier_base = clone(best_classifiers[diag][2])
    new_classifier = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), new_classifier_base)
    new_classifier.fit(X, y)
    return new_classifier

def get_top_n_features(feature_subsets, diag, n):
    features_up_top_n = feature_subsets[diag][n]
    return features_up_top_n

def get_cv_scores_on_feature_subsets(feature_subsets, datasets, best_classifiers):
    cv_scores_on_feature_subsets = {}
    
    for diag in datasets.keys():
        print("Getting CV scores on feature subsets for " + diag)
        cv_scores_on_feature_subsets[diag] = []
        for nb_features in feature_subsets[diag].keys():
            X_train, y_train = datasets[diag]["X_train"], datasets[diag]["y_train"]
            top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
            new_classifier = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='median'), StandardScaler(), clone(best_classifiers[diag][2]))
            cv_scores = cross_val_score(new_classifier, X_train[top_n_features], y_train, cv = StratifiedKFold(n_splits=10), scoring='roc_auc')
            cv_scores_on_feature_subsets[diag].append(cv_scores.mean())
    return cv_scores_on_feature_subsets

def re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}

    for nb_features in feature_subsets[diag].keys():
        X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]

        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
        new_classifier = fit_classifier_on_subset_of_features(best_classifiers, diag, X_train[top_n_features], y_train)
        classifiers_on_feature_subsets[nb_features] = new_classifier
    
    return classifiers_on_feature_subsets

def re_train_models_on_feature_subsets(feature_subsets, datasets, best_classifiers):
    classifiers_on_feature_subsets = {}
    for diag in datasets.keys():
        print("Re-training models on feature subsets for output: " + diag)
        classifiers_on_feature_subsets[diag] = re_train_models_on_feature_subsets_per_output(diag, feature_subsets, datasets, best_classifiers)
        
    return classifiers_on_feature_subsets

def calculate_thresholds_for_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, datasets):
    X_train, y_train = datasets[diag]["X_train_train"], datasets[diag]["y_train_train"]
    X_val, y_val = datasets[diag]["X_val"], datasets[diag]["y_val"]

    thresholds_on_feature_subsets = {}

    for nb_features in feature_subsets[diag].keys():
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)

        thresholds_on_feature_subsets[nb_features] = models.calculate_thresholds(
            classifiers_on_feature_subsets[diag][nb_features], 
            X_train[top_n_features], 
            y_train,
            X_val[top_n_features], 
            y_val
            )

    return thresholds_on_feature_subsets

def calculate_thresholds_for_feature_subsets(feature_subsets, classifiers_on_feature_subsets, datasets):
    thresholds_on_feature_subsets = {}
    for diag in datasets.keys():
        thresholds_on_feature_subsets[diag] = calculate_thresholds_for_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, datasets)
    return thresholds_on_feature_subsets

def get_performances_on_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, use_test_set):

    if use_test_set == 1:
        X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
    else:
        X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

    metrics_on_subsets = {}
    
    for nb_features in feature_subsets[diag].keys():
        # Create new pipeline with the params of the best classifier (need to re-train the imputer on less features)
        top_n_features = get_top_n_features(feature_subsets, diag, nb_features)
        new_classifier = classifiers_on_feature_subsets[diag][nb_features]
        thresholds = thresholds_on_feature_subsets[diag][nb_features][0]
        metrics_on_subsets[nb_features] = {}
        for threshold in thresholds:
            metrics, metric_names = get_metrics(new_classifier, threshold, X_test[top_n_features], y_test)
            relevant_metrics = [
                metrics[-1], # AUC ROC
                metrics[metric_names.index("Recall (Sensitivity)")],
                metrics[metric_names.index("TNR (Specificity)")]]
            metrics_on_subsets[nb_features][threshold] = relevant_metrics
        optimal_threshold = thresholds_on_feature_subsets[diag][nb_features][1]

    return metrics_on_subsets, optimal_threshold

def get_performances_on_feature_subsets(feature_subsets, datasets, best_classifiers, use_test_set):
    cv_scores_on_feature_subsets = get_cv_scores_on_feature_subsets(feature_subsets, datasets, best_classifiers)
    classifiers_on_feature_subsets = re_train_models_on_feature_subsets(feature_subsets, datasets, best_classifiers)
    thresholds_on_feature_subsets = calculate_thresholds_for_feature_subsets(feature_subsets, classifiers_on_feature_subsets, datasets)

    performances_on_subsets = {}
    optimal_thresholds = {}
    
    for diag in datasets.keys():
        performances_on_subsets[diag], optimal_thresholds[diag] = get_performances_on_feature_subsets_per_output(diag, feature_subsets, classifiers_on_feature_subsets, thresholds_on_feature_subsets, datasets, use_test_set)

    return performances_on_subsets, cv_scores_on_feature_subsets, optimal_thresholds