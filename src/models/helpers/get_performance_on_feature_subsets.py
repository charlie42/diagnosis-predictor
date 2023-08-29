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
    metric_names = ['TP','TN','FP','FN','Prevalence','Accuracy','PPV (Precision)','NPV','FDR','FOR','check_Pos','check_Neg','Recall (Sensitivity)','FPR','FNR','TNR (Specificity)','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','MCC','BM','MK','Predicted Positive Ratio','ROC AUC']   

    return (mat_met, metric_names)

def get_metrics(estimator, threshold, X, y):
       
    y_pred_prob = estimator.predict_proba(X)
    y_pred = (y_pred_prob[:,1] >= threshold).astype(bool) 

    metrics, metric_names = get_matrix_metrics(y, y_pred)

    roc_auc = models.get_roc_auc(X, y, estimator)
    metrics.append(roc_auc)
    
    return metrics, metric_names

def get_performances_on_feature_subsets_per_output(diag, feature_subsets, estimators_on_feature_subsets, datasets, use_test_set):

    if diag in datasets.keys():
        if use_test_set == 1:
            X_test, y_test = datasets[diag]["X_test"], datasets[diag]["y_test"]
        else:
            X_test, y_test = datasets[diag]["X_val"], datasets[diag]["y_val"]

        metrics_on_subsets = {}
        
        for nb_features in feature_subsets[diag].keys():
            # Create new pipeline with the params of the best estimator (need to re-train the imputer on less features)
            print("Getting metrics on feature subsets for " + diag + " with " + str(nb_features) + " features")
            top_n_features = models.get_top_n_features(feature_subsets, diag, nb_features)
            new_estimator = estimators_on_feature_subsets[diag][nb_features]
            metrics_on_subsets[nb_features] = {}
            thresholds = np.arange(0.1, 1, 0.01)
            for threshold in thresholds:
                metrics, metric_names = get_metrics(new_estimator, threshold, X_test[top_n_features], y_test)
                relevant_metrics = [
                    metrics[-1], # AUC ROC
                    metrics[metric_names.index("Recall (Sensitivity)")],
                    metrics[metric_names.index("TNR (Specificity)")],
                    metrics[metric_names.index("PPV (Precision)")],
                    metrics[metric_names.index("NPV")]]
                metrics_on_subsets[nb_features][threshold] = relevant_metrics
        
    return metrics_on_subsets

def get_performances_on_feature_subsets(feature_subsets, datasets, estimators_on_feature_subsets, use_test_set):
    performances_on_subsets = {}
    
    for i, diag in enumerate(feature_subsets):
        if diag in datasets.keys():
            print("Getting performances on feature subsets for " + diag + " (" + str(i+1) + "/" + str(len(feature_subsets)) + ")")
            performances_on_subsets[diag] = get_performances_on_feature_subsets_per_output(diag, feature_subsets, estimators_on_feature_subsets, datasets, use_test_set)

    return performances_on_subsets