import math
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd

metric_names = ['TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall (Sensitivity)','FPR','FNR','TNR (Specificity)','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK','Predicted Positive Ratio','ROC AUC']   

def get_matrix_metrics(real_values,pred_values,beta):
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
    DOR        = round( LRPos/LRNeg)
    #DOR        = 1 # FIX, LINE ABOVE
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1   
    Predicted_Positive_Ratio = round( (TP+FP) / Population,2)
    
    mat_met = [TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK,Predicted_Positive_Ratio]
    return (mat_met)

def get_metrics(classifier, threshold, X, y, beta):
       
    y_pred_prob = classifier.predict_proba(X)
    y_pred = (y_pred_prob[:,1] >= threshold).astype(bool) 

    metrics = get_matrix_metrics(y, y_pred, beta=beta)

    roc_auc = roc_auc_score(y, y_pred_prob[:,1])
    metrics.append(roc_auc)
    
    return metrics

# Do cross-validation to get more reliable ROC AUC scores (f1 harder to obtain with cross validation - need to change threshold)
def get_cv_auc_values(classifier, datasets, diag):
    cv = StratifiedKFold(n_splits=5)
    X_train, y_train = datasets[diag]["X_train"], datasets[diag]["y_train"]
    auc = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    return (auc.mean(), auc.std())

def check_performance(best_classifiers, datasets, best_thresholds, beta, use_test_set=False):
    results = []
    for diag in best_classifiers:
        print(diag)
        classifier = best_classifiers[diag]
        threshold = best_thresholds[diag]
        if use_test_set:
            X, y = datasets[diag]["X_test"], datasets[diag]["y_test"]
        else:
            X, y = datasets[diag]["X_val"], datasets[diag]["y_val"]

        metrics = get_metrics(classifier, threshold, X, y, beta)
        results.append([
            diag, 
            *metrics])

    results = pd.DataFrame(results, columns=["Diag"]+metric_names)

    if use_test_set == False: # If using validation set, also get cross validation AUC
            cv_auc_mean, cv_auc_std = get_cv_auc_values(classifier, datasets, diag)
            results["ROC AUC Mean CV"] = pd.Series(cv_auc_mean)
            results["ROC AUC Std CV"] = pd.Series(cv_auc_std)

    return results.sort_values(by="ROC AUC", ascending=False)

