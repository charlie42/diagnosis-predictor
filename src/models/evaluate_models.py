import math
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import sys

# Calculate probability threshold
def calculate_threshold(classifier, X_train_train, y_train_train, X_val, y_val, b):
    from numpy import nanargmax
    
    # Fit model on validation set
    classifier.fit(X_train_train, y_train_train)
    
    # Get predicted probabilities values
    y_val_pred_prob = classifier.predict_proba(X_val)
    
    # Calculate precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_prob[:,1])
    
    # Calculate F-scores
    fscores = ((1+b**2) * precisions * recalls) / ((b**2 * precisions) + recalls)
    
    # Locate the index of the largest F-score
    ix = nanargmax(fscores)
    
    threshold = thresholds[ix]
    
    return threshold

# Find best thresholds
def find_best_thresholds(beta, best_classifiers, datasets):
    best_thresholds = {}
    for diag in best_classifiers:
        print(diag)
        best_classifier_for_diag = best_classifiers[diag]
        X_train_train, y_train_train, X_val, y_val = \
            datasets[diag]["X_train_train"], \
            datasets[diag]["y_train_train"], \
            datasets[diag]["X_val"], \
            datasets[diag]["y_val"]
        threshold = calculate_threshold(
            best_classifier_for_diag, 
            X_train_train, y_train_train, X_val, y_val, 
            beta
        )
        best_thresholds[diag] = threshold
    print(best_thresholds)
    return best_thresholds

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
def get_cv_auc_values(best_classifiers, datasets):
    auc_cv_mean = []
    auc_cv_std = []
    for diag in best_classifiers:    
        cv = StratifiedKFold(n_splits=5)
        classifier = best_classifiers[diag]
        X_train, y_train = datasets[diag]["X_train"], datasets[diag]["y_train"]
        auc = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        auc_cv_mean.append(auc.mean())
        auc_cv_std.append(auc.std())
    return (auc_cv_mean, auc_cv_std)

def add_number_of_positive_examples(results, datasets):
    for diag in datasets:
        full_dataset_y = datasets[diag]["y_train"].append(datasets[diag]["y_test"]) # Reconstruct full dataset from train and test
        results.loc[results["Diag"] == diag, "# of Positive Examples"] = full_dataset_y.sum()
    return results

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
    results = add_number_of_positive_examples(results, datasets)

    if use_test_set == False: # If using validation set, also get cross validation AUC
        auc_cv_mean, auc_cv_std = get_cv_auc_values(best_classifiers, datasets)
        results["ROC AUC Mean CV"] = auc_cv_mean
        results["ROC AUC Std CV"] = auc_cv_std
        results["(ROC AUC Mean CV) - 1 SD"] = np.array(auc_cv_mean) - np.array(auc_cv_std)

    return results.sort_values(by="ROC AUC", ascending=False)

# Get diagnoses with good performance
def find_well_performing_diags(results, min_roc_auc_cv):
    well_performing_diags = results[
        (results["ROC AUC Mean CV"] >= min_roc_auc_cv)
    ]["Diag"].values
    return well_performing_diags

def main(beta, threshold_positive_examples):

    # Need this to be able to import local packages
    import sys, os, inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import models, data

    beta, threshold_positive_examples = int(beta), int(threshold_positive_examples)

    models_dir = "models/"
    data_processed_dir = "data/processed/"
    reports_dir = "reports/"

    full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")

    # Get list of column names with "Diag: " prefix, where number of 
    # positive examples is > threshold
    diag_cols = [x for x in full_dataset.columns if x.startswith("Diag: ") and 
                full_dataset[x].sum() > threshold_positive_examples] 

    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')

    datasets = data.create_datasets(full_dataset, diag_cols)

    # Find best probability thresholds for each diagnosis
    best_thresholds = find_best_thresholds(
            beta=beta, 
            best_classifiers=best_classifiers, 
            datasets=datasets
    )
    from joblib import dump
    dump(best_thresholds, models_dir+'best-thresholds.joblib', compress=1)

    # Print performances of models on validation set
    performance_table = models.check_performance(best_classifiers, datasets, best_thresholds, beta=beta, use_test_set=False)
    print(performance_table[['Diag','Recall (Sensitivity)','TNR (Specificity)','ROC AUC Mean CV']].sort_values("ROC AUC Mean CV"))
    performance_table.to_csv(reports_dir+"performance_table_all_featuers.csv", index=False)    

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])