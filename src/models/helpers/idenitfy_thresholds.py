from sklearn.metrics import roc_curve
import numpy as np

# Calculate probability threshold
def calculate_thresholds(classifier, X_train_train, y_train_train, X_val, y_val):
    from numpy import nanargmax

    # Fit model on train set
    classifier.fit(X_train_train, y_train_train)
    
    # Get predicted probabilities values
    y_val_pred_prob = classifier.predict_proba(X_val)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob[:,1], drop_intermediate=False)

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))

    # locate the index of the largest g-mean
    ix = nanargmax(gmeans)
    
    threshold = thresholds[ix]
    
    return [thresholds, threshold]

# Find best thresholds
def find_best_thresholds(best_classifiers, datasets):
    best_thresholds = {}
    for diag in datasets.keys():
        best_classifier_for_diag = best_classifiers[diag]
        X_train_train, y_train_train, X_val, y_val = \
            datasets[diag]["X_train_train"], \
            datasets[diag]["y_train_train"], \
            datasets[diag]["X_val"], \
            datasets[diag]["y_val"]
        thresholds = calculate_thresholds(
            best_classifier_for_diag, 
            X_train_train, y_train_train, X_val, y_val, 
        )
        best_thresholds[diag] = thresholds
    print("Thesholds: ", best_thresholds)
    return best_thresholds