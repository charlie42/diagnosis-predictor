from sklearn.metrics import roc_curve
import numpy as np

# Calculate probability threshold
def calculate_thresholds(estimator, X_train_train, y_train_train, X_val, y_val):
    from numpy import nanargmax

    # Fit model on train set
    estimator.fit(X_train_train, y_train_train)
    
    # Get predicted probabilities values
    y_val_pred_prob = estimator.predict_proba(X_val)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob[:,1], drop_intermediate=False)

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))

    # locate the index of the largest g-mean
    ix = nanargmax(gmeans)
    
    threshold = thresholds[ix]
    
    return [thresholds, threshold]

# Find best thresholds
def find_best_thresholds(best_estimators, datasets):
    best_thresholds = {}
    for i, diag in enumerate(datasets):
        print("Finding best threshold for " + diag + " (" + str(i+1) + "/" + str(len(datasets)) + ")" )
        best_estimator_for_diag = best_estimators[diag]
        X_train_train, y_train_train, X_val, y_val = \
            datasets[diag]["X_train_train"], \
            datasets[diag]["y_train_train"], \
            datasets[diag]["X_val"], \
            datasets[diag]["y_val"]
        thresholds = calculate_thresholds(
            best_estimator_for_diag, 
            X_train_train, y_train_train, X_val, y_val, 
        )
        best_thresholds[diag] = thresholds
    print("Thesholds: ", best_thresholds)
    return best_thresholds