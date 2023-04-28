def get_only_healthy_controls(X, y, diag_cols):
    # Remove people with comorbidities from negative cases 
    
    # Get indices of Y to remove: indices of X where diag is 0 and any other diag is 1
    print(X.columns)
    [print(x) for x in X.columns if "diag" in x]
    indices_to_remove = X.index[(y == 0) & (X[diag_cols].sum(axis=1) > 0)].tolist()

    print(indices_to_remove)
    print(X[indices_to_remove][diag_cols])
    print(y[indices_to_remove])

    # Remove rows from X and y
    X = X.drop(indices_to_remove)
    y = y.drop(indices_to_remove)

    return X, y 