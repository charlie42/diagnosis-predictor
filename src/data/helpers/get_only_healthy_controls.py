def get_only_healthy_controls(datasets, diag_cols):
    # Remove people with comorbidities from negative cases
    for diag in diag_cols:
        # Get indices of Ys to drop (where X.diag )
