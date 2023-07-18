def get_optimal_nb_features(auc_table, max_nb_features=27):

    optimal_nbs_features = {}

    for diag in auc_table.columns:
        # Get max score at number of features in the longest subcsale among those that perform best for each diag (from HBN-scripts repo)
        max_score = auc_table[diag].iloc[0:max_nb_features].max() 

        # Get optimal score (95% of max score)
        optimal_score = max_score * 0.95

        # Get index of the first row with a score >= optimal_score
        optimal_nbs_features[diag] = auc_table[diag][auc_table[diag] >= optimal_score].index[0]

    return optimal_nbs_features