
def find_thresholds_sens_over_n(val_set_performances, n):
    results = {}
    print("D", val_set_performances)
    for diag in val_set_performances:

        results[diag] = {}
        for subset in val_set_performances[diag]:

            results[diag][subset] = {}
            subset_performances = val_set_performances[diag][subset]
            
            # Find the threshold where sensitivity the closest to n and sensitivity is > specificity 
            # If there are multiple thresholds with the same sensitivity, return the one with the highest specificity
            # If at the threshold where sensitivity is the closest to n, sensitivity is not > specificity, return the threshold with min sensisitivity where sensitivity is > specificity
            # subset_performances format is {threshold: [auc, sens, spec]}
            # n is a float between 0 and 1


            # Find the threshold where sensitivity is the closest to n
            closest_to_n = 1
            closest_to_n_threshold = None
            for threshold in subset_performances:
                if abs(subset_performances[threshold][1] - n) < closest_to_n:
                    closest_to_n = abs(subset_performances[threshold][1] - n)
                    closest_to_n_threshold = threshold

            # Find the threshold with the highest specificity where sensitivity is the closest to n
            highest_spec = 0
            highest_spec_threshold = None
            for threshold in subset_performances:
                if subset_performances[threshold][1] == closest_to_n and subset_performances[threshold][2] > highest_spec:
                    highest_spec = subset_performances[threshold][2]
                    highest_spec_threshold = threshold

            # If there is no threshold where sensitivity is the closest to n and sensitivity is > specificity, find the threshold with the highest specificity where sensitivity is > specificity
            if highest_spec_threshold == None:
                for threshold in subset_performances:
                    if subset_performances[threshold][1] > subset_performances[threshold][2] and subset_performances[threshold][1] < closest_to_n:
                        closest_to_n = subset_performances[threshold][1]
                        closest_to_n_threshold = threshold

            results[diag][subset] = closest_to_n_threshold

    return results