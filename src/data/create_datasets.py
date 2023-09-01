import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from joblib import dump
import sys, os, inspect
import argparse

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, features

N_SAMPLES_ALL = 2323
N_SAMPLES_LEARNING = 1406

def build_output_dir_name(params_for_dir_name):

    only_parent_report, first_assessment_to_drop, use_other_diags_as_input, only_free_assessments, learning, NIH, fix_n_all, fix_n_learning = params_for_dir_name

    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = {
        "only_parent_report": only_parent_report,
        "first_assessment_to_drop": first_assessment_to_drop, 
        "use_other_diags_as_input": use_other_diags_as_input, 
        "only_free_assessments": only_free_assessments, 
        "learning?": learning, 
        "NIH?": NIH,
        "fix_n_all": fix_n_all,  # Fix number of training examples when using less assessments
        "fix_n_learning": fix_n_learning # Fix number of training examples when using less assessments
    }
    params_part = util.build_param_string_for_dir_name(params)
    
    return datetime_part + "___" + params_part

def set_up_directories(params_for_dir_name):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data_archive/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(params_for_dir_name)

    data_statistics_dir = data_dir + "reports/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_statistics_dir)
    util.create_dir_if_not_exists(data_statistics_dir+"figures/")

    data_output_dir = data_dir + "data/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_output_dir)

    return {"data_statistics_dir": data_statistics_dir, "data_output_dir": data_output_dir}


def get_input_cols_per_diag(full_dataset, diag, use_other_diags_as_input, learning):
    
    input_cols = [x for x in full_dataset.columns if 
                            not x == "Diag.No Diagnosis Given"  # Will be negatively correlated with any diagnosis
                            and not x == diag] # Output
    
    if use_other_diags_as_input == 0: # Drop all diag cols
        input_cols = [x for x in input_cols if 
                            not x.startswith("Diag.")]
        
    # If learning, use NIH T scores in input
    if learning:
        input_cols += [x for x in input_cols if 
                            not x.startswith("NIH") and x.endswith("_P")]


    input_cols = data.customize_input_cols_per_output(input_cols, diag)
    print(f"Input assessemnts used for {diag}: ", list(set([x.split(",")[0] for x in input_cols])))
    
    return input_cols

def keep_only_healthy_controls(X, y):
    # Remove people where y=0 and Diag.No Diagnosis Given is 0 (they have other diagnoses)
    
    # Get indices to remove
    indices_to_remove = X[(y == 0) & (X["Diag.No Diagnosis Given"] == 0)].index

    # Remove rows
    X_new = X.drop(indices_to_remove)
    y_new = y.drop(indices_to_remove)

    return X_new, y_new

def split_datasets_per_diag(data, diag_cols, split_percentage, use_other_diags_as_input, clinical_config, learning):
    datasets = {}
    for diag in diag_cols:
        
        output_col = diag
        
        # Split train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(data, data[output_col], 
                                                            test_size=split_percentage, stratify=data[output_col], 
                                                            random_state=1)
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=split_percentage, 
                                                                      stratify=y_train, random_state=1)
        
        X_test_only_healthy_controls, y_test_only_healthy_controls = keep_only_healthy_controls(X_test, y_test)
        X_val_only_healthy_controls, y_val_only_healthy_controls = keep_only_healthy_controls(X_val, y_val)

        # Drop columns from input that we don't want there
        input_cols = get_input_cols_per_diag(data, diag, use_other_diags_as_input, learning)
        X_train = X_train[input_cols]
        X_test = X_test[input_cols]
        X_train_train = X_train_train[input_cols]
        X_val = X_val[input_cols]
        X_test_only_healthy_controls = X_test_only_healthy_controls[input_cols]
        X_val_only_healthy_controls = X_val_only_healthy_controls[input_cols]
    
        datasets[diag] = { "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "X_train_train": X_train_train,
                        "X_val": X_val,
                        "y_train_train": y_train_train,
                        "y_val": y_val,
                        "X_test_only_healthy_controls": X_test_only_healthy_controls,
                        "y_test_only_healthy_controls": y_test_only_healthy_controls,
                        "X_val_only_healthy_controls": X_val_only_healthy_controls,
                        "y_val_only_healthy_controls": y_val_only_healthy_controls}
        
    return datasets

def get_positive_examples_in_ds(full_dataset, diags):
    positive_ex_in_ds = {}
    for diag in diags:
        positive_ex_in_ds[diag] = full_dataset[full_dataset[diag] == 1].shape[0]
    return positive_ex_in_ds
    
def find_diags_w_enough_positive_examples_in_val_set(positive_examples_in_ds, all_diags, split_percentage, min_pos_examples_val_set):
    diags_w_enough_positive_examples_in_val_set = []
    for diag in all_diags:
        # First get # of positive examples in the train set, then from those, get # of positive examples in the validation set 
        # (first we split the dataset into train and test set, then we split the train set into train and validation set)
        positive_examples_val_set = positive_examples_in_ds[diag] * (1-split_percentage) * split_percentage 
        if positive_examples_val_set >= min_pos_examples_val_set:
            diags_w_enough_positive_examples_in_val_set.append(diag)
    return diags_w_enough_positive_examples_in_val_set

def add_cols_from_total_and_subscale_to_input(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config):
    if "add cols to input" in clinical_config and clinical_config["add cols to input"]:
        cols_to_add = clinical_config["add cols to input"]
        cog_cols_to_add = [x for x in cols_to_add if x in cog_tasks_ds.columns]
        subscales_cols_to_add = [x for x in cols_to_add if x in subscales_ds.columns]

        item_level_ds = item_level_ds.merge(cog_tasks_ds[cog_cols_to_add], left_index=True, right_index=True)
        item_level_ds = item_level_ds.merge(subscales_ds[subscales_cols_to_add], left_index=True, right_index=True)
        
    return item_level_ds

def update_datasets_with_new_diags(item_level_ds, cog_tasks_ds, subscales_ds, total_score_ds, consensus_diags, dir):
    all_diags_in_item_lvl_ds = [x for x in item_level_ds.columns if "Diag." in x]
    new_diags = [x for x in all_diags_in_item_lvl_ds if x not in consensus_diags]
    print(f"New diags: {new_diags}")

    print(f"Old shape of cog_tasks_ds: {cog_tasks_ds.shape}")

    # Add new diags to cog_tasks_ds and subscales_ds
    cog_tasks_ds = cog_tasks_ds.merge(item_level_ds[new_diags], left_index=True, right_index=True)
    subscales_ds = subscales_ds.merge(item_level_ds[new_diags], left_index=True, right_index=True)
    total_score_ds = total_score_ds.merge(item_level_ds[new_diags], left_index=True, right_index=True)

    print(f"New shape of cog_tasks_ds: {cog_tasks_ds.shape}")

    # Rewrite csv files
    cog_tasks_ds.to_csv(dir + "cog_tasks.csv")
    subscales_ds.to_csv(dir + "subscale_scores.csv")
    total_score_ds.to_csv(dir + "total_scores.csv")

    return cog_tasks_ds, subscales_ds

def fix_n(item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds, dirs, n):
    # Gen random n indices
    indices_to_keep = np.random.choice(item_level_ds.index, n, replace=False)

    item_level_ds = item_level_ds.loc[indices_to_keep]
    cog_tasks_ds = cog_tasks_ds.loc[indices_to_keep]
    subscales_ds = subscales_ds.loc[indices_to_keep]
    total_scores_ds = total_scores_ds.loc[indices_to_keep]

    # Rewrite csv files
    item_level_ds.to_csv(dirs["data_output_dir"] + "item_lvl.csv")
    cog_tasks_ds.to_csv(dirs["data_output_dir"] + "cog_tasks.csv")
    subscales_ds.to_csv(dirs["data_output_dir"] + "subscale_scores.csv")
    total_scores_ds.to_csv(dirs["data_output_dir"] + "total_scores.csv")

    return item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distrib-only', action='store_true', help='Only generate assessment distribution, do not create datasets')
    parser.add_argument('--parent-only', action='store_true', help='Only use parent-report assessments')
    parser.add_argument('--use-other-diags', action='store_true', help='Use other diagnoses as input')
    parser.add_argument('--free-only', action='store_true', help='Only use free assessments')
    parser.add_argument('--learning', action='store_true', help='Use additional assessments like C3SR (reduces # of examples)')
    parser.add_argument('--nih', action='store_true', help='Use NIH toolbox scores')
    parser.add_argument('--fix-n-all', action='store_true', help='Fix number of training examples when using less assessments')
    parser.add_argument('--fix-n-learning', action='store_true', help='Fix number of training examples when using less assessments (learning))')
    
    args = parser.parse_args()
    only_assessment_distribution = args.distrib_only
    only_parent_report = args.parent_only
    use_other_diags_as_input = args.use_other_diags
    only_free_assessments = args.free_only
    learning = args.learning
    nih = args.nih
    fix_n_all = args.fix_n_all
    fix_n_learning = args.fix_n_learning

    clinical_config = util.read_config("clinical", learning)
    
    first_assessment_to_drop = clinical_config["first assessment to drop"]

    if learning == 1 and nih == 0: # Don't add NIH scores to input
        clinical_config["add cols to input"] = [x for x in clinical_config["add cols to input"] if not x.startswith("NIH")]

    params_for_dir_name = [
        only_parent_report,
        first_assessment_to_drop, 
        use_other_diags_as_input, 
        only_free_assessments, 
        learning, # use additional assessments like C3SR (reduces # of examples)
        nih, # use NIH toolbox scores
        fix_n_all,  # Fix number of training examples when using less assessments
        fix_n_learning # Fix number of training examples when using less assessments
    ] 
    dirs = set_up_directories(params_for_dir_name)

    data.make_HBN(only_assessment_distribution, only_parent_report, first_assessment_to_drop, only_free_assessments, dirs, learning)

    if only_assessment_distribution == 0:

        item_level_ds = pd.read_csv(dirs["data_output_dir"] + "item_lvl.csv")
        cog_tasks_ds = pd.read_csv(dirs["data_output_dir"] + "cog_tasks.csv") 
        subscales_ds = pd.read_csv(dirs["data_output_dir"] + "subscale_scores.csv")
        total_scores_ds = pd.read_csv(dirs["data_output_dir"] + "total_scores.csv")

        # Fix n when using less assessments: read and rewrite csv files
        if fix_n_all:
            item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds = fix_n(item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds, dirs, N_SAMPLES_ALL)
        elif fix_n_learning:
            item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds = fix_n(item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds, dirs, N_SAMPLES_LEARNING)

        consensus_diags = [x for x in item_level_ds.columns if x.startswith("Diag.")]

        item_level_ds = features.make_new_diag_cols(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config)
        cog_tasks_ds, subscales_ds = update_datasets_with_new_diags(item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds, consensus_diags, dir = dirs["data_output_dir"])
        item_level_ds = add_cols_from_total_and_subscale_to_input(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config)

        # Drop ID
        item_level_ds.drop("ID", axis=1, inplace=True)

        item_level_ds.to_csv(dirs["data_output_dir"] + "item_lvl_new.csv")
        # Print dataset shape
        print("Full dataset shape: Number of rows: ", item_level_ds.shape[0], "Number of columns: ", item_level_ds.shape[1])

        # Get list of column names with "Diag." prefix, where number of 
        # positive examples is > threshold
        min_pos_examples_val_set = 20
        split_percentage = 0.2
        all_diags = [x for x in item_level_ds.columns if x.startswith("Diag.")]
        if clinical_config["predict consensus diags"] is False: # Remove consensus diags, only keep new diags
            all_diags = [x for x in all_diags if x not in consensus_diags]
        positive_examples_in_ds = get_positive_examples_in_ds(item_level_ds, all_diags)
        dump(positive_examples_in_ds, dirs["data_output_dir"]+'positive_examples_in_ds.joblib', compress=1)
        
        diag_cols = find_diags_w_enough_positive_examples_in_val_set(positive_examples_in_ds, all_diags, split_percentage, min_pos_examples_val_set)

        # Create datasets for each diagnosis (different input and output columns)
        datasets = split_datasets_per_diag(item_level_ds, diag_cols, split_percentage, use_other_diags_as_input, clinical_config, learning)

        dump(datasets, dirs["data_output_dir"]+'datasets.joblib', compress=1)        
        
if __name__ == "__main__":
    main()