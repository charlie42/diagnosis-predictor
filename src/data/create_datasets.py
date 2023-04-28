import pandas as pd

from sklearn.model_selection import train_test_split

from joblib import dump
import sys, os, inspect

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data

def build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = {"first_assessment_to_drop": first_assessment_to_drop, "use_other_diags_as_input": use_other_diags_as_input}
    params_part = util.build_param_string_for_dir_name(params)
    
    return datetime_part + "___" + params_part

def set_up_directories(first_assessment_to_drop, use_other_diags_as_input):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input)

    data_statistics_dir = data_dir + "reports/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_statistics_dir)
    util.create_dir_if_not_exists(data_statistics_dir+"figures/")

    data_output_dir = data_dir + "data/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_output_dir)

    return {"data_statistics_dir": data_statistics_dir, "data_output_dir": data_output_dir}

def customize_input_cols_per_diag(input_cols, diag):
    # Remove "Diag.Intellectual Disability-Mild" when predicting "Diag.Borderline Intellectual Functioning"
    #   and vice versa because they are highly correlated, same for other diagnoses
    
    if diag == "Diag.Intellectual Disability-Mild":
        input_cols = [x for x in input_cols if x != "Diag.Borderline Intellectual Functioning"]
    if diag == "Diag.Borderline Intellectual Functioning":
        input_cols = [x for x in input_cols if x != "Diag.Intellectual Disability-Mild"]
    if diag == "Diag.No Diagnosis Given":
        input_cols = [x for x in input_cols if not x.startswith("Diag.")]
    if diag == "Diag.ADHD-Combined Type":
        input_cols = [x for x in input_cols if x not in ["Diag.ADHD-Inattentive Type", 
                                                         "Diag.ADHD-Hyperactive/Impulsive Type",
                                                         "Diag.Other Specified Attention-Deficit/Hyperactivity Disorder",
                                                         "Diag.Unspecified Attention-Deficit/Hyperactivity Disorder"]]
    if diag == "Diag.ADHD-Inattentive Type":
        input_cols = [x for x in input_cols if x not in ["Diag.ADHD-Combined Type", 
                                                         "Diag.ADHD-Hyperactive/Impulsive Type",
                                                         "Diag.Other Specified Attention-Deficit/Hyperactivity Disorder",
                                                         "Diag.Unspecified Attention-Deficit/Hyperactivity Disorder"]]
                      
    return input_cols

def get_input_cols_per_diag(full_dataset, diag, use_other_diags_as_input):
    
    if use_other_diags_as_input == 1:
        input_cols = [x for x in full_dataset.columns if 
                            not x in ["WHODAS_P,WHODAS_P_Total", "CIS_P,CIS_P_Score", "WHODAS_SR,WHODAS_SR_Score", "CIS_SR,CIS_SR_Total"]
                            and not x == diag
                            and not x == "Diag.No Diagnosis Given"]
    else:
        input_cols = [x for x in full_dataset.columns if 
                            not x in ["WHODAS_P,WHODAS_P_Total", "CIS_P,CIS_P_Score", "WHODAS_SR,WHODAS_SR_Score", "CIS_SR,CIS_SR_Total"]
                            and not x.startswith("Diag.")]
    
    input_cols = customize_input_cols_per_diag(input_cols, diag)
    
    return input_cols

def keep_only_healthy_controls(X, y):
    # Remove people where y=0 and Diag.No Diagnosis Given is 0 (they have other diagnoses)
    
    # Get indices to remove
    indices_to_remove = X[(y == 0) & (X["Diag.No Diagnosis Given"] == 0)].index

    # Remove rows
    X_new = X.drop(indices_to_remove)
    y_new = y.drop(indices_to_remove)

    return X_new, y_new

def split_datasets_per_diag(full_dataset, diag_cols, split_percentage, use_other_diags_as_input):
    datasets = {}
    for diag in diag_cols:
        
        output_col = diag
        
        # Split train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(full_dataset, full_dataset[output_col], 
                                                            test_size=split_percentage, stratify=full_dataset[output_col], 
                                                            random_state=1)
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=split_percentage, 
                                                                      stratify=y_train, random_state=1)
        
        X_test_only_healthy_controls, y_test_only_healthy_controls = keep_only_healthy_controls(X_test, y_test)
        X_val_only_healthy_controls, y_val_only_healthy_controls = keep_only_healthy_controls(X_val, y_val)

        # Drop columns from input that we don't want there
        input_cols = get_input_cols_per_diag(full_dataset, diag, use_other_diags_as_input)
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

def find_diags_w_enough_positive_examples_in_val_set(full_dataset, all_diags, split_percentage, min_pos_examples_val_set):
    diags_w_enough_positive_examples_in_val_set = []
    for diag in all_diags:
        positive_examples_full_ds = full_dataset[full_dataset[diag] == 1].shape[0]
        # First get # of positive examples in the train set, then from those, get # of positive examples in the validation set 
        # (first we split the dataset into train and test set, then we split the train set into train and validation set)
        positive_examples_val_set = positive_examples_full_ds * (1-split_percentage) * split_percentage 
        if positive_examples_val_set >= min_pos_examples_val_set:
            diags_w_enough_positive_examples_in_val_set.append(diag)
    return diags_w_enough_positive_examples_in_val_set

def main(only_assessment_distribution, first_assessment_to_drop, use_other_diags_as_input):
    dirs = set_up_directories(first_assessment_to_drop, use_other_diags_as_input)

    data.make_full_dataset(only_assessment_distribution, first_assessment_to_drop, dirs)
    full_dataset = pd.read_csv(dirs["data_output_dir"] + "item_lvl.csv")

    # Print dataset shape
    print("Full dataset shape: ", full_dataset.shape)

    # Get list of column names with "Diag." prefix, where number of 
    # positive examples is > threshold
    min_pos_examples_val_set = 20
    split_percentage = 0.2
    all_diags = [x for x in full_dataset.columns if x.startswith("Diag.")]
    diag_cols = find_diags_w_enough_positive_examples_in_val_set(full_dataset, all_diags, split_percentage, min_pos_examples_val_set)

    # Create datasets for each diagnosis (different input and output columns)
    datasets = split_datasets_per_diag(full_dataset, diag_cols, split_percentage, use_other_diags_as_input)
    print("Train set shape: ", datasets[diag_cols[0]]["X_train_train"].shape)

    dump(datasets, dirs["data_output_dir"]+'datasets.joblib', compress=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])