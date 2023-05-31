import pandas as pd

from sklearn.model_selection import train_test_split

from joblib import dump
import sys, os, inspect

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, features

def build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = {"first_assessment_to_drop": first_assessment_to_drop, "use_other_diags_as_input": use_other_diags_as_input, 
              "only_free_assessments": only_free_assessments}
    params_part = util.build_param_string_for_dir_name(params)
    
    return datetime_part + "___" + params_part

def set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments)

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
    print("Input assessemnts used: ", list(set([x.split(",")[0] for x in input_cols])))
    
    return input_cols

def keep_only_healthy_controls(X, y, diag):
    # Simulate exclusion criteria: remove people who have diagnoses other than diag (cols starting with Diag.)

    # Get indices of people with other diagnoses
    other_diags = [x for x in X.columns if x.startswith("Diag.") and not x == diag]
    indices_to_remove = X[X[other_diags].sum(axis=1) > 0].index

    # Remove rows
    X_new = X.drop(indices_to_remove)
    y_new = y.drop(indices_to_remove)

    print(f"{X_new.shape[0]} rows left after removing people with other diagnoses than {diag}.")

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
        
        X_test_only_healthy_controls, y_test_only_healthy_controls = keep_only_healthy_controls(X_test, y_test, diag)
        X_val_only_healthy_controls, y_val_only_healthy_controls = keep_only_healthy_controls(X_val, y_val, diag)

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

def save_dataset_stats(datasets, diag_cols, full_dataset, dir):
    stats = {}
    stats["n_rows_full_ds"] = full_dataset.shape[0]
    stats["n_rows_train_ds"] = datasets[diag_cols[0]]["X_train_train"].shape[0]
    stats["n_rows_val_ds"] = datasets[diag_cols[0]]["X_val"].shape[0]
    stats["n_rows_test_ds"] = datasets[diag_cols[0]]["X_test"].shape[0]
    stats["n_rows_test_ds_only_healthy_controls"] = datasets[diag_cols[0]]["X_test_only_healthy_controls"].shape[0]
    stats["n_rows_val_ds_only_healthy_controls"] = datasets[diag_cols[0]]["X_val_only_healthy_controls"].shape[0]
    stats["n_input_cols"] = datasets[diag_cols[0]]["X_train_train"].shape[1] - len(diag_cols)
    # To df
    stats_df = pd.DataFrame.from_dict(stats, orient="index")
    stats_df.columns = ["Value"]
    stats_df.to_csv(dir + "dataset_stats.csv")

def main(only_assessment_distribution, first_assessment_to_drop, use_other_diags_as_input, only_free_assessments):
    only_assessment_distribution = int(only_assessment_distribution)
    use_other_diags_as_input = int(use_other_diags_as_input)
    only_free_assessments = int(only_free_assessments)

    dirs = set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments)

    data.make_full_dataset(only_assessment_distribution, first_assessment_to_drop, only_free_assessments, dirs)

    if only_assessment_distribution == 0:
        full_dataset = pd.read_csv(dirs["data_output_dir"] + "item_lvl.csv")
        full_dataset = features.make_new_diag_cols(full_dataset)

        # Print dataset shape
        print("Full dataset shape: Number of rows: ", full_dataset.shape[0], "Number of columns: ", full_dataset.shape[1])

        # Get list of column names with "Diag." prefix, where number of 
        # positive examples is > threshold
        min_pos_examples_val_set = 20
        split_percentage = 0.2
        all_diags = [x for x in full_dataset.columns if x.startswith("Diag.")]
        positive_examples_in_ds = get_positive_examples_in_ds(full_dataset, all_diags)
        
        diag_cols = find_diags_w_enough_positive_examples_in_val_set(positive_examples_in_ds, all_diags, split_percentage, min_pos_examples_val_set)

        # Create datasets for each diagnosis (different input and output columns)
        datasets = split_datasets_per_diag(full_dataset, diag_cols, split_percentage, use_other_diags_as_input)

        save_dataset_stats(datasets, diag_cols, full_dataset, dirs["data_statistics_dir"])
            
        dump(datasets, dirs["data_output_dir"]+'datasets.joblib', compress=1)

        # Save number of positive examples for each diagnosis to csv (convert dict to df)
        pos_examples_col_name = f"Positive examples out of {full_dataset.shape[0]}"
        pd.DataFrame(positive_examples_in_ds.items(), columns=["Diag", pos_examples_col_name]).sort_values(pos_examples_col_name, ascending=False).to_csv(dirs["data_statistics_dir"]+"number-of-positive-examples.csv")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])