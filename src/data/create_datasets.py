import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from joblib import dump
import sys, os, inspect

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data, features

def build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments, learning):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params = {"first_assessment_to_drop": first_assessment_to_drop, "use_other_diags_as_input": use_other_diags_as_input, 
              "only_free_assessments": only_free_assessments, "learning?": learning}
    params_part = util.build_param_string_for_dir_name(params)
    
    return datetime_part + "___" + params_part

def set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments, learning):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments, learning)

    data_statistics_dir = data_dir + "reports/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_statistics_dir)
    util.create_dir_if_not_exists(data_statistics_dir+"figures/")

    data_output_dir = data_dir + "data/create_datasets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_output_dir)

    return {"data_statistics_dir": data_statistics_dir, "data_output_dir": data_output_dir}

def customize_input_cols_per_diag(input_cols, diag):
    # Remove "Diag.Intellectual Disability-Mild" when predicting "Diag.Borderline Intellectual Functioning"
    #   and vice versa because they are highly correlated, same for other diagnoses
    #   (only useful when use_other_diags_as_input = 1)
    
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
        
    # Remove NIH scores for NVLD (used for diagnosis)
    if "NVLD" in diag:
        input_cols = [x for x in input_cols if not x.startswith("NIH")]
                      
    return input_cols

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


    input_cols = customize_input_cols_per_diag(input_cols, diag)
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
                                                            random_state=0)
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=split_percentage, 
                                                                      stratify=y_train, random_state=0)
        
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

def make_corr_df(full_dataset):
    # Make table with correlation between all columns, sort by highest correlations
    corr_df = full_dataset.corr()
    corr_df_unstacked = corr_df.unstack().sort_values(kind="quicksort", ascending=False).reset_index()
    corr_df_unstacked.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation Coefficient"}, inplace=True)
    corr_df_unstacked.drop(corr_df_unstacked.iloc[corr_df_unstacked[corr_df_unstacked["Feature 1"] == corr_df_unstacked["Feature 2"]].index].index, inplace=True)
    corr_df_unstacked.drop_duplicates(subset=["Correlation Coefficient"], inplace=True)
    corr_df_unstacked.reset_index(drop=True, inplace=True)

    # Percentage of columns with >0.3 or <-0.3 correlation with another column, except _WAS_MISSING ones
    n_cols_over_03 = 0
    for col in corr_df:
        if "_WAS_MISSING" in col:
            continue
        # Get max_corr and column with highest correlation except if 1.0 (correlation with itself)
        # Drop rows with correlation 1.0 (correlation with itself)
        new_corr_col = corr_df[col].drop(corr_df[col].loc[corr_df[col] == 1.0].index)
        max_corr = new_corr_col.abs().max()

        if max_corr > 0.3:
            n_cols_over_03 += 1

    percentage = n_cols_over_03 / corr_df.shape[0]
    print(f"Percentage of columns with >0.3 or <-0.3 correlation with another column: {percentage}")


    return corr_df_unstacked

def save_dataset_stats(datasets, diag_cols, item_level_ds, dir):
    stats = {}
    stats["n_rows_full_ds"] = item_level_ds.shape[0]
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

    corr_df = make_corr_df(item_level_ds)
    corr_df.to_csv(dir + "corr_df.csv")

    item_level_ds.describe(include = 'all').T.to_csv(dir + "column_stats.csv")

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


def main(only_assessment_distribution, use_other_diags_as_input, only_free_assessments, learning):
    only_assessment_distribution = int(only_assessment_distribution)
    use_other_diags_as_input = int(use_other_diags_as_input)
    only_free_assessments = int(only_free_assessments)
    learning = int(learning)

    clinical_config = util.read_config("clinical", learning)
    
    first_assessment_to_drop = clinical_config["first assessment to drop"]

    dirs = set_up_directories(first_assessment_to_drop, use_other_diags_as_input, only_free_assessments, learning)

    data.make_full_dataset(only_assessment_distribution, first_assessment_to_drop, only_free_assessments, dirs, learning)

    if only_assessment_distribution == 0:
        item_level_ds = pd.read_csv(dirs["data_output_dir"] + "item_lvl.csv")
        cog_tasks_ds = pd.read_csv(dirs["data_output_dir"] + "cog_tasks.csv") 
        subscales_ds = pd.read_csv(dirs["data_output_dir"] + "subscale_scores.csv")
        total_scores_ds = pd.read_csv(dirs["data_output_dir"] + "total_scores.csv")

        consensus_diags = [x for x in item_level_ds.columns if x.startswith("Diag.")]

        item_level_ds = features.make_new_diag_cols(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config)
        cog_tasks_ds, subscales_ds = update_datasets_with_new_diags(item_level_ds, cog_tasks_ds, subscales_ds, total_scores_ds, consensus_diags, dir = dirs["data_output_dir"])
        item_level_ds = add_cols_from_total_and_subscale_to_input(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config)

        # Drop ID
        item_level_ds.drop("ID", axis=1, inplace=True)

        # Print dataset shape
        print("Full dataset shape: Number of rows: ", item_level_ds.shape[0], "Number of columns: ", item_level_ds.shape[1])

        # Get list of column names with "Diag." prefix, where number of 
        # positive examples is > threshold
        min_pos_examples_val_set = 20
        split_percentage = 0.2
        all_diags = [x for x in item_level_ds.columns if x.startswith("Diag.")]
        if clinical_config["predict consensus diags"] == False: # Remove consensus diags, only keep new diags
            all_diags = [x for x in all_diags if x not in consensus_diags]
        positive_examples_in_ds = get_positive_examples_in_ds(item_level_ds, all_diags)
        
        diag_cols = find_diags_w_enough_positive_examples_in_val_set(positive_examples_in_ds, all_diags, split_percentage, min_pos_examples_val_set)

        # Create datasets for each diagnosis (different input and output columns)
        datasets = split_datasets_per_diag(item_level_ds, diag_cols, split_percentage, use_other_diags_as_input, clinical_config, learning)

        save_dataset_stats(datasets, diag_cols, item_level_ds, dirs["data_statistics_dir"])
            
        dump(datasets, dirs["data_output_dir"]+'datasets.joblib', compress=1)

        # Save number of positive examples for each diagnosis to csv (convert dict to df)
        pos_examples_col_name = f"Positive examples out of {item_level_ds.shape[0]}"
        pd.DataFrame(positive_examples_in_ds.items(), columns=["Diag", pos_examples_col_name]).sort_values(pos_examples_col_name, ascending=False).to_csv(dirs["data_statistics_dir"]+"number-of-positive-examples.csv")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])