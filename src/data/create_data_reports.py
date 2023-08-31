import sys, os, inspect
from joblib import dump, load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util


def build_output_dir_name(params_from_create_datasets):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params_part = util.build_param_string_for_dir_name(params_from_create_datasets)
    
    return datetime_part + "___" + params_part

def set_up_directories():

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data_archive/"
    util.create_dir_if_not_exists(data_dir)

    # Input dirs
    input_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")

    # Create directory inside the output directory with the run timestamp and params:
    #    - [params from create_datasets.py]
    params_from_create_datasets = util.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_create_datasets)

    models_dir = data_dir + "models/" + "create_data_reports/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(models_dir)

    reports_dir = data_dir + "reports/" + "create_data_reports/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(reports_dir) 

    return {"input_data_dir": input_data_dir, "models_dir": models_dir, "reports_dir": reports_dir}

def set_up_load_directories():
    # When loading existing models, can't take the newest directory, we just created it, it will be empty. 
    #   Need to take the newest non-empty directory.

    data_dir = "../diagnosis_predictor_data_archive/"
    
    load_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    load_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "reports/create_data_reports/")
    
    return {"load_data_dir": load_data_dir, "load_reports_dir": load_reports_dir}


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

def save_pos_ex_stats(positive_examples_in_ds, diag_cols, item_level_ds, dir):
    pos_examples_col_name = f"Positive examples out of {item_level_ds.shape[0]}"
    pd.DataFrame(positive_examples_in_ds.items(), columns=["Diag", pos_examples_col_name]).sort_values(pos_examples_col_name, ascending=False).to_csv(dir+"number-of-positive-examples.csv")

    # Save only those diags that are used
    positive_examples_in_ds_for_used = {k: v for k, v in positive_examples_in_ds.items() if k in diag_cols}
    pd.DataFrame(positive_examples_in_ds_for_used.items(), columns=["Diag", pos_examples_col_name]).sort_values(pos_examples_col_name, ascending=False).to_csv(dir+"number-of-positive-examples-used.csv")

def plot_age_distributions(item_level_ds, diag_cols, dir):
    # Plot age distribution of whole dataset and of each diagnosis 
    age_col = "Basic_Demos,Age"

    fig, axes = plt.subplots(int(np.ceil(len(diag_cols)/2)), 2, figsize=(20, 20)) #calculate grid for nb of diags such that it is square
    # Add gap between subplots
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes = axes.flatten()

    # Set age range to min and max in itm_lvl_ds
    axes[0].set_xlim([item_level_ds[age_col].min(), item_level_ds[age_col].max()])

    axes[0].hist(item_level_ds[age_col], bins=50)
    axes[0].set_title("Age distribution of whole dataset")

    for i, diag in enumerate(diag_cols):
        if diag == "Diag.Any Diag":
            continue
        axes[i+1].hist(item_level_ds[item_level_ds[diag] == 1][age_col], bins=50)
        axes[i+1].set_title(f"Age distribution of {diag}")

    plt.savefig(dir + "age_distributions.png")

def main():
    dirs = set_up_directories()

    datasets = load(dirs["input_data_dir"] + "datasets.joblib")
    item_level_ds = pd.read_csv(dirs["input_data_dir"] + "item_lvl_new.csv")
    positive_examples_in_ds = load(dirs["input_data_dir"] + "positive_examples_in_ds.joblib")

    diag_cols = list(datasets.keys())
    
    save_dataset_stats(datasets, diag_cols, item_level_ds, dirs["reports_dir"])

    # Save number of positive examples for each diagnosis to csv (convert dict to df)
    save_pos_ex_stats(positive_examples_in_ds, diag_cols, item_level_ds, dirs["reports_dir"])

    # Plot age distribution of whole dataset and of each diagnosis
    plot_age_distributions(item_level_ds, diag_cols, dirs["reports_dir"])


if __name__ == "__main__":
    main()