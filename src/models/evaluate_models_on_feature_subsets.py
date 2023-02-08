import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

import pandas as pd

from joblib import dump, load

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

DEBUG_MODE = False

def build_output_dir_name(params_from_previous_script):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + models.build_param_string_for_dir_name(params_from_previous_script)

def set_up_directories():

    data_dir = "../diagnosis_predictor_data/"

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/train_models/")
    input_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/identify_feature_subsets/")
    
    # Output dirs
    params_from_previous_script = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_previous_script)
    
    output_models_dir = data_dir + "models/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_models_dir)

    output_reports_dir = data_dir + "reports/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir,  "input_models_dir": input_models_dir, "output_models_dir": output_models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def set_up_load_directories():
    data_dir = "../diagnosis_predictor_data/"
    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/evaluate_models_on_feature_subsets/")
    return {"load_reports_dir": load_reports_dir}

def make_and_write_cv_auc_table(auc_on_subsets, dir):
    auc_on_subsets = pd.DataFrame.from_dict(auc_on_subsets)
    auc_on_subsets.index = range(1, len(auc_on_subsets)+1)
    auc_on_subsets = auc_on_subsets.rename(columns={"index": "Diagnosis"})

    auc_on_subsets.to_csv(dir+'cv-auc-on-subsets.csv')

    return auc_on_subsets

def make_performance_tables_opt_threshold(performances_on_feature_subsets, optimal_thresholds):
    # Create a table for each performance metric (AUC, Sensitivity, Specificity) for each number of features, using the optimal threshold
    # Build list of lists where each list is a row in the table
    auc_table = []
    sens_table = []
    spec_table = []

    for diag in performances_on_feature_subsets:
        
        diag_row_auc = [diag] # Each row in the table will have the diagnosis and then each columns will be the performance for each number of features
        diag_row_sens = [diag]
        diag_row_spec = [diag]
        
        for nb_features in performances_on_feature_subsets[diag]:

            optimal_threshold = optimal_thresholds[diag][nb_features]
            diag_row_auc = diag_row_auc + [performances_on_feature_subsets[diag][nb_features][optimal_threshold][0]] 
            diag_row_sens = diag_row_sens + [performances_on_feature_subsets[diag][nb_features][optimal_threshold][1]]
            diag_row_spec = diag_row_spec + [performances_on_feature_subsets[diag][nb_features][optimal_threshold][2]]
            
        auc_table = auc_table + [diag_row_auc]
        sens_table = sens_table + [diag_row_sens]
        spec_table = spec_table + [diag_row_spec]

    auc_df = pd.DataFrame(auc_table, columns=["Diagnosis"] + list(performances_on_feature_subsets[diag].keys()))
    sens_df = pd.DataFrame(sens_table, columns=["Diagnosis"] + list(performances_on_feature_subsets[diag].keys()))
    spec_df = pd.DataFrame(spec_table, columns=["Diagnosis"] + list(performances_on_feature_subsets[diag].keys()))

    # Sort diagnoses by performance on max number of features (sort values by last column)
    auc_df = auc_df.sort_values(by=auc_df.columns[-1], ascending=False)
    sens_df = sens_df.sort_values(by=sens_df.columns[-1], ascending=False)
    spec_df = spec_df.sort_values(by=spec_df.columns[-1], ascending=False)

    # Transpose so that each column is a diagnosis and each row is a number of features
    auc_df = auc_df.transpose()
    sens_df = sens_df.transpose()
    spec_df = spec_df.transpose()

    # Rename columns and index
    auc_df.columns = auc_df.iloc[0]
    auc_df = auc_df.drop(auc_df.index[0])
    auc_df.index.name = "Number of features"

    sens_df.columns = sens_df.iloc[0]
    sens_df = sens_df.drop(sens_df.index[0])
    sens_df.index.name = "Number of features"

    spec_df.columns = spec_df.iloc[0]
    spec_df = spec_df.drop(spec_df.index[0])
    spec_df.index.name = "Number of features"

    # Reverse order order of rows so max number of features is at the top
    auc_df = auc_df.iloc[::-1]
    sens_df = sens_df.iloc[::-1]
    spec_df = spec_df.iloc[::-1]

    # Remove Diag. from diagnosis names
    auc_df.columns = [col.replace("Diag. ", "") for col in auc_df.columns]
    sens_df.columns = [col.replace("Diag. ", "") for col in sens_df.columns]
    spec_df.columns = [col.replace("Diag. ", "") for col in spec_df.columns]
    
    return auc_df, sens_df, spec_df

def make_performance_tables_opt_nb_features(performances_on_feature_subsets, optimal_nbs_features):
    # Create a table for each performance metric (AUC, Sensitivity, Specificity) for each threshold, using the optimal number of features
    # Build list of lists where each list is a row in the table
    sens_spec_tables = {}

    for diag in performances_on_feature_subsets:

        sens_spec_table_diag = []
        
        optimal_nb_features = optimal_nbs_features[diag]
        
        for threshold in performances_on_feature_subsets[diag][optimal_nb_features]: # Each row in the table will have the threshold, specificity, and sensitivity for that threshold
            row = [threshold, performances_on_feature_subsets[diag][optimal_nb_features][threshold][1], performances_on_feature_subsets[diag][optimal_nb_features][threshold][2]]
            sens_spec_table_diag = sens_spec_table_diag + [row]

        sens_spec_tables[diag] = pd.DataFrame(sens_spec_table_diag, columns=["Threshold", "Sensitivity", "Specificity"])

        # Replace Thresholds with numbers 1, 2, 3, etc. to obscure the actual thresholds
        sens_spec_tables[diag]["Threshold"] = range(1, len(sens_spec_tables[diag]) + 1)
        sens_spec_tables[diag].set_index("Threshold", inplace=True)

        # If more than 25 thresholds, evenly take ~25 thresholds
        if len(sens_spec_tables[diag]) > 25:
            step = int(len(sens_spec_tables[diag])/25) 
            sens_spec_tables[diag] = sens_spec_tables[diag][::step]

    return sens_spec_tables

def get_and_write_optimal_nbs_features(auc_table, dir):
    optimal_nbs_features = {}

    for diag in auc_table.columns:
        max_score = auc_table[diag].max()
        optimal_score = max_score - 0.01
        # Get index of the first row with a score >= optimal_score
        optimal_nbs_features[diag] = auc_table[diag][auc_table[diag] >= optimal_score].index[0]

    print(optimal_nbs_features)
    util.write_dict_to_file(optimal_nbs_features, dir, "optimal-nb-features.txt")

    return optimal_nbs_features

def make_and_write_test_set_performance_tables(performances_on_feature_subsets, dir, optimal_thresholds, optimal_nbs_features):

    # Make AUC, Sens, Spec tables for optimal thresholds
    [auc_test_set_table_optimal_threshold, sens_test_set_table_optimal_threshold, spec_test_set_table_optimal_threshold] = make_performance_tables_opt_threshold(performances_on_feature_subsets, optimal_thresholds)
    auc_test_set_table_optimal_threshold.to_csv(dir+'auc-on-subsets-test-set-optimal-threshold.csv')
    sens_test_set_table_optimal_threshold.to_csv(dir+'sens-on-subsets-test-set-optimal-threshold.csv')
    spec_test_set_table_optimal_threshold.to_csv(dir+'spec-on-subsets-test-set-optimal-threshold.csv')

    # Make AUC, Sens, Spec tables for all thresholds on optimal number of features
    sens_spec_test_set_tables_optimal_nb_features = make_performance_tables_opt_nb_features(performances_on_feature_subsets, optimal_nbs_features)
    path = dir + "sens-spec-on-subsets-test-set-optimal-nb-features/"
    util.create_dir_if_not_exists(path)
    for diag in sens_spec_test_set_tables_optimal_nb_features:
        sens_spec_test_set_tables_optimal_nb_features[diag].to_csv(path+diag+'.csv')

    # Make a table with AUC, Sens, Spec for optimal threhsolds on optimal number of features for each diagnosis
    auc_sens_spec_test_set_opt_thres_opt_nb_features = []
    for diag in auc_test_set_table_optimal_threshold.columns:
        auc_test_set_table_optimal_threshold = auc_test_set_table_optimal_threshold[auc_test_set_table_optimal_threshold["Number of features"] == optimal_nbs_features[diag]].iloc[0][diag]
        sens_test_set_table_optimal_threshold = sens_test_set_table_optimal_threshold[sens_test_set_table_optimal_threshold["Number of features"] == optimal_nbs_features[diag]].iloc[0][diag]
        spec_test_set_table_optimal_threshold = spec_test_set_table_optimal_threshold[spec_test_set_table_optimal_threshold["Number of features"] == optimal_nbs_features[diag]].iloc[0][diag]
        auc_sens_spec_test_set_opt_thres_opt_nb_features.append([diag, optimal_nbs_features[diag], auc_test_set_table_optimal_threshold, sens_test_set_table_optimal_threshold, spec_test_set_table_optimal_threshold])
    
    auc_sens_spec_test_set_opt_thres_opt_nb_features = pd.DataFrame(auc_sens_spec_test_set_opt_thres_opt_nb_features, columns=["Diagnosis", "Number of features", "AUC", "Sensitivity", "Specificity"])
    auc_sens_spec_test_set_opt_thres_opt_nb_features.to_csv(dir+'auc-sens-spec-on-subsets-test-set-optimal-threshold-optimal-nb-features.csv')
    

def main(models_from_file = 1):
    models_from_file = int(models_from_file)

    dirs = set_up_directories()

    feature_subsets = load(dirs["input_reports_dir"]+'feature-subsets.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')
    best_classifiers = load(dirs["input_models_dir"]+'best-classifiers.joblib')

    if DEBUG_MODE == True:
        # In debug mode, only use first diagnosis
        datasets = {list(datasets.keys())[0]: datasets[list(datasets.keys())[0]]}
        feature_subsets = {list(feature_subsets.keys())[0]: feature_subsets[list(feature_subsets.keys())[0]]}
        best_classifiers = {list(best_classifiers.keys())[0]: best_classifiers[list(best_classifiers.keys())[0]]}

    if models_from_file == 1:
        load_dirs = set_up_load_directories()

        performances_on_feature_subsets = load(load_dirs["load_reports_dir"]+'performances-on-feature-subsets.joblib')    
        cv_scores_on_feature_subsets = load(load_dirs["load_reports_dir"]+'cv-scores-on-feature-subsets.joblib')
        optimal_thresholds = load(load_dirs["load_reports_dir"]+'optimal-thresholds.joblib')

        # Save reports to newly created directories
        dump(performances_on_feature_subsets, dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')
        dump(cv_scores_on_feature_subsets, dirs["output_reports_dir"]+'cv-scores-on-feature-subsets.joblib')
        dump(optimal_thresholds, dirs["output_reports_dir"]+'optimal-thresholds.joblib')
    else:
        performances_on_feature_subsets, cv_scores_on_feature_subsets, optimal_thresholds = models.get_performances_on_feature_subsets(feature_subsets, datasets, best_classifiers, use_test_set = 1)
        dump(performances_on_feature_subsets, dirs["output_reports_dir"]+'performances-on-feature-subsets.joblib')
        dump(cv_scores_on_feature_subsets, dirs["output_reports_dir"]+'cv-scores-on-feature-subsets.joblib')
        dump(optimal_thresholds, dirs["output_reports_dir"]+'optimal-thresholds.joblib')
    
    cv_auc_table = make_and_write_cv_auc_table(cv_scores_on_feature_subsets, dirs["output_reports_dir"])
    optimal_nbs_features = get_and_write_optimal_nbs_features(cv_auc_table, dirs["output_reports_dir"])
    make_and_write_test_set_performance_tables(performances_on_feature_subsets, dirs["output_reports_dir"], optimal_thresholds, optimal_nbs_features)

if __name__ == "__main__":
    main(sys.argv[1])