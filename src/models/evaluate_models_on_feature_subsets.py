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
    # Part with the datetimeS
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + util.build_param_string_for_dir_name(params_from_previous_script)

def set_up_directories():

    data_dir = "../diagnosis_predictor_data/"

    # Input dirs
    input_data_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    print("Reading data from: " + input_data_dir)
    input_models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    print("Reading models from: " + input_models_dir)
    input_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir+ "reports/identify_feature_subsets/")
    print("Reading reports from: " + input_reports_dir)
    
    # Output dirs
    params_from_previous_script = util.get_params_from_current_data_dir_name(input_models_dir)
    current_output_dir_name = build_output_dir_name(params_from_previous_script)
    
    output_models_dir = data_dir + "models/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_models_dir)

    output_reports_dir = data_dir + "reports/" + "evaluate_models_on_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir,  "input_models_dir": input_models_dir, "output_models_dir": output_models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def set_up_load_directories():
    data_dir = "../diagnosis_predictor_data/"
    load_reports_dir = util.get_newest_non_empty_dir_in_dir(data_dir+ "reports/evaluate_models_on_feature_subsets/")
    load_models_dir = util.get_newest_non_empty_dir_in_dir(data_dir + "models/" + "evaluate_models_on_feature_subsets/")
    return {"load_reports_dir": load_reports_dir, "load_models_dir": load_models_dir}

def make_performance_tables_one_threshold(performances_on_feature_subsets, opt_threshold):
    # Create a table for each performance metric (AUC, Sensitivity, Specificity) for each number of features, using the threshold where sensitivity is 0.8 and > specificity
    # Build list of lists where each list is a row in the table
    auc_table = []
    sens_table = []
    spec_table = []

    for diag in performances_on_feature_subsets:
        
        diag_row_auc = [diag] # Each row in the table will have the diagnosis and then each columns will be the performance for each number of features
        diag_row_sens = [diag]
        diag_row_spec = [diag]
        
        for nb_features in performances_on_feature_subsets[diag]:

            # Find threshold where sensitivity is 0.8 and > specificity
            threshold = opt_threshold[diag][nb_features]

            diag_row_auc = diag_row_auc + [performances_on_feature_subsets[diag][nb_features][threshold][0]] 
            diag_row_sens = diag_row_sens + [performances_on_feature_subsets[diag][nb_features][threshold][1]]
            diag_row_spec = diag_row_spec + [performances_on_feature_subsets[diag][nb_features][threshold][2]]
            
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
            row = [threshold, 
                   performances_on_feature_subsets[diag][optimal_nb_features][threshold][1], 
                   performances_on_feature_subsets[diag][optimal_nb_features][threshold][2],
                   performances_on_feature_subsets[diag][optimal_nb_features][threshold][3],
                   performances_on_feature_subsets[diag][optimal_nb_features][threshold][4]]
            sens_spec_table_diag = sens_spec_table_diag + [row]

        sens_spec_tables[diag] = pd.DataFrame(sens_spec_table_diag, columns=["Threshold", "Sensitivity", "Specificity", "PPV", "NPV"])

        # Replace Thresholds with numbers 1, 2, 3, etc. to obscure the actual thresholds
        sens_spec_tables[diag]["Threshold"] = range(1, len(sens_spec_tables[diag]) + 1)
        sens_spec_tables[diag].set_index("Threshold", inplace=True)

        # Drop duplicates
        sens_spec_tables[diag] = sens_spec_tables[diag].drop_duplicates()

    return sens_spec_tables

def read_opt_nbs_features(reports_dir):
    return load(reports_dir+'optimal-nb-features.joblib')

def make_perf_test_set_one_thres_opt_nb_features(auc_test_set, sens_test_set_one_threshold, spec_test_set_one_threshold, opt_nbs_features):
    result = []
    diags = auc_test_set.columns
    for diag in diags:
        diag_auc = auc_test_set[diag].loc[opt_nbs_features[diag]]
        diag_sens_one_threshold = sens_test_set_one_threshold[diag].loc[opt_nbs_features[diag]]
        diag_spec_one_threshold = spec_test_set_one_threshold[diag].loc[opt_nbs_features[diag]]
        result.append([diag, opt_nbs_features[diag], diag_auc, diag_sens_one_threshold, diag_spec_one_threshold])
    result_df = pd.DataFrame(result, columns=["Diagnosis", "Number of features", "AUC", "Sensitivity", "Specificity"])
    return result_df

def make_and_write_test_set_performance_tables(performances_on_feature_subsets_test_set, performances_on_feature_subsets_val_set, opt_nbs_features, opt_thresholds, output_dir):

    # Make AUC table, Sens table, Spec table for one threshold, all subsets 
    [auc_test_set, sens_test_set_one_threshold, spec_test_set_one_threshold] = make_performance_tables_one_threshold(performances_on_feature_subsets_test_set, opt_thresholds)
    [auc_val_set, sens_val_set_one_threshold, spec_val_set_one_threshold] = make_performance_tables_one_threshold(performances_on_feature_subsets_val_set, opt_thresholds)

    # Make a table for each diag on optimal number of features with all thresholds, sens, spec, ppv, npv
    sens_spec_test_set_optimal_nb_features = make_performance_tables_opt_nb_features(performances_on_feature_subsets_test_set, opt_nbs_features) #dict of dataframes, key=diag
    sens_spec_val_set_optimal_nb_features = make_performance_tables_opt_nb_features(performances_on_feature_subsets_val_set, opt_nbs_features) 

    # Make a table with AUC, Sens, Spec, ppv, npv for one threhsold on optimal number of features (combine the two above)
    perf_test_set_one_thres_opt_nb_features = make_perf_test_set_one_thres_opt_nb_features(auc_test_set, sens_test_set_one_threshold, spec_test_set_one_threshold, opt_nbs_features)
    perf_val_set_one_thres_opt_nb_features = make_perf_test_set_one_thres_opt_nb_features(auc_val_set, sens_val_set_one_threshold, spec_val_set_one_threshold, opt_nbs_features)
    
    # Write to csv
    auc_test_set.to_csv(output_dir+'auc-on-subsets-test-set.csv', float_format='%.3f')
    auc_val_set.to_csv(output_dir+'auc-on-subsets-val-set.csv', float_format='%.3f')
    sens_test_set_one_threshold.to_csv(output_dir+'sens-on-subsets-test-set-one-threshold.csv', float_format='%.3f')
    sens_val_set_one_threshold.to_csv(output_dir+'sens-on-subsets-val-set-one-threshold.csv', float_format='%.3f')
    spec_test_set_one_threshold.to_csv(output_dir+'spec-on-subsets-test-set-one-threshold.csv', float_format='%.3f')
    spec_val_set_one_threshold.to_csv(output_dir+'spec-on-subsets-val-set-one-threshold.csv', float_format='%.3f')
    perf_test_set_one_thres_opt_nb_features.to_csv(output_dir+'perf-on-subsets-test-set-one-threshold-optimal-nb-features.csv', float_format='%.3f')
    perf_val_set_one_thres_opt_nb_features.to_csv(output_dir+'perf-on-subsets-val-set-one-threshold-optimal-nb-features.csv', float_format='%.3f')

    path = output_dir + "sens-spec-on-subsets-test-set-optimal-nb-features/"
    util.create_dir_if_not_exists(path)
    for diag in sens_spec_test_set_optimal_nb_features:
        sens_spec_test_set_optimal_nb_features[diag].to_csv(path+diag+'.csv', float_format='%.3f')

    path = output_dir + "sens-spec-on-subsets-val-set-optimal-nb-features/"
    util.create_dir_if_not_exists(path)
    for diag in sens_spec_val_set_optimal_nb_features:
        sens_spec_val_set_optimal_nb_features[diag].to_csv(path+diag+'.csv', float_format='%.3f')

    return [auc_test_set, sens_test_set_one_threshold, spec_test_set_one_threshold, perf_test_set_one_thres_opt_nb_features, sens_spec_test_set_optimal_nb_features]

def make_and_save_saturation_plot(auc_table, dir):
    import matplotlib.pyplot as plt

    # Plot a line of AUROCs at each number of features, one line per diagnosis. x axis is number of features (row index of auc_table), y axis is AUROC value
    fig, ax = plt.subplots()
    for diag in auc_table.columns:
        ax.plot(auc_table[diag], label=diag.split(".")[1])
    ax.set_xlabel("Number of features")
    ax.set_ylabel("AUROC")
    ax.set_ylim([0.5, 1.0])
    # Print legend under the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')

    fig.savefig(dir+'saturation-plot.png', bbox_inches='tight', dpi=800)

def re_write_subsets_w_auroc(feature_subsets, estimators_on_subsets, optimal_nbs_features, performance_table, output_dir):
    # optimal_nbs_features provided to drop subsets over the optimal number of features
    models.write_feature_subsets_to_file(feature_subsets, estimators_on_subsets, output_dir, performance_table, optimal_nbs_features)

def main(models_from_file = 1):
    models_from_file = int(models_from_file)

    clinical_config = util.read_config("clinical")
    number_of_features_to_check = clinical_config["max items in screener"]
    minimum_sensitivity = clinical_config["minimum sensitivity"]

    dirs = set_up_directories()

    feature_subsets = load(dirs["input_reports_dir"]+'feature-subsets.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')
    best_estimators = load(dirs["input_models_dir"]+'best-estimators.joblib')

    if DEBUG_MODE == True:
        # In debug mode, only use first diagnosis
        datasets = {list(datasets.keys())[0]: datasets[list(datasets.keys())[0]]}
        feature_subsets = {list(feature_subsets.keys())[0]: feature_subsets[list(feature_subsets.keys())[0]]}
        best_estimators = {list(best_estimators.keys())[0]: best_estimators[list(best_estimators.keys())[0]]}

    if models_from_file == 1:
        load_dirs = set_up_load_directories()

        performances_on_feature_subsets_test_set = load(load_dirs["load_reports_dir"]+'performances-on-feature-subsets-test-set.joblib')    
        performances_on_feature_subsets_val_set = load(load_dirs["load_reports_dir"]+'performances-on-feature-subsets-val-set.joblib')    
        estimators_on_feature_subsets = load(load_dirs["load_models_dir"]+'estimators-on-feature-subsets.joblib')
        
        # Save reports to newly created directories
        dump(performances_on_feature_subsets_test_set, dirs["output_reports_dir"]+'performances-on-feature-subsets-test-set.joblib')
        dump(performances_on_feature_subsets_val_set, dirs["output_reports_dir"]+'performances-on-feature-subsets-val-set.joblib')
        dump(estimators_on_feature_subsets, dirs["output_models_dir"]+'estimators-on-feature-subsets.joblib')

    else:
        estimators_on_feature_subsets = models.re_train_models_on_feature_subsets(feature_subsets, datasets, best_estimators)
        performances_on_feature_subsets_test_set = models.get_performances_on_feature_subsets(feature_subsets, 
                                                                                    datasets, 
                                                                                    estimators_on_feature_subsets, 
                                                                                    use_test_set = 1)

        performances_on_feature_subsets_val_set = models.get_performances_on_feature_subsets(feature_subsets, 
                                                                                    datasets, 
                                                                                    estimators_on_feature_subsets, 
                                                                                    use_test_set = 0)

        dump(estimators_on_feature_subsets, dirs["output_models_dir"]+'estimators-on-feature-subsets.joblib')
        dump(performances_on_feature_subsets_test_set, dirs["output_reports_dir"]+'performances-on-feature-subsets-test-set.joblib')
        dump(performances_on_feature_subsets_val_set, dirs["output_reports_dir"]+'performances-on-feature-subsets-val-set.joblib')
    
    output_reports_dir = dirs["output_reports_dir"]
    
    # Generate reports
    opt_nbs_features = read_opt_nbs_features(dirs["input_reports_dir"])
    opt_thresholds = models.find_thresholds_sens_over_n(performances_on_feature_subsets_val_set, minimum_sensitivity)

    auc_test_set, sens_test_set_one_threshold, spec_test_set_one_threshold, perf_test_set_one_thres_opt_nb_features, sens_spec_test_set_optimal_nb_features = make_and_write_test_set_performance_tables(
        performances_on_feature_subsets_test_set,
        performances_on_feature_subsets_val_set, 
        opt_nbs_features,
        opt_thresholds,
        output_reports_dir)
    make_and_save_saturation_plot(auc_test_set, output_reports_dir)
    re_write_subsets_w_auroc(feature_subsets, 
                             estimators_on_feature_subsets,
                             opt_nbs_features, 
                             performances_on_feature_subsets_test_set,
                             output_reports_dir)

if __name__ == "__main__":
    main(sys.argv[1])