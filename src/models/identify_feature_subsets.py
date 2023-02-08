import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

from joblib import load, dump
import json

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
    models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/train_models/")

    # Output dirs
    params_from_previous_script = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_previous_script)
    output_reports_dir = data_dir + "reports/" + "identify_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def set_up_load_directories():
    data_dir = "../diagnosis_predictor_data/"
    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/identify_feature_subsets/")
    return {"load_reports_dir": load_reports_dir}

def get_feature_subsets(best_classifiers, datasets, number_of_features_to_check, dirs):
    feature_subsets = {}
    for diag in best_classifiers.keys():
        base_model_type = util.get_base_model_name_from_pipeline(best_classifiers[diag])
        base_model = util.get_estimator_from_pipeline(best_classifiers[diag])
        print(diag, base_model_type)
        if DEBUG_MODE and base_model_type != "logisticregression": # Don't do RF models in debug mode, takes long
            continue
        # If base model is exposes feature importances, use RFE to get first 50 feature, then use SFS to get the rest.
        if not (base_model_type == "svc" and base_model.kernel != "linear"):
            feature_subsets[diag] = models.get_feature_subsets_from_rfe_then_sfs(diag, best_classifiers, datasets, number_of_features_to_check)
        # If base model doesn't expose feature importances, use SFS to get feature subsets directly (will take very long)
        else:
            feature_subsets[diag] = models.get_feature_subsets_from_sfs(diag, best_classifiers, datasets, number_of_features_to_check)
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    return feature_subsets

def write_feature_subsets_to_text_file(feature_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"
    util.write_two_lvl_dict_to_file(feature_subsets, path)
    
def main(number_of_features_to_check = 126, importances_from_file = 0):
    number_of_features_to_check = int(number_of_features_to_check)
    importances_from_file = int(importances_from_file)

    dirs = set_up_directories()

    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    # If number of features to check is higher than the number of items in the total feature set, set it to the number of items in the total feature set
    number_of_features_to_check = min(number_of_features_to_check, len(list(datasets.values())[0]["X_train"].columns))

    if importances_from_file == 1:
        load_dirs = set_up_load_directories()
        feature_subsets = load(load_dirs["load_reports_dir"]+'feature-subsets.joblib')

        # Save reports to newly created directories
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    else:
        feature_subsets = get_feature_subsets(best_classifiers, datasets, number_of_features_to_check, dirs)
        
    write_feature_subsets_to_text_file(feature_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])