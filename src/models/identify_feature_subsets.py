import os, sys, inspect
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning" # Seems to be the only way to suppress multi-thread sklearn warnings

# Colorful error messages
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme='Neutral', call_pdb=False)

from joblib import load, dump
import pandas as pd
import warnings

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import models, util

DEBUG_MODE = True

def build_output_dir_name(params_from_previous_script):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    return datetime_part + "___" + util.build_param_string_for_dir_name(params_from_previous_script)

def set_up_directories():

    data_dir = "../diagnosis_predictor_data/"

    # Input dirs
    input_data_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "data/create_datasets/")
    print("Reading data from: " + input_data_dir)
    models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/train_models/")
    print("Reading models from: " + models_dir)
    input_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/train_models/")
    print("Reading reports from: " + input_reports_dir)

    # Output dirs
    params_from_previous_script = models.get_params_from_current_data_dir_name(input_data_dir)
    current_output_dir_name = build_output_dir_name(params_from_previous_script)

    output_reports_dir = data_dir + "reports/" + "identify_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_reports_dir)

    output_models_dir = data_dir + "models/" + "identify_feature_subsets/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(output_models_dir)

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, 
            "output_reports_dir": output_reports_dir, "output_models_dir": output_models_dir}

def set_up_load_directories():
    data_dir = "../diagnosis_predictor_data/"

    load_reports_dir = models.get_newest_non_empty_dir_in_dir(data_dir+ "reports/identify_feature_subsets/")
    load_models_dir = models.get_newest_non_empty_dir_in_dir(data_dir + "models/identify_feature_subsets/")

    return {"load_reports_dir": load_reports_dir, "load_models_dir": load_models_dir}

def get_feature_subsets(best_estimators, datasets, number_of_features_to_check, dirs):
    feature_subsets = {}
    for diag in best_estimators.keys():
        base_model_type = util.get_base_model_name_from_pipeline(best_estimators[diag])
        base_model = util.get_estimator_from_pipeline(best_estimators[diag])
        print(diag, base_model_type)
        if DEBUG_MODE and base_model_type != "logisticregression": # Don't do RF models in debug mode, takes long
            continue
        # If base model is exposes feature importances, use RFE to get first 50 feature, then use SFS to get the rest.
        if not (base_model_type == "svc" and base_model.kernel != "linear"):
            feature_subsets[diag] = models.get_feature_subsets_from_rfe_then_sfs(diag, best_estimators, datasets, number_of_features_to_check)
        # If base model doesn't expose feature importances, use SFS to get feature subsets directly (will take very long)
        else:
            feature_subsets[diag] = models.get_feature_subsets_from_sfs(diag, best_estimators, datasets, number_of_features_to_check)
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    return feature_subsets

def fix_data_dict(names_df):
    # Replace ICU_X with ICU_P_X where X is item number (except ICU_SR lines)
    names_df.index = names_df.index.str.replace(r"ICU_(?!(SR))", "ICU_P_")
    return names_df

def make_sign_dict(feature_list, estimator):
    # Re-train models on feature subsets (train_train set), get signs of coefficients
    #   If sign is positive, item is positively correlated with diagnosis
    #   If sign is negative, item is negatively correlated with diagnosis

    sign_dict = {}

    # If model doesn't have coeffieicents, make values empty
    if util.get_base_model_name_from_pipeline(estimator) not in ["logisticregression", "svc"]:
        for item in feature_list:
            sign_dict[item] = ""
            warnings.warn("Model doesn't have coefficients, can't prepend coefficient sign to item names")
        return sign_dict

    if util.get_base_model_name_from_pipeline(estimator) == "logisticregression":
        coef = estimator.named_steps["logisticregression"].coef_[0]
    else:
        coef = estimator.named_steps["svc"].coef_[0]
    
    # Get signs of coefficients
    for i, item in enumerate(feature_list):
        if coef[i] > 0:
            sign_dict[item] = "+"
        else:
            sign_dict[item] = "-"

    return sign_dict

def make_name_dict(feature_list):
    # Append item name to each item ID 
    #   Remove name from assessment from feauture name, only keep item name (already contains assessment name)
    #   Don't append name to basic demographics, self explanatory item IDs
    names_df = fix_data_dict(pd.read_csv("references/item-names.csv", index_col=1, encoding = "ISO-8859-1", names=["questions", "keys"], sep=","))
    name_dict = {}
    missing_names = ("WAS_MISSING", "financialsupport", "Panic_A01A", "Panic_A02A", "Panic_A01B", "Panic_A02B") 
    for item in feature_list:
        if item.startswith("Basic_Demos") or item.endswith(missing_names):
            name_dict[item] = ""
        else:
            name_dict[item] = names_df.loc[item.split(",")[1]]["questions"]
    return name_dict

def append_names_and_sign_to_feature_subsets(feature_subsets, estimators_on_subsets):
    feature_subsets_with_names_and_signs = {}
    for diag in feature_subsets.keys():
        feature_subsets_with_names_and_signs[diag] = {}

        if diag not in list(estimators_on_subsets.keys()):
            feature_subsets_with_names_and_signs[diag] = feature_subsets[diag]
            warnings.warn(f"Estimators on subsets for {diag} not found, skipping appending names and signs")
            continue

        for subset in feature_subsets[diag].keys():

            sign_dict = make_sign_dict(feature_subsets[diag][subset], estimators_on_subsets[diag][subset])
            name_dict = make_name_dict(feature_subsets[diag][subset])

            feature_subsets_with_names_and_signs[diag][subset] = [f'{sign_dict[x]} {x} ({name_dict[x]})' 
                                                                  for x in feature_subsets[diag][subset]]
    return feature_subsets_with_names_and_signs

def write_feature_subsets_with_names_and_signs(feature_subsets, estimators_on_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"

    feature_subsets_with_names_and_sign = append_names_and_sign_to_feature_subsets(feature_subsets, estimators_on_subsets)

    util.write_two_lvl_dict_to_file(feature_subsets_with_names_and_sign, path)
    
def main(number_of_features_to_check = 126, importances_from_file = 0):
    number_of_features_to_check = int(number_of_features_to_check)
    importances_from_file = int(importances_from_file)

    dirs = set_up_directories()

    best_estimators = load(dirs["models_dir"]+'best-estimators.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    if importances_from_file == 1:
        load_dirs = set_up_load_directories()
        feature_subsets = load(load_dirs["load_reports_dir"]+'feature-subsets.joblib')
        estimators_on_subsets = load(load_dirs["load_models_dir"]+'estimators-on-subsets.joblib')

        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
        dump(estimators_on_subsets, dirs["output_models_dir"]+'estimators-on-subsets.joblib')
    else:
        feature_subsets = get_feature_subsets(best_estimators, datasets, number_of_features_to_check, dirs)
        estimators_on_subsets = models.re_train_models_on_feature_subsets(feature_subsets, datasets, best_estimators) 
                
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
        dump(estimators_on_subsets, dirs["output_models_dir"]+'estimators-on-subsets.joblib')

    write_feature_subsets_with_names_and_signs(feature_subsets, estimators_on_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])