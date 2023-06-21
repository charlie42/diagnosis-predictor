import warnings
import pandas as pd

# To import from parent directory
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def fix_data_dict(names_df):
    # Replace ICU_X with ICU_P_X where X is item number (except ICU_SR lines)
    names_df.index = names_df.index.str.replace(r"ICU_(?!(SR))", "ICU_P_")
    return names_df

def make_coef_dict(feature_list, estimator):
    # Re-train models on feature subsets (train_train set), get coefficients
    
    coef_dict = {}

    # If model doesn't have coeffieicents, make values empty
    if util.get_base_model_name_from_pipeline(estimator) not in ["logisticregression", "svc"]:
        for item in feature_list:
            coef_dict[item] = ""
            warnings.warn("Model doesn't have coefficients, can't prepend coefficients to item names")
        return coef_dict

    if util.get_base_model_name_from_pipeline(estimator) == "logisticregression":
        coef = estimator.named_steps["logisticregression"].coef_[0]
    else:
        coef = estimator.named_steps["svc"].coef_[0]
    
    # Make dict with coefficients
    for item, coef_value in zip(feature_list, coef):
        coef_dict[item] = coef_value

    return coef_dict

def make_name_and_value_dict(feature_list):
    # Append item name to each item ID 
    #   Remove name from assessment from feauture name, only keep item name (already contains assessment name)
    #   Don't append name to basic demographics, self explanatory item IDs
    name_and_value_df = fix_data_dict(pd.read_csv("references/item-names.csv", index_col=1, encoding = "ISO-8859-1", names=["questions","keys","datadic","value","valueLabels"], sep=","))
    name_and_value_dict = {}
    missing_names = ("WAS_MISSING", "preg_symp", "financialsupport", "Panic_A01A", "Panic_A02A", "Panic_A01B", "Panic_A02B") 
    for item in feature_list:
        if item.startswith("Basic_Demos") or item.endswith(missing_names):
            name_and_value_dict[item] = ["", ""]
        else:
            id = item.split(",")[1]
            if id not in name_and_value_df.index:
                warnings.warn(f"Item {id} not found in item-names.csv, skipping appending name and value")
                name_and_value_dict[item] = ["", ""]
            else:
                name_and_value_dict[item] = [
                    name_and_value_df.loc[id]["questions"],
                    name_and_value_df.loc[id]["valueLabels"]
                    ]
    return name_and_value_dict

def append_names_and_coef_to_feature_subsets(feature_subsets, estimators_on_subsets):
    feature_subsets_with_names_and_coefs = {}
    for diag in feature_subsets.keys():
        feature_subsets_with_names_and_coefs[diag] = {}

        if diag not in list(estimators_on_subsets.keys()):
            feature_subsets_with_names_and_coefs[diag] = feature_subsets[diag]
            warnings.warn(f"Estimators on subsets for {diag} not found, skipping appending names and coefs")
            continue

        for subset in feature_subsets[diag].keys():

            coef_dict = make_coef_dict(feature_subsets[diag][subset], estimators_on_subsets[diag][subset])
            name_and_value_dict = make_name_and_value_dict(feature_subsets[diag][subset])

            feature_subsets_with_names_and_coefs[diag][subset] = [f'({coef_dict[x]:.2f}*) {x}: {name_and_value_dict[x][0]} - {name_and_value_dict[x][1]}' 
                                                                  for x in feature_subsets[diag][subset]]
    return feature_subsets_with_names_and_coefs

def add_performances_to_subsets(feature_subsets_with_names_and_coef, performances):
    result = {}
    for diag in feature_subsets_with_names_and_coef:
        result[diag] = {}
        for subset in feature_subsets_with_names_and_coef[diag]:
            auroc = performances[diag][subset][list(performances[diag][subset].keys())[0]][0] # AUROC is first item in list of metrics, and 
                                                                                              # is the same for all thresholds (so we take 0th threshold)
            result[diag][subset] = [f'AUROC: {auroc:.2}', feature_subsets_with_names_and_coef[diag][subset]]
    return result

def write_feature_subsets_to_file(feature_subsets, estimators_on_subsets, output_reports_dir, performances = None, optimal_nbs_features = None):
    path = output_reports_dir+"feature-subsets/"

    feature_subsets_with_names_and_coef = append_names_and_coef_to_feature_subsets(feature_subsets, estimators_on_subsets)

    if performances:
        result = add_performances_to_subsets(feature_subsets_with_names_and_coef, performances)        
    else:
        result = feature_subsets_with_names_and_coef

    if optimal_nbs_features:
        # Drop subsets over the optimal number of features:
        #  drop keys that are over the optimal number of features

        for diag in result.keys():
            result[diag] = {n_subset: result[diag][n_subset] for n_subset in result[diag] if int(n_subset) <= optimal_nbs_features[diag]}
        
        


        
    util.write_two_lvl_dict_to_file(result, path)