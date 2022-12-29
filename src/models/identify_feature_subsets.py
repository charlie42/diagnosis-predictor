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

def set_up_directories(keep_old_importances=0):

    input_data_dir = "data/train_models/"
    models_dir = "models/" + "train_models/"
    input_reports_dir = "reports/" + "train_models/"

    output_reports_dir = "reports/" + "identify_feature_subsets/"
    util.create_dir_if_not_exists(output_reports_dir)

    if keep_old_importances == 0:
        util.clean_dirs([output_reports_dir]) # Remove old reports

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def get_feature_subsets(ignore_non_linear_models, best_classifiers, datasets, number_of_features_to_check):
    # If base model is LR, get feature subsets from LR coefficients (top n features), otherwise from SFS
    feature_subsets = {}
    for diag in best_classifiers.keys():
        if "Impairment" not in diag: # DEBUG
            continue
        base_model_type = util.get_base_model_name_from_pipeline(best_classifiers[diag])
        base_model = util.get_estimator_from_pipeline(best_classifiers[diag])
        print(diag, base_model_type)
        # If base model is linear: LR or SVM with linear kernel, use RFE to get feature subsets
        if base_model_type == "logisticregression" or (base_model_type == "svc" and base_model.kernel == "linear"):
            feature_subsets[diag] = models.get_feature_subsets_for_linear_models(diag, best_classifiers, datasets, number_of_features_to_check)
        # If base model is RF, use RFE to get first 50 feature, then use SFS to get the rest (to avoid high cordinality bias in RF feature importances).
        #     Since only a few features have higher cardinality than the rest (NIH scores), RFE should be sufficient to get the top 50 features.
        elif ignore_non_linear_models == 0 and base_model_type == "randomforestclassifier":
            feature_subsets[diag] = models.get_feature_subsets_for_rf(diag, best_classifiers, datasets, number_of_features_to_check)
        # If base model doesn't expose feature importances (e.g. non-linear SVC), use SFS to get feature subsets
        else:
            feature_subsets[diag] = models.get_feature_subsets_from_sfs(diag, best_classifiers, datasets, number_of_features_to_check)
    return feature_subsets

def write_feature_subsets_to_text_file(feature_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"
    util.write_two_lvl_dict_to_file(feature_subsets, path)
    
def main(number_of_features_to_check = 50, importances_from_file = 0, ignore_non_linear_models = 0):
    number_of_features_to_check = int(number_of_features_to_check)
    importances_from_file = int(importances_from_file)
    ignore_non_linear_models = int(ignore_non_linear_models)

    dirs = set_up_directories(importances_from_file)

    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    if importances_from_file == 1:
        feature_subsets = load(dirs["output_reports_dir"]+'feature-subsets.joblib')
    else:
        feature_subsets = get_feature_subsets(ignore_non_linear_models, best_classifiers, datasets, number_of_features_to_check)
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    
    write_feature_subsets_to_text_file(feature_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])