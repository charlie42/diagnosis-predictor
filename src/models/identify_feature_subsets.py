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
        util.clean_dirs([output_reports_dir]) # Remove old models and reports

    return {"input_data_dir": input_data_dir,  "models_dir": models_dir, "input_reports_dir": input_reports_dir, "output_reports_dir": output_reports_dir}

def get_base_model_from_classifier(classifier):
    return list(classifier.named_steps.keys())[-1]

def get_feature_subsets(ignore_non_lr_diags, best_classifiers, datasets, number_of_features_to_check):
    # If base model is LR, get feature subsets from LR coefficients (top n features), otherwise from SFS
    feature_subsets = {}
    for diag in best_classifiers.keys():
        base_model = get_base_model_from_classifier(best_classifiers[diag])
        print(diag, base_model)
        if base_model == "logisticregression":
            feature_subsets[diag] = models.get_feature_subsets_from_lr(diag, best_classifiers, datasets, number_of_features_to_check)
        elif base_model != "logisticregression" and ignore_non_lr_diags == 0:
            feature_subsets[diag] = models.get_feature_subsets_from_sfs(diag, best_classifiers, datasets, number_of_features_to_check)
    return feature_subsets

def write_feature_subsets_to_text_file(feature_subsets, output_reports_dir):
    path = output_reports_dir+"feature-subsets/"
    util.create_dir_if_not_exists(path)
    for diag in feature_subsets.keys():
        diag_path = path + diag + '/'
        util.create_dir_if_not_exists(diag_path)
        with open(diag_path+'feature-subsets.txt', 'w') as file:
            file.write(json.dumps(feature_subsets[diag], indent=2))

def main(number_of_features_to_check = 50, importances_from_file = 0, ignore_non_lr_diags = 0):
    number_of_features_to_check = int(number_of_features_to_check)
    importances_from_file = int(importances_from_file)
    ignore_non_lr_diags = int(ignore_non_lr_diags)

    dirs = set_up_directories(importances_from_file)

    best_classifiers = load(dirs["models_dir"]+'best-classifiers.joblib')
    datasets = load(dirs["input_data_dir"]+'datasets.joblib')

    if importances_from_file == 1:
        feature_subsets = load(dirs["output_reports_dir"]+'feature-subsets.joblib')
    else:
        feature_subsets = get_feature_subsets(ignore_non_lr_diags, best_classifiers, datasets, number_of_features_to_check)
        dump(feature_subsets, dirs["output_reports_dir"]+'feature-subsets.joblib')
    
    write_feature_subsets_to_text_file(feature_subsets, dirs["output_reports_dir"])
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])