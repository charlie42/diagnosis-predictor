import sys, os, inspect
import pandas as pd
import matplotlib.pyplot as plt

def get_importances_from_models(best_classifiers, datasets, diag_cols):
    for diag in diag_cols:
        print(diag)
        print(list(best_classifiers[diag].named_steps.keys())[-1])
        X_train = datasets[diag]["X_train"]
        if list(best_classifiers[diag].named_steps.keys())[-1] == "randomforestclassifier":
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].feature_importances_
            importances = pd.DataFrame(zip(X_train.columns, importances), columns=["Feature", "Importance"])
        else:
            importances = best_classifiers[diag].named_steps[list(best_classifiers[diag].named_steps.keys())[-1]].coef_
            importances = pd.DataFrame(zip(X_train.columns, abs(importances[0])), columns=["Feature", "Importance"])
        importances = importances[importances["Importance"]>0].sort_values(by="Importance", ascending=False).reset_index(drop=True)
        print(importances)
        importances.plot(y="Importance")
        plt.show()

def main(auc_threshold):

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    import data

    data_processed_dir = "data/processed/"
    models_dir = "models/"
    reports_dir = "reports/"

    full_dataset = pd.read_csv(data_processed_dir + "item_lvl_w_impairment.csv")
    performance_table = pd.read_csv(reports_dir + "performance_table_all_featuers.csv")

    auc_threshold = float(auc_threshold)

    # Get list of column names with "Diag: " prefix, where number of 
    # performance is > threshold (ROC AUC reference: https://gpsych.bmj.com/content/gpsych/30/3/207.full.pdf)
    diag_cols = performance_table[performance_table["ROC AUC Mean CV"] >= auc_threshold]["Diag"].tolist()

    from joblib import load
    best_classifiers = load(models_dir+'best-classifiers.joblib')

    # Create datasets for each diagnosis (different input and output columns)
    datasets = data.create_datasets(full_dataset, diag_cols)

    get_importances_from_models(best_classifiers, datasets, diag_cols)

if __name__ == "__main__":
    main(sys.argv[1])