# diagnosis-predictor

## 1 step:

`python -W ignore src/data/make_dataset.py SCARED_SR`

Arguments: first_assessment_to_drop

## 2 step:

`python -W ignore src/models/train_models.py 2.5 0.02 0`

Arguments: beta = 2.5, performance_margin = 0.02, models_from_file = 1

## 3 step:

`python -W ignore src/models/evaluate_models.py 2.5 0.02 0.8 1`

Arguments: beta = 2.5, performance_margin = 0.02, auc_threshold = 0.8, use_test_set=1

## 4 step:

`python -W ignore src/models/identify_feature_subsets.py 2.5 0.8 100 0.02 0`

Arguments: beta = 2.5, auc_threshold = 0.8, number_of_features_to_check = 100, performance_margin = 0.02, sfs_importances_from_file = 1