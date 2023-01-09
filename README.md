# diagnosis-predictor

File structure: http://drivendata.github.io/cookiecutter-data-science/#directory-structure 

## 1 step:

`python -W ignore src/data/make_dataset.py 1 ICU_P`

Arguments: only_assessment_distribution, first_assessment_to_drop

## 2 step:

`python -W ignore src/models/train_models.py 0.02 0 0 SWAN`

Arguments: performance_margin = 0.02, use_other_diags_as_input = 0, models_from_file = 1, input_questionnaire = None

## 3 step:

`python -W ignore src/models/evaluate_original_models.py 1`

Arguments: use_test_set=1

## 4 step:

`python -W ignore src/models/identify_feature_subsets.py 126 0`

Arguments: number_of_features_to_check = 126, importances_from_file = 0

## 5 step

`python -W ignore src/models/evaluate_models_on_feature_subsets.py 0`

Arguments: models_from_file = 1

