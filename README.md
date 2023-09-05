# diagnosis-predictor

File structure: http://drivendata.github.io/cookiecutter-data-science/#directory-structure 

Predicting HBN consensus diagnoses, and test-based disagnoses for:
- Specific Learning Disorder with Impairment in Reading
- Specific Learning Disorder with Impairment in Mathematics
- Specific Learning Disorder with Impairment in Written Expression
- Intellectual Disability-Mild
- Borderline Intellectual Functioning
- Processing Speed Deficit 
- NVLD

To run on an empty system:

- Install python 3.9
- Install pip
- Navigate to the directory root
- Install pipenv using pip
- Run `pipenv install`

`requirements.txt` is also available for installing dependencies with pip directly

## 1 step:

`pipenv run python -W ignore src/data/create_datasets.py 0 0 0 0 0`

Arguments: only_assessment_distribution, only_parent_report, use_other_diags_as_input = 0, only_free_assessments = 0, learning = 0

### 2 step (optional):

`pipenv run python -W ignore src/data/create_data_reports.py`

## 2 step:

`pipenv run python -W ignore src/models/train_models.py 0`

Arguments: models_from_file = 1

## 3 step (optional):

`pipenv run python -W ignore src/models/evaluate_original_models.py 1`

Arguments: use_test_set=1

## 4 step:

`pipenv run python -W ignore src/models/identify_feature_subsets.py 0`

Arguments: importances_from_file = 0

## 5 step

`pipenv run python -W ignore src/models/evaluate_models_on_feature_subsets.py 0`

Arguments: models_from_file = 1

