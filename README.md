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

- Navigate to the project directory
- Install python 3.9
- Install pip
- `pip install pipenv`
- `pipenv install`

## 1 step:

`pipenv run python -W ignore src/data/create_datasets.py [OPTIONS]`

Arguments: 
- `--distrib-only`: Generate only the assessment distribution without creating datasets.
- `--parent-only`: Use only parent-report assessments.
- `--use-other-diags`: Include other diagnoses as input.
- `--free-only`: Include only free assessments.
- `--learning`: Include additional assessments like C3SR (which reduces the number of examples).
- `--nih`: Include NIH toolbox scores.
- `--fix-n-all` Fix number of training examples to the smallest one up to WHODAS
- `--fix-n-learning` Fix number of training examples to the smallest one with C3SR

### 2 step (optional):

`pipenv run python -W ignore src/data/create_data_reports.py [OPTIONS]`

Arguments: 
- `--plot-col-value-distrib`: Plot value distribution of every column (takes time).

## 3 step:

`pipenv run python -W ignore src/models/train_models.py [OPTIONS]`

Arguments: 
- `--performance-margin`: Margin of error for ROC AUC (for prefering logistic regression over other models). Defalult value is 0.02.
- `--from-file`: Load existing models from file instead of training new ones

## 4 step (optional):

`pipenv run python -W ignore src/models/evaluate_original_models.py [OPTIONS]`

Arguments: 
-  `--val-set`: Use the validation set instead of the test set

## 5 step:

`pipenv run python -W ignore src/models/identify_feature_subsets.py [OPTIONS]`

Arguments: 
-  `--from-file`: Whether to import feature importances from file or not

## 6 step

`pipenv run python -W ignore src/models/evaluate_models_on_feature_subsets.py [OPTIONS]`

Arguments: 
-  `--from-file`: Whether to load existing models instead of training new models

