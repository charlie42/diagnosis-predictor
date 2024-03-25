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
Copy data from LORIS to `data/raw/` directory, `LORIS-realease.csv`

## 2 step:
Check that you're happy with the clinical and technical config values (`config/` directory)

## 3 step:

Run `pipenv run python -W ignore src/data/create_datasets.py 0 0 0 0 0` to prepare the data

Arguments: 
- `only_assessment_distribution`: do not create data, only vizualize assessment distribution 
- `only_parent_report`: only use parent-report assessments
- `use_other_diags_as_input` = 0: use presence of other diagnoses as input variables
- `only_free_assessments` = 0: use only non-proprietary assessments
- `learning` = 0: use `learning.yml` config file with extended set of input assessments

### Optional:

Run `pipenv run python -W ignore src/data/create_data_reports.py` to create reports on the dataset

## 4 step:

Run `pipenv run python -W ignore src/models/train_models.py` train models and generate report file

Output files will be saved in a sister directory `diagnosis_predictor_data`
