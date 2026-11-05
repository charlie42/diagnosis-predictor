# diagnosis-predictor

Code for Faure et al. 2026, *BMC Medical Informatics and Decision Making*: https://doi.org/10.1186/s12911-025-03329-5

## Author and citation

Author: Charlie Faure.

Faure C, Leventhal BL, Koyama M, Panda S, Vogelstein JT, Milham MP, Lindner AB, Klein A. Accurate and efficient data-driven psychiatric assessment using machine learning. BMC Medical Informatics and Decision Making. 2026;26(1):40. https://doi.org/10.1186/s12911-025-03329-5

See `CITATION.cff` for a machine-readable citation.


File structure: http://drivendata.github.io/cookiecutter-data-science/#directory-structure 

Predicting Healthy Brain Network (HBN) consensus diagnoses, and test-based diagnoses (`predict test-based diags` in `config/clinical/general.yml`) for:
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
Request access to HBN phenotypic data (http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/), download it from LORIS and save it as `data/raw/LORIS-release-10.csv`

## 2 step:
Check that you're happy with the clinical and technical config values (`config/` directory)

## 3 step:

Run `pipenv run python -W ignore src/data/create_datasets.py 0 0 0 0 0` to prepare the data

Arguments: 
- `only_assessment_distribution`: do not create data, only visualize assessment distribution 
- `only_parent_report`: only use parent-report assessments
- `use_other_diags_as_input` = 0: use presence of other diagnoses as input variables
- `only_free_assessments` = 0: use only non-proprietary assessments
- `learning` = 0: use `learning.yml` config file with extended set of input assessments

### Optional:

Run `pipenv run python -W ignore src/data/create_data_reports.py` to create reports on the dataset

## 4 step:

Run `pipenv run python -W ignore src/models/train_models.py` train models and generate report file

Output files will be saved in a sister directory `diagnosis_predictor_data`
