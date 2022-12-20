<<<<<<< HEAD
import os, shutil, json, numpy

# File Utilities
=======
import os, shutil

>>>>>>> 3a87fde (CODE: refactor - add debug flag and clean models and reports dir in train_models)
def clean_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def clean_dirs(folders):
    for folder in folders:
        clean_dir(folder)

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_numpy_int64_to_int_in_dict(dict):
    return {key: int(value) for key, value in dict.items()}

def write_dict_to_file(dict, path, file_name):
    # If dict values are numpy int64, convert to int
    if type(list(dict.values())[0]) == numpy.int64:
        dict = convert_numpy_int64_to_int_in_dict(dict)

    create_dir_if_not_exists(path)
    with open(path+file_name, 'w') as file:
        file.write(json.dumps(dict, indent=2))

def write_two_lvl_dict_to_file(dict, path):
    create_dir_if_not_exists(path)
    for key in dict.keys():
        write_dict_to_file(dict[key], path, key+".txt")

def remove_chars_forbidden_in_file_names(string):
    forbidden_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in forbidden_chars:
        string = string.replace(char, '.')
    return string


# Model Utilities
def get_base_model_name_from_estimator(classifier):
    return classifier.__class__.__name__.lower()

def get_estimator_from_pipeline(pipeline):
    return pipeline.steps[-1][1]

def get_base_model_name_from_pipeline(pipeline):
    classifier = get_estimator_from_pipeline(pipeline)
    return get_base_model_name_from_estimator(classifier)



# Datetime utils
def get_string_with_current_datetime():
    import datetime

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string with the format 'YYYY-MM-DD HH.MM.SS' (Can't use ':' in file names)
    date_time_str = now.strftime('%Y-%m-%d %H.%M.%S')

    return date_time_str
