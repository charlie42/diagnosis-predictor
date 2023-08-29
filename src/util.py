import os, shutil, json, numpy, datetime

# File Utilities
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

def build_param_string_for_dir_name(params):
    param_string = ""
    for param_name, param_value in params.items():
        param_string += param_name + "__" + str(param_value) + "___"
    # Drop last "___"
    param_string = param_string[:-3]
    return param_string

def get_params_from_current_data_dir_name(current_data_dir_name):

    # Get paramers from the dir name created by train_models.py. Format: "[DATETIME]__first_param_1__second_param_TRUE"

    # Remove the last underscore
    current_data_dir_name = current_data_dir_name[:-1]
    
    # Split the string on the triple underscores
    parts = current_data_dir_name.split("___")
    
    # The first element is the datetime, so we can ignore it
    # The remaining elements are the parameters, so we can assign them to a list
    params = parts[1:]
    
    # Initialize an empty dictionary to store the param names and values
    param_dict = {}
    
    # Iterate through the list of params
    for param in params:
        # Split the param on the underscore to separate the name from the value
        print(param.rsplit("__", 1))
        name, value = param.rsplit("__", 1)
        
        # Add the name and value to the dictionary
        param_dict[name] = value
    
    # Return the dictionary
    return param_dict

def get_newest_non_empty_dir_in_dir(path):
    dir_names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    non_empty_dir_names = [d for d in dir_names if len(os.listdir(path+d)) > 0]
    # Find non-empty dir with the latest timestamp, dir name format: 2023-01-05 11.03.00___first_dropped_assessment__ICU_P___other_diag_as_input__0___debug_mode__True
    
    timestamps = [d.split("___")[0] for d in non_empty_dir_names]
    timestamps = [datetime.datetime.strptime(t, "%Y-%m-%d_%H.%M.%S") for t in timestamps]
    print("DEBUG timestamps", timestamps, "non_empty_dir_names", non_empty_dir_names, "path", path)
    newest_dir_name = non_empty_dir_names[timestamps.index(max(timestamps))]
    return path + newest_dir_name + "/"


# Model Utilities
def get_base_model_name_from_estimator(estimator):
    return estimator.__class__.__name__.lower()

def get_estimator_from_pipeline(pipeline):
    return pipeline.steps[-1][1]

def get_base_model_name_from_pipeline(pipeline):
    estimator = get_estimator_from_pipeline(pipeline)
    return get_base_model_name_from_estimator(estimator)



# Datetime utils
def get_string_with_current_datetime():
    import datetime

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time as a string with the format 'YYYY-MM-DD HH.MM.SS' (Can't use ':' in file names)
    date_time_str = now.strftime('%Y-%m-%d_%H.%M.%S')

    return date_time_str


# Config utils
import yaml
def read_config(type, learning = 0):
    config = yaml.safe_load(open(f"config/{type}/general.yml", "r"))
    
    if type == "clinical" and learning == 1:
        learning_config = yaml.safe_load(open(f"config/{type}/learning.yml", "r"))

        # Rewrite all keys in clnical with values in learning
        config.update(learning_config)
    return config
