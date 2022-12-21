import os, shutil, json, numpy

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
        print(key)
        write_dict_to_file(dict[key], path, key+".txt")