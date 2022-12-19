from sklearn.model_selection import train_test_split

def customize_input_cols_per_diag(input_cols, diag):
    # Remove "Diag.Intellectual Disability-Mild" when predicting "Diag.Borderline Intellectual Functioning"
    #   and vice versa because they are highly correlated, same for other diagnoses
    
    if diag == "Diag.Intellectual Disability-Mild":
        input_cols = [x for x in input_cols if x != "Diag.Borderline Intellectual Functioning"]
    if diag == "Diag.Borderline Intellectual Functioning":
        input_cols = [x for x in input_cols if x != "Diag.Intellectual Disability-Mild"]
    if diag == "Diag.No Diagnosis Given":
        input_cols = [x for x in input_cols if not x.startswith("Diag.")]
    if diag == "Diag.ADHD-Combined Type":
        input_cols = [x for x in input_cols if x not in ["Diag.ADHD-Inattentive Type", 
                                                         "Diag.ADHD-Hyperactive/Impulsive Type",
                                                         "Diag.Other Specified Attention-Deficit/Hyperactivity Disorder",
                                                         "Diag.Unspecified Attention-Deficit/Hyperactivity Disorder"]]
    if diag == "Diag.ADHD-Inattentive Type":
        input_cols = [x for x in input_cols if x not in ["Diag.ADHD-Combined Type", 
                                                         "Diag.ADHD-Hyperactive/Impulsive Type",
                                                         "Diag.Other Specified Attention-Deficit/Hyperactivity Disorder",
                                                         "Diag.Unspecified Attention-Deficit/Hyperactivity Disorder"]]
                      
    return input_cols

def get_input_and_output_cols_for_diag(full_dataset, diag, use_other_diags_as_input, input_questionnaires):
    
    input_cols = [x for x in full_dataset.columns if x.split(",")[0] in input_questionnaires]
    
    output_col = diag
    
    return input_cols, output_col

def create_datasets(full_dataset, diag_cols, split_percentage, use_other_diags_as_input, input_questionnaire):
    datasets = {}
    for diag in diag_cols:
        
        input_cols, output_col = get_input_and_output_cols_for_diag(full_dataset, diag, use_other_diags_as_input, input_questionnaire)
        
        # Split train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(full_dataset[input_cols], full_dataset[output_col], test_size=split_percentage, stratify=full_dataset[output_col], random_state=1)
        X_train_train, X_val, y_train_train, y_val = train_test_split(X_train, y_train, test_size=split_percentage, stratify=y_train, random_state=1)
    
        datasets[diag] = { "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "X_train_train": X_train_train,
                        "X_val": X_val,
                        "y_train_train": y_train_train,
                        "y_val": y_val}
    return datasets