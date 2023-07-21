def customize_input_cols_per_diag(input_cols, diag):
    # Remove "Diag.Intellectual Disability-Mild" when predicting "Diag.Borderline Intellectual Functioning"
    #   and vice versa because they are highly correlated, same for other diagnoses
    #   (only useful when use_other_diags_as_input = 1)
    
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
        
    # Remove NIH scores for NVLD (used for diagnosis)
    if "NVLD" in diag:
        input_cols = [x for x in input_cols if not x.startswith("NIH")]
                      
    return input_cols