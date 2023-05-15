def make_new_diag_cols(data):
    data["Diag.Any Diag"] = data["Diag.No Diagnosis Given"].apply(lambda x: 1 if x == 0 else 0)
    return data