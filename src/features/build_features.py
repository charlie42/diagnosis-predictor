import logging
# To import from parent directory
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import data

def build_nvld(data, ignore_reading_condition=False):

    wisc_vci = "WISC,WISC_VCI"
    wisc_bd = "WISC,WISC_BD_Scaled"
    wisc_mr = "WISC,WISC_MR_Scaled"
    wisc_vsi = "WISC,WISC_VSI"
    wisc_vci = "WISC,WISC_VCI"
    wisc_fri = "WISC,WISC_FRI"
    wasi_bd = "WASI,WASI_BD_T"
    wasi_mr = "WASI,WASI_Matrix_T"
    wasi_vci = "WASI,WASI_VCI_Comp"
    wasi_pri = "WASI,WASI_PRI_Comp"
    wais_bd = "WAIS,WAIS_BD_PERC"
    wais_mr = "WAIS,WAIS_MR_PERC"
    wais_vci = "WAIS,WAIS_VCI_COMP"
    wais_pri = "WAIS,WAIS_PRI_COMP"
    word = "WIAT,WIAT_Word_P"
    cbcl = "CBCL,CBCL_SP_T"
    cbcl_pre = "CBCL_Pre,CBCLPre_SP_T"
    num = "WIAT,WIAT_Num_P"
    flanker_p = "NIH_Scores,NIH7_Flanker_P"
    card_p = "NIH_Scores,NIH7_Card_P"
    flanker = "NIH_final,NIH_Flanker_Age_Corr_Stnd"
    card = "NIH_final,NIH_Card_Age_Corr_Stnd"
    peg = "Pegboard,peg_z_d"

    cols = [wisc_vci, 
            wisc_bd, 
            wisc_mr,
            wisc_vsi,
            wisc_vci,
            wisc_fri,
            wasi_bd,
            wasi_mr,
            wasi_vci,
            wasi_pri,
            wais_bd,
            wais_mr,
            wais_vci,
            wais_pri,
            word,
            cbcl,
            cbcl_pre,
            num,
            flanker,
            card,
            flanker_p,
            card_p,
            peg]

    for col in cols:
        if col in data.columns:
            data[col] = data[col].astype(float)
        
    # Step 1
    if wisc_bd in data.columns:
        spatial_deficit = (data[wisc_bd] <= 7) | (data[wisc_mr] <= 7)
        discrepancy =((data[wisc_vci] - data[wisc_fri]) > 15) | ((data[wisc_vci] - data[wisc_vsi]) > 15)
    elif wasi_bd in data.columns:
        spatial_deficit = (data[wasi_bd] <= 40) | (data[wasi_mr] <= 40)
        discrepancy = ((data[wasi_vci] - data[wasi_pri]) > 15)
    else:
        spatial_deficit = (data[wais_bd] <= 16 | data[wais_mr] <= 16)
        discrepancy = (data[wais_vci] - data[wais_pri]) > 15

    spatial_condition = (spatial_deficit | discrepancy)
    reading_condition = (data[word] >= 16)
    step_1_condition = spatial_condition if ignore_reading_condition else spatial_condition & reading_condition

    # Step 2
    if flanker_p in data.columns:
        EF_condition = (data[flanker_p] < 16) | (data[card_p] < 16)
    else:
        EF_condition = (data[flanker] <= 85) | (data[card] <= 85)
    
    if cbcl in data.columns:
        social_condition = (data[cbcl] >= 70)
    else:
        social_condition = (data[cbcl_pre] >= 70)

    math_condition = (data[num] <= 16)
    motor_condition = (data[peg] <= -0.800)
    step2_condition = ((social_condition.astype(int) + math_condition.astype(int) + EF_condition.astype(int) + motor_condition.astype(int)) >= 2)

    return step_1_condition & step2_condition 

def only_keep_item_lvl_cols(data, item_level_ds, test_based_diags):
    # Only keep columns that are in item_level_ds in data (item level responses + diagnoses + new diagnoses)
    item_level_cols = list(item_level_ds.columns) + list(test_based_diags.values())
    print(item_level_cols)
    data = data[item_level_cols]
    return data

def join_by_id_and_diag(df1, df2):
    diag_cols = [col for col in df1.columns if "Diag." in col]
    df = df1.merge(df2, on=diag_cols + ["ID"], how="inner")
    return df

def make_new_diag_cols(item_level_ds, cog_tasks_ds, subscales_ds, clinical_config):

    data = item_level_ds

    if clinical_config["predict test-based diags"]: # Create learning test-based diags

        data = join_by_id_and_diag(item_level_ds, cog_tasks_ds)
        print(data.columns)
        data = join_by_id_and_diag(data, subscales_ds)
        print(data.columns)

        test_based_diags = {
            "read": "Diag.Specific Learning Disorder with Impairment in Reading (test)",
            "math": "Diag.Specific Learning Disorder with Impairment in Mathematics (test)",
            "write": "Diag.Specific Learning Disorder with Impairment in Written Expression (test)",
            "int-mild": "Diag.Intellectual Disability-Mild (test)",
            "int-borderline": "Diag.Borderline Intellectual Functioning (test)",
            "ps": "Diag.Processing Speed Deficit (test)",
            "nvld": "Diag.NVLD (test)",
            "nvld-no-read": "Diag.NVLD without reading condition (test)",
        }

        # Create new diganosis columns: positive if consensus diagnosis is positive OR if WIAT or WISC score is within range
        data[test_based_diags["read"]] = (data["WIAT,WIAT_Word_Stnd"] < 85) & (data["WISC,WISC_FSIQ"] > 70)
        data[test_based_diags["math"]] = (data["WIAT,WIAT_Num_Stnd"] < 85) & (data["WISC,WISC_FSIQ"] > 70)
        data[test_based_diags["write"]] = (data["WIAT,WIAT_Spell_Stnd"] < 85)  & (data["WISC,WISC_FSIQ"] > 70)
        data[test_based_diags["int-mild"]] = (data["WISC,WISC_FSIQ"] < 70) 
        data[test_based_diags["int-borderline"]] = ((data["WISC,WISC_FSIQ"] < 85) & (data["WISC,WISC_FSIQ"] > 70))
        data[test_based_diags["ps"]] = (data["WISC,WISC_PSI"] < 85) 
        try:
            data[test_based_diags["nvld"]] = build_nvld(data)
            data[test_based_diags["nvld-no-read"]] = build_nvld(data, ignore_reading_condition=True)
        except Exception as e:
            print(f"Coulnd't create NVLD diagnosis: {e}")
            test_based_diags.pop("nvld")
            test_based_diags.pop("nvld-no-read")

        # Drop non-item-lvl columns
        data = only_keep_item_lvl_cols(data, item_level_ds, test_based_diags) # only needed them for building new learning diags
        print(data.columns)
    
    data["Diag.Any Diag"] = data["Diag.No Diagnosis Given"].apply(lambda x: 1 if x == 0 else 0)
        
    return data