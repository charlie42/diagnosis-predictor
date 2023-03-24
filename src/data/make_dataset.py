# Notes:
# Data: http://data.healthybrainnetwork.org/ (LORIS, all_with_new_diag_and_nih query)
# Diagnosis is contained in the "Diagnosis_ClinicianConsensus" column (not in "ConsensusDx")

import pandas as pd
import numpy as np
from collections import Counter
from re import M
import os, inspect
import matplotlib.pyplot as plt
import sys

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util

def build_output_dir_name(first_assessment_to_drop):
    # Part with the datetime
    datetime_part = util.get_string_with_current_datetime()

    # Part with the params
    params_part = "first_dropped_assessment__" + first_assessment_to_drop
    
    return datetime_part + "___" + params_part

def set_up_directories(first_assessment_to_drop):

    # Create directory in the parent directory of the project (separate repo) for output data, models, and reports
    data_dir = "../diagnosis_predictor_data/"
    util.create_dir_if_not_exists(data_dir)

    # Create directory inside the output directory with the run timestamp and first_assessment_to_drop param
    current_output_dir_name = build_output_dir_name(first_assessment_to_drop)

    data_statistics_dir = data_dir + "reports/make_dataset/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_statistics_dir)
    util.create_dir_if_not_exists(data_statistics_dir+"figures/")

    data_output_dir = data_dir + "data/make_dataset/" + current_output_dir_name + "/"
    util.create_dir_if_not_exists(data_output_dir)

    return data_statistics_dir, data_output_dir
    
def remove_admin_cols(full):
    # Remove uninteresting columns
    columns_to_drop = []

    column_suffixes_to_drop = ["Administration", "Data_entry", "Days_Baseline", "START_DATE", "Season", "Site", "Study", "Year", "Commercial_Use", "Release_Number"]
    for suffix in column_suffixes_to_drop:
        cols_w_suffix = [x for x in full.columns if suffix in x]
        columns_to_drop.extend(cols_w_suffix)

    present_columns_to_drop = full.filter(columns_to_drop)
    full = full.drop(present_columns_to_drop, axis = 1)
    return full 

def get_ID_from_EID(full, EID_cols):

    # Get only EID cols
    full_for_EID_check = full[EID_cols]

    # In EID cols df, fill missing EIDs with EIDs from other questionnaires 
    full_for_EID_check = full_for_EID_check.ffill(axis=1).bfill(axis=1)

    # Drop lines with different EID within one row
    full = full[full_for_EID_check.eq(full_for_EID_check.iloc[:, 0], axis=0).all(1)]

    # Fill ID field with the first non-null questionnaire-specific EID
    full["ID"] = full_for_EID_check.iloc[:, 0]

    return full

# Drop rows with underscores in ID (NDARZZ007YMP_1, NDARAA075AMK_Visit_1)
def drop_rows_w_underscore_in_id(full):

    rows_with_underscore_in_id = full[full["ID"].str.contains("_")]
    non_empty_columns_in_underscore = rows_with_underscore_in_id.columns[
        ~rows_with_underscore_in_id.isna().all()
    ].tolist() 
    non_empty_questionnaires_in_underscore = set([x.split(",")[0] for x in non_empty_columns_in_underscore])
    
    non_empty_questionnaires_in_underscore.remove("Identifiers")
    non_empty_questionnaires_in_underscore.remove("ID")
    full_wo_underscore = full[~full["ID"].str.contains("_")]

    # Drop questionnaires present in rows with underscores from data ({'DailyMeds', 'TRF', 'TRF_P', 'TRF_Pre'})
    for questionnaire in non_empty_questionnaires_in_underscore:
        full_wo_underscore = full_wo_underscore.drop(full_wo_underscore.filter(regex=(questionnaire+",")), axis=1)

    return full_wo_underscore

def remove_incomplete_and_missing_diag(full_wo_underscore):
    full_wo_underscore = full_wo_underscore[full_wo_underscore["Diagnosis_ClinicianConsensus,DX_01"] != "No Diagnosis Given: Incomplete Eval"]
    full_wo_underscore = full_wo_underscore[full_wo_underscore["Diagnosis_ClinicianConsensus,EID"].notna()]
    return full_wo_underscore

def get_assessment_answer_count(full_wo_underscore, EID_cols):
    assessment_answer_counts = full_wo_underscore[EID_cols].count().sort_values(ascending=False).to_frame()
    assessment_answer_counts["Ratio"] = assessment_answer_counts[0]/full_wo_underscore["ID"].nunique()*100
    assessment_answer_counts.columns = ["N of Participants", "% of Participants Filled"]
    return assessment_answer_counts

def get_relevant_id_cols_by_popularity(assessment_answer_counts):
    # Get list of assessments sorted by popularity
    EID_columns_by_popularity = assessment_answer_counts.index

    # Get relevant assessments: 
    #   relevant cognitive tests, Questionnaire Measures of Emotional and Cognitive Status, and 
    #   Questionnaire Measures of Family Structure, Stress, and Trauma (from Assessment_List_Jan2019.xlsx)
    relevant_EID_list = [x+",EID" for x in ["Basic_Demos", "PreInt_EduHx", "PreInt_DevHx", "SympChck", "SCQ", "Barratt", 
        "ASSQ", "ARI_P", "SDQ", "SWAN", "SRS", "CBCL", "ICU_P", "APQ_P", "PCIAT", "DTS", "ESWAN", "MFQ_P", "APQ_SR", 
        "WHODAS_P", "CIS_P", "PSI", "RBS", "PhenX_Neighborhood", "WHODAS_SR", "CIS_SR", "SCARED_SR", 
        "C3SR", "CCSC", "CPIC", "YSR", "PhenX_SchoolRisk", "CBCL_Pre", "SRS_Pre", "ASR"]]

    # Get relevant ID columns sorted by popularity    
    EID_columns_by_popularity = [x for x in EID_columns_by_popularity if x in relevant_EID_list]

    return EID_columns_by_popularity

def get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity):
    cumul_number_of_examples_list = []
    for i in range(1, len(EID_columns_by_popularity)+1):
        columns = EID_columns_by_popularity[0:i] # top i assessments
        cumul_number_of_examples = full_wo_underscore[columns].notnull().all(axis=1).sum()
        cumul_number_of_examples_list.append([cumul_number_of_examples, [x.split(",")[0] for x in columns]])
    cumul_number_of_examples_df = pd.DataFrame(cumul_number_of_examples_list)
    cumul_number_of_examples_df.columns = ("Respondents", "Assessments")
    cumul_number_of_examples_df["N of Assessments"] = cumul_number_of_examples_df["Assessments"].str.len()
    cumul_number_of_examples_df["Last Assessment"] = cumul_number_of_examples_df["Assessments"].str[-1]
    return cumul_number_of_examples_df

def plot_comul_number_of_examples(cumul_number_of_examples_df, data_statistics_dir):
    plt.figure(figsize=(16,8))
    plt.xticks(cumul_number_of_examples_df["N of Assessments"])
    plt.scatter(cumul_number_of_examples_df["N of Assessments"], cumul_number_of_examples_df["Respondents"])
    plt.savefig(data_statistics_dir+'figures/cumul_assessment_distrib.png')  

def get_columns_until_dropped(full_wo_underscore, EID_columns_until_dropped):
    columns_until_dropped = []
    assessments_until_dropped = [x.split(",")[0]+"," for x in EID_columns_until_dropped]
    for assessment in assessments_until_dropped:
        columns = [column for column in full_wo_underscore.columns if column.startswith(assessment)]
        columns_until_dropped.extend(columns)
    return columns_until_dropped

def get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped, columns_until_dropped):
    diag_colunms = ["Diagnosis_ClinicianConsensus,DX_01", "Diagnosis_ClinicianConsensus,DX_02", "Diagnosis_ClinicianConsensus,DX_03", 
        "Diagnosis_ClinicianConsensus,DX_04", "Diagnosis_ClinicianConsensus,DX_05", "Diagnosis_ClinicianConsensus,DX_06", 
        "Diagnosis_ClinicianConsensus,DX_07", "Diagnosis_ClinicianConsensus,DX_08", "Diagnosis_ClinicianConsensus,DX_09", 
        "Diagnosis_ClinicianConsensus,DX_10"]
    data_up_to_dropped = full_wo_underscore.loc[full_wo_underscore[EID_columns_until_dropped].dropna(how="any").index][columns_until_dropped+["ID"]+diag_colunms]

    return data_up_to_dropped

def convert_numeric_col_to_numeric_type(col):
    if col.name != "ID" and "Diagnosis_ClinicianConsensus" not in col.name:
        return pd.to_numeric(col, errors='coerce') # Non-numeric values are converted to NaN and removed later in remove_cols_w_missing_over_n function
    else:
        return col

def get_missing_values_df(data_up_to_dropped):
    missing_report_up_to_dropped = data_up_to_dropped.isna().sum().to_frame(name="Amount missing")
    missing_report_up_to_dropped["Persentage missing"] = missing_report_up_to_dropped["Amount missing"]/data_up_to_dropped["ID"].nunique() * 100
    missing_report_up_to_dropped = missing_report_up_to_dropped[~missing_report_up_to_dropped.index.str.contains("Diagnosis_ClinicianConsensus")] # remove dx because it's expected to be missing
    missing_report_up_to_dropped = missing_report_up_to_dropped[missing_report_up_to_dropped["Persentage missing"] > 0]
    return missing_report_up_to_dropped[missing_report_up_to_dropped["Persentage missing"] > 0].sort_values(ascending=False, by="Amount missing")

def remove_cols_w_missing_over_n(data_up_to_dropped, n, missing_values_df):
    cols_to_remove = list(missing_values_df[missing_values_df["Persentage missing"] > n].index)
    data_up_to_dropped = data_up_to_dropped.drop(cols_to_remove, axis=1)
    return data_up_to_dropped

def add_missingness_markers(data_up_to_dropped, n, missing_values_df):
    missing_cols_to_mark = list(missing_values_df[(missing_values_df["Persentage missing"] <= 40) & (missing_values_df["Persentage missing"] > n)].index)
    for col in missing_cols_to_mark:
        data_up_to_dropped[col+ "_WAS_MISSING"] = data_up_to_dropped[col].isna()
    return data_up_to_dropped

def transform_dx_cols(data_up_to_dropped):
    og_diag_cols = [x for x in data_up_to_dropped.columns if "DX_" in x]

    # Get list of diagnoses
    diags = []
    for col in og_diag_cols:
        diags.extend(list(data_up_to_dropped[col].value_counts().index))
    diags = list(set(diags))
    diags.remove(' ')

    # Make new columns
    for diag in diags:
        data_up_to_dropped["Diag." + util.remove_chars_forbidden_in_file_names(diag)] = (data_up_to_dropped[og_diag_cols] == diag).any(axis=1)
        
    # Drop original diag columns
    data_up_to_dropped = data_up_to_dropped.drop(og_diag_cols, axis=1)

    return data_up_to_dropped

def transform_devhx_eduhx_cols(data_up_to_dropped):

    list_of_preg_symp_cols = [x for x in data_up_to_dropped.columns if "preg_symp" in x]
    
    # If any of the preg_symp columns are 1, then the preg_symp column is 1
    data_up_to_dropped["preg_symp"] = (data_up_to_dropped[list_of_preg_symp_cols] == 1).any(axis=1)

    # Drop original preg_symp columns
    data_up_to_dropped = data_up_to_dropped.drop(list_of_preg_symp_cols, axis=1) 

    data_up_to_dropped = data_up_to_dropped.drop(["PreInt_EduHx,NeuroPsych", "PreInt_EduHx,IEP", "PreInt_EduHx,learning_disability", "PreInt_EduHx,EI", "PreInt_EduHx,CPSE"], axis=1)

    return data_up_to_dropped

def separate_item_lvl_from_scale_scores(data_up_to_dropped, columns_until_dropped):
    total_score_cols_w_raw = ["SCQ,SCQ_Total", 
                        "Barratt,Barratt_Total", 
                        "ASSQ,ASSQ_Total",
                        "ARI_P,ARI_P_Total_Score", 
                        "SWAN,SWAN_Total",
                        "SRS,SRS_Total", 
                        "SRS,SRS_Total_T", 
                        "CBCL,CBCL_Total",
                        "CBCL,CBCL_Total_T",
                        "ICU_P,ICU_P_Total",
                        "APQ_P,APQ_P_Total",
                        "PCIAT,PCIAT_Total",
                        "DTS,DTS_Total",
                        "MFQ_P,MFQ_P_Total",
                        "APQ_SR,APQ_SR_Total",
                        #"WHODAS_P,WHODAS_P_Total", # Don't remove impairment scores - to predict
                        #"CIS_P,CIS_P_Score", # Don't remove impairment scores - to predict
                        "PSI,PSI_Total",
                        "PSI,PSI_Total_T",
                        "RBS,RBS_Total",
                        #"WHODAS_SR,WHODAS_SR_Score", # Don't remove impairment scores - to predict
                        #"CIS_SR,CIS_SR_Total" # Don't remove impairment scores - to predict
                    ]
    total_score_raw_cols = [x.strip("_T") for x in total_score_cols_w_raw if x.endswith("_T")]
    subscale_score_cols_w_raw = ["Barratt,Barratt_Total_Edu", "Barratt,Barratt_Total_Occ",
                        "SWAN,SWAN_HY", "SWAN,SWAN_IN",
                        "SRS,SRS_AWR_T", "SRS,SRS_AWR", "SRS,SRS_COG_T", "SRS,SRS_COG", "SRS,SRS_COM_T", "SRS,SRS_COM", "SRS,SRS_DSMRRB_T", "SRS,SRS_DSMRRB", "SRS,SRS_MOT_T", "SRS,SRS_MOT", "SRS,SRS_RRB_T", "SRS,SRS_RRB", "SRS,SRS_SCI_T", "SRS,SRS_SCI",
                        "CBCL,CBCL_AB_T", "CBCL,CBCL_AB", "CBCL,CBCL_AD_T", "CBCL,CBCL_AD", "CBCL,CBCL_AP_T", "CBCL,CBCL_AP", "CBCL,CBCL_Ext_T", "CBCL,CBCL_Ext", "CBCL,CBCL_Int_T", "CBCL,CBCL_Int", "CBCL,CBCL_RBB_T", "CBCL,CBCL_RBB", "CBCL,CBCL_SC_T", "CBCL,CBCL_SC", "CBCL,CBCL_SP_T", "CBCL,CBCL_SP", "CBCL,CBCL_TP_T", "CBCL,CBCL_TP", "CBCL,CBCL_WD_T", "CBCL,CBCL_WD", "CBCL,CBCL_C", "CBCL,CBCL_OP",
                        "ICU_P,ICU_P_Callous", "ICU_P,ICU_P_Uncaring", "ICU_P,ICU_P_Unemotional",
                        "APQ_P,APQ_P_CP", "APQ_P,APQ_P_ID", "APQ_P,APQ_P_INV", "APQ_P,APQ_P_OPD", "APQ_P,APQ_P_PM", "APQ_P,APQ_P_PP",
                        "DTS,DTS_absorption", "DTS,DTS_appraisal", "DTS,DTS_regulation", "DTS,DTS_tolerance",
                        "APQ_SR,APQ_SR_CP", "APQ_SR,APQ_SR_ID", "APQ_SR,APQ_SR_INV_D", "APQ_SR,APQ_SR_INV_M", "APQ_SR,APQ_SR_OPD", "APQ_SR,APQ_SR_PM", "APQ_SR,APQ_SR_PP",
                        "PSI,PSI_DC_T", "PSI,PSI_DC", "PSI,PSI_PCDI_T", "PSI,PSI_PCDI", "PSI,PSI_PD_T", "PSI,PSI_PD",
                        "RBS,RBS_Score_01", "RBS,RBS_Score_02", "RBS,RBS_Score_03", "RBS,RBS_Score_04", "RBS,RBS_Score_05",  
                        "C3SR,C3SR_AG", "C3SR,C3SR_AG_T", "C3SR,C3SR_FR", "C3SR,C3SR_FR_T", "C3SR,C3SR_HY", "C3SR,C3SR_HY_T", "C3SR,C3SR_IN", "C3SR,C3SR_IN_T", "C3SR,C3SR_LP", "C3SR,C3SR_LP_T", "C3SR,C3SR_NI", "C3SR,C3SR_PI",
                        ]
    subscale_score_raw_cols = [x.strip("_T") for x in subscale_score_cols_w_raw if x.endswith("_T")]

    # Item level columns = all columns except those of total and subscale scores (includes diag cols)
    item_level_cols = [x for x in columns_until_dropped if (x not in total_score_cols_w_raw) and (x not in subscale_score_cols_w_raw)]
    item_level_col_subset = [x for x in data_up_to_dropped.columns if (x not in total_score_cols_w_raw) and (x not in subscale_score_cols_w_raw)]
    data_up_to_dropped_item_lvl = data_up_to_dropped[item_level_col_subset]

    # Total columns = all columns except those for item_level (calculated above), all subscale scores, and raw total scores (only keep t-scores)
    total_score_col_subset = [x for x in data_up_to_dropped.columns if (x not in item_level_cols) and (x not in subscale_score_cols_w_raw) and (x not in total_score_raw_cols)]
    data_up_to_dropped_total_scores = data_up_to_dropped[total_score_col_subset]

    # Subscale columns = all columns except those for item_level (calculated above), all total scores, and raw subscale scores (only keep t-scores)
    subscale_score_col_subset = [x for x in data_up_to_dropped.columns if (x not in item_level_cols) and (x not in total_score_cols_w_raw) and (x not in subscale_score_raw_cols)]
    data_up_to_dropped_subscale_scores = data_up_to_dropped[subscale_score_col_subset]

    return data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores

def remove_irrelavent_missing_markers(data_up_to_dropped, data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores):
    was_missing_cols = [x for x in data_up_to_dropped.columns if "_WAS_MISSING" in x]
    was_missing_col_originals = [x.split("_WAS_MISSING")[0] for x in was_missing_cols]

    for col in was_missing_col_originals:
        if col not in data_up_to_dropped_item_lvl.columns and col +"_WAS_MISSING" in data_up_to_dropped_item_lvl.columns:
            data_up_to_dropped_item_lvl = data_up_to_dropped_item_lvl.drop(col+"_WAS_MISSING", axis=1)
        
    for col in was_missing_col_originals:
        if col not in data_up_to_dropped_total_scores.columns and col +"_WAS_MISSING" in data_up_to_dropped_total_scores.columns:
            data_up_to_dropped_total_scores = data_up_to_dropped_total_scores.drop(col+"_WAS_MISSING", axis=1)
        
    for col in was_missing_col_originals:
        if col not in data_up_to_dropped_subscale_scores.columns and col +"_WAS_MISSING" in data_up_to_dropped_subscale_scores.columns:
            data_up_to_dropped_subscale_scores = data_up_to_dropped_subscale_scores.drop(col+"_WAS_MISSING", axis=1)

    return data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores

def export_datasets(data_up_to_dropped, data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_output_dir):
    ## Predicting impairment (remove impairment questionnaires from input)
    impairment_assessments = ["WHODAS_SR,", "WHODAS_P,", "CIS_SR,", "CIS_P,"]

    impairment_columns = [] # All item-level impairment (keep total impairment scores for output)
    for impairment_assessment in  impairment_assessments:
        impairment_columns.extend([x for x in data_up_to_dropped if x.startswith(impairment_assessment) and x not in ["WHODAS_P,WHODAS_P_Total", "CIS_P,CIS_P_Score", "WHODAS_SR,WHODAS_SR_Score", "CIS_SR,CIS_SR_Total"]])

    data_up_to_dropped_item_lvl_wo_impairment = data_up_to_dropped_item_lvl[[x for x in data_up_to_dropped_item_lvl.columns if x not in impairment_columns]]
    data_up_to_dropped_subscale_scores_wo_impairment = data_up_to_dropped_subscale_scores[[x for x in data_up_to_dropped_subscale_scores.columns if x not in impairment_columns]]
    data_up_to_dropped_total_scores_wo_impairment = data_up_to_dropped_total_scores[[x for x in data_up_to_dropped_total_scores.columns if x not in impairment_columns]]

    data_up_to_dropped_item_lvl_wo_impairment.to_csv(data_output_dir + "item_lvl_wo_impairment.csv", index=False)
    data_up_to_dropped_subscale_scores_wo_impairment.to_csv(data_output_dir + "subscale_scores_wo_impairment.csv", index=False)
    data_up_to_dropped_total_scores_wo_impairment.to_csv(data_output_dir + "total_scores_wo_impairment.csv", index=False)

    ## Predicting diagnoses
    data_up_to_dropped_item_lvl_w_impairment = data_up_to_dropped_item_lvl
    data_up_to_dropped_subscale_scores_w_impairment = data_up_to_dropped_subscale_scores
    data_up_to_dropped_total_scores_w_impairment = data_up_to_dropped_total_scores

    data_up_to_dropped_item_lvl_w_impairment.to_csv(data_output_dir + "item_lvl_w_impairment.csv", index=False)
    data_up_to_dropped_subscale_scores_w_impairment.to_csv(data_output_dir + "subscale_scores_w_impairment.csv", index=False)
    data_up_to_dropped_total_scores_w_impairment.to_csv(data_output_dir + "total_scores_w_impairment.csv", index=False)

def main(only_assessment_distribution, first_assessment_to_drop):
    only_assessment_distribution = int(only_assessment_distribution)

    data_statistics_dir, data_output_dir = set_up_directories(first_assessment_to_drop)

    # LORIS saved query (all data)
    full = pd.read_csv("data/raw/LORIS-release-10.csv", dtype=object)
    
    # Replace NaN (currently ".") values with np.nan
    full = full.replace(".", np.nan)

    # Drop first row (doesn't have ID)
    full = full.iloc[1: , :]

    # Drop empty columns
    full = full.dropna(how='all', axis=1)

    full = remove_admin_cols(full)

    # Get ID columns (contain quetsionnaire names, e.g. 'ACE,EID', will be used to check if an assessment is filled)
    EID_cols = [x for x in full.columns if ",EID" in x]

    # Get ID col from EID cols
    full = get_ID_from_EID(full, EID_cols)

    full_wo_underscore = drop_rows_w_underscore_in_id(full)

    # Drop questionnaires present in rows with underscores from data from list of ID columns
    EID_cols = [x for x in EID_cols if 'TRF' not in x]
    EID_cols = [x for x in EID_cols if 'DailyMeds' not in x]

    # Remove incomplete DX and missing DX
    full_wo_underscore = remove_incomplete_and_missing_diag(full_wo_underscore)    

    # Get list of assessments in data
    assessment_list = set([x.split(",")[0] for x in EID_cols])

    # Check how many people filled each assessments
    assessment_answer_counts = get_assessment_answer_count(full_wo_underscore, EID_cols)
    assessment_answer_counts.to_csv(data_statistics_dir + "assessment-filled-distrib.csv")

    # Get relevant ID columns sorted by popularity
    EID_columns_by_popularity = get_relevant_id_cols_by_popularity(assessment_answer_counts)    

    # Get cumulative distribution of assessments: number of people who took all top 1, top 2, top 3, etc. popular assessments 
    cumul_number_of_examples_df = get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity)
    cumul_number_of_examples_df.to_csv(data_statistics_dir + "assessment-filled-distrib-cumul.csv")

    # Plot cumulative distribution of assessments
    plot_comul_number_of_examples(cumul_number_of_examples_df, data_statistics_dir)

    ### => The first drop-off in number of respondents is at ICU_P, 
    # then SCARED_SR (the biggest drop off, and it's the first assessment with an age restriction). Last drop off is at CPIC.

    if only_assessment_distribution != 1:
    
        # List of most popular assessments until the first one from the drop list 
        EID_columns_until_dropped = [x for x in EID_columns_by_popularity[:EID_columns_by_popularity.index(first_assessment_to_drop+",EID")]]

        # Get data up to the dropped assessment
        # Get only people who took the most popular assessments until the first one from the drop list 
        columns_until_dropped = get_columns_until_dropped(full_wo_underscore, EID_columns_until_dropped)
        data_up_to_dropped = get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped, columns_until_dropped)

        # Remove EID columns: not needed anymore
        data_up_to_dropped = data_up_to_dropped.drop(EID_columns_until_dropped, axis=1)

        # Aggregare demographics input columns: remove per parent data from Barratt
        data_up_to_dropped = data_up_to_dropped.drop(["Barratt,Barratt_P1_Edu", "Barratt,Barratt_P1_Occ", "Barratt,Barratt_P2_Edu", "Barratt,Barratt_P2_Occ"], axis=1)

        # Transform PreInt_DevHx columns
        data_up_to_dropped = transform_devhx_eduhx_cols(data_up_to_dropped)

        # Convert numeric columns to numeric type (all except ID and DX)
        data_up_to_dropped = data_up_to_dropped.apply(lambda col: convert_numeric_col_to_numeric_type(col))

        # Save report of missing values
        missing_values_df = get_missing_values_df(data_up_to_dropped)
        missing_values_df.to_csv(data_statistics_dir + "missing-values-report.csv")

        # Remove columns with more than 40% missing data
        data_up_to_dropped = remove_cols_w_missing_over_n(data_up_to_dropped, 40, missing_values_df)

        # Special case: replace missing "CBCL,CBCL_56H" with 0 ("Other")
        data_up_to_dropped[["CBCL,CBCL_56H"]] = data_up_to_dropped[["CBCL,CBCL_56H"]].fillna(value=0)

        # Add missingness marker for columns with more than 5% missing data 
        data_up_to_dropped = add_missingness_markers(data_up_to_dropped, 5, missing_values_df)

        # Transform diagnosis columns
        data_up_to_dropped = transform_dx_cols(data_up_to_dropped)

        # Remove ID column - not needed anymore
        data_up_to_dropped = data_up_to_dropped.drop("ID", axis=1)

        # Convert new boolean columns to numeric
        data_up_to_dropped = data_up_to_dropped.replace({True: 1, False: 0})

        # Separate subscale and total scores
        data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores = separate_item_lvl_from_scale_scores(data_up_to_dropped, columns_until_dropped)

        # Remove _WAS_MISSING columns that are not linked to any columns from each dataset
        data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores = remove_irrelavent_missing_markers(data_up_to_dropped, 
            data_up_to_dropped_item_lvl, 
            data_up_to_dropped_total_scores, 
            data_up_to_dropped_subscale_scores)

        # Export final datasets
        export_datasets(data_up_to_dropped, data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_output_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])