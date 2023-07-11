# Notes:
# Data: http://data.healthybrainnetwork.org/ (LORIS, all_with_new_diag_and_nih query)
# Diagnosis is contained in the "Diagnosis_ClinicianConsensus" column (not in "ConsensusDx")

import pandas as pd
import numpy as np
from re import M
import os, inspect
import matplotlib.pyplot as plt
import sys

# To import from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import util, data

def remove_proprietary_assessments(relevant_assessment_list, proprietary_assessments):
    return [x for x in relevant_assessment_list if x not in proprietary_assessments]
    
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

def get_relevant_id_cols_by_popularity(assessment_answer_counts, relevant_assessment_list):

    relevant_EID_list = [x+",EID" for x in relevant_assessment_list]

    # Get list of assessments sorted by popularity
    EID_columns_by_popularity = assessment_answer_counts.index

    # Get relevant ID columns sorted by popularity    
    EID_columns_by_popularity = [x for x in EID_columns_by_popularity if x in relevant_EID_list]

    return EID_columns_by_popularity

def get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity):
    cumul_number_of_examples_list = []
    for i in range(1, len(EID_columns_by_popularity)+1):
        columns = EID_columns_by_popularity[0:i] # top i assessments
        cumul_number_of_examples = full_wo_underscore[columns].notnull().all(axis=1).sum()
        min_age_among_non_null = full_wo_underscore[full_wo_underscore[columns].notnull().all(axis=1)]["Basic_Demos,Age"].astype(float).min()
        cumul_number_of_examples_list.append([cumul_number_of_examples, [x.split(",")[0] for x in columns], min_age_among_non_null])
    cumul_number_of_examples_df = pd.DataFrame(cumul_number_of_examples_list)
    cumul_number_of_examples_df.columns = ("Respondents", "Assessments", "Min Age")
    cumul_number_of_examples_df["N of Assessments"] = cumul_number_of_examples_df["Assessments"].str.len()
    cumul_number_of_examples_df["Last Assessment"] = cumul_number_of_examples_df["Assessments"].str[-1]
    return cumul_number_of_examples_df

def plot_comul_number_of_examples(cumul_number_of_examples_df, data_statistics_dir):
    plt.figure(figsize=(16,8))
    plt.xticks(cumul_number_of_examples_df["N of Assessments"])
    plt.scatter(cumul_number_of_examples_df["N of Assessments"], cumul_number_of_examples_df["Respondents"])
    # Add vertical lines for each point
    for i in range(0, len(cumul_number_of_examples_df)):
        plt.axvline(x=cumul_number_of_examples_df["N of Assessments"][i], color='gray', linestyle='--')
    plt.xlabel("Number of Assessments")
    plt.ylabel("Number of Respondents")
    plt.title("Cumulative Number of Respondents with Complete Data")
    plt.savefig(data_statistics_dir+'figures/cumul_assessment_distrib.png', dpi=600)  

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
        "Diagnosis_ClinicianConsensus,DX_10"] + [x for x in full_wo_underscore.columns if x.endswith(tuple(["_ByHx", "_Confirmed", "_Presum", "_RC"]))]
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
    og_diag_cols = [x for x in data_up_to_dropped.columns if "DX_" in x and not x.endswith(tuple(["_ByHx", "_Confirmed", "_Presum", "_RC"]))]

    # Get list of diagnoses
    diags = []
    for col in og_diag_cols:
        diags.extend(list(data_up_to_dropped[col].value_counts().index))
    diags = list(set(diags))
    diags.remove(' ')

    # For each diag, only keep the value if the corresponding _ByHx column is 0, and either the _Confirmed, _Presum, or _RC column is 1, otherwise set to NaN, always keep No Diagnosis Given
    for og_diag_col in og_diag_cols:
        col_root = og_diag_col.split("_ByHx")[0]
        byhx_col = col_root+"_ByHx"
        confirmed_col = col_root+"_Confirmed"
        presum_col = col_root+"_Presum"
        rc_col = col_root+"_RC"
        data_up_to_dropped[og_diag_col] = np.where((data_up_to_dropped[col_root] == "No Diagnosis Given") | (data_up_to_dropped[byhx_col].astype(float) == 0) & ((data_up_to_dropped[confirmed_col].astype(float) == 1) | (data_up_to_dropped[presum_col].astype(float) == 1) | (data_up_to_dropped[rc_col].astype(float) == 1)), data_up_to_dropped[og_diag_col], np.nan)

    # Make new columns
    for diag in diags:
        data_up_to_dropped["Diag." + util.remove_chars_forbidden_in_file_names(diag)] = (data_up_to_dropped[og_diag_cols] == diag).any(axis=1)
        
    # Drop original diag columns
    data_up_to_dropped = data_up_to_dropped.drop(og_diag_cols + [x for x in data_up_to_dropped.columns if x.endswith(tuple(["_ByHx", "_Confirmed", "_Presum", "_RC"]))], axis=1)

    return data_up_to_dropped

def transform_devhx_eduhx_cols(data_up_to_dropped):

    list_of_preg_symp_cols = [x for x in data_up_to_dropped.columns if "preg_symp" in x]
    
    if list_of_preg_symp_cols:
        # If any of the preg_symp columns are 1, then the preg_symp column is 1, otherwise 0
        data_up_to_dropped["preg_symp"] = (data_up_to_dropped[list_of_preg_symp_cols].astype(str) == "1").any(axis=1)

        print(data_up_to_dropped[[x for x in data_up_to_dropped.columns if "preg_symp" in x]])

        # Drop original preg_symp columns
        data_up_to_dropped = data_up_to_dropped.drop(list_of_preg_symp_cols, axis=1) 

    if "PreInt_EduHx,NeuroPsych" in data_up_to_dropped.columns:
        data_up_to_dropped = data_up_to_dropped.drop(["PreInt_EduHx,NeuroPsych", "PreInt_EduHx,IEP", "PreInt_EduHx,learning_disability", "PreInt_EduHx,EI", "PreInt_EduHx,CPSE"], axis=1)

    return data_up_to_dropped

def get_t_score_otherwise_raw(cols):
    scales = set([x.strip("_T") for x in cols])
    scales_plus_t = [x+"_T" for x in scales]

    result = []
    for scale in scales:
        if scale+"_T" in cols:
            result.append(scale+"_T")
        else:
            result.append(scale)

    return result

def separate_item_lvl_from_scale_scores(data_up_to_dropped, clinical_config):

    all_cols = data_up_to_dropped.columns

    total_score_cols = ["SCQ,SCQ_Total", 
                        "Barratt,Barratt_Total", 
                        "ASSQ,ASSQ_Total",
                        "ARI_P,ARI_P_Total_Score", 
                        "ARI_S,ARI_S_Total_Score", 
                        "SWAN,SWAN_Total",
                        "SRS,SRS_Total", 
                        "SRS,SRS_Total_T", 
                        "CBCL,CBCL_Total",
                        "CBCL,CBCL_Total_T",
                        "ICU_P,ICU_P_Total",
                        "ICU_SR,ICU_SR_Total",
                        "APQ_P,APQ_P_Total",
                        "APQ_P,APQ_SR_OPD",
                        "PCIAT,PCIAT_Total",
                        "DTS,DTS_Total",
                        "MFQ_P,MFQ_P_Total",
                        "APQ_SR,APQ_SR_Total",
                        "APQ_SR,APQ_SR_OPD",
                        "WHODAS_P,WHODAS_P_Total", 
                        "CIS_P,CIS_P_Score", 
                        "PSI,PSI_Total",
                        "PSI,PSI_Total_T",
                        "RBS,RBS_Total",
                        "SCARED_P,SCARED_P_Total",
                        "SCARED_SR,SCARED_SR_Total",
                        "WHODAS_SR,WHODAS_SR_Score", 
                        "CIS_SR,CIS_SR_Total",
                        # "C3SR,C3SR_Total", # Doesn't have a total
                        # "CCSC,CCSC_Total",  # Doesn't have a total
                        # "CPIC,CPIC_Total", # Doesn't have a total
                        "YSR,YSR_Total",
                        "YSR,YSR_Total_T",
                        "CBCL_Pre,CBCL_Pre_Total",
                        "CBCL_Pre,CBCL_Pre_Total_T",
                        "SRS_Pre,SRS_Pre_Total",
                        "SRS_Pre,SRS_Pre_Total_T",
                        "ASR,ASR_Total",
                        "ASR,ASR_Total_T",
                    ]
    # Get T scores if present, otherwise get raw scores
    total_scores_t_otherwise_raw = get_t_score_otherwise_raw(total_score_cols)

    subscale_score_cols = ["Barratt,Barratt_Total_Edu", "Barratt,Barratt_Total_Occ",
                        "SWAN,SWAN_HY", "SWAN,SWAN_IN",
                        "SRS,SRS_AWR_T", "SRS,SRS_AWR", "SRS,SRS_COG_T", "SRS,SRS_COG", "SRS,SRS_COM_T", "SRS,SRS_COM", "SRS,SRS_DSMRRB_T", "SRS,SRS_DSMRRB", "SRS,SRS_MOT_T", "SRS,SRS_MOT", "SRS,SRS_RRB_T", "SRS,SRS_RRB", "SRS,SRS_SCI_T", "SRS,SRS_SCI",
                        "CBCL,CBCL_AB_T", "CBCL,CBCL_AB", "CBCL,CBCL_AD_T", "CBCL,CBCL_AD", "CBCL,CBCL_AP_T", "CBCL,CBCL_AP", "CBCL,CBCL_Ext_T", "CBCL,CBCL_Ext", "CBCL,CBCL_Int_T", "CBCL,CBCL_Int", "CBCL,CBCL_RBB_T", "CBCL,CBCL_RBB", "CBCL,CBCL_SC_T", "CBCL,CBCL_SC", "CBCL,CBCL_SP_T", "CBCL,CBCL_SP", "CBCL,CBCL_TP_T", "CBCL,CBCL_TP", "CBCL,CBCL_WD", "CBCL,CBCL_WD_T", "CBCL,CBCL_C", "CBCL,CBCL_C_T", "CBCL,CBCL_OP", "CBCL,CBCL_OP_T",
                        "ICU_P,ICU_P_Callous", "ICU_P,ICU_P_Uncaring", "ICU_P,ICU_P_Unemotional",
                        "ICU_SR,ICU_SR_Callous", "ICU_SR,ICU_SR_Uncaring", "ICU_SR,ICU_SR_Unemotional",
                        "PANAS_PositiveAffect", "PANAS_NegativeAffect",
                        "APQ_P,APQ_P_CP", "APQ_P,APQ_P_ID", "APQ_P,APQ_P_INV", "APQ_P,APQ_P_PM", "APQ_P,APQ_P_PP",
                        "DTS,DTS_absorption", "DTS,DTS_appraisal", "DTS,DTS_regulation", "DTS,DTS_tolerance",
                        "APQ_SR,APQ_SR_CP", "APQ_SR,APQ_SR_ID", "APQ_SR,APQ_SR_INV_D", "APQ_SR,APQ_SR_INV_M", "APQ_SR,APQ_SR_PM", "APQ_SR,APQ_SR_PP",
                        "PSI,PSI_DC_T", "PSI,PSI_DC", "PSI,PSI_PCDI_T", "PSI,PSI_PCDI", "PSI,PSI_PD_T", "PSI,PSI_PD",
                        "RBS,RBS_Score_01", "RBS,RBS_Score_02", "RBS,RBS_Score_03", "RBS,RBS_Score_04", "RBS,RBS_Score_05",  
                        "SCARED_P,SCARED_P_GD", "SCARED_P,SCARED_P_PN", "SCARED_P,SCARED_P_SC", "SCARED_P,SCARED_P_SH", "SCARED_P,SCARED_P_SP",
                        "SCARED_SR,SCARED_SR_GD", "SCARED_SR,SCARED_SR_PN", "SCARED_SR,SCARED_SR_SC", "SCARED_SR,SCARED_SR_SH", "SCARED_SR,SCARED_SR_SP",
                        "C3SR,C3SR_AG", "C3SR,C3SR_AG_T", "C3SR,C3SR_FR", "C3SR,C3SR_FR_T", "C3SR,C3SR_HY", "C3SR,C3SR_HY_T", "C3SR,C3SR_IN", "C3SR,C3SR_IN_T", "C3SR,C3SR_LP", "C3SR,C3SR_LP_T", "C3SR,C3SR_NI", "C3SR,C3SR_PI",
                        "CCSC,CCSC_PFC", "CCSC,CCSC_CDM", "CCSC,CCSC_DPS", "CCSC,CCSC_SU", "CCSC,CCSC_AC", "CCSC,CCSC_AA", "CCSC,CCSC_REP", "CCSC,CCSC_WT", "CCSC,CCSC_PCR", "CCSC,CCSC_CON", "CCSC,CCSC_OPT", "CCSC,CCSC_POS", "CCSC,CCSC_REL", "CCSC,CCSC_SS", "CCSC,CCSC_SUPMF", "CCSC,CCSC_SUPOA", "CCSC,CCSC_SUPEER", "CCSC,CCSC_SUPSIB",
                        "CPIC,CPIC_Frequency_Total", "CPIC,CPIC_Intensity_Total", "CPIC,CPIC_Resolution_Total", "CPIC,CPIC_Content_Total", "CPIC,CPIC_Perceived_Threat_Total", "CPIC,CPIC_Self_Blame_Total", "CPIC,CPIC_Triangulation_Total", "CPIC,CPIC_Stability_Total",
                        "YSR,YSR_AB", "YSR,YSR_AB_T", "YSR,YSR_AD", "YSR,YSR_AD_T", "YSR,YSR_AP", "YSR,YSR_AP_T", "YSR,YSR_WD", "YSR,YSR_WD_T", "YSR,YSR_RBB", "YSR,YSR_RBB_T", "YSR,YSR_SC", "YSR,YSR_SC_T", "YSR,YSR_SP", "YSR,YSR_SP_T", "YSR,YSR_TP", "YSR,YSR_TP_T", "YSR,YSR_Ext", "YSR,YSR_Ext_T", "YSR,YSR_Int", "YSR,YSR_Int_T", "YSR,YSR_OP", "YSR,YSR_C", "YSR,YSR_Total", "YSR,YSR_Total_T",
                        "CBCL_Pre,CBCL_Pre_AB", "CBCL_Pre,CBCL_Pre_AB_T", "CBCL_Pre,CBCL_Pre_AD", "CBCL_Pre,CBCL_Pre_AD_T", "CBCL_Pre,CBCL_Pre_AP", "CBCL_Pre,CBCL_Pre_AP_T", "CBCL_Pre,CBCL_Pre_SC", "CBCL_Pre,CBCL_Pre_SC_T", "CBCL_Pre,CBCL_Pre_SP", "CBCL_Pre,CBCL_Pre_SP_T", "CBCL_Pre,CBCL_Pre_WD", "CBCL_Pre,CBCL_Pre_WD_T", "CBCL_Pre,CBCL_Pre_Ext", "CBCL_Pre,CBCL_Pre_Ext_T", "CBCL_Pre,CBCL_Pre_Int", "CBCL_Pre,CBCL_Pre_Int_T", "CBCL_Pre,CBCL_Pre_DSM_ADHP", "CBCL_Pre,CBCL_Pre_DSM_ADHP_T", "CBCL_Pre,CBCL_Pre_DSM_AnxP", "CBCL_Pre,CBCL_Pre_DSM_AnxP_T", "CBCL_Pre,CBCL_Pre_DSM_AP", "CBCL_Pre,CBCL_Pre_DSM_AP_T", "CBCL_Pre,CBCL_Pre_DSM_ODP", "CBCL_Pre,CBCL_Pre_DSM_ODP_T", "CBCL_Pre,CBCL_Pre_DSM_PDP", "CBCL_Pre,CBCL_Pre_DSM_PDP_T", "CBCL_Pre,CBCL_Pre_OP", "CBCL_Pre,CBCL_Pre_OP_T", "CBCL_Pre,CBCL_Pre_Total", "CBCL_Pre,CBCL_Pre_Total_T",
                        "SRS_Pre,SRS_Pre_AWR_T", "SRS_Pre,SRS_Pre_AWR", "SRS_Pre,SRS_Pre_COG_T", "SRS_Pre,SRS_Pre_COG", "SRS_Pre,SRS_Pre_COM_T", "SRS_Pre,SRS_Pre_COM", "SRS_Pre,SRS_Pre_DSMRRB_T", "SRS_Pre,SRS_Pre_DSMRRB", "SRS_Pre,SRS_Pre_MOT_T", "SRS_Pre,SRS_Pre_MOT", "SRS_Pre,SRS_Pre_RRB_T", "SRS_Pre,SRS_Pre_RRB", "SRS_Pre,SRS_Pre_SCI_T", "SRS,SRS_Pre_SCI",
                        "ASR,ASR_AD", "ASR,ASR_AD_T", "ASR,ASR_WD", "ASR,ASR_WD_T", "ASR,ASR_SC", "ASR,ASR_SC_T", "ASR,ASR_TP", "ASR,ASR_TP_T", "ASR,ASR_AP", "ASR,ASR_AP_T", "ASR,ASR_RBB", "ASR,ASR_RBB_T", "ASR,ASR_AB", "ASR,ASR_AB_T", "ASR,ASR_OP", "ASR,ASR_Int", "ASR,ASR_Int_T", "ASR,ASR_Ext", "ASR,ASR_Ext_T", "ASR,ASR_Intrusive", "ASR,ASR_Intrusive_T", "ASR,ASR_C", 
                        "SDQ,Conduct_Problems_Total", "SDQ,Difficulties_Total", "SDQ,Emotional_Problems_Total", "SDQ,Externalising_Total", "SDQ,Generating_Impact_Total", "SDQ,Hyperactivity_Total", "SDQ,Internalising_Total", "SDQ,Peer_Problems_Total", "SDQ,Prosocial_Total",
                        ]
    subscale_scores_t_otherwise_raw = get_t_score_otherwise_raw(subscale_score_cols)

    cog_task_cols = get_cog_task_cols(data_up_to_dropped, clinical_config)

    diag_cols = [x for x in all_cols if x.startswith("Diag.")]

    # Item level columns = all columns except those of total, subscale scores, and cog task cols (includes diag cols for output)
    item_level_cols = [x for x in all_cols if (x not in total_score_cols) and (x not in subscale_score_cols) and (x not in cog_task_cols)]
    data_up_to_dropped_item_lvl = data_up_to_dropped[item_level_cols]

    # Total columns 
    total_scores_present = [x for x in total_scores_t_otherwise_raw+diag_cols if x in data_up_to_dropped.columns]
    data_up_to_dropped_total_scores = data_up_to_dropped[total_scores_present + ["ID"]]

    # Subscale columns
    subscale_scores_present = [x for x in subscale_scores_t_otherwise_raw+diag_cols if x in data_up_to_dropped.columns]
    data_up_to_dropped_subscale_scores = data_up_to_dropped[subscale_scores_present + ["ID"]]

    # Cog task columns
    cog_tasks_present = [x for x in cog_task_cols+diag_cols if x in data_up_to_dropped.columns]
    data_up_to_dropped_cog_task_scores = data_up_to_dropped[cog_tasks_present + ["ID"]]

    # Set ID as index in all dataframes
    data_up_to_dropped_item_lvl = data_up_to_dropped_item_lvl.set_index("ID")
    data_up_to_dropped_total_scores = data_up_to_dropped_total_scores.set_index("ID")
    data_up_to_dropped_subscale_scores = data_up_to_dropped_subscale_scores.set_index("ID")
    data_up_to_dropped_cog_task_scores = data_up_to_dropped_cog_task_scores.set_index("ID")

    return data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores

def remove_irrelavent_missing_markers(data_up_to_dropped, data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores):
    was_missing_cols = [x for x in data_up_to_dropped.columns if "_WAS_MISSING" in x]
    was_missing_col_originals = [x.split("_WAS_MISSING")[0] for x in was_missing_cols]

    for col in was_missing_col_originals:
        if col not in data_up_to_dropped_item_lvl.columns and col +"_WAS_MISSING" in data_up_to_dropped_item_lvl.columns:
            data_up_to_dropped_item_lvl = data_up_to_dropped_item_lvl.drop(col+"_WAS_MISSING", axis=1)

        if col not in data_up_to_dropped_total_scores.columns and col +"_WAS_MISSING" in data_up_to_dropped_total_scores.columns:
            data_up_to_dropped_total_scores = data_up_to_dropped_total_scores.drop(col+"_WAS_MISSING", axis=1)
        
        if col not in data_up_to_dropped_subscale_scores.columns and col +"_WAS_MISSING" in data_up_to_dropped_subscale_scores.columns:
            data_up_to_dropped_subscale_scores = data_up_to_dropped_subscale_scores.drop(col+"_WAS_MISSING", axis=1)
        
        if col in data_up_to_dropped_cog_task_scores.columns and col +"_WAS_MISSING" in data_up_to_dropped_cog_task_scores.columns:
            data_up_to_dropped_cog_task_scores = data_up_to_dropped_cog_task_scores.drop(col+"_WAS_MISSING", axis=1)    


    return data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores

def export_datasets(data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores, data_output_dir):
    data_up_to_dropped_item_lvl.to_csv(data_output_dir + "item_lvl.csv")
    data_up_to_dropped_subscale_scores.to_csv(data_output_dir + "subscale_scores.csv")
    data_up_to_dropped_total_scores.to_csv(data_output_dir + "total_scores.csv")
    data_up_to_dropped_cog_task_scores.to_csv(data_output_dir + "cog_tasks.csv")

def get_cog_task_cols(data, clinical_config):
    cog_task_cols = [x for x in data.columns if x.startswith(tuple(clinical_config["cog batteries"]))]
    return cog_task_cols

def generate_assessment_reports(full_wo_underscore, EID_columns_by_popularity, assessment_answer_counts, dir):
    
    # Remove EID from assessment names
    assessment_answer_counts.index = [x.split(",")[0] for x in assessment_answer_counts.index]
    assessment_answer_counts.to_csv(dir + "assessment-filled-distrib.csv")

    relevant_assessments_by_popularity = [x.split(",")[0] for x in EID_columns_by_popularity]
    assessment_answer_counts.loc[relevant_assessments_by_popularity].to_csv(dir + "relevant-assessment-filled-distrib.csv")

    # Get cumulative distribution of assessments: number of people who took all top 1, top 2, top 3, etc. popular assessments 
    cumul_number_of_examples_df = get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity)
    cumul_number_of_examples_df.to_csv(dir + "assessment-filled-distrib-cumul.csv", float_format='%.3f')

    # Plot cumulative distribution of assessments
    plot_comul_number_of_examples(cumul_number_of_examples_df, dir)

def make_full_dataset(only_assessment_distribution, first_assessment_to_drop, only_free_assessments, dirs, learning):

    clinical_config = util.read_config("clinical", learning)

    relevant_assessments_list = clinical_config["relevant assessments"]
    proprietary_assessments = clinical_config["proprietary assessments"]
    res_only_assessments = clinical_config["research only assessments"]
    report_assessments = clinical_config["report assessments"]

    if clinical_config["use only research only assessments"] == 1:
        #relevant_assessments_list = [x for x in relevant_assessments_list if x in res_only_assessments]
        relevant_assessments_list = [x for x in relevant_assessments_list if x not in report_assessments]

    if only_free_assessments == 1:
        relevant_assessments_list = remove_proprietary_assessments(relevant_assessments_list, proprietary_assessments)

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
    print("EID_cols", EID_cols)

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
    # Get relevant ID columns sorted by popularity
    EID_columns_by_popularity = get_relevant_id_cols_by_popularity(assessment_answer_counts, relevant_assessments_list)    

    generate_assessment_reports(full_wo_underscore, EID_columns_by_popularity, assessment_answer_counts, dir = dirs["data_statistics_dir"])

    if only_assessment_distribution != 1:
    
        # List of most popular assessments until the first one from the drop list 
        EID_columns_until_dropped = [x for x in EID_columns_by_popularity[:EID_columns_by_popularity.index(first_assessment_to_drop+",EID")]]

        # Get data up to the dropped assessment
        # Get only people who took the most popular assessments until the first one from the drop list 
        columns_until_dropped = get_columns_until_dropped(full_wo_underscore, EID_columns_until_dropped)
        data_up_to_dropped = get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped, columns_until_dropped)

        # Remove EID columns: not needed anymore
        data_up_to_dropped = data_up_to_dropped.drop(EID_columns_until_dropped, axis=1)

        # Aggregare demographics input columns: remove PER parent data from Barratt, only keep aggregated scores
        if "Barratt,Barratt_P1_Edu" in data_up_to_dropped.columns:
            data_up_to_dropped = data_up_to_dropped.drop(["Barratt,Barratt_P1_Edu", "Barratt,Barratt_P1_Occ", "Barratt,Barratt_P2_Edu", "Barratt,Barratt_P2_Occ"], axis=1)

        # Transform PreInt_DevHx columns
        data_up_to_dropped = transform_devhx_eduhx_cols(data_up_to_dropped)

        # Convert numeric columns to numeric type (all except ID and DX)
        data_up_to_dropped = data_up_to_dropped.apply(lambda col: convert_numeric_col_to_numeric_type(col))

        # Save report of missing values
        missing_values_df = get_missing_values_df(data_up_to_dropped)
        missing_values_df.to_csv(dirs["data_statistics_dir"] + "missing-values-report.csv", float_format='%.3f')

        # Remove columns with more than 40% missing data
        data_up_to_dropped = remove_cols_w_missing_over_n(data_up_to_dropped, 40, missing_values_df)

        # Special case: replace missing "CBCL,CBCL_56H" with 0 ("Other")
        if "CBCL,CBCL_56H" in data_up_to_dropped.columns:
            data_up_to_dropped[["CBCL,CBCL_56H"]] = data_up_to_dropped[["CBCL,CBCL_56H"]].fillna(value=0)

        # Add missingness marker for columns with more than 5% missing data 
        data_up_to_dropped = add_missingness_markers(data_up_to_dropped, 5, missing_values_df)

        # Transform diagnosis columns
        data_up_to_dropped = transform_dx_cols(data_up_to_dropped)

        # Convert new boolean columns to numeric
        data_up_to_dropped = data_up_to_dropped.replace({True: 1, False: 0})

        # Separate subscale and total scores
        data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores = separate_item_lvl_from_scale_scores(data_up_to_dropped, clinical_config)

        # Remove _WAS_MISSING columns that are not linked to any columns from each dataset
        data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores = remove_irrelavent_missing_markers(data_up_to_dropped, 
            data_up_to_dropped_item_lvl, 
            data_up_to_dropped_total_scores, 
            data_up_to_dropped_subscale_scores,
            data_up_to_dropped_cog_task_scores)

        # Export final datasets
        export_datasets(data_up_to_dropped_item_lvl, data_up_to_dropped_total_scores, data_up_to_dropped_subscale_scores, data_up_to_dropped_cog_task_scores, dirs["data_output_dir"])