# Notes:
# Data: http://data.healthybrainnetwork.org/ (LORIS, all_with_new_diag_and_nih query)
# Diagnosis is contained in the "Diagnosis_ClinicianConsensus" column (not in "ConsensusDx")

import pandas as pd
import numpy as np
from collections import Counter
from re import M
import os
import matplotlib.pyplot as plt
import sys

def create_repositories():
    data_statistics_dir = "reports/"
    if not os.path.exists(data_statistics_dir):
        os.mkdir(data_statistics_dir)

    data_output_dir = "data/processed/"
    if not os.path.exists(data_output_dir):
        os.mkdir(data_output_dir)
        
    pd.set_option("display.max_columns", None)

    return data_statistics_dir, data_output_dir

def remove_irrelevant_nih_cols(full):
    NIH_cols = [x for x in full.columns if "NIH" in x]
    NIH_scores_cols = [x for x in NIH_cols if x.startswith("NIH_Scores,")]

    # Drop percentile scores, only keep actual score
    NIH_cols_to_drop = [x for x in NIH_scores_cols if x.endswith("_P")]
    full = full.drop(NIH_cols_to_drop, axis = 1)

    # Drop non-numeric columns
    full = full.drop(["NIH_Scores,NIH7_Incomplete_Reason"], axis = 1)

    return full
    
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
    relevant_EID_list = [x+",EID" for x in ["Basic_Demos", "WIAT", "NIH_Scores", "SympChck", "SCQ", "Barratt", 
        "ASSQ", "ARI_P", "SDQ", "SWAN", "SRS", "CBCL", "ICU_P", "APQ_P", "PCIAT", "DTS", "ESWAN", "MFQ_P", "APQ_SR", 
        "WISC", "WHODAS_P", "CIS_P", "PSI", "RBS", "PhenX_Neighborhood", "WHODAS_SR", "CIS_SR", "SCARED_SR", 
        "C3SR", "CCSC", "CPIC", "YSR", "PhenX_SchoolRisk", "CBCL_Pre", "SRS_Pre", "ASR"]]

    # Get relevant ID columns sorted by popularity    
    EID_columns_by_popularity = [x for x in EID_columns_by_popularity if x in relevant_EID_list]

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

def get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped):
    columns_until_dropped = []
    assessments_until_dropped = [x.split(",")[0]+"," for x in EID_columns_until_dropped]
    for assessment in assessments_until_dropped:
        columns = [column for column in full_wo_underscore.columns if column.startswith(assessment)]
        columns_until_dropped.extend(columns)
        
    diag_colunms = ["Diagnosis_ClinicianConsensus,DX_01", "Diagnosis_ClinicianConsensus,DX_02", "Diagnosis_ClinicianConsensus,DX_03", 
        "Diagnosis_ClinicianConsensus,DX_04", "Diagnosis_ClinicianConsensus,DX_05", "Diagnosis_ClinicianConsensus,DX_06", 
        "Diagnosis_ClinicianConsensus,DX_07", "Diagnosis_ClinicianConsensus,DX_08", "Diagnosis_ClinicianConsensus,DX_09", 
        "Diagnosis_ClinicianConsensus,DX_10"]
    data_up_to_dropped = full_wo_underscore.loc[full_wo_underscore[EID_columns_until_dropped].dropna(how="any").index][columns_until_dropped+["ID"]+diag_colunms]

    return data_up_to_dropped

def remove_irrelevant_output_cols(data_up_to_dropped):
    WIAT_cols_to_keep = ["WIAT,WIAT_Word_Stnd", "WIAT,WIAT_Num_Stnd"]
    WIAT_cols_to_drop = [x for x in data_up_to_dropped.columns if "WIAT" in x and x not in WIAT_cols_to_keep] 
    data_up_to_dropped = data_up_to_dropped.drop(WIAT_cols_to_drop, axis=1)

    WISC_cols_to_keep = ["WISC,WISC_Coding_Scaled", "WISC,WISC_SS_Scaled", "WISC,WISC_FSIQ"]
    WISC_cols_to_drop = [x for x in data_up_to_dropped.columns if "WISC" in x and x not in WISC_cols_to_keep] 
    data_up_to_dropped = data_up_to_dropped.drop(WISC_cols_to_drop, axis=1)

def convert_numeric_col_to_numeric_type(col):
    if col.name != "ID" and "Diagnosis_ClinicianConsensus" not in col.name:
        return pd.to_numeric(col)
    else:
        return col

def main(first_assessment_to_drop):
    data_statistics_dir, data_output_dir = create_repositories()

    # LORIS saved query (all data)
    full = pd.read_csv("data/input/LORIS-release-10.csv", dtype=object)
    
    # Replace NaN (currently ".") values with np.nan
    full = full.replace(".", np.nan)

    # Drop first row (doesn't have ID)
    full = full.iloc[1: , :]

    # Drop empty columns
    full = full.dropna(how='all', axis=1)

    # Remove irrelevant NIH toolbox columns
    full = remove_irrelevant_nih_cols(full)

    full = remove_admin_cols(full)

    ## Fill missing EIDs with EIDs from other questionnaires 
    full_for_EID_check = full_for_EID_check.ffill(axis=1).bfill(axis=1)

    # Remove lines with different EID within one row
    full = full[full_for_EID_check.eq(full_for_EID_check.iloc[:, 0], axis=0).all(1)]

    # Get ID columns (contain quetsionnaire names, e.g. 'ACE,EID', will be used to check if an assessment is filled)
    EID_cols = [x for x in full.columns if ",EID" in x]

    # Fill ID field with the first non-null questionnaire-specific EID
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
    assessment_answer_counts.to_csv(data_output_dir + "assessment-filled-distrib.csv")

    # Get relevant ID columns sorted by popularity
    EID_columns_by_popularity = get_relevant_id_cols_by_popularity(assessment_answer_counts)    

    # Get cumulative distribution of assessments: number of people who took all top 1, top 2, top 3, etc. popular assessments 
    cumul_number_of_examples_df = get_cumul_number_of_examples_df(full_wo_underscore, EID_columns_by_popularity)
    cumul_number_of_examples_df.to_csv(data_statistics_dir + "assessment-filled-distrib-cumul.csv")

    # Plot cumulative distribution of assessments
    plot_comul_number_of_examples(cumul_number_of_examples_df, data_statistics_dir)

    ### The first drop-off in number of respondents is at SCARED_SR, the first assessment with an age restriction. Second drop off is at CPIC.
    
    # List of most popular assessments until the first one from the drop list 
    EID_columns_until_dropped = [x for x in EID_columns_by_popularity[:EID_columns_by_popularity.index(first_assessment_to_drop)]]

    # Get data up to the dropped assessment
    # Get only people who took the most popular assessments until the first one from the drop list (CPIC)
    data_up_to_dropped = get_data_up_to_dropped(full_wo_underscore, EID_columns_until_dropped)

    # Remove EID columns: not needed anymore
    data_up_to_dropped = data_up_to_dropped.drop(EID_columns_until_dropped, axis=1)

    # Remove non-used output columns (WISC and WIAT)
    data_up_to_dropped = remove_irrelevant_output_cols(data_up_to_dropped)

    # Aggregare demographics input columns: remove per parent data from Barratt
    data_up_to_dropped = data_up_to_dropped.drop(["Barratt,Barratt_P1_Edu", "Barratt,Barratt_P1_Occ", "Barratt,Barratt_P2_Edu", "Barratt,Barratt_P2_Occ"], axis=1)

    # Convert numeric columns to numeric type (all except ID and DX)
    data_up_to_dropped = data_up_to_dropped.apply(lambda col: convert_numeric_col_to_numeric_type(col))

if __name__ == "__main__":
    main(sys.argv[1])