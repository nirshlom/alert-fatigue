import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np


# Load the data: df_main_active_adult.csv this order level data only active adults
# This is the data to work when analysing the active adult orders
df_active_adult = pd.read_csv('alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv')
print(f'original df_active_adult has {df_active_adult.shape[0]} rows and {df_active_adult.shape[1]} columns')
# original df_active_adult has 2,543,301 rows and 66 columns

#test for duplicated ids:
df_active_adult['id1'].duplicated().sum() > 0

df_active_adult.columns = map(str.lower, df_active_adult.columns)

df_active_adult.shape
#(2543301, 66) R version, py version: (2595123, 165)

# Grouping and summarizing the data: this data in patient level
src_tbl1_active_by_patient_gb = (
    df_active_adult
    .groupby(["id1", "age_num", "age_cat", "gender_text_en_cat"])
    .agg(
        hospitalname_en_cat_cnt=pd.NamedAgg(column="hospitalname_en_cat", aggfunc=pd.Series.nunique),  # Count distinct
        survivalrate10years_age_adj_mean=pd.NamedAgg(column="survivalrate10years_age_adj", aggfunc="mean"),  # Mean
        medical_record_cat_cnt=pd.NamedAgg(column="medical_record", aggfunc=pd.Series.nunique),  # Count distinct
        nummedamount_calc_mean=pd.NamedAgg(column="nummedamount_calc", aggfunc="mean"),  # Mean
        hosp_days_mean=pd.NamedAgg(column="hosp_days", aggfunc="mean"),  # Mean
        chronic_num_calc_mean=pd.NamedAgg(column="chronic_num_calc", aggfunc="mean")  # Mean
    )
    .reset_index()
)

src_tbl1_active_by_patient_gb.shape
print(f'src_tbl1_active_by_patient_gb has {src_tbl1_active_by_patient_gb.shape[0]} unique patients'
      f' and {src_tbl1_active_by_patient_gb.shape[1]} variables')
#src_tbl1_active_by_patient_gb has 155899 unique patients and 10 variables

src_tbl1_active_by_patient_gb_to_merge = src_tbl1_active_by_patient_gb[[
    'id1' ,'hospitalname_en_cat_cnt', 'survivalrate10years_age_adj_mean',
    'medical_record_cat_cnt', 'nummedamount_calc_mean',
    'hosp_days_mean', 'chronic_num_calc_mean'
]]

src_active_patients_merged = pd.merge(
    src_tbl1_active_by_patient_gb_to_merge,
    df_active_adult,
    on=["id1"], how="left")

src_active_patients_merged.shape
print(f'src_active_patients_merged has {src_active_patients_merged.shape[0]} orders '
      f'and {src_active_patients_merged.shape[1]} columns')

src_active_patients_merged.head()

# src_tbl1_active_by_patient_gb.to_csv('alert_analysis/data/src_tbl1_active_by_patient_gb.csv', index=False)
# src_active_patients_merged.to_csv('alert_analysis/data/src_active_patients_merged.csv', index=False)

#TODO: create profile report for df_active_adult
profile = ProfileReport(df_active_adult, title="Data Profiling Report", explorative=True)
profile.to_file("df_active_adult_profiling.html") # open "1_data_profiling.html" file if you can't see the iframe
profile.to_notebook_iframe()


#TODO: add the following to the data_handler.py
src_active_patients_merged = pd.read_csv('alert_analysis/data/src_active_patients_merged.csv')
src_active_patients_merged.shape

##TODO: create profile report for src_tbl1_active_by_patient_gb
profile = ProfileReport(src_active_patients_merged, title="Data Profiling Report", explorative=True)
profile.to_file("src_active_patients_merged_profiling.html")
profile.to_notebook_iframe()

#TODO: create the profile report seperately for the following columns:
category_col = "gender_text_en_cat"

# Loop through each unique category and generate a report
for category in src_active_patients_merged[category_col].unique():
    subset_df = src_active_patients_merged[src_tbl1_active_by_patient_gb[category_col] == category]  # Filter dataset for the category

    # Generate Pandas Profiling report for this subset
    profile = ProfileReport(subset_df, title=f"Pandas Profiling Report - {category}")

    # Save the report to an HTML file
    profile.to_file(f"profile_report_{category}.html")

print("Reports generated successfully!")
#  I would like to....


src_active_patients_merged['atc_group'].value_counts()

src_active_patients_merged = pd.read_csv('alert_analysis/data/src_active_patients_merged.csv')

#TODO: add the columns: Module_Alert_Rn, Alert_Message, DiagnosisInReception,
# HospDiagnosis, Other_Text, Response, Answer_Text, hosp_days, num_of_alerts_per_order_id - ###### Done ######

# calling the raw data:
data_main = pd.read_csv('alert_analysis/data/main_data_2022/emek.data - Copy.csv')
data_main.shape
data_main.columns

#TODO: cretae a new flat column below/ exceeds dose, the full logic is in Hiba_project_fatigue_alert.rmd
# add Module_Alert_Rn and Alert_Message ###### Done ######
#add the logic below
# ifelse(data_distincted_active_mode_stoping$Module_Alert_Rn == "DRC - Frequency 1" & grepl("exceeds",
#                                                                                           data_distincted_active_mode_stoping$Alert_Message), "DRC - Frequency - exceeds ",
#
# ifelse(data_distincted_active_mode_stoping$Module_Alert_Rn == "DRC - Frequency 1" & grepl("below",
#                                                                                           data_distincted_active_mode_stoping$Alert_Message), "DRC - Frequency - below",

# Apply the same logic to "DRC - Single Dose 1"
# Apply the same logic to "DRC - Max Daily Dose 1"

#This is the main outcome to analyse
df_active_adult['Alert_type'].value_counts()





#TODO: April 27
# 1. add to patient level data: number of diagnoses (diagnosisinreception, hospdiagnosis), add this to tableone analysis # TODO: Done
# 2. add unitname_cat to patient level data and tableone # TODO: Done   
# from the main data, remove PARAMEDICAL (column name = sectortext_en_cat) # TODO: Done
# create the alert table according to year-month # TODO: not done
# create pie chart- figure 1 (distribution of alert types)  - I should create a new column that combines all the features listed in the doc
# figure 3, answer_text_en
# figure 8, specified in the doc
# create table 8: general statistics of alert # TODO: nit Done



# Define categorical and continuous variables
categorical_vars = [
    "DRC_Single_Dose_1",
    "DRC_Frequency_1",
    "DRC_Max_Daily_Dose_1",
    "Renal_alerts_CAT",
    "DDI_Contraindicated_Drug_Combination_CAT",
    "DDI_Severe_Interaction_CAT",
    "DDI_Moderate_Interaction_CAT",
    "DAM_CAT",
    "Technical_alerts_CAT"
]

# continuous_vars = [
#     "AGE_num",
#     "survivalrate10years_age_adj_mean",
#     "NumMedAmount_calc_mean",
#     "hosp_days_mean",
#     "chronic_num_calc_mean",
#     "Medical_Record_cnt"
# ]

# Generate summary table
alert_table = TableOne(
    df_main_active_adult,
    categorical=categorical_vars,
    groupby="date_time_prescribe",
    #continuous=continuous_vars,
    pval=True,  # Add p-values
    missing=True  # Show missing values
)

# Print the table
print(alert_table.tabulate(tablefmt="pipe"))
