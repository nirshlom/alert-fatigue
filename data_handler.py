import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib as plt
import numpy as np

# Load the data: df_main_flat.csv(this is order based data) and df_main_active_adult.csv
df = pd.read_csv('alert_analysis/data/main_data_2022/df_main_flat.csv')
df.columns = map(str.lower, df.columns)
print(f'original df has {df.shape[0]} rows and {df.shape[1]} columns')
# original df has 3615984 rows and 66 columns

# Load the data: df_main_active_adult.csv this order level data only active adults
# This is the data to work when analysing the active adult orders
df_active_adult = pd.read_csv('alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv')
print(f'original df_active_adult has {df_active_adult.shape[0]} rows and {df_active_adult.shape[1]} columns')
# original df_active_adult has 2,543,301 rows and 66 columns

#test for duplicated ids:
df_active_adult['id1'].duplicated().sum() > 0
df_active_adult['Order_ID_new_update'].duplicated().sum() > 0

df_active_adult.columns = map(str.lower, df_active_adult.columns)

# df_active_adult = df_active_adult[
#     (df_active_adult['severityleveltostoporder_cat'] != "Silence Mode") &
#     (df_active_adult['adult_child_cat'] == "adult") &
#     (~df_active_adult['hospital_cat'].isin(["243", "113", "29"])) &
#     (~df_active_adult['unitname_cat'].isin(["Day_care", "ICU", "Pediatric", "Rehabilitation"]))
# ]

df_active_adult.shape
#(2543301, 66)

# Grouping and summarizing the data: this data in patient level
src_tbl1_active_by_patient_gb = (
    df_active_adult
    .groupby(["id1", "age_num", "age_cat", "gender_text_en_cat"])
    .agg(
        hospitalname_en_cat_cnt=pd.NamedAgg(column="hospitalname_en_cat", aggfunc=pd.Series.nunique),  # Count distinct
        survivalrate10years_age_adj_mean=pd.NamedAgg(column="survivalrate10years_age_adj", aggfunc="mean"),  # Mean
        medical_record_cat_cnt=pd.NamedAgg(column="medical_record_cat", aggfunc=pd.Series.nunique),  # Count distinct
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
select_cols = [
        "Hospital_cat",
        "HospitalName_cat",
        "HospitalName_EN_cat",
        "UnitName_cat",
        "SeverityLevelToStopOrder_cat",
        "Medical_Record",
        "Admission_No",
        "AGE_num",
        "Gender_Text_cat",
        "Gender_Text_EN_cat",
        "SectorText_cat",
        "SectorText_EN_cat",
        "Order_ID_new_update",
        "Time_Prescribing_Order",  # additional column not used in grouping
        "Field",
        "IS_New_Order",
        "Drug_Header_cat",
        "Module_Alert_Rn",
        "Alert_Message",
        "Alert_Severity",
        "OrderOrigin",
        "ShiftType_cat",
        "DayHebrew_cat",
        "DayEN_cat",
        "HospAmount_new",
        "NumMedAmount",
        "Other_Text",  # additional column
        "ResponseType_cat",  # additional column
        "Details",
        "ATC",
        "ATC_cln",
        "Basic_Name",
        "DiagnosisInReception",
        "HospDiagnosis",
        "id1",
        "id2",
        "Alert_Rn_Severity_cat",
        "Response",  # additional column
        "Answer_Text_EN",  # additional column
        "Module_Severity_Rn",
        "Time_Mabat_Request_convert_ms_res",  # additional column
        "Time_Mabat_Response_convert_ms_res",  # additional column
        "diff_time_mabat_ms_y",  # additional column
        "DRC_SUB_GROUP",  # additional column
        "NeoDRC_SUB_GROUP"  # additional column
    ]
profile = ProfileReport(df_active_adult[select_cols], title="Data Profiling Report", explorative=False)
profile.to_file("df_active_adult_profiling_ud.html") # open "1_data_profiling.html" file if you can't see the iframe
profile.to_notebook_iframe()

test_df = df_active_adult.loc[(df_active_adult['ResponseType_cat'] == 'Ignore') & (df_active_adult['diff_time_mabat_ms_y'] <= 2500)][['Order_ID_new_update', 'diff_time_mabat_ms_y']]
df_active_adult.loc[(df_active_adult['ResponseType_cat'] == 'Ignore') & df_active_adult['diff_time_mabat_ms_y'].isnull()]

test_df = df_active_adult.loc[df_active_adult['diff_time_mabat_ms_y'] < 20000]

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df' and the column you're interested in is 'AGE_num'
sns.histplot(test_df['diff_time_mabat_ms_y'].dropna(), bins=60, kde=False)  # drop NaN values and plot the histogram
plt.xlabel('Time Diff MS')
plt.ylabel('Frequency')
plt.title('Histogram of diff_time_mabat_ms_y')
plt.show()


plot_histogram(test_df, 'diff_time_mabat_ms_y', title='diff time response')

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
# HospDiagnosis, Other_Text, Response, Answer_Text, hosp_days, num_of_alerts_per_order_id

# calling the raw data:
data_main = pd.read_csv('alert_analysis/data/main_data_2022/emek.data - Copy.csv')
data_main.shape
data_main.columns

#TODO: cretae a new flat column below/ exceeds dose, the full logic is in Hiba_project_fatigue_alert.rmd
# add Module_Alert_Rn and Alert_Message
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




