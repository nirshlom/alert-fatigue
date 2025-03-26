import pandas as pd
data_main = pd.read_csv('alert_analysis/data_process/data_main_prep.csv')

# -------------------------------
# Step 1: Grouping and deduplication
# -------------------------------
# Define the columns to group by (using the same columns as in the R group_by)
group_cols = [
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
    "Order_ID_new",
    #"Time_Prescribing_Order",
    "Field",
    "IS_New_Order",
    "Drug_Header_cat",
    #"Row_ID",
    #"Time_Mabat_Request",
    "Module_Alert_Rn",
    "Alert_Message",
    "Alert_Severity",
    "OrderOrigin",
    "ShiftType_cat",
    "DayHebrew_cat",
    "DayEN_cat",
    "HospAmount_new",
    "NumMedAmount",
    #"Other_Text",
    #"ResponseType_cat",
    "Details",
    "ATC",
    "ATC_cln",
    #"ATC_cat",#not relevant here any longer
    "Basic_Name",
    "DiagnosisInReception",
    "HospDiagnosis",
    "id1",
    "id2",
    "Alert_Rn_Severity_cat",
    # "Response",
    # "Answer_Text",
    "Module_Severity_Rn",
    #"Time_Mabat_Request_convert_ms_res",
    #"Time_Mabat_Response_convert_ms_res",
    #"diff_time_mabat_ms"  ## from some reason it is not removing the time duplicatin - we will run it at the end only on that column and join
]

# Replace NaN in grouping columns with a placeholder (e.g., "NA") (to be aligned with the R code)
data_main[group_cols] = data_main[group_cols].fillna("NA")

# Group by the columns and take the first row in each group
data_distincted_main = data_main.groupby(group_cols, as_index=False, sort=False).first()
# data_distincted_main = data_main.groupby(group_cols, as_index=False).first()

# -------------------------------
# Step 2: Selecting (reordering) columns
# -------------------------------
# Define the final column order (this includes additional columns that weren't used for grouping)
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
    "Order_ID_new",
    "Time_Prescribing_Order",  # additional column not used in grouping
    "Field",
    "IS_New_Order",
    "Drug_Header_cat",
    #"Row_ID",
    #"Time_Mabat_Request",
    "Module_Alert_Rn",
    "Alert_Message",
    "Alert_Severity",
    "OrderOrigin",
    "ShiftType_cat",
    "DayHebrew_cat",
    "DayEN_cat",
    "HospAmount_new",
    "NumMedAmount",
    "Other_Text",        # additional column
    "ResponseType_cat",  # additional column
    "Details",
    "ATC",
    "ATC_cln",
    #"ATC_cat", #not relevant here any longer
    "Basic_Name",
    "DiagnosisInReception",
    "HospDiagnosis",
    "id1",
    "id2",
    "Alert_Rn_Severity_cat",
    "Response",          # additional column
    "Answer_Text_EN",    # additional column
    "Module_Severity_Rn",
    "Time_Mabat_Request_convert_ms_res",  # additional column
    "Time_Mabat_Response_convert_ms_res",   # additional column
    "diff_time_mabat_ms",                   # additional column
    "DRC_SUB_GROUP",                        # additional column
    "NeoDRC_SUB_GROUP"                      # additional column
]

# Reorder/select the columns
data_distincted_main = data_distincted_main[select_cols]

# -------------------------------
# Step 3: Creating an ATC count column and filtering rows
# -------------------------------
# Count the number of ATC codes in the 'ATC' column by splitting on commas
data_distincted_main['ATC_cnt'] = data_distincted_main['ATC'].apply(lambda x: len(str(x).split(',')))

# Remove rows with 3 or more ATC codes
data_distincted_main = data_distincted_main[data_distincted_main['ATC_cnt'] < 3]

# Keep rows where either:
# - ATC_cnt equals 1, OR
# - ATC_cnt equals 2 AND the string "B05XA03" is present in the ATC column
mask = ((data_distincted_main['ATC_cnt'] == 2) & (data_distincted_main['ATC'].str.contains("B05XA03"))) | (data_distincted_main['ATC_cnt'] == 1)
data_distincted_main = data_distincted_main[mask]

# -------------------------------
# (Optional) For testing: Select only certain columns for a quick look
# -------------------------------
# test = data_distincted_main[["ShiftType_cat", "Time_Mabat_Request_convert_ms_res"]]

# Now, data_distincted_main is the optimized, deduplicated and filtered DataFrame.

import pandas as pd
import numpy as np
import re

# -------------------------------------------
# Helper function to extract the first word
# -------------------------------------------
def extract_first_word(text):
    """
    Replace '(' with ' (' to ensure there's a space before any '(',
    then split the string and return the first token in uppercase.
    """
    # Convert to string (in case of NaN or other types)
    text = str(text)
    # Ensure there's a space before any '(' to correctly isolate the first word
    text = re.sub(r'\(', ' (', text)
    # Split the string into words and return the first word in uppercase if available
    parts = text.split()
    return parts[0].upper() if parts else ''

# -------------------------------------------
# 1. Create first-word columns in data_distincted_main
# -------------------------------------------
# For Drug_Header_cat and Basic_Name columns, extract the first word and convert to uppercase.
data_distincted_main["Drug_Header_cat_1st_word"] = data_distincted_main["Drug_Header_cat"].apply(extract_first_word)
data_distincted_main["Basic_Name_1st_word"] = data_distincted_main["Basic_Name"].apply(extract_first_word)

# -------------------------------------------
# 2. Read and clean the Excel file for basic_atc
# -------------------------------------------
# Read the Excel file, treating "NULL" as missing values
basic_atc = pd.read_excel("/Users/nirshlomo/Dropbox/PHD/alert-fatigue/alert_analysis/data/BASIC NAME - ATC CODE - Copy.xlsx",
                            na_values="NULL")

# Select only the relevant columns
basic_atc = basic_atc[["ATC5", "BASIC NAME"]]

# Remove duplicate rows
basic_atc = basic_atc.drop_duplicates()

# Rename the column "BASIC NAME" to "BASIC_NAME_EXT"
basic_atc = basic_atc.rename(columns={"BASIC NAME": "BASIC_NAME_EXT"})

# Remove rows with any missing values
basic_atc = basic_atc.dropna()

# -------------------------------------------
# 3. Extract the first word from BASIC_NAME_EXT in uppercase
# -------------------------------------------
basic_atc["BASIC_NAME_EXT_1st_word"] = basic_atc["BASIC_NAME_EXT"].apply(
    lambda x: x.split()[0].upper() if isinstance(x, str) and x.split() else ''
)

# Group by the first word and take the first row in each group,
# then select only the columns BASIC_NAME_EXT_1st_word and ATC5.
basic_atc_cln = basic_atc.groupby("BASIC_NAME_EXT_1st_word", as_index=False).first()[["BASIC_NAME_EXT_1st_word", "ATC5"]]

# -------------------------------------------
# 4. Left join data_distincted_main with basic_atc_cln
# -------------------------------------------
# Merge on the condition: Drug_Header_cat_1st_word == BASIC_NAME_EXT_1st_word
data_distincted_main_new = pd.merge(data_distincted_main, basic_atc_cln, how="left",
                                    left_on="Drug_Header_cat_1st_word", right_on="BASIC_NAME_EXT_1st_word")

# -------------------------------------------
# 5. Create and update the ATC_NEW column based on conditions
# -------------------------------------------
# First condition: if Drug_Header_cat_1st_word != Basic_Name_1st_word, then use ATC5; otherwise use ATC_cln.
data_distincted_main_new["ATC_NEW"] = np.where(
    data_distincted_main_new["Drug_Header_cat_1st_word"] != data_distincted_main_new["Basic_Name_1st_word"],
    data_distincted_main_new["ATC5"],
    data_distincted_main_new["ATC_cln"]
)

# Second condition: if (Drug_Header_cat_1st_word != Basic_Name_1st_word) and OrderOrigin is 'Chronic Meds', then ensure ATC_NEW is ATC5.
mask = (data_distincted_main_new["Drug_Header_cat_1st_word"] != data_distincted_main_new["Basic_Name_1st_word"]) & \
       (data_distincted_main_new["OrderOrigin"] == 'Chronic Meds')
data_distincted_main_new.loc[mask, "ATC_NEW"] = data_distincted_main_new.loc[mask, "ATC5"]

# Third condition: if OrderOrigin is 'Chronic Meds' and ATC5 is missing, then set ATC_NEW to ATC_cln.
mask2 = (data_distincted_main_new["OrderOrigin"] == 'Chronic Meds') & (data_distincted_main_new["ATC5"].isna())
data_distincted_main_new.loc[mask2, "ATC_NEW"] = data_distincted_main_new.loc[mask2, "ATC_cln"]

# -------------------------------------------
# 6. Validate the join by selecting a subset of columns for testing
# -------------------------------------------
test = data_distincted_main_new[[
    "OrderOrigin", "Hospital_cat", "Order_ID_new", "Details", "Alert_Message",
    "Drug_Header_cat", "Drug_Header_cat_1st_word", "Basic_Name", "Basic_Name_1st_word",
    "ATC", "ATC_cln", "ATC5", "ATC_NEW", "Module_Severity_Rn", "Alert_Severity"
]]

# -------------------------------------------
# 7. Write the final DataFrame to a CSV file
# -------------------------------------------
data_distincted_main_new.to_csv(
    'alert_analysis/data_process/data_distincted_main_new_raw_1.csv',
                                index=False
)

import numpy as np

# -------------------------------
# 1. Sort the DataFrame
# -------------------------------
# Sort data by 'Order_ID_new' and then by 'ATC_NEW'
data_distincted_main_new.sort_values(by=["Order_ID_new", "ATC_NEW"], inplace=True)

# -------------------------------
# 2. Create the cnt_chronic_id Column
# -------------------------------
# For each group (by Order_ID_new), compute a cumulative sum that increments
# every time the value in ATC_NEW changes from the previous row.
data_distincted_main_new['cnt_chronic_id'] = data_distincted_main_new.groupby('Order_ID_new')['ATC_NEW'].transform(
    lambda s: (s != s.shift(1).fillna(s.iloc[0])).cumsum()
)

# -------------------------------
# 3. Update Order_ID_new_update Column
# -------------------------------
# If OrderOrigin equals 'Chronic Meds' and ATC_NEW is not missing,
# then update Order_ID_new by appending '_' and the counter.
# Otherwise, retain the original Order_ID_new.
data_distincted_main_new['Order_ID_new_update'] = np.where(
    (data_distincted_main_new['OrderOrigin'] == 'Chronic Meds') & (data_distincted_main_new['ATC_NEW'].notna()),
    data_distincted_main_new['Order_ID_new'].astype(str) + "_" + data_distincted_main_new['cnt_chronic_id'].astype(str),
    data_distincted_main_new['Order_ID_new']
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create ATC_GROUP Column
# -------------------------------
# Define a mapping of ATC_NEW prefixes to their group names.
atc_mapping = [
    ("A02", "DRUGS FOR ACID RELATED DISORDERS"),
    ("A04", "ANTIEMETICS AND ANTINAUSEANTS"),
    ("A06", "DRUGS FOR CONSTIPATION"),
    ("A10", "DRUGS USED IN DIABETES"),
    ("A11", "VITAMINS"),
    ("A12", "MINERAL SUPPLEMENTS"),
    ("B01", "ANTITHROMBOTIC AGENTS"),
    ("B02", "ANTIHEMORRHAGICS"),
    ("B03", "ANTIANEMIC PREPARATIONS"),
    ("B05", "BLOOD SUBSTITUTES AND PERFUSION SOLUTIONS"),
    ("C01", "CARDIAC THERAPY"),
    ("C02", "ANTIHYPERTENSIVES"),
    ("C03", "DIURETICS"),
    ("C07", "BETA BLOCKING AGENTS"),
    ("C08", "CALCIUM CHANNEL BLOCKERS"),
    ("C09", "AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM"),
    ("C10", "LIPID MODIFYING AGENTS"),
    ("G04", "UROLOGICALS"),
    ("H02", "CORTICOSTEROIDS FOR SYSTEMIC USE"),
    ("H03", "THYROID THERAPY"),
    ("J01", "ANTIBACTERIALS FOR SYSTEMIC USE"),
    ("J02", "ANTIMYCOTICS FOR SYSTEMIC USE"),
    ("J05", "ANTIVIRALS FOR SYSTEMIC USE"),
    ("L01", "ANTINEOPLASTIC AGENTS"),
    ("L03", "IMMUNOSTIMULANTS"),
    ("L04", "IMMUNOSUPPRESSANTS"),
    ("N01", "ANESTHETICS"),
    ("N02", "ANALGESICS"),
    ("N03", "ANTIEPILEPTICS"),
    ("N04", "ANTI-PARKINSON "),
    ("N05", "PSYCHOLEPTICS"),
    ("N06", "PSYCHOANALEPTICS"),
    ("N07", "OTHER NERVOUS SYSTEM DRUGS"),
    ("R03", "DRUGS FOR OBSTRUCTIVE AIRWAY DISEASES"),
    ("V03", "ALL OTHER THERAPEUTIC PRODUCTS")
]

# Ensure missing values in ATC_NEW are filled (if applicable)
data_distincted_main_new["ATC_NEW"] = data_distincted_main_new["ATC_NEW"].fillna("")

# Convert each condition to a NumPy boolean array
conditions = [
    data_distincted_main_new["ATC_NEW"].str.startswith(prefix).to_numpy()
    for prefix, _ in atc_mapping
]
choices = [group for _, group in atc_mapping]

# Now np.select should work correctly:
data_distincted_main_new["ATC_GROUP"] = np.select(conditions, choices, default="OTHER")

# -------------------------------
# 2. Plot Histogram for Original NumMedAmount (Optional)
# -------------------------------
plt.hist(data_distincted_main_new["NumMedAmount"], bins=400)
plt.title("Histogram of NumMedAmount")
plt.xlabel("NumMedAmount")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 3. Calculate the Number of Medications per Patient
# -------------------------------
# Convert Medical_Record to a categorical type for efficiency
data_distincted_main_new['Medical_Record_cat'] = data_distincted_main_new['Medical_Record'].astype('category')

# In the R code, the first summarization groups by Medical_Record_cat and Order_ID_new_update,
# then counts the number of unique order groups per patient.
# We calculate the number of unique Order_ID_new_update per Medical_Record_cat.
num_med_df = (data_distincted_main_new
              .groupby('Medical_Record_cat')['Order_ID_new_update']
              .nunique()
              .reset_index(name='NumMedAmount_calc'))

# -------------------------------
# 4. Join the Calculated Medication Count Back to the Main DataFrame
# -------------------------------
data_distincted_main_new = data_distincted_main_new.merge(num_med_df, on='Medical_Record_cat', how='left')

# -------------------------------
# 5. Plot Histogram for Calculated Number of Medications (Optional)
# -------------------------------
plt.hist(data_distincted_main_new["NumMedAmount_calc"], bins=400)
plt.title("Histogram of Calculated NumMedAmount")
plt.xlabel("NumMedAmount_calc")
plt.ylabel("Frequency")
plt.show()

import numpy as np
import pandas as pd

# -------------------------------
# 1. Create num_of_diagnosis Column
# -------------------------------
# Split the 'HospDiagnosis' string by ";" and count the resulting parts. if 'NA', set to 0.
data_distincted_main_new["num_of_diagnosis"] = data_distincted_main_new["HospDiagnosis"].apply(
    lambda x: 0 if x == "NA" else len(x.split(";"))
)

# -------------------------------
# 2. Create diseaseSplit Column
# -------------------------------
# Split 'HospDiagnosis' into a list of diagnoses for each row.
data_distincted_main_new["diseaseSplit"] = data_distincted_main_new["HospDiagnosis"].str.split(";")

# -------------------------------
# 3. Clean diseaseSplit Column:
#    - If the split result equals 'NANA' or 'NA', set it to None.
#    - Otherwise, convert all diagnoses to uppercase.
# -------------------------------
def fix_disease_split(ds):
    """
    Expects ds to be a list of diagnosis strings.
    If the list has exactly one element that is 'NANA' or 'NA' (case-insensitive), return None.
    Otherwise, convert each diagnosis to uppercase.
    """
    if isinstance(ds, list):
        # Check if the list is exactly one element and that element is 'NANA' or 'NA'
        if len(ds) == 1 and ds[0].strip().upper() in ['NANA', 'NA']:
            return None
        # Otherwise, convert each diagnosis to uppercase
        return [s.upper() for s in ds]
    return ds

data_distincted_main_new["diseaseSplit"] = data_distincted_main_new["diseaseSplit"].apply(fix_disease_split)

# -------------------------------
# 4. Define List of Keywords (Diseases) to Count
# -------------------------------
my_list = ['LIVER', 'HEPATIC', 'CIRRHOSIS', 'HEPATITIS', 'PORTAL HYPERTENSION', 'JAUNDICE',
           'MYOCARDIAL', 'HEART FAILURE',
           'PERIPHERAL VASCULAR DISEASE', 'PVD',
           'CEREBROVASCULAR', 'STROKE', 'CEREBRAL INFARCTION', 'TIA', 'CVA',
           'DEMENTIA',
           'CHRONIC PULMONARY DISEASE', 'COPD', 'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
           'RHEUMATOLOGIC DISEASE', 'FIBROMYALGIA', 'GOUT', 'ARTHRITIS', 'ANKYLOSING SPONDYLITIS', 'SCLERODERMA',
           'DUODENAL ULCER', 'PEPTIC ULCER', 'GASTRIC ULCER',
           'DIABETES',
           'HEMIPLEGIA', 'HEMIPARESIS',
           'PARAPLEGIA',
           'RENAL', 'PYELONEPHRITIS', 'HEMODIALYSIS', 'KIDNEY', 'CKD', 'ARF', 'AKI',
           'MALIGNANCY', 'MALIGNANT', 'CARCINOMA', 'NEOPLASM', 'ADENOCARCINOMA',
           'LEUKEMIA', 'AML', 'CML',
           'LYMPHOMA',
           'HIV',
           'METASTATIC', 'METASTASIS']

# -------------------------------
# 5. Create Count Columns for Each Keyword
# -------------------------------
def count_keyword(ds, keyword):
    """
    Counts how many times the substring `keyword` appears in the list ds.
    """
    if ds is None:
        return 0
    return sum(1 for s in ds if keyword in s)

# Create a new column for each keyword in my_list with the suffix "_count"
for item in my_list:
    col_name = f"{item}_count"
    data_distincted_main_new[col_name] = data_distincted_main_new["diseaseSplit"].apply(lambda ds: count_keyword(ds, item))

# -------------------------------
# 6. Create Group Count Columns Based on Specific Disease Count Columns
# -------------------------------

# Liver group:
# If any of ["LIVER_count", "HEPATIC_count", "JAUNDICE_count", "PORTAL HYPERTENSION_count", "CIRRHOSIS_count"] is > 0,
# then if either "PORTAL HYPERTENSION_count" or "CIRRHOSIS_count" is > 0, set to 3, else 1. Otherwise, set to 0.
liver_cols = ["LIVER_count", "HEPATIC_count", "JAUNDICE_count", "PORTAL HYPERTENSION_count", "CIRRHOSIS_count"]
portal_cirrhosis = ["PORTAL HYPERTENSION_count", "CIRRHOSIS_count"]

data_distincted_main_new["liver_group_cnt"] = np.where(
    data_distincted_main_new[liver_cols].fillna(0).gt(0).sum(axis=1) > 0,
    np.where(
        data_distincted_main_new[portal_cirrhosis].fillna(0).gt(0).sum(axis=1) > 0,
        3,
        1
    ),
    0
)

# PVD group:
# If either "PERIPHERAL VASCULAR DISEASE_count" or "PVD_count" is > 0, set to 1; otherwise, 0.
data_distincted_main_new["PVD_group_cnt"] = np.where(
    data_distincted_main_new[["PERIPHERAL VASCULAR DISEASE_count", "PVD_count"]].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Cerebrovascular group:
# If any of ["CEREBROVASCULAR_count", "STROKE_count", "CEREBRAL INFARCTION_count", "TIA_count", "CVA_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["CEREBROVASCULAR_group_cnt"] = np.where(
    data_distincted_main_new[['CEREBROVASCULAR_count', 'STROKE_count', 'CEREBRAL INFARCTION_count', 'TIA_count', 'CVA_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# COPD group:
# If any of ["CHRONIC PULMONARY DISEASE_count", "COPD_count", "CHRONIC OBSTRUCTIVE PULMONARY DISEASE_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["COPD_group_cnt"] = np.where(
    data_distincted_main_new[['CHRONIC PULMONARY DISEASE_count', 'COPD_count', 'CHRONIC OBSTRUCTIVE PULMONARY DISEASE_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Gout group:
# If any of ["RHEUMATOLOGIC DISEASE_count", "FIBROMYALGIA_count", "GOUT_count", "ARTHRITIS_count", "ANKYLOSING SPONDYLITIS_count", "SCLERODERMA_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["GOUT_group_cnt"] = np.where(
    data_distincted_main_new[['RHEUMATOLOGIC DISEASE_count', 'FIBROMYALGIA_count', 'GOUT_count', 'ARTHRITIS_count', 'ANKYLOSING SPONDYLITIS_count', 'SCLERODERMA_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Ulcer group:
# If any of ["DUODENAL ULCER_count", "PEPTIC ULCER_count", "GASTRIC ULCER_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["ULCER_group_cnt"] = np.where(
    data_distincted_main_new[['DUODENAL ULCER_count', 'PEPTIC ULCER_count', 'GASTRIC ULCER_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Hemiplegia group:
# If any of ["HEMIPLEGIA_count", "HEMIPARESIS_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["HEMIPLEGIA_group_cnt"] = np.where(
    data_distincted_main_new[['HEMIPLEGIA_count', 'HEMIPARESIS_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Renal group:
# If any of ["RENAL_count", "PYELONEPHRITIS_count", "HEMODIALYSIS_count", "KIDNEY_count", "CKD_count", "ARF_count", "AKI_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["RENAL_group_cnt"] = np.where(
    data_distincted_main_new[['RENAL_count', 'PYELONEPHRITIS_count', 'HEMODIALYSIS_count', 'KIDNEY_count', 'CKD_count', 'ARF_count', 'AKI_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Malignancy group:
# If any of ["MALIGNANCY_count", "MALIGNANT_count", "CARCINOMA_count", "NEOPLASM_count", "ADENOCARCINOMA_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["MALIGNANCY_group_cnt"] = np.where(
    data_distincted_main_new[['MALIGNANCY_count', 'MALIGNANT_count', 'CARCINOMA_count', 'NEOPLASM_count', 'ADENOCARCINOMA_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Leukemia group:
# If any of ["LEUKEMIA_count", "AML_count", "CML_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["LEUKEMIA_group_cnt"] = np.where(
    data_distincted_main_new[['LEUKEMIA_count', 'AML_count', 'CML_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

# Metastatic group (6 POINTS GROUP):
# If any of ["METASTATIC_count", "METASTASIS_count"] is > 0, set to 1; otherwise, 0.
data_distincted_main_new["METASTATIC_group_cnt"] = np.where(
    data_distincted_main_new[['METASTATIC_count', 'METASTASIS_count']].fillna(0).gt(0).sum(axis=1) > 0,
    1,
    0
)

import numpy as np
import pandas as pd

# =================== Calculate Charlson Comorbidity Index Scores ===================

# 1. Calculate the 1- and 3-point sum
cols_sum1 = [
    'MYOCARDIAL_count',
    "HEART FAILURE_count",
    "PVD_group_cnt",
    "CEREBROVASCULAR_group_cnt",
    'DEMENTIA_count',
    "COPD_group_cnt",
    "GOUT_group_cnt",
    "ULCER_group_cnt",
    "liver_group_cnt"
]
data_distincted_main_new["charls_sum1_3_points"] = data_distincted_main_new[cols_sum1].sum(axis=1)

# 2. Calculate the 2-point sum and multiply by 2
cols_sum2 = [
    "HEMIPLEGIA_group_cnt",
    "RENAL_group_cnt",
    "MALIGNANCY_group_cnt",
    "LEUKEMIA_group_cnt",
    "LYMPHOMA_count",
    'DIABETES_count'
]
data_distincted_main_new["charls_sum2points"] = data_distincted_main_new[cols_sum2].sum(axis=1) * 2

# 3. Calculate the 6-point sum and multiply by 6
cols_sum6 = ['HIV_count', 'METASTATIC_group_cnt']
data_distincted_main_new["charls_sum6points"] = data_distincted_main_new[cols_sum6].sum(axis=1) * 6

# 4. Total Charlson score is the sum of the above scores
data_distincted_main_new["Charlson_score"] = data_distincted_main_new[
    ["charls_sum1_3_points", "charls_sum2points", "charls_sum6points"]
].sum(axis=1)

# 5. Adjust Charlson score for age using nested conditions:
#    - If AGE_num is between 50 and 59, add 1.
#    - If between 60 and 69, add 2.
#    - If between 70 and 79, add 3.
#    - If 80 or above, add 4.

# Convert 'AGE_num' to numeric, turning anything non-convertible (like "NA") into NaN.
data_distincted_main_new["AGE_num"] = pd.to_numeric(
    data_distincted_main_new["AGE_num"], errors="coerce"
)

conditions = [
    (data_distincted_main_new["AGE_num"] >= 50) & (data_distincted_main_new["AGE_num"] <= 59),
    (data_distincted_main_new["AGE_num"] >= 60) & (data_distincted_main_new["AGE_num"] <= 69),
    (data_distincted_main_new["AGE_num"] >= 70) & (data_distincted_main_new["AGE_num"] <= 79),
    (data_distincted_main_new["AGE_num"] >= 80)
]
choices = [
    data_distincted_main_new["Charlson_score"] + 1,
    data_distincted_main_new["Charlson_score"] + 2,
    data_distincted_main_new["Charlson_score"] + 3,
    data_distincted_main_new["Charlson_score"] + 4
]
data_distincted_main_new["Charlson_score_age_adj"] = np.select(conditions, choices, default=data_distincted_main_new["Charlson_score"])

# 6. Calculate 10-year survival rate based on the age-adjusted Charlson score.
# Note: In R, the expression is: (0.983^(exp(1)^(score*0.9)))*100, where '^' is right-associative.
# In Python, this translates directly using ** for exponentiation.
data_distincted_main_new["SurvivalRate10years_age_adj"] = (0.983 ** (np.exp(1) ** (data_distincted_main_new["Charlson_score_age_adj"] * 0.9))) * 100
data_distincted_main_new["SurvivalRate10years_age_adj_"] = (0.983 ** (2.71828 ** (data_distincted_main_new["Charlson_score_age_adj"] * 0.9))) * 100

# 7. (Optional) Create a test DataFrame to check Charlson-related columns
test_Charlson = data_distincted_main_new[[
    'diseaseSplit', 'AGE_num', 'charls_sum1_3_points', 'charls_sum2points',
    'charls_sum6points', 'Charlson_score', 'Charlson_score_age_adj',
    'SurvivalRate10years_age_adj', 'SurvivalRate10years_age_adj_'
]]

# =================== Drop Unrequired Columns ===================
# Remove all columns that contain "_count"
cols_to_remove = [col for col in data_distincted_main_new.columns if "_count" in col]
df_main_cln = data_distincted_main_new.drop(columns=cols_to_remove)

# Then remove all columns that contain "_group_cnt" from the new DataFrame
cols_to_remove = [col for col in df_main_cln.columns if "_group_cnt" in col]
df_main_cln = df_main_cln.drop(columns=cols_to_remove)

# df_main_cln now contains the cleaned DataFrame without the _count and _group_cnt columns.
df_main_cln.to_csv('alert_analysis/data_process/data_main_cln.csv', index=False)



