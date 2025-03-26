import pandas as pd
import numpy as np
df_main_cln = pd.read_csv('alert_analysis/data_process/data_main_cln.csv')
#TODO #ADD THE AGE
# 1. Create the 'adult_child_cat' column
#    - If AGE_num < 19, label as "child"; otherwise "adult"
df_main_cln['adult_child_cat'] = np.where(df_main_cln['AGE_num'] < 19, 'child', 'adult')

# 2. Create the 'Age_cat' column using bins and labels
bins = [0, 1, 6, 11, 16, 19, 31, 45, 56, 65, 76, 86, float('inf')]
labels = [
    '< 1',      # For ages > 0 and < 1
    '1-5',      # For ages >= 1 and < 6
    '6-10',     # For ages >= 6 and < 11
    '11-15',    # For ages >= 11 and < 16
    '16-18',    # For ages >= 16 and < 19
    '19-30',    # For ages >= 19 and < 31
    '31-44',    # For ages >= 31 and < 45
    '45-55',    # For ages >= 45 and < 56
    '56-64',    # For ages >= 56 and < 65
    '65-75',    # For ages >= 65 and < 76
    '76-85',    # For ages >= 76 and < 86
    '> 85'      # For ages >= 86
]

df_main_cln['Age_cat'] = pd.cut(
    df_main_cln['AGE_num'],
    bins=bins,
    labels=labels,
    right=False  # Left-inclusive, right-exclusive intervals
)

import pandas as pd
import numpy as np
import re

# -----------------------------------------------------------------------------
# 1) IF ELSE LOGIC: EXTRACT TEXT UP TO THE WORD "may" WHEN ALERT_Rn_Severity_cat
#    CONTAINS "DDI" OR "DT," OTHERWISE USE NaN
# -----------------------------------------------------------------------------

# Explanation:
# - We check whether "DDI" or "DT" (case-insensitive) is present in 'Alert_Rn_Severity_cat'.
# - If either is found, we extract the text from the start of 'Alert_Message' up
#   to (but not including) the word "may" (also case-insensitive).
# - Otherwise, the value is set to NaN (similar to R's NA).

df_main_cln['drugs_from_AlretMessage'] = np.where(
    df_main_cln['Alert_Rn_Severity_cat'].str.contains("DDI", case=False) |
    df_main_cln['Alert_Rn_Severity_cat'].str.contains("DT", case=False),
    # Use a lambda to apply a regex replacement on each row's Alert_Message.
    df_main_cln['Alert_Message'].apply(
        lambda x: re.sub(r"(?i)(.*)\bmay\b.*", r"\1", x) if pd.notna(x) else x
    ),
    # Otherwise fill with NaN.
    np.nan
)

# -----------------------------------------------------------------------------
# 2) EXTRACT ONLY WORDS THAT CONTAIN AT LEAST ONE CAPITAL LETTER
# -----------------------------------------------------------------------------

# Explanation:
# - We define a regex pattern that captures words containing at least one uppercase letter.
# - For each row in 'drugs_from_AlretMessage', we use re.findall() to get all matches
#   and then join them with commas.

def extract_capital_words(text):
    if pd.isna(text):
        return np.nan
    # Find all words that have at least one uppercase letter
    matches = re.findall(r"\b[A-Za-z]*[A-Z]+[A-Za-z]*\b", str(text))
    # Join the matches by commas
    return ",".join(matches)

df_main_cln['drugs_from_AlretMessage'] = df_main_cln['drugs_from_AlretMessage'].apply(extract_capital_words)

# -----------------------------------------------------------------------------
# 3) REMOVE UNWANTED WORDS (THE "words_to_drop" LIST)
# -----------------------------------------------------------------------------

# Explanation:
# - We create a pattern that captures each word from the 'words_to_drop' list as a whole word.
# - Then we replace any match (case-insensitive) in 'drugs_from_AlretMessage' with an empty string.

words_to_drop = [
    "The", "TEVA", "KWIK", "PEN", "BAYER", "SOLOSTAR", "FLASH", "PLUS", "RTH", "Duplicate",
    "Therapy", "TURBUHALER", "CHILD", "ORAL", "INOVAMED", "CFC", "FREE", "NEW", "KERN",
    "PHARMA", "STRAWB", "(EU)", "CREAM", "OINTMENT", "VELO", "INOVAMED", "UNIT", "RETARD",
    "PENFILL", "novoRAPID", "APIDRA", "ROMPHAR", "ROMPHARM", "CLARIS", "FRESENIUS", "FLASH",
    "DIASPORAL", ",GRA", "AVENIR", "MYLAN", "RATIO", "SALF", "LANTUS", "RESPIR", "FLEX",
    "BASAGLAR", "CRT", "TOUJEO", "LIPURO", "HCL", "DECANOAS", "KALCEKS", "ODT", "AOV",
    "TRIMA", "DEXCEL", "PANPHA", "ROTEXMEDICA", "ROTEX", "ROTEXMED", "FORTE", "HumuLIN",
    "ADVANCE", "BANANA", "COFFEE", "FIBER", "VANILLA", "TREGLUDEC", "CHOCOLATE", "PENFILL",
    "PWD", "RESPULES", "Drops", ",DRP"
]

# Build a single regex pattern that matches any of these words as whole words.
drop_pattern = r"\b(" + "|".join(words_to_drop) + r")\b"

# Replace them with "" (empty string), ignoring case.
df_main_cln['drugs_from_AlretMessage'] = df_main_cln['drugs_from_AlretMessage'].str.replace(
    drop_pattern, "", case=False, regex=True
)

# -----------------------------------------------------------------------------
# 4) FIX COMMON PHRASES (SERIES OF gsub EQUIVALENTS)
# -----------------------------------------------------------------------------

# Explanation:
# - We perform a series of string replacements (one by one) that convert specific substrings
#   (like "ACTILYSE,TPA") into desired formats (like "ACTILYSE-TPA").

replacements = [
    ('ACTILYSE,TPA','ACTILYSE-TPA'),
    ('V,DALGIN','V-DALGIN'),
    ('FOLIC,ACID','FOLIC-ACID'),
    ('COD,ACAMOL','COD-ACAMOL'),
    ('PROCTO,GLYVENOL','PROCTO-GLYVENOL'),
    ('Betacorten,G','Betacorten-G'),
    ('MICRO,KALIUM','MICRO-KALIUM'),
    ('VASODIP,COMBO','VASODIP-COMBO'),
    ('DEPO','DEPO-medrol'),
    ('VITA,CAL','VITA-CAL'),
    ('TAZO,PIP','TAZO-PIP'),
    ('Solu,CORTEF','Solu-CORTEF'),
    ('DEPALEPT,CHRONO','DEPALEPT-CHRONO'),
    ('PIPERACILLIN,TAZOBACTAM','PIPERACILLIN-TAZOBACTAM'),
    ('SOLU','SOLU-medrol'),
    ('SOPA,K','SOPA-K'),
    ('V,OPTIC','V-OPTIC'),
    ('SLOW,K','SLOW-K'),
    ('JARDIANCE DUO','JARDIANCE-DUO')
]

for old, new in replacements:
    df_main_cln['drugs_from_AlretMessage'] = df_main_cln['drugs_from_AlretMessage'].str.replace(old, new, regex=False)

# -----------------------------------------------------------------------------
# 5) REMOVE DUPLICATES WITHIN THE SAME CELL
# -----------------------------------------------------------------------------

# Explanation:
# - Some rows may have repeated items separated by commas.
# - Split the string by commas, keep unique values, then rejoin them with commas.

def remove_duplicates_in_cell(text):
    if pd.isna(text):
        return np.nan
    items = text.split(",")
    # Filter out empty strings
    items = [itm for itm in items if itm.strip() != ""]
    # unique() while preserving order: use dict.fromkeys trick
    items_unique = list(dict.fromkeys(items))
    return ",".join(items_unique)

df_main_cln['drugs_from_AlretMessage'] = df_main_cln['drugs_from_AlretMessage'].apply(remove_duplicates_in_cell)

# -----------------------------------------------------------------------------
# 6) REPLACE THE STRING 'NA' WITH ACTUAL NaN
# -----------------------------------------------------------------------------

df_main_cln.loc[df_main_cln['drugs_from_AlretMessage'] == 'NA', 'drugs_from_AlretMessage'] = np.nan

# -----------------------------------------------------------------------------
# 7) CREATE A TEST DATAFRAME WITH SELECT COLUMNS (IF DESIRED)
# -----------------------------------------------------------------------------

# Explanation:
# - Just subset or “select” a few columns for inspection.

test_data_from_alert = df_main_cln[['Order_ID_new_update',
                                    'Alert_Rn_Severity_cat',
                                    'Alert_Message',
                                    'drugs_from_AlretMessage']]

# Print or inspect the resulting subset.
print(test_data_from_alert)

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1) SELECT SPECIFIC COLUMNS FROM df_main_cln
# -----------------------------------------------------------------------------
# This replicates src_for_flat <- df_main_cln[, (names(df_main_cln) %in% c(...))]
# In Python, we use df[column_list] to subset.

columns_to_keep = [
    "Order_ID_new_update",
    "Hospital_cat",
    "HospitalName_EN_cat",
    "UnitName_cat",
    "Medical_Record_cat",
    "SeverityLevelToStopOrder_cat",
    "OrderOrigin",
    "Time_Prescribing_Order",
    "Details",
    "ATC_NEW",
    "ATC_GROUP",
    "ResponseType_cat",
    "Alert_Rn_Severity_cat",
    "Response",
    "Answer_Text_EN",
    "diff_time_mabat_ms",
    "ShiftType_cat",
    "DayEN_cat",
    "HospAmount_new",
    "NumMedAmount",
    "NumMedAmount_calc",
    "id1",
    "AGE_num",
    "Age_cat",
    "Gender_Text_EN_cat",
    "Charlson_score_age_adj",
    "SurvivalRate10years_age_adj",
    "DiagnosisInReception",
    "HospDiagnosis",
    "SectorText_EN_cat",
    "id2",
    "adult_child_cat",
    "DRC_SUB_GROUP",
    "NeoDRC_SUB_GROUP",
    "drugs_from_AlretMessage",
    # The below columns are added in March 2025
    "Module_Alert_Rn",
    "Alert_Message",
    "Other_Text",
]

src_for_flat = df_main_cln[columns_to_keep]

# -----------------------------------------------------------------------------
# 2) REMOVE DUPLICATES
# -----------------------------------------------------------------------------
# Equivalent to R:
#   src_for_flat_cln <- src_for_flat %>% group_by_all() %>% slice(1)
# In pandas, group_by_all + slice(1) is effectively just drop_duplicates on all columns.

src_for_flat_cln = src_for_flat.drop_duplicates()

# -----------------------------------------------------------------------------
# 3) CREATE A VERSION WITHOUT 'diff_time_mabat_ms'
# -----------------------------------------------------------------------------
# Equivalent to removing 'diff_time_mabat_ms' from the column set.
# R code:
#   src_for_flat_cln_wo_mabat_ms <- src_for_flat_cln[, (names(src_for_flat_cln) %in% c(...))]
# We simply drop the column in Python.

columns_without_mabat = [col for col in columns_to_keep if col != "diff_time_mabat_ms"]
src_for_flat_cln_wo_mabat_ms = src_for_flat_cln[columns_without_mabat]

# -----------------------------------------------------------------------------
# 4) PIVOT TO WIDE FORMAT BY 'Alert_Rn_Severity_cat'
# -----------------------------------------------------------------------------
# Equivalent to:
#   flat_by_sevirity <- dcast(src_for_flat_cln_wo_mabat_ms,
#        Order_ID_new_update + Hospital_cat + ... + adult_child_cat ~ Alert_Rn_Severity_cat)
# The left-hand side of ~ are "index" columns; the right-hand side is the "columns".
# In R's reshape2::dcast with no value.var or fun.aggregate, it typically counts occurrences.
# In pandas, we replicate that with pivot_table using aggfunc='size' or a lambda returning x.size.

# index_cols = [
#     "Order_ID_new_update",
#     "Hospital_cat",
#     "HospitalName_EN_cat",
#     "UnitName_cat",
#     "Medical_Record_cat",
#     "SeverityLevelToStopOrder_cat",
#     "OrderOrigin",
#     "Time_Prescribing_Order",
#     "Details",
#     "ATC_NEW",
#     "ATC_GROUP",
#     "ResponseType_cat",
#     "Response",
#     "Answer_Text_EN",
#     "ShiftType_cat",
#     "DayEN_cat",
#     "HospAmount_new",
#     "NumMedAmount",
#     "NumMedAmount_calc",
#     "id1",
#     "AGE_num",
#     "Age_cat",
#     "Gender_Text_EN_cat",
#     "Charlson_score_age_adj",
#     "SurvivalRate10years_age_adj",
#     "DiagnosisInReception",
#     "HospDiagnosis",
#     "SectorText_EN_cat",
#     "id2",
#     "adult_child_cat"
# ]
#
# flat_by_sevirity = pd.pivot_table(
#     src_for_flat_cln_wo_mabat_ms,
#     index=index_cols,
#     columns="Alert_Rn_Severity_cat",
#     aggfunc='size',    # Equivalent to R's default of counting rows if no value.var is specified
#     fill_value=0
# ).reset_index()

# -----------------------------------------------------------------------------
# PIVOT ONLY ON A MINIMAL SET OF COLUMNS:
# -----------------------------------------------------------------------------
# 1) Pivot only on a minimal set of columns:
#    Here, we pivot so each Order_ID_new_update forms one row,
#    and each Alert_Rn_Severity_cat forms a column counting occurrences.
alert_counts = (
    pd.pivot_table(
        src_for_flat_cln_wo_mabat_ms,
        index=["Order_ID_new_update"],            # minimal index
        columns="Alert_Rn_Severity_cat",          # pivot columns
        aggfunc="size",
        fill_value=0
    )
    .reset_index()
)
# 2) Then join back to your original wide set of columns
#    Instead of re-creating them in the pivot index, we just do a merge.
flat_by_sevirity = pd.merge(
    src_for_flat_cln_wo_mabat_ms.drop_duplicates(subset=["Order_ID_new_update"]),
    alert_counts,
    on="Order_ID_new_update",
    how="left"
)


flat_by_sevirity.to_csv('alert_analysis/data_process/flat_by_sevirity.csv', index=False)
flat_by_sevirity.shape # (3737601, 47)
assert flat_by_sevirity.shape[0] == 3737601, "The number of rows is incorrect."

# -----------------------------------------------------------------------------
# OPTIONAL: INSPECT THE RESULT
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1) ADD THE SUM OF SELECTED ALERT COLUMNS (EXCLUDING NA)
# -----------------------------------------------------------------------------
# R Equivalent:
#   flat_by_sevirity$num_of_alerts_per_order_id <- rowSums(flat_by_sevirity[,
#       c("DAM", "DDI-Contraindicated Drug Combination", "DDI-Moderate Interaction",
#       "DDI-Severe Interaction","DRC","Renal alerts","Technical alert")])
#
# Simply sum across these columns (axis=1) for each row.
# (Commented-out "DT" remains omitted if your data doesn't contain it.)

alert_cols = [
    "DAM",
    "DDI-Contraindicated Drug Combination",
    "DDI-Moderate Interaction",
    "DDI-Severe Interaction",
    "DRC",
    "Renal alerts",
    "Technical alert"
]

flat_by_sevirity["num_of_alerts_per_order_id"] = flat_by_sevirity[alert_cols].sum(axis=1)

# -----------------------------------------------------------------------------
# 2) RENAME COLUMNS & CONVERT TO FACTOR (CATEGORICAL) TYPES
# -----------------------------------------------------------------------------
#   flat_by_sevirity <- flat_by_sevirity %>%
#       rename(Renal_alerts = `Renal alerts`) %>%
#       rename(Technical_alerts = `Technical alert`) %>%
#       rename(DDI_Severe_Interaction = "DDI-Severe Interaction") %>%
#       rename(DDI_Moderate_Interaction = "DDI-Moderate Interaction") %>%
#       rename(DDI_Contraindicated_Drug_Combination = "DDI-Contraindicated Drug Combination")
#
# R: factor(...) => Python: .astype('category')

flat_by_sevirity.rename(
    columns={
        "Renal alerts": "Renal_alerts",
        "Technical alert": "Technical_alerts",
        "DDI-Severe Interaction": "DDI_Severe_Interaction",
        "DDI-Moderate Interaction": "DDI_Moderate_Interaction",
        "DDI-Contraindicated Drug Combination": "DDI_Contraindicated_Drug_Combination"
    },
    inplace=True
)

flat_by_sevirity["Renal_alerts"] = flat_by_sevirity["Renal_alerts"].astype("category")
flat_by_sevirity["Technical_alerts"] = flat_by_sevirity["Technical_alerts"].astype("category")

# -----------------------------------------------------------------------------
# 3) CREATE BINARY (0/1) COLUMNS FOR ALERT CATEGORIES
# -----------------------------------------------------------------------------
#   flat_by_sevirity$DAM_CAT       <- factor(ifelse(DAM != 0, 1, 0))
#   flat_by_sevirity$DRC_CAT       <- factor(ifelse(DRC != 0, 1, 0))
#   etc.
#
# In Python, we can use np.where(...) and convert to categorical.

flat_by_sevirity["DAM_CAT"] = (
        flat_by_sevirity["DAM"] != 0
).astype(int).astype("category")

flat_by_sevirity["DDI_Contraindicated_Drug_Combination_CAT"] = (
    (flat_by_sevirity["DDI_Contraindicated_Drug_Combination"] != 0)
    .astype(int)
    .astype("category")
)

flat_by_sevirity["DDI_Moderate_Interaction_CAT"] = (
    (flat_by_sevirity["DDI_Moderate_Interaction"] != 0)
    .astype(int)
    .astype("category")
)

flat_by_sevirity["DDI_Severe_Interaction_CAT"] = (
    (flat_by_sevirity["DDI_Severe_Interaction"] != 0)
    .astype(int)
    .astype("category")
)

flat_by_sevirity["DRC_CAT"] = (
    (flat_by_sevirity["DRC"] != 0)
    .astype(int)
    .astype("category")
)

# If you need "DT_CAT" when the data has "DT", just replicate similarly.
# flat_by_sevirity["DT_CAT"] = np.where(flat_by_sevirity["DT"] != 0, 1, 0).astype("category")

flat_by_sevirity["Technical_alerts_CAT"] = (
    (flat_by_sevirity["Technical_alerts"] != 0)  # Boolean (True/False)
    .astype(int)                                   # Convert True/False -> 1/0
    .astype("category")                            # Cast to categorical dtype
)

# If you have a "NeoDRC" column from pivot:
# flat_by_sevirity["NeoDRC_CAT"] = np.where(flat_by_sevirity["NeoDRC"] != 0, 1, 0).astype("category")
# (Uncomment if "NeoDRC" was indeed part of your pivot.)

flat_by_sevirity["Renal_alerts_CAT"] = (
    (flat_by_sevirity["Renal_alerts"] != 0)
    .astype(int)
    .astype("category")
)

# -----------------------------------------------------------------------------
# 4) ADD "chronic_num_calc" USING GROUPED LOGIC
# -----------------------------------------------------------------------------
#   R Equivalent:
#     flat_by_sevirity <- flat_by_sevirity %>%
#       group_by(Medical_Record_cat, id1) %>%
#       mutate(chronic_num_calc = sum(ifelse(OrderOrigin == "Chronic Meds", 1, 0))) %>%
#       ungroup() %>%
#       mutate(chronic_num_calc = ifelse(OrderOrigin != "Chronic Meds", 0, chronic_num_calc))
#
# In Python, we can use groupby + transform, then conditionally reset values.

flat_by_sevirity["chronic_num_calc"] = (
    flat_by_sevirity
    .groupby(["Medical_Record_cat", "id1"])["OrderOrigin"]
    .transform(lambda x: (x == "Chronic Meds").sum())
)

# Now, set chronic_num_calc = 0 if OrderOrigin != "Chronic Meds"
mask_not_chronic = flat_by_sevirity["OrderOrigin"] != "Chronic Meds"
flat_by_sevirity.loc[mask_not_chronic, "chronic_num_calc"] = 0

# -----------------------------------------------------------------------------
# 5) COMPUTE "hosp_days" USING ADMISSION & DISCHARGE DATE DIFFERENCE
# -----------------------------------------------------------------------------
#   R Equivalent:
#     flat_by_sevirity$date_time_prescribe <- as.Date(substr(flat_by_sevirity$Time_Prescribing_Order,1,10))
#     flat_by_sevirity <- flat_by_sevirity %>%
#       group_by(Medical_Record_cat,id1) %>%
#       mutate(hosp_days = as.numeric(difftime(max(date_time_prescribe),
#                         min(date_time_prescribe), units = "days"))+1) %>%
#       ungroup()
#
# Python Implementation:

# 5a) Extract date substring and convert to datetime
flat_by_sevirity["date_time_prescribe"] = flat_by_sevirity["Time_Prescribing_Order"].str[:10]
flat_by_sevirity["date_time_prescribe"] = pd.to_datetime(flat_by_sevirity["date_time_prescribe"], errors="coerce")

# 5b) Group by [Medical_Record_cat, id1], find max - min + 1
flat_by_sevirity["hosp_days"] = (
    flat_by_sevirity
    .groupby(["Medical_Record_cat", "id1"])["date_time_prescribe"]
    .transform(lambda x: (x.max() - x.min()).days + 1)
)

# -----------------------------------------------------------------------------
# DONE. The DataFrame flat_by_sevirity now reflects all the transformations
# from your original R code snippet.
# -----------------------------------------------------------------------------

# OPTIONAL: inspect results
print(flat_by_sevirity.head(10))

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# A) JOIN BACK THE MAX diff_time_mabat_ms
# -----------------------------------------------------------------------------
# R: df_diff_time_ms <- src_for_flat_cln %>%
#      group_by(Order_ID_new_update) %>%
#      summarise(diff_time_mabat_ms= max(diff_time_mabat_ms))
#
# Then flat_by_sevirity <- flat_by_sevirity %>%
#      left_join(df_diff_time_ms, by ="Order_ID_new_update" )

df_diff_time_ms = (
    src_for_flat_cln
    .groupby("Order_ID_new_update", as_index=False)
    .agg({"diff_time_mabat_ms": "max"})
)

flat_by_sevirity = flat_by_sevirity.merge(
    df_diff_time_ms,
    on="Order_ID_new_update",
    how="left"
)

# -----------------------------------------------------------------------------
# B) JOIN BACK DRC_SUB_GROUP
# -----------------------------------------------------------------------------
# R: df_DRC_SUB_GROUP <- dcast(src_for_flat_cln, Order_ID_new_update ~ DRC_SUB_GROUP)
#     df_DRC_SUB_GROUP <- df_DRC_SUB_GROUP[, !names(df_DRC_SUB_GROUP) %in% c("NA")]
#
#     flat_by_sevirity <- flat_by_sevirity %>%
#       left_join(df_DRC_SUB_GROUP, by ="Order_ID_new_update")
#
# In Python, we use pivot_table (similar to dcast):
#   index="Order_ID_new_update"
#   columns="DRC_SUB_GROUP"
#   If no value.var is specified, we typically count rows (aggfunc="size").
#   Remove the column "NA" if it exists, then left join.

df_DRC_SUB_GROUP = (
    pd.pivot_table(
        src_for_flat_cln,
        index="Order_ID_new_update",
        columns="DRC_SUB_GROUP",
        aggfunc="size",    # Count the number of rows for each group
        fill_value=0
    )
    .reset_index()
)

# Remove any column literally named "NA" if it exists
if "NA" in df_DRC_SUB_GROUP.columns:
    df_DRC_SUB_GROUP.drop(columns=["NA"], inplace=True)

# Left-join onto flat_by_sevirity
flat_by_sevirity = flat_by_sevirity.merge(
    df_DRC_SUB_GROUP,
    on="Order_ID_new_update",
    how="left"
)

# -----------------------------------------------------------------------------
# C) JOIN BACK NeoDRC_SUB_GROUP
# -----------------------------------------------------------------------------
# R: df_NeoDRC_SUB_GROUP <- dcast(src_for_flat_cln, Order_ID_new_update ~ NeoDRC_SUB_GROUP)
#     df_NeoDRC_SUB_GROUP <- df_NeoDRC_SUB_GROUP[, !names(df_NeoDRC_SUB_GROUP) %in% c("NA")]
#
#     flat_by_sevirity <- flat_by_sevirity %>%
#       left_join(df_NeoDRC_SUB_GROUP, by ="Order_ID_new_update")

df_NeoDRC_SUB_GROUP = (
    pd.pivot_table(
        src_for_flat_cln,
        index="Order_ID_new_update",
        columns="NeoDRC_SUB_GROUP",
        aggfunc="size",
        fill_value=0
    )
    .reset_index()
)

if "NA" in df_NeoDRC_SUB_GROUP.columns:
    df_NeoDRC_SUB_GROUP.drop(columns=["NA"], inplace=True)

flat_by_sevirity = flat_by_sevirity.merge(
    df_NeoDRC_SUB_GROUP,
    on="Order_ID_new_update",
    how="left"
)

# -----------------------------------------------------------------------------
# DONE. flat_by_sevirity NOW CONTAINS:
# 1) The max(diff_time_mabat_ms) per Order_ID_new_update
# 2) The wide expansions for DRC_SUB_GROUP and NeoDRC_SUB_GROUP
# -----------------------------------------------------------------------------

print(flat_by_sevirity.head(10))

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1) CREATE A NEW FACTOR 'Alert_type'
# -----------------------------------------------------------------------------
# R logic:
# flat_by_sevirity$Alert_type <- factor(ifelse( (Renal_alerts_CAT == 1 | DRC_CAT == 1 | DAM_CAT ==1 |
#                                                NeoDRC_CAT ==1 | DDI_Contraindicated_Drug_Combination_CAT ==1 ),
#                                                "Error_Alert",
#                                                ifelse( (Technical_alerts_CAT ==1 | DDI_Moderate_Interaction_CAT ==1 |
#                                                         DDI_Severe_Interaction_CAT ==1),
#                                                        "Non_Error_alert",
#                                                        "Non_alert")))
#
# If you also have DT_CAT in your data, add it to the condition for "Error_Alert" as per the commented code.
#

def determine_alert_type(row):
    """
    This function replicates the nested ifelse logic in R for setting 'Alert_type'.
    """
    if (
        (row.get("Renal_alerts_CAT") == 1)
        or (row.get("DRC_CAT") == 1)
        or (row.get("DAM_CAT") == 1)
        or (row.get("NeoDRC_CAT") == 1)
        or (row.get("DDI_Contraindicated_Drug_Combination_CAT") == 1)
    ):
        return "Error_Alert"
    elif (
        (row.get("Technical_alerts_CAT") == 1)
        or (row.get("DDI_Moderate_Interaction_CAT") == 1)
        or (row.get("DDI_Severe_Interaction_CAT") == 1)
    ):
        return "Non_Error_alert"
    else:
        return "Non_alert"

flat_by_sevirity["Alert_type"] = flat_by_sevirity.apply(determine_alert_type, axis=1)
flat_by_sevirity["Alert_type"] = flat_by_sevirity["Alert_type"].astype("category")

# -----------------------------------------------------------------------------
# 2) CREATE test_respType2 (subset of columns) & GROUP BY
# -----------------------------------------------------------------------------
# R logic:
#   test_respType2 <- flat_by_sevirity[, c("ResponseType_cat","Order_ID_new_update","Alert_type","Technical_alerts_CAT")]
#   test_respType2_gb <- test_respType2 %>% group_by(ResponseType_cat, Alert_type, Technical_alerts_CAT) %>%
#                       summarise(total_count = n())

test_respType2 = flat_by_sevirity[
    ["ResponseType_cat", "Order_ID_new_update", "Alert_type", "Technical_alerts_CAT"]
].copy()

test_respType2_gb = (
    test_respType2
    .groupby(["ResponseType_cat", "Alert_type", "Technical_alerts_CAT"], as_index=False)
    .size()  # equivalent to summarise(total_count = n())
    .rename(columns={"size": "total_count"})
)

# -----------------------------------------------------------------------------
# 3) CREATE NEW COLUMN 'Alert_status'
# -----------------------------------------------------------------------------
# R logic:
#   flat_by_sevirity$Alert_status <- factor(ifelse( (Alert_type=='Error_Alert' | Technical_alerts_CAT ==1),
#                                                   "Stoping_alert",
#                                                   ifelse( (DDI_Moderate_Interaction_CAT==1 | DDI_Severe_Interaction_CAT==1),
#                                                           "Non_stoping_alert",
#                                                           "Non_alert")))

def determine_alert_status(row):
    if row["Alert_type"] == "Error_Alert" or row.get("Technical_alerts_CAT") == 1:
        return "Stoping_alert"
    elif (row.get("DDI_Moderate_Interaction_CAT") == 1) or (row.get("DDI_Severe_Interaction_CAT") == 1):
        return "Non_stoping_alert"
    else:
        return "Non_alert"

flat_by_sevirity["Alert_status"] = flat_by_sevirity.apply(determine_alert_status, axis=1)
flat_by_sevirity["Alert_status"] = flat_by_sevirity["Alert_status"].astype("category")

#flat_by_sevirity.to_csv('alert_analysis/data_process/flat_by_sevirity_alerts.csv', index=False)
flat_by_sevirity = pd.read_csv('alert_analysis/data_process/flat_by_sevirity_alerts.csv')
assert flat_by_sevirity.shape[0] == 3737601, "The number of rows is incorrect."

# Create test_Alert_status subset & group by
test_Alert_status = flat_by_sevirity[
    ["Order_ID_new_update", "ResponseType_cat", "Alert_type", "Alert_status"]
].copy()

test_Alert_status_gb = (
    test_Alert_status
    .groupby(["ResponseType_cat", "Alert_type", "Alert_status"], as_index=False)
    .size()
    .rename(columns={"size": "total_count"})
)

# -----------------------------------------------------------------------------
# 4) FIX 'ResponseType_cat' FOR CERTAIN CONDITIONS
# -----------------------------------------------------------------------------
# R logic:
#   flat_by_sevirity$ResponseType_cat[Alert_type=='Non_alert' & Alert_status=='Non_alert'] <- "Non_alert"
#   flat_by_sevirity$ResponseType_cat[Alert_type=='Non_Error_alert' & Alert_status=='Non_stoping_alert'] <- "Non_stoping_alert"

mask_non_alert = (
    (flat_by_sevirity["Alert_type"] == "Non_alert") &
    (flat_by_sevirity["Alert_status"] == "Non_alert")
)
flat_by_sevirity.loc[mask_non_alert, "ResponseType_cat"] = "Non_alert"

mask_non_stoping_alert = (
    (flat_by_sevirity["Alert_type"] == "Non_Error_alert") &
    (flat_by_sevirity["Alert_status"] == "Non_stoping_alert")
)
flat_by_sevirity.loc[mask_non_stoping_alert, "ResponseType_cat"] = "Non_stoping_alert"

# Re-check group by
test_Alert_status = flat_by_sevirity[
    ["Order_ID_new_update","ResponseType_cat","Alert_type","Alert_status"]
].copy()

test_Alert_status_gb = (
    test_Alert_status
    .groupby(["ResponseType_cat","Alert_type","Alert_status"], as_index=False)
    .size()
    .rename(columns={"size":"total_count"})
)

# -----------------------------------------------------------------------------
# 5) RENAME COLUMNS & FILTER OUT ROWS WHERE DRC_Frequency_1, ETC. < 2
# -----------------------------------------------------------------------------
# R logic:
#   flat_by_sevirity <- flat_by_sevirity %>% rename(DRC_Frequency_1 = `DRC - Frequency 1`)
#   ...  rename(DRC_Single_Dose_1 = `DRC - Single Dose 1`)
#   ...  rename(DRC_Max_Daily_Dose_1 = `DRC - Max Daily Dose 1`)
#   flat_by_sevirity <- flat_by_sevirity[flat_by_sevirity$DRC_Frequency_1 < 2 &
#                                        flat_by_sevirity$DRC_Single_Dose_1 < 2  &
#                                        flat_by_sevirity$DRC_Max_Daily_Dose_1 < 2, ]

rename_map = {
    "DRC - Frequency 1": "DRC_Frequency_1",
    "DRC - Single Dose 1": "DRC_Single_Dose_1",
    "DRC - Max Daily Dose 1": "DRC_Max_Daily_Dose_1"
}

flat_by_sevirity.rename(columns=rename_map, inplace=True)

# Filter rows
condition_filter = (
    (flat_by_sevirity["DRC_Frequency_1"] < 2) &
    (flat_by_sevirity["DRC_Single_Dose_1"] < 2) &
    (flat_by_sevirity["DRC_Max_Daily_Dose_1"] < 2)
)
flat_by_sevirity_filtered = flat_by_sevirity[condition_filter]
flat_by_sevirity = flat_by_sevirity[condition_filter]

# -----------------------------------------------------------------------------
# 6) GROUP BY Order_ID_new_update => total_count, REMOVE DUPLICATES
# -----------------------------------------------------------------------------
# R logic:
#   unique_VALIDATION <- flat_by_sevirity %>% group_by(Order_ID_new_update) %>%
#                       summarise(total_count = n())
#   unique_to_delete <- unique_VALIDATION[unique_VALIDATION$total_count == 2, "Order_ID_new_update"]
#   test_final2 <- flat_by_sevirity[!flat_by_sevirity$Order_ID_new_update %in% unique_to_delete$Order_ID_new_update &
#                                   !is.na(flat_by_sevirity$Order_ID_new_update), ]

unique_VALIDATION = (
    flat_by_sevirity
    .groupby("Order_ID_new_update", as_index=False)
    .size()
    .rename(columns={"size": "total_count"})
)

unique_to_delete = unique_VALIDATION.loc[unique_VALIDATION["total_count"] == 2, "Order_ID_new_update"]

test_final2 = flat_by_sevirity[
    (~flat_by_sevirity["Order_ID_new_update"].isin(unique_to_delete)) &
    (flat_by_sevirity["Order_ID_new_update"].notna())
].copy()

# -----------------------------------------------------------------------------
# 7) CREATE test_hiba SUBSET (REMOVE NA in ATC_NEW), THEN GROUP & INSPECT
# -----------------------------------------------------------------------------
# R logic:
#   test_hiba <- test_final2[!is.na(test_final2$ATC_NEW), ]
#   unique_VALIDATION <- test_hiba %>% group_by(Order_ID_new_update) %>% summarise(total_count = n())

test_hiba = test_final2[test_final2["ATC_NEW"].notna()].copy()

unique_VALIDATION_hiba = (
    test_hiba
    .groupby("Order_ID_new_update", as_index=False)
    .size()
    .rename(columns={"size": "total_count"})
)

# -----------------------------------------------------------------------------
# 8) CREATE test_Time_diff SUBSET
# -----------------------------------------------------------------------------
# R logic:
#   test_Time_diff <- flat_by_sevirity[, c("Order_ID_new_update","HospitalName_EN_cat",
#                                         "ResponseType_cat","Alert_type","Alert_status",
#                                         "diff_time_mabat_ms")]

test_Time_diff = flat_by_sevirity[
    ["Order_ID_new_update",
     "HospitalName_EN_cat",
     "ResponseType_cat",
     "Alert_type",
     "Alert_status",
     "diff_time_mabat_ms"]
].copy()

# -----------------------------------------------------------------------------
# 9) WRITE OUT test_hiba TO CSV (optional)
# -----------------------------------------------------------------------------
#   write.csv(test_hiba, "C:/Users/Keren/Desktop/Fatigue_alert/Data/Main_data_2022/df_main_flat.csv", row.names=FALSE)
# In Python:
# test_hiba.to_csv(r"C:/Users/Keren/Desktop/Fatigue_alert/Data/Main_data_2022/df_main_flat.csv", index=False)

# For demonstration only (uncomment if/when you wish to write the file):
# test_hiba.to_csv("C:/Users/Keren/Desktop/Fatigue_alert/Data/Main_data_2022/df_main_flat.csv", index=False)

# -----------------------------------------------------------------------------
# DONE.
# The transformations in 'test_final2', 'test_hiba', 'test_Time_diff' mirror
# the final steps of your R code.
# -----------------------------------------------------------------------------

print("Done. Final shapes:")
print("flat_by_sevirity:", flat_by_sevirity.shape)
print("test_final2:", test_final2.shape)
print("test_hiba:", test_hiba.shape)
print("test_Time_diff:", test_Time_diff.shape)

