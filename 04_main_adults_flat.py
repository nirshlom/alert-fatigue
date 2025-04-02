import pandas as pd

# 1. Read the CSV file into a pandas DataFrame.
df_main_flat = pd.read_csv("/alert_analysis/data/main_data_2022/df_main_flat_py_version.csv")

# 2. Convert a list of columns to categorical data type (similar to R factors).
columns_to_convert = [
    "Hospital_cat", "HospitalName_EN_cat", "UnitName_cat", "SeverityLevelToStopOrder_cat",
    "ATC_GROUP", "ResponseType_cat", "ShiftType_cat", "DayEN_cat", "Gender_Text_EN_cat",
    "SectorText_EN_cat", "adult_child_cat", "DRC_CAT", "DT_CAT", "Technical_alerts_CAT",
    "NeoDRC_CAT", "Renal_alerts_CAT", "Alert_type", "Alert_status", "Medical_Record_cat"
]

for col in columns_to_convert:
    df_main_flat[col] = df_main_flat[col].astype('category')

# 3. Convert 'Age_cat' column to a categorical type with a specified order.
age_order = ["< 1", "1-5", "6-10", "11-15", "16-18", "19-30", "31-44", "45-55", "56-64", "65-75", "76-85", "> 85"]
df_main_flat['Age_cat'] = pd.Categorical(df_main_flat['Age_cat'], categories=age_order, ordered=True)

# 4. Print a summary of the DataFrame before dropping missing AGE_num values.
print("Data Summary before dropping missing AGE_num values:")
print(df_main_flat.describe(include='all'))

# 5. Drop rows where the 'AGE_num' column has missing values.
df_main_flat = df_main_flat.dropna(subset=['AGE_num'])

# Print summary after dropping missing values.
print("\nData Summary after dropping missing AGE_num values:")
print(df_main_flat.describe(include='all'))

# 6. Print the categories of 'ResponseType_cat' and 'ShiftType_cat'.
print("\nResponseType_cat categories:")
print(df_main_flat['ResponseType_cat'].cat.categories)
print("\nShiftType_cat categories:")
print(df_main_flat['ShiftType_cat'].cat.categories)

# 7. Calculate the load index per shift by grouping on id2, date_time_prescribe, and ShiftType_cat.
load_index = (
    df_main_flat
    .groupby(['id2', 'date_time_prescribe', 'ShiftType_cat'])
    .size()
    .reset_index(name='load_index_OrderId_Per_Shift')
    .sort_values(by=['id2', 'date_time_prescribe'])
)
print("\nLoad Index per Shift:")
print(load_index)

# --------------------------------------------------------
# Filtering DataFrame to obtain active adult records based on multiple conditions.
conditions = (
    (df_main_flat['SeverityLevelToStopOrder_cat'] != "Silence Mode") &
    (df_main_flat['adult_child_cat'] == "adult") &
    (df_main_flat['Hospital_cat'] != "243") &
    (df_main_flat['Hospital_cat'] != "113") &
    (df_main_flat['Hospital_cat'] != "29") &
    (df_main_flat['UnitName_cat'] != "Day_care") &
    (df_main_flat['UnitName_cat'] != "ICU") &
    (df_main_flat['UnitName_cat'] != "Pediatric") &
    (df_main_flat['UnitName_cat'] != "Rehabilitation")
)
df_main_active_adult = df_main_flat.loc[conditions]

# Create conditions that mimic the R filtering, ensuring NA values are excluded.
conditions = (
    df_main_flat['SeverityLevelToStopOrder_cat'].notna() & (df_main_flat['SeverityLevelToStopOrder_cat'] != "Silence Mode") &
    df_main_flat['adult_child_cat'].notna() & (df_main_flat['adult_child_cat'] == "adult") &
    df_main_flat['Hospital_cat'].notna() & (df_main_flat['Hospital_cat'] != "243") &
    (df_main_flat['Hospital_cat'] != "113") &
    (df_main_flat['Hospital_cat'] != "29") &
    df_main_flat['UnitName_cat'].notna() & (df_main_flat['UnitName_cat'].str.strip() != "Day_care") &
    (df_main_flat['UnitName_cat'] != "ICU") &
    (df_main_flat['UnitName_cat'] != "Pediatric") &
    (df_main_flat['UnitName_cat'] != "Rehabilitation")
)

# Subset the DataFrame using the conditions.
df_main_active_adult = df_main_flat.loc[conditions]

# Print a summary of the filtered active adult DataFrame.
print("\nSummary of df_main_active_adult:")
#print(df_main_active_adult.describe(include='all'))

# Change the reference (i.e., rename the categories) of UnitName_cat:
#  - Convert to categorical and drop unused categories.
#  - Then reassign new category names.
df_main_active_adult['UnitName_cat'] = df_main_active_adult['UnitName_cat'].astype('category')
df_main_active_adult['UnitName_cat'] = df_main_active_adult['UnitName_cat'].cat.remove_unused_categories()

new_categories = [
    "Internal", "Cardiology", "Emergency", "Geriatric", "Gynecology",
    "Hematology", "Internal-Covid19", "Nephrology", "Oncology", "Surgery"
]
df_main_active_adult['UnitName_cat'] = df_main_active_adult['UnitName_cat'].cat.set_categories(new_categories)

# (Optional) Print the first few rows of the filtered DataFrame.
print("\nHead of df_main_active_adult:")
print(df_main_active_adult.head())

# Save the filtered DataFrame to a CSV file.
df_main_active_adult.to_csv("alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv", index=False)