#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
os.getcwd()
# os.chdir(r'/Users/eveadam/Dropbox/PHD/alert-fatigue/model_pipeline')
# Load data
df = pd.read_csv("../alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv")
#hiba_new_columns

#create a new column called atc_group_ud
selected_atc_groups = [
    'ANALGESICS',
    'ANTITHROMBOTIC AGENTS',
    'ANTIBACTERIALS FOR SYSTEMIC USE',
    'DRUGS USED IN DIABETES',
    'PSYCHOLEPTICS'
]
df['atc_group_ud'] = df['atc_group'].apply(lambda x: x if x in selected_atc_groups else 'OTHER')

#frequency table of atc_group_ud
freq_table = df['atc_group_ud'].value_counts().reset_index()
freq_table.columns = ['atc_group_ud', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)

print(freq_table)


#create a new column called prescription_day

df['prescription_day'] = np.where(df['day'].str.capitalize().isin(['Friday', 'Saturday']), 'WEEKEND', 'WEEKDAY')

#create a new column called chronic_med_ud

df['chronic_med_ud'] = np.select(
    [
        df['chronic_med_count'] == 0,
        (df['chronic_med_count'] > 0) & (df['chronic_med_count'] < 5),
        df['chronic_med_count'] >= 5
    ],
    ['0', '<5', '>=5'],
    default='0'
) 

#frequency table of chronic_med_ud
freq_table = df['chronic_med_ud'].value_counts().reset_index()
freq_table.columns = ['chronic_med_ud', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
print(freq_table)

#create a new column called hospital_category

df['hospital_category'] = np.select(
    [
        df['hospital_code'].isin([19, 22, 237]),
        df['hospital_code'].isin([23, 26, 28]),
        df['hospital_code'].isin([20, 24])
    ],
    ['Small hospital', 'Medium hospital', 'Large hospital'],
    default='Other'
)
#frequency table of hospital_category

freq_table = df['hospital_category'].value_counts().reset_index()
freq_table.columns = ['hospital_category', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
print(freq_table)

#create a new column called unit_category_ud
other_units_categories = [
     'Gynecology', 'Cardiology',
     'Geriatric', 'Hematology', 'Nephrology', 'Oncology'
]
internal_categories = ['Internal', 'Internal-Covid19']
df['unit_category_ud'] = df['unit_category'].apply(lambda x: 'OTHER' if x in other_units_categories else  'new_internal' if x in internal_categories else x)

#frequency table of unit_category_ud
freq_table = df['unit_category_ud'].value_counts().reset_index()
freq_table.columns = ['unit_category_ud', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
print(freq_table)

#create new colums for co-morbidities: chronic_disease_group_ud

df['kidney_disease'] = np.where(df['renal_group_count'] > 0, 1, 0)
df['hepatic_disease'] = np.where(df['liver_group_count'] > 0, 1, 0)
df['diabetes_disease'] = np.where(df['diabetes_count'] > 0, 1, 0)
df['ischemic_heart_disease'] = np.where(
    (df['myocardial_count'] > 1) |
    (df['hospital_diagnosis'].str.contains('ISCHEMIC HEART DISEASE', case=False, na=False)),
    1,
    0
)

df['copd_disease'] = np.where(df['copd_group_count'] > 0, 1, 0)
df['cerebrovascular_disease'] = np.where(df['cerebrovascular_group_count'] > 0, 1, 0)
df['peptic_ulcer_disease'] = np.where(df['ulcer_group_count'] > 0, 1, 0)
df['dementia_disease'] = np.where(df['dementia_count'] > 0, 1, 0)

df['oncological_disease'] = np.where(
    (df['malignancy_group_count'] > 0) |
    (df['metastatic_group_count'] > 0) |
    (df['hospital_diagnosis'].str.contains('CANCER', case=False, na=False)),
    1,
    0
)

df['hemato_oncological_disease'] = np.where(
    (df['leukemia_group_count'] > 0) |
    (df['lymphoma_count'] > 0) |
    (df['hospital_diagnosis'].str.contains('LYMPHOMA|MULTIPLE MYELOMA', case=False, na=False)),
    1,
    0
)

df['hypertension_disease'] = np.where(
    df['hospital_diagnosis'].str.contains('ESSENTIAL HYPERTENSION|HYPERTENSION', case=False, na=False),
    1,
    0
)

df['atrial_fibrillation_disease'] = np.where(
    df['hospital_diagnosis'].str.contains(
        'ATRIAL FIBRILLATION|PAF|PAROXYSMAL ATRIAL FIBRILLATION',
        case=False,
        na=False
    ),
    1,
    0
)


df['hyperlipidemia_disease'] = np.where(
    df['hospital_diagnosis'].str.contains(
        'HYPERLIPIDEMIA|DYSLIPIDEMIA|HYPERCHOLESTEROL', case=False, na=False),
    1,
    0
)


df['congestive_heart_failure_disease'] = np.where(
    df['hospital_diagnosis'].str.contains(
        'CONGESTIVE HEART FAILURE|CHF|HEART FAILURE',
        case=False,
        na=False
    ),
    1,
    0
)

df['obesity_disease'] = np.where(
    df['hospital_diagnosis'].str.contains('OBESITY|MORBID OBESITY', case=False, na=False),
    1,
    0
)


# 1️⃣ Count number of orders per shift per ID2 - not used for now to measure doctor_workload ( לא  מתחשב בתאריך ההוראה)
#df['doctor_workload'] = df.groupby(['id2', 'shift_type'])['order_id'].transform('count')


# oldClean and sample with merged columns for binary classification=0 
#df = df[df['gender'] != 'gender']  # Remove header contamination
#df['alert_status_binary'] = (df['alert_status'] == 'Stoping_alert').astype(int)
#df['alert_type_binary'] = (df['alert_type'] == 'Error_Alert').astype(int)
#df_sample = df.sample(n=int(len(df) * 1.0), random_state=42)
#df_sample = df.sample(n=int(len(df) * 0.10), random_state=42)

# new Clean and sample with non-merged columns for binary classification=0 (alert_type_binary and alert_status_binary)
df['alert_type_binary'] = df['alert_type'].apply(
    lambda x: 1 if x == 'Error_Alert'
              else 0 if x == 'Non_alert'
              else pd.NA
)

df['alert_status_binary'] = df['alert_status'].apply(
    lambda x: 1 if x == 'Stoping_alert'
              else 0 if x == 'Non_alert'
              else pd.NA
)

#before saving check if all categorical parameters are defined correctly (if makes problem to remove it)
print("\n" + "="*60)
print("VALIDATING CATEGORICAL PARAMETERS")
print("="*60)

validation_errors = []

# Define expected values for each categorical column
expected_values = {
    'atc_group_ud': ['ANALGESICS', 'ANTITHROMBOTIC AGENTS', 'ANTIBACTERIALS FOR SYSTEMIC USE', 
                     'DRUGS USED IN DIABETES', 'PSYCHOLEPTICS', 'OTHER'],
    'prescription_day': ['WEEKEND', 'WEEKDAY'],
    'chronic_med_ud': ['0', '<5', '>=5'],
    'hospital_category': ['Small hospital', 'Medium hospital', 'Large hospital', 'Other'],
    'alert_type_binary': [0, 1],  # Can also have NA
    'alert_status_binary': [0, 1],  # Can also have NA
}

# Binary disease columns (should be 0 or 1)
binary_disease_cols = [
    'kidney_disease', 'hepatic_disease', 'diabetes_disease', 'ischemic_heart_disease',
    'copd_disease', 'cerebrovascular_disease', 'peptic_ulcer_disease', 'dementia_disease',
    'oncological_disease', 'hemato_oncological_disease', 'hypertension_disease',
    'atrial_fibrillation_disease', 'hyperlipidemia_disease', 'congestive_heart_failure_disease',
    'obesity_disease'
]

# Check categorical columns with expected values
for col, expected in expected_values.items():
    if col not in df.columns:
        validation_errors.append(f"❌ Column '{col}' is missing")
        continue
    
    # For binary columns, check for NA values separately
    if col in ['alert_type_binary', 'alert_status_binary']:
        unique_vals = set(df[col].dropna().unique())
        unexpected = unique_vals - set(expected)
        if unexpected:
            validation_errors.append(f"❌ Column '{col}' has unexpected values: {unexpected}")
        else:
            print(f"✓ {col}: Valid values found")
    else:
        unique_vals = set(df[col].dropna().unique())
        unexpected = unique_vals - set(expected)
        if unexpected:
            validation_errors.append(f"❌ Column '{col}' has unexpected values: {unexpected}")
        else:
            print(f"✓ {col}: Valid values found")
    
    # Check for missing values (except binary alert columns which can have NA)
    if col not in ['alert_type_binary', 'alert_status_binary']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            validation_errors.append(f"❌ Column '{col}' has {missing_count:,} missing values")

# Check binary disease columns
for col in binary_disease_cols:
    if col not in df.columns:
        validation_errors.append(f"❌ Column '{col}' is missing")
        continue
    
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        unexpected = unique_vals - {0, 1}
        validation_errors.append(f"❌ Column '{col}' has unexpected values: {unexpected}")
    else:
        print(f"✓ {col}: Valid binary values (0, 1)")
    
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        validation_errors.append(f"❌ Column '{col}' has {missing_count:,} missing values")

# Check unit_category_ud - this one is more complex, just check it exists and has no missing values
if 'unit_category_ud' in df.columns:
    missing_count = df['unit_category_ud'].isna().sum()
    if missing_count > 0:
        validation_errors.append(f"❌ Column 'unit_category_ud' has {missing_count:,} missing values")
    else:
        print(f"✓ unit_category_ud: No missing values")
        print(f"  Unique values: {df['unit_category_ud'].nunique()} categories")
else:
    validation_errors.append(f"❌ Column 'unit_category_ud' is missing")

# Summary
print("\n" + "="*60)
if validation_errors:
    print("VALIDATION FAILED - Found the following errors:")
    for error in validation_errors:
        print(f"  {error}")
    print("\n❌ Cannot save file due to validation errors. Please fix the issues above.")
    raise ValueError("Validation failed. See errors above.")
else:
    print("✓ All categorical parameters are properly defined!")
    print("="*60 + "\n")

df_sample = df.sample(n=int(len(df) * 1.0), random_state=42)
#df_sample = df.sample(n=int(len(df) * 0.10), random_state=42)

# Save 10% sample
#df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_new_clean_sample_10pct.csv", index=False)
#print(f"Saved {len(df_sample):,} rows")

# Save 100% sample
df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_new_clean_sample_100pct.csv", index=False)
print(f"Saved {len(df_sample):,} rows")