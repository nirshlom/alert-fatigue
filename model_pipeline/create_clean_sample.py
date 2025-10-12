#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
os.getcwd()
# Load data
df = pd.read_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv")

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
        df['chronic_med_count'] < 5,
        df['chronic_med_count'] >= 5
    ],
    ['0', '<5', '>=5'],
    default='Unknown'
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
    ['Small hospitals', 'Medium hospitals', 'Large hospitals'],
    default='Other'
)
#frequency table of hospital_category

freq_table = df['hospital_category'].value_counts().reset_index()
freq_table.columns = ['hospital_category', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
print(freq_table)

#create a new column called unit_category_ud
selected_unit_categories = [
    'Internal', 'Surgery', 'Gynecology', 'Cardiology', 'Emergency',
    'Internal-Covid19', 'Geriatric', 'Hematology', 'Nephrology', 'Oncology'
]
df['unit_category_ud'] = df['unit_category'].apply(lambda x: x if x in selected_unit_categories else 'OTHER')

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
        'HYPERLIPIDEMIA|DYSLIPIDEMIA', case=False, na=False),
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


# Clean and sample
df = df[df['gender'] != 'gender']  # Remove header contamination
df['alert_status_binary'] = (df['alert_status'] == 'Stoping_alert').astype(int)
df_sample = df.sample(n=int(len(df) * 0.1), random_state=42)

# Save
df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv", index=False)
print(f"Saved {len(df_sample):,} rows")
