#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

# Get the script's directory and construct path relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "alert_analysis", "data", "main_data_2022", "df_main_active_adult_renamed.csv")

# Load data
df = pd.read_csv(data_path, low_memory=False)
#hiba_new_columns

#create a new column called atc_group_ud
# Create the list of selected ATC groups for direct categories
selected_atc_groups = [
    'ANALGESICS',
    'ANTITHROMBOTIC AGENTS',
    'ANTIBACTERIALS FOR SYSTEMIC USE',
    'DRUGS USED IN DIABETES'
    ]

# Define cardiovascular ATC groups
cardiovascular_groups = [
    'CALCIUM CHANNEL BLOCKERS',
    'CARDIAC THERAPY',
    'AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM',
    'LIPID MODIFYING AGENTS',
    'ANTIHYPERTENSIVES',
    'BETA BLOCKING AGENTS',
    'DIURETICS'
]

# Function to categorize each ATC group
def classify_atc(x):
    if x in selected_atc_groups:
        return x
    elif x in cardiovascular_groups:
        return 'CARDIOVASCULAR DRUGS'
    else:
        return 'OTHER'

# Create new column
df['atc_group_ud'] = df['atc_group'].apply(classify_atc)

# Build frequency table
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



# Fix alert_rn_severity: If module_alert_rn contains DAM patterns, set alert_rn_severity to DAM
def fix_dam_severity(df,
                    col_severity="alert_rn_severity",
                    col_module="module_alert_rn"):
    """
    Fix alert_rn_severity: If module_alert_rn contains DAM patterns (e.g., "DAM - 1", "DAM - 2"),
    ensure alert_rn_severity is set to "DAM" instead of "Technical alert".
    """
    d = df.copy()
    
    if col_severity in d.columns and col_module in d.columns:
        # Check if module_alert_rn contains DAM patterns
        module_str = d[col_module].astype(str)
        dam_patterns = module_str.str.contains("DAM", case=False, na=False)
        
        # Find rows where module_alert_rn contains DAM but alert_rn_severity is not DAM
        mask_dam_module = dam_patterns & (d[col_severity] != "DAM")
        
        if mask_dam_module.any():
            # Show examples BEFORE fixing
            if 'drug_order_id' in d.columns:
                examples_before = d.loc[mask_dam_module, ['drug_order_id', col_module, col_severity]].head(10)
                print(f"Found {mask_dam_module.sum()} rows where module_alert_rn contains DAM but alert_rn_severity is not DAM:")
                print(examples_before.to_string())
            
            # Set alert_rn_severity to DAM for these rows
            num_fixed = mask_dam_module.sum()
            d.loc[mask_dam_module, col_severity] = "DAM"
            print(f"\nFixed {num_fixed} rows: Changed alert_rn_severity to 'DAM'")
    
    return d

# Fix DAM severity misclassifications
df = fix_dam_severity(df)

#create a new column called response_type_ud
def build_response_type_ud(df,
                           col_response_type="response_type",
                           col_alert_msg="alert_message",
                           col_severity="alert_rn_severity",
                           col_rrc="response_reasons_codes",
                           out_col="response_type_ud"):
    """
    Create response_type_ud categorical column based on response type, alert message,
    severity, and response reasons codes.
    
    Categories:
    - "Change": ResponseType contains "change" OR (alert_type=Error_Alert AND response_reasons_codes=null)
    - "No_response_need": Severe/moderate severity OR alert message empty
    - "Ignore": Specific severities with numeric response_reasons_codes
    - "Change": Specific severities with empty response_reasons_codes
    - "No_response_fit": Default fallback
    """
    d = df.copy()

    # --- helpers ---
    def norm_series(s):
        return s.astype(str).str.strip()

    # normalize main columns
    rt = norm_series(d[col_response_type]) if col_response_type in d.columns else pd.Series([""]*len(d), index=d.index)
    sev = norm_series(d[col_severity]) if col_severity in d.columns else pd.Series([""]*len(d), index=d.index)

    # Alert message empty?
    if col_alert_msg in d.columns:
        am = d[col_alert_msg]
        am_str = norm_series(am)
        alert_empty = am.isna() | am_str.eq('') | am_str.str.lower().isin(['nan', 'none'])
    else:
        alert_empty = pd.Series([True]*len(d), index=d.index)  # if missing column, treat as empty

    # response_reasons_codes numeric?
    if col_rrc in d.columns:
        rrc = d[col_rrc]
        rrc_str = norm_series(rrc)
        rrc_empty = rrc.isna() | rrc_str.eq('') | rrc_str.str.lower().isin(['nan', 'none'])
        rrc_num = pd.to_numeric(rrc_str, errors="coerce")  # will be NaN if not numeric
        rrc_is_number = rrc_num.notna()
    else:
        rrc_empty = pd.Series([True]*len(d), index=d.index)
        rrc_is_number = pd.Series([False]*len(d), index=d.index)

    # --- init ---
    d[out_col] = "No_response_fit"

    # rule 1: Change in original response type
    mask_change_text = rt.str.contains("change", case=False, na=False)
    d.loc[mask_change_text, out_col] = "Change"

    # rule 2: severity severe/moderate/undetermined-alt therapy => No_response_need
    no_need_sev = {
        "DDI-Severe Interaction",
        "DDI-Moderate Interaction",
        "DDI-Undetermined Severity - Alternative Therapy Interaction"
    }
    mask_no_need_sev = sev.isin(no_need_sev) & ~mask_change_text
    d.loc[mask_no_need_sev, out_col] = "No_response_need"

    # rule 3: alert message null/empty => No_response_need (unless already Change)
    mask_no_need_alert_empty = alert_empty & ~mask_change_text & ~mask_no_need_sev
    d.loc[mask_no_need_alert_empty, out_col] = "No_response_need"

    # rule 4: specific severities => Ignore if rrc number else Change if rrc empty
    # IMPORTANT: If rrc is numeric, it should be "Ignore" even if response_type contains "change"
    sev_group = {
        "DAM", "DT", "DRC", "NeoDRC",
        "Technical alert",
        "DDI-Contraindicated Drug Combination",
        "Renal alerts"
    }
    base_mask = sev.isin(sev_group) & ~mask_no_need_sev

    # Priority: If rrc is numeric, set to "Ignore" (even if response_type contains "change")
    mask_ignore = base_mask & rrc_is_number
    d.loc[mask_ignore, out_col] = "Ignore"

    # Then: If rrc is empty and not already set to Ignore, set to "Change"
    mask_change_by_rules = base_mask & rrc_empty & (d[out_col] != "Ignore")
    d.loc[mask_change_by_rules, out_col] = "Change"

    # rule 5: Change where alert_type=Error_Alert and response_reasons_codes=null
    # This should not override "Ignore" if rrc is numeric
    if 'alert_type' in d.columns:
        mask_error_alert_rrc_null = (
            (d['alert_type'] == "Error_Alert") & 
            rrc_empty & 
            (d[out_col] != "Ignore") &
            (d[out_col] != "No_response_need")
        )
        d.loc[mask_error_alert_rrc_null, out_col] = "Change"

    # Convert to categorical dtype
    d[out_col] = d[out_col].astype('category')
    
    return d

# Apply the function to create response_type_ud
df = build_response_type_ud(df)

# Fix NULL alert_rn_severity for DRC - Message patterns
def fix_null_drc_message_severity(df,
                                   col_severity="alert_rn_severity",
                                   col_module="module_alert_rn"):
    """Fix NULL alert_rn_severity: If module_alert_rn contains 'DRC - Message', set to 'Technical alert'."""
    d = df.copy()
    
    if col_severity in d.columns and col_module in d.columns:
        # Check if module_alert_rn contains DRC - Message patterns
        module_str = d[col_module].astype(str)
        drc_message_patterns = module_str.str.contains("DRC.*Message", case=False, na=False, regex=True)
        
        # Find rows where module contains DRC - Message but severity is NULL
        mask_drc_message_null = drc_message_patterns & d[col_severity].isna()
        
        if mask_drc_message_null.any():
            num_fixed = mask_drc_message_null.sum()
            d.loc[mask_drc_message_null, col_severity] = "Technical alert"
            print(f"Fixed {num_fixed} rows: Set NULL alert_rn_severity to 'Technical alert' for DRC - Message patterns")
    
    return d

# Fix NULL severity for DRC - Message
df = fix_null_drc_message_severity(df)

# Fill dosing direction columns for DRC 2 and 3 patterns
def fill_drc_dosing_directions(df,
                               col_subgroup="dosing_subgroup",
                               col_alert_msg="alert_message"):
    """Fill dosing direction columns for DRC patterns 2 and 3 based on dosing_subgroup and alert_message."""
    d = df.copy()
    
    if col_subgroup not in d.columns or col_alert_msg not in d.columns:
        return d
    
    # Ensure alert_message is string
    d[col_alert_msg] = d[col_alert_msg].astype(str)
    
    # Check if direction columns exist, if not create them
    if 'dosing_frequency_direction' not in d.columns:
        d['dosing_frequency_direction'] = None
    if 'dosing_max_daily_dose_direction' not in d.columns:
        d['dosing_max_daily_dose_direction'] = None
    if 'dosing_single_dose_direction' not in d.columns:
        d['dosing_single_dose_direction'] = None
    
    # DRC - Frequency 2, 3
    mask_freq_2_3 = d[col_subgroup].isin(["DRC - Frequency 2", "DRC - Frequency 3"])
    d.loc[mask_freq_2_3 & d[col_alert_msg].str.contains("exceeds", case=False, na=False), 'dosing_frequency_direction'] = "exceeds"
    d.loc[mask_freq_2_3 & d[col_alert_msg].str.contains("below", case=False, na=False), 'dosing_frequency_direction'] = "below"
    
    # DRC - Max Daily Dose 2, 3
    mask_max_2_3 = d[col_subgroup].isin(["DRC - Max Daily Dose 2", "DRC - Max Daily Dose 3"])
    d.loc[mask_max_2_3 & d[col_alert_msg].str.contains("exceeds", case=False, na=False), 'dosing_max_daily_dose_direction'] = "exceeds"
    d.loc[mask_max_2_3 & d[col_alert_msg].str.contains("below", case=False, na=False), 'dosing_max_daily_dose_direction'] = "below"
    
    # DRC - Single Dose 2, 3
    mask_single_2_3 = d[col_subgroup].isin(["DRC - Single Dose 2", "DRC - Single Dose 3"])
    d.loc[mask_single_2_3 & d[col_alert_msg].str.contains("exceeds", case=False, na=False), 'dosing_single_dose_direction'] = "exceeds"
    d.loc[mask_single_2_3 & d[col_alert_msg].str.contains("below", case=False, na=False), 'dosing_single_dose_direction'] = "below"
    
    print("Filled dosing direction columns for DRC patterns 2 and 3")
    return d

# Fill dosing directions for DRC 2 and 3
df = fill_drc_dosing_directions(df)

# Re-check and fix alert_type and alert_status
def fix_alert_type_and_status(df):
    """Re-check and fix alert_type and alert_status based on updated rules."""
    d = df.copy()
    
    # Ensure required columns exist
    required_cols = ['alert_message', 'alert_rn_severity']
    missing_cols = [col for col in required_cols if col not in d.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, skipping alert_type/status fixes")
        return d
    
    # Check if alert_message is empty/null
    alert_msg_str = d['alert_message'].astype(str).str.strip()
    alert_empty = d['alert_message'].isna() | (alert_msg_str == '') | alert_msg_str.str.lower().isin(['nan', 'none'])
    
    # Normalize severity
    sev = d['alert_rn_severity'].astype(str).str.strip()
    
    # Fix alert_type
    if 'alert_type' in d.columns:
        # Non_alert where alert_message is null
        d.loc[alert_empty, 'alert_type'] = "Non_alert"
        
        # Error_Alert where alert_rn_severity = DRC OR NeoDRC OR DAM OR DT OR Renal alerts OR DDI-Contraindicated Drug Combination
        error_severities = {
            "DRC", "NeoDRC", "DAM", "DT", "Renal alerts",
            "DDI-Contraindicated Drug Combination"
        }
        # Also check for DDI-Contraindicated pattern
        mask_error = sev.isin(error_severities) | sev.str.startswith("DDI-Contraindicated", na=False)
        d.loc[mask_error & ~alert_empty, 'alert_type'] = "Error_Alert"
        
        # Non_Error_alert where DDI-Severe Interaction OR DDI-Moderate Interaction OR DDI-Undetermined Severity OR Technical alert
        non_error_severities = {
            "DDI-Severe Interaction",
            "DDI-Moderate Interaction",
            "DDI-Undetermined Severity - Alternative Therapy Interaction",
            "Technical alert"
        }
        # Also check for DDI patterns
        mask_non_error = sev.isin(non_error_severities) | (
            sev.str.startswith("DDI-", na=False) & 
            ~sev.str.startswith("DDI-Contraindicated", na=False)
        )
        d.loc[mask_non_error & ~alert_empty, 'alert_type'] = "Non_Error_alert"
        
        d['alert_type'] = d['alert_type'].astype('category')
        print("Fixed alert_type based on updated rules")
    
    # Fix alert_status
    if 'alert_status' in d.columns:
        # Non_alert where alert_message is null
        d.loc[alert_empty, 'alert_status'] = "Non_alert"
        
        # Stopping_alert where alert_rn_severity = DRC OR NeoDRC OR DAM OR DT OR Renal alerts OR DDI-Contraindicated OR Technical alert
        stopping_severities = {
            "DRC", "NeoDRC", "DAM", "DT", "Renal alerts",
            "DDI-Contraindicated Drug Combination", "Technical alert"
        }
        mask_stopping = sev.isin(stopping_severities) | sev.str.startswith("DDI-Contraindicated", na=False)
        d.loc[mask_stopping & ~alert_empty, 'alert_status'] = "Stopping_alert"
        
        # Non_stopping_alert where DDI-Severe Interaction OR DDI-Moderate Interaction OR DDI-Undetermined Severity
        non_stopping_severities = {
            "DDI-Severe Interaction",
            "DDI-Moderate Interaction",
            "DDI-Undetermined Severity - Alternative Therapy Interaction"
        }
        mask_non_stopping = sev.isin(non_stopping_severities) | (
            sev.str.startswith("DDI-", na=False) & 
            ~sev.str.startswith("DDI-Contraindicated", na=False)
        )
        d.loc[mask_non_stopping & ~alert_empty, 'alert_status'] = "Non_stopping_alert"
        
        d['alert_status'] = d['alert_status'].astype('category')
        print("Fixed alert_status based on updated rules")
    
    return d

# Fix alert_type and alert_status
df = fix_alert_type_and_status(df)

# Re-apply response_type_ud rule 5 after alert_type is fixed
# Rule 5: Change where alert_type=Error_Alert and response_reasons_codes=null
if 'alert_type' in df.columns and 'response_reasons_codes' in df.columns:
    rrc_str = df['response_reasons_codes'].astype(str).str.strip()
    rrc_empty = df['response_reasons_codes'].isna() | (rrc_str == '') | rrc_str.str.lower().isin(['nan', 'none'])
    
    mask_error_alert_rrc_null = (
        (df['alert_type'] == "Error_Alert") & 
        rrc_empty & 
        (df['response_type_ud'] != "Ignore") &
        (df['response_type_ud'] != "No_response_need")
    )
    df.loc[mask_error_alert_rrc_null, 'response_type_ud'] = "Change"
    if mask_error_alert_rrc_null.any():
        print(f"Applied rule 5: Set {mask_error_alert_rrc_null.sum()} rows to 'Change' where alert_type=Error_Alert and response_reasons_codes=null")

#frequency table of response_type_ud
freq_table = df['response_type_ud'].value_counts().reset_index()
freq_table.columns = ['response_type_ud', 'count']
freq_table['percent'] = (freq_table['count'] / freq_table['count'].sum() * 100).round(2)
print(freq_table)

# new Clean and sample with non-merged columns for binary classification=0 (alert_type_binary and alert_status_binary)
df['alert_type_binary'] = df['alert_type'].apply(
    lambda x: 1 if x == 'Error_Alert'
              else 0 if x == 'Non_alert'
              else pd.NA
)

df['alert_status_binary'] = df['alert_status'].apply(
    lambda x: 1 if x == 'Stopping_alert'
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
                     'DRUGS USED IN DIABETES', 'CARDIOVASCULAR DRUGS', 'OTHER'],
    'prescription_day': ['WEEKEND', 'WEEKDAY'],
    'chronic_med_ud': ['0', '<5', '>=5'],
    'hospital_category': ['Small hospital', 'Medium hospital', 'Large hospital', 'Other'],
    'response_type_ud': ['Change', 'No_response_need', 'Ignore', 'No_response_fit'],
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
        validation_errors.append(f"[ERROR] Column '{col}' is missing")
        continue
    
    # For binary columns, check for NA values separately
    if col in ['alert_type_binary', 'alert_status_binary']:
        unique_vals = set(df[col].dropna().unique())
        unexpected = unique_vals - set(expected)
        if unexpected:
            validation_errors.append(f"[ERROR] Column '{col}' has unexpected values: {unexpected}")
        else:
            print(f"[OK] {col}: Valid values found")
    else:
        unique_vals = set(df[col].dropna().unique())
        unexpected = unique_vals - set(expected)
        if unexpected:
            validation_errors.append(f"[ERROR] Column '{col}' has unexpected values: {unexpected}")
        else:
            print(f"[OK] {col}: Valid values found")
    
    # Check for missing values (except binary alert columns which can have NA)
    if col not in ['alert_type_binary', 'alert_status_binary']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            validation_errors.append(f"[ERROR] Column '{col}' has {missing_count:,} missing values")

# Check binary disease columns
for col in binary_disease_cols:
    if col not in df.columns:
        validation_errors.append(f"[ERROR] Column '{col}' is missing")
        continue
    
    unique_vals = set(df[col].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        unexpected = unique_vals - {0, 1}
        validation_errors.append(f"[ERROR] Column '{col}' has unexpected values: {unexpected}")
    else:
        print(f"[OK] {col}: Valid binary values (0, 1)")
    
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        validation_errors.append(f"[ERROR] Column '{col}' has {missing_count:,} missing values")

# Check unit_category_ud - this one is more complex, just check it exists and has no missing values
if 'unit_category_ud' in df.columns:
    missing_count = df['unit_category_ud'].isna().sum()
    if missing_count > 0:
        validation_errors.append(f"[ERROR] Column 'unit_category_ud' has {missing_count:,} missing values")
    else:
        print(f"[OK] unit_category_ud: No missing values")
        print(f"  Unique values: {df['unit_category_ud'].nunique()} categories")
else:
    validation_errors.append(f"[ERROR] Column 'unit_category_ud' is missing")

# Summary
print("\n" + "="*60)
if validation_errors:
    print("VALIDATION FAILED - Found the following errors:")
    for error in validation_errors:
        print(f"  {error}")
    print("\n[ERROR] Cannot save file due to validation errors. Please fix the issues above.")
    raise ValueError("Validation failed. See errors above.")
else:
    print("[OK] All categorical parameters are properly defined!")
    print("="*60 + "\n")

# Drop all rows where alert_rn_severity = "NeoDRC" (adult-only data) i found some NeoDRC rows in the data because soroka hospital data was not good
if 'alert_rn_severity' in df.columns:
    rows_before = len(df)
    mask_neodrc = df['alert_rn_severity'] == "NeoDRC"
    if mask_neodrc.any():
        num_dropped = mask_neodrc.sum()
        df = df[~mask_neodrc].copy()
        rows_after = len(df)
        print(f"Dropped {num_dropped:,} rows where alert_rn_severity = 'NeoDRC' (adult-only data)")
        print(f"Rows before: {rows_before:,}, Rows after: {rows_after:,}")
    else:
        print("No NeoDRC rows found to drop")

# Drop rows where alert_status = "Stoping_alert" OR dosing_message=2 OR response_type_ud="No_response_fit"
rows_before = len(df)
required_cols = ['alert_status', 'dosing_message', 'response_type_ud']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"Warning: Missing columns {missing_cols}, skipping row drop for alert_status/dosing_message/response_type_ud")
else:
    # Build mask with OR logic - drop if ANY condition is true
    mask_to_drop = pd.Series([False] * len(df), index=df.index)
    
    # Condition 1: alert_status = "Stoping_alert"
    if 'alert_status' in df.columns:
        mask_to_drop |= df['alert_status'].isin(['Stoping_alert'])
    
    # Condition 2: dosing_message = 2
    if 'dosing_message' in df.columns:
        mask_to_drop |= (df['dosing_message'] == 2)
    
    # Condition 3: response_type_ud = "No_response_fit"
    if 'response_type_ud' in df.columns:
        mask_to_drop |= (df['response_type_ud'] == 'No_response_fit')
    
    if mask_to_drop.any():
        num_dropped = mask_to_drop.sum()
        df = df[~mask_to_drop].copy()
        rows_after = len(df)
        print(f"Dropped {num_dropped:,} rows where alert_status='Stoping_alert' OR dosing_message=2 OR response_type_ud='No_response_fit'")
        print(f"Rows before: {rows_before:,}, Rows after: {rows_after:,}")
    else:
        print("No rows found matching the criteria (alert_status='Stoping_alert' OR dosing_message=2 OR response_type_ud='No_response_fit')")

df_sample = df.sample(n=int(len(df) * 1.0), random_state=42)
#df_sample = df.sample(n=int(len(df) * 0.10), random_state=42)

# Save 10% sample
#df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_new_clean_sample_10pct.csv", index=False)
#print(f"Saved {len(df_sample):,} rows")

# save 100% sample
df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_new_clean_sample_100pct.csv", index=False)
print(f"Saved {len(df_sample):,} rows")