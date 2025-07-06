import pandas as pd
import os


def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file into a DataFrame."""
    print(f"Reading data from {file_path} ...")
    df = pd.read_csv(file_path)
    assert not df.empty, "DataFrame is empty after reading the file."
    print(f"Successfully read {len(df)} rows.")
    return df


def print_data_summary(df: pd.DataFrame, message: str):
    """Print summary of the DataFrame with a header message."""
    print(message)
    print(df.describe(include='all'))


def my_range(x):
    return x.max() - x.min()


def get_count_columns(df: pd.DataFrame) -> list:
    """Get all columns that end with _count from the input dataframe."""
    count_columns = [col for col in df.columns if col.endswith('_count')]
    return count_columns


def convert_unit_cat(df: pd.DataFrame, col):
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)
    return df

def group_and_save_patient_data(df: pd.DataFrame) -> None:
    """Group data by patient and save to CSV."""
    # Get all count columns
    count_columns = get_count_columns(df)
    
    # Rename "Non_alert" to "Non_alert_response" in response_type column
    df['response_type'] = df['response_type'].replace('Non_alert', 'Non_alert_response')
    
    # Check for required columns
    required_columns = [
        'id1',  # patient ID
        'gender',  # renamed from Gender_Text_EN_cat
        'age',  # renamed from AGE_num
        'age_category',  # renamed from Age_cat
        'unit_category',  # renamed from UnitName_cat
        'chronic_diagnosis',  # renamed from DiagnosisInReception
        'hospital_diagnosis',  # renamed from HospDiagnosis
        'alert_type',  # renamed from Alert_type
        'response_type',  # renamed from ResponseType_cat
        'response_reasons_other_text',  # renamed from Other_Text
        'dosing_frequency',  
        'dosing_single_dose',  
        'dosing_max_daily_dose',  
        'neo_dosing_frequency',  
        'neo_dosing_single_dose',  
        'neo_dosing_max_daily_dose',  
        'hospital_name',
        'medication_orders_hospatalization',
        'survival_rate_10y_age_adj',
        'charlson_score_age_adj',
        'hospital_days',
        'num_of_chronic_diagnosis',
    ] + count_columns  # Add all count columns
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Define unit categories
    unit_categories = [
        "Internal", "Surgery", "Gynecology", "Cardiology", "Emergency",
        "Internal-Covid19", "Geriatric", "Hematology", "Nephrology", "Oncology"
    ]
    
    # Define alert types
    alert_types = ["Non_alert", "Non_Error_alert", "Error_Alert"]
    
    # Define response types
    response_types = ["Non_alert_response", "Non_stoping_alert", "Ignore", "Change"]
    
    # Create base aggregation dictionary
    agg_dict = {
        'gender': 'first',
        'age': 'first',
        'age_category': 'first',
        'response_reasons_other_text': lambda x: '; '.join(x.dropna().unique()),
        'dosing_frequency': 'sum',
        'dosing_single_dose': 'sum',
        'dosing_max_daily_dose': 'sum',
        'neo_dosing_frequency': 'sum',
        'neo_dosing_single_dose': 'sum',
        'neo_dosing_max_daily_dose': 'sum',
        'hospital_name': lambda x: x.nunique(),  # Count unique hospitals
        'survival_rate_10y_age_adj': 'mean',
        'charlson_score_age_adj': 'mean',
        'hospital_days': 'mean'
    }
    
    # Add count columns with mean aggregation
    for col in count_columns:
        agg_dict[col] = 'mean'
    
    # First do the basic grouping
    grouped = df.groupby('id1').agg(agg_dict).reset_index()
    print(f"\nNumber of unique patients: {len(grouped)}")
    
    # Count units by type for each patient
    print("\nUnit type counts in original data:")
    print(df['unit_category'].value_counts())
    
    unit_counts = df.groupby(['id1', 'unit_category']).size().unstack(fill_value=0)
    print("\nUnit counts per patient (first 5 rows):")
    print(unit_counts.head())
    
    # Merge unit counts with grouped data
    for unit in unit_categories:
        grouped[f'unit_{unit}'] = grouped['id1'].map(unit_counts[unit])
    
    # Count alerts by type for each patient
    print("\nAlert type counts in original data:")
    print(df['alert_type'].value_counts())
    
    alert_counts = df.groupby(['id1', 'alert_type']).size().unstack(fill_value=0)
    print("\nAlert counts per patient (first 5 rows):")
    print(alert_counts.head())
    
    # Merge alert counts with grouped data
    for alert in alert_types:
        grouped[f'is_{alert}'] = grouped['id1'].map(alert_counts[alert])
    
    # Count responses by type for each patient
    print("\nResponse type counts in original data:")
    print(df['response_type'].value_counts())
    
    response_counts = df.groupby(['id1', 'response_type']).size().unstack(fill_value=0)
    print("\nResponse counts per patient (first 5 rows):")
    print(response_counts.head())
    
    # Merge response counts with grouped data
    for response in response_types:
        grouped[f'is_{response}'] = grouped['id1'].map(response_counts[response])
    
    # Fill NaN values with 0 for the new count columns
    count_cols = [col for col in grouped.columns if col.startswith(('unit_', 'is_'))]
    grouped[count_cols] = grouped[count_cols].fillna(0)
    
    #TODO: after the aggregation, convert all count_columns from get_count_columns(df)  to boolean. 0=False, >0=True. rename the columns to include _bool suffix specifically for columns with disease name in the column name.
    
    print("\nFinal counts in grouped data (first 5 rows):")
    print(grouped[['id1'] + [f'unit_{unit}' for unit in unit_categories] + [f'is_{alert}' for alert in alert_types] + [f'is_{response}' for response in response_types]].head())
    
    # Rename columns to include aggregation type
    rename_dict = {
        'dosing_frequency': 'dosing_frequency_sum',
        'dosing_single_dose': 'dosing_single_dose_sum',
        'dosing_max_daily_dose': 'dosing_max_daily_dose_sum',
        'neo_dosing_frequency': 'neo_dosing_frequency_sum',
        'neo_dosing_single_dose': 'neo_dosing_single_dose_sum',
        'neo_dosing_max_daily_dose': 'neo_dosing_max_daily_dose_sum',
        'hospital_name': 'unique_hospitals',
        'hospital_days': 'hospital_days_mean',
        'survival_rate_10y_age_adj': 'survival_rate_10y_age_adj_mean',
        'charlson_score_age_adj': 'charlson_score_age_adj_mean'
    }
    
    # Add rename mappings for count columns
    for col in count_columns:
        rename_dict[col] = f'{col}_mean'
    
    # Apply the renaming
    grouped = grouped.rename(columns=rename_dict)
    
    # Get disease-related count columns
    disease_count_columns = [col for col in get_count_columns(df) if any(disease in col.lower() for disease in [
        'liver', 'hepatic', 'cirrhosis', 'hepatitis', 'portal', 'jaundice',
        'myocardial', 'heart', 'vascular', 'cerebrovascular', 'stroke',
        'cerebral', 'tia', 'cva', 'dementia', 'pulmonary', 'copd',
        'rheumatic', 'fibromyalgia', 'gout', 'arthritis', 'ankylosing',
        'scleroderma', 'ulcer', 'diabetes', 'hemiplegia', 'hemiparesis',
        'paraplegia', 'renal', 'pyelonephritis', 'hemodialysis', 'kidney',
        'ckd', 'arf', 'aki', 'malignancy', 'malignant', 'carcinoma',
        'neoplasm', 'adenocarcinoma', 'leukemia', 'aml', 'cml', 'lymphoma',
        'hiv', 'metastatic', 'metastasis'
    ])]
    
    # Convert disease-related count columns to boolean and drop the mean columns
    for col in disease_count_columns:
        mean_col = f"{col}_mean"  # because we renamed them with _mean suffix
        bool_col = f"{col}_bool"
        grouped[bool_col] = grouped[mean_col] > 0
        grouped = grouped.drop(columns=[mean_col])  # drop the mean column

    # Convert all unit columns to binary (0/1) format
    unit_columns = [col for col in grouped.columns if col.startswith('unit_')]
    for col in unit_columns:
        grouped = convert_unit_cat(df=grouped, col=col)
    
    # Save to CSV
    output_path = 'alert_analysis/data/main_data_2022/df_patients_level_data.csv'
    grouped.to_csv(output_path, index=False)
    print(f"✅ Patient-level data saved to {output_path}")
    print(f"Shape of patient-level data: {grouped.shape}")
    
    # Assert the number of patients is correct
    assert grouped.shape[0] == 156443, f"Expected 156443 patients, got {grouped.shape[0]}"


def main():
    # File paths
    input_file = "alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv"
    output_file = "alert_analysis/data/main_data_2022/df_patients_level_data.csv"
    
    # Read data
    df = read_data(input_file)
    print("\nOriginal columns:", list(df.columns))
    
    # Print summary of the data
    #print_data_summary(df, "Data Summary:")
    
    # Save the DataFrame
    print(f"\nSaving data to {output_file} ...")
    group_and_save_patient_data(df)
    print("Data saved successfully.")


if __name__ == "__main__":
    main()