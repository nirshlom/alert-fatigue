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

def group_and_save_patient_data(data,
                                output_dir="alert_analysis/data/main_data_2022/",
                                file_name="grouped_patient_data.csv"):
    """
    Groups and summarizes the patient-level data from `df_active_adult` and saves the
    result as a CSV file in the specified directory.

    The data is grouped by the columns:
        - id1
        - age
        - age_cat
        - gender_text_en_cat

    And aggregates are applied as follows:
        - hospital_name_cnt: Count of unique values in 'hospital_name'
        - survival_rate_10y_age_adj_mean: Mean of 'survival_rate_10y_age_adj'
        - medical_record_cnt: Count of unique values in 'medical_record'
        - medication_orders_hospatalization_mean: Mean of 'medication_orders_hospatalization'
        - hospital_days_mean: Mean of 'hospital_days'
        - chronic_med_count_mean: Mean of 'chronic_med_count'

    After grouping, the function prints the shape of the resulting DataFrame and
    saves it as a CSV file in the provided output directory.

    Parameters:
        df_active_adult (pd.DataFrame): Input DataFrame containing patient data.
        output_dir (str): Directory in which to save the output CSV file.
        file_name (str): Name of the output CSV file.

    Returns:
        pd.DataFrame: The grouped and summarized DataFrame.
    """
    grouped_df = (
        data
        .groupby(["id1", "age", "Age_cat", "Gender_Text_EN_cat"])
        .agg(
            hospital_name_cnt=pd.NamedAgg(column="hospital_name", aggfunc=pd.Series.nunique),
            survival_rate_10y_age_adj_mean=pd.NamedAgg(column="survival_rate_10y_age_adj", aggfunc="mean"),
            survival_rate_10y_age_adj_range=pd.NamedAgg(column="survival_rate_10y_age_adj", aggfunc=my_range),
            medical_record_cnt=pd.NamedAgg(column="medical_record", aggfunc=pd.Series.nunique),
            medication_orders_hospatalization_mean=pd.NamedAgg(column="medication_orders_hospatalization", aggfunc="mean"),
            medication_orders_hospatalization_range=pd.NamedAgg(column="medication_orders_hospatalization", aggfunc=my_range),
            hospital_days_mean=pd.NamedAgg(column="hospital_days", aggfunc="mean"),
            hospital_days_range=pd.NamedAgg(column="hospital_days", aggfunc=my_range),
            chronic_med_count_mean=pd.NamedAgg(column="chronic_med_count", aggfunc="mean"),
            chronic_med_count_range=pd.NamedAgg(column="chronic_med_count", aggfunc=my_range),
            num_of_chronic_diagnosis_mean=pd.NamedAgg(column="num_of_chronic_diagnosis", aggfunc="mean"),
            num_of_chronic_diagnosis_range=pd.NamedAgg(column="num_of_chronic_diagnosis", aggfunc=my_range),
            diagnosis_count_mean=pd.NamedAgg(column="diagnosis_count", aggfunc="mean"),
            diagnosis_count_range=pd.NamedAgg(column="diagnosis_count", aggfunc=my_range),

        )
        .reset_index()
    )

    print(f'src_tbl1_active_by_patient_gb has {grouped_df.shape[0]} unique patients'
          f' and {grouped_df.shape[1]} variables')

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full file path and save the grouped DataFrame to CSV.
    file_path = os.path.join(output_dir, file_name)
    grouped_df.to_csv(file_path, index=False)

    return grouped_df


def main():
    # File paths
    input_file = "alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv"
    output_file = "alert_analysis/data/main_data_2022/df_patients_level_data.csv"
    
    # Read data
    df = read_data(input_file)
    print("\nOriginal columns:", list(df.columns))
    
    # Print summary of the data
    print_data_summary(df, "Data Summary:")
    
    # Save the DataFrame
    print(f"\nSaving data to {output_file} ...")
    df.to_csv(output_file, index=False)
    print("Data saved successfully.")


if __name__ == "__main__":
    main()