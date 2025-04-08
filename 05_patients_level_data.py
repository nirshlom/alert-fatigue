import pandas as pd
import os


def read_active_adult_data(file_path='alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv'):
    """
    Reads the active adult data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing the active adult data.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)


def group_and_save_patient_data(data,
                                output_dir="alert_analysis/data/main_data_2022/",
                                file_name="grouped_patient_data.csv"):
    """
    Groups and summarizes the patient-level data from `df_active_adult` and saves the
    result as a CSV file in the specified directory.

    The data is grouped by the columns:
        - id1
        - age_num
        - age_cat
        - gender_text_en_cat

    And aggregates are applied as follows:
        - hospitalname_en_cat_cnt: Count of unique values in 'hospitalname_en_cat'
        - survivalrate10years_age_adj_mean: Mean of 'survivalrate10years_age_adj'
        - medical_record_cat_cnt: Count of unique values in 'medical_record'
        - nummedamount_calc_mean: Mean of 'nummedamount_calc'
        - hosp_days_mean: Mean of 'hosp_days'
        - chronic_num_calc_mean: Mean of 'chronic_num_calc'

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
        .groupby(["id1", "age_num", "age_cat", "gender_text_en_cat"])
        .agg(
            hospitalname_en_cat_cnt=pd.NamedAgg(column="HospitalName_en_cat", aggfunc=pd.Series.nunique),
            survivalrate10years_age_adj_mean=pd.NamedAgg(column="SurvivalRate10years_age_adj", aggfunc="mean"),
            medical_record_cat_cnt=pd.NamedAgg(column="Medical_Record", aggfunc=pd.Series.nunique),
            nummedamount_calc_mean=pd.NamedAgg(column="NumMedAmount_calc", aggfunc="mean"),
            hosp_days_mean=pd.NamedAgg(column="hosp_days", aggfunc="mean"),
            chronic_num_calc_mean=pd.NamedAgg(column="chronic_num_calc", aggfunc="mean")
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

if __name__ == "__main__":
    # Read the active adult data
    df_active_adult = read_active_adult_data()

    # Group and save the patient data
    grouped_patient_df = group_and_save_patient_data(df_active_adult)