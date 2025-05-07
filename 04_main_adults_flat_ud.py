import pandas as pd

def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file into a DataFrame."""
    print(f"Reading data from {file_path} ...")
    df = pd.read_csv(file_path)
    assert not df.empty, "DataFrame is empty after reading the file."
    print(f"Successfully read {len(df)} rows.")
    return df

def convert_columns_to_category(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Convert specified columns to the categorical dtype."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
        else:
            print(f"Warning: Column {col} not found in DataFrame.")
    return df

def convert_age_cat(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """Convert an age column to an ordered categorical with a specified order."""
    if col in df.columns:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    else:
        print(f"Warning: {col} column not found in DataFrame.")
    return df

def print_data_summary(df: pd.DataFrame, message: str):
    """Print summary of the DataFrame with a header message."""
    print(message)
    print(df.describe(include='all'))

def drop_missing_age(df: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """Drop rows with missing values in the specified age column."""
    before_drop = len(df)
    df = df.dropna(subset=[age_col])
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows with missing {age_col} values.")
    # Assert that no missing values remain.
    assert df[age_col].isna().sum() == 0, f"Missing values remain in {age_col}"
    return df

def print_category_info(df: pd.DataFrame, col: str):
    """Print the categories of a given categorical column."""
    if col in df.columns:
        print(f"\nCategories for {col}:")
        print(df[col].cat.categories)
    else:
        print(f"Warning: {col} column not found in DataFrame.")

def compute_load_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the load index per shift by grouping the data."""
    print("\nComputing load index per shift...")
    load_index = (
        df.groupby(['id2', 'date_time_prescribe', 'ShiftType_cat'])
          .size()
          .reset_index(name='load_index_OrderId_Per_Shift')
          .sort_values(by=['id2', 'date_time_prescribe'])
    )
    print("Load Index per Shift:")
    print(load_index)
    return load_index

def filter_active_adult(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to obtain active adult records based on:
      - SeverityLevelToStopOrder_cat not "Silence Mode"
      - adult_child_cat equals "adult"
      - Hospital_cat not equal to "243", "113", or "29"
      - UnitName_cat not equal to "Day_care", "ICU", "Pediatric", or "Rehabilitation"
    NA values are excluded from each condition.
    """
    print("\nFiltering active adult records...")
    conditions = (
        df['SeverityLevelToStopOrder_cat'].notna() & (df['SeverityLevelToStopOrder_cat'] != "Silence Mode") &
        df['adult_child_cat'].notna() & (df['adult_child_cat'] == "adult") &
        df['Hospital_cat'].notna() & (df['Hospital_cat'] != "243") &
        (df['Hospital_cat'] != "113") &
        (df['Hospital_cat'] != "29") &
        df['UnitName_cat'].notna() & (df['UnitName_cat'].str.strip() != "Day_care") &
        (df['UnitName_cat'] != "ICU") &
        (df['UnitName_cat'] != "Pediatric") &
        (df['UnitName_cat'] != "Rehabilitation")
    )
    df_filtered = df.loc[conditions].copy()
    print(f"Filtered data contains {len(df_filtered)} rows.")
    return df_filtered

def update_unitname_category(df: pd.DataFrame, new_categories: list) -> pd.DataFrame:
    """
    Update the UnitName_cat column by:
      - Converting to categorical (if not already)
      - Removing unused categories
      - Setting new category levels
    """
    if 'UnitName_cat' in df.columns:
        df['UnitName_cat'] = df['UnitName_cat'].astype('category')
        df['UnitName_cat'] = df['UnitName_cat'].cat.remove_unused_categories()
        df['UnitName_cat'] = df['UnitName_cat'].cat.set_categories(new_categories)
    else:
        print("Warning: UnitName_cat column not found in DataFrame.")
    return df

def save_data(df: pd.DataFrame, file_path: str):
    """Save the DataFrame to a CSV file."""
    print(f"\nSaving filtered data to {file_path} ...")
    df.to_csv(file_path, index=False)
    print("Data saved successfully.")

def filter_rows_by_conditions(df: pd.DataFrame, conditions_dict: dict) -> pd.DataFrame:
    """
    Filter DataFrame rows based on specified column conditions.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        conditions_dict (dict): Dictionary where keys are column names and values are conditions.
                              Conditions can be:
                              - Single value: exact match
                              - List: any of the values
                              - Tuple: range (min, max)
                              - Callable: custom function
    
    Returns:
        pd.DataFrame: Filtered DataFrame
    
    Example:
        conditions = {
            'Age_cat': ['19-30', '31-44'],  # Age categories to include
            'Hospital_cat': '123',           # Exact hospital match
            'AGE_num': (18, 65),            # Age range between 18 and 65
            'SeverityLevelToStopOrder_cat': lambda x: x != 'Silence Mode'  # Custom condition
        }
        filtered_df = filter_rows_by_conditions(df, conditions)
    """
    mask = pd.Series(True, index=df.index)
    
    for column, condition in conditions_dict.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        if callable(condition):
            # Handle custom function conditions
            mask &= condition(df[column])
        elif isinstance(condition, tuple) and len(condition) == 2:
            # Handle range conditions
            min_val, max_val = condition
            mask &= (df[column] >= min_val) & (df[column] <= max_val)
        elif isinstance(condition, list):
            # Handle list of values
            mask &= df[column].isin(condition)
        else:
            # Handle single value
            mask &= (df[column] == condition)
    
    filtered_df = df[mask].copy()
    print(f"Filtered from {len(df)} to {len(filtered_df)} rows")
    return filtered_df

def main():
    # File paths
    #input_file = "/alert_analysis/data/main_data_2022/df_main_flat_py_version.csv"
    input_file = 'alert_analysis/data/main_data_2022/df_main_flat_py_version.csv'
    output_file = "alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv"

    # 1. Read data
    df = read_data(input_file)
    print(list(df.columns))
    'sectortext_en_cat' in [col.lower() for col in df.columns]

    # 2. Convert specified columns to categorical data type.
    columns_to_convert = [
        "Hospital_cat", "HospitalName_EN_cat", "UnitName_cat", "SeverityLevelToStopOrder_cat",
        "ATC_GROUP", "ResponseType_cat", "ShiftType_cat", "DayEN_cat", "Gender_Text_EN_cat",
        "SectorText_EN_cat", "adult_child_cat", "DRC_CAT", "DT_CAT", "Technical_alerts_CAT",
        "NeoDRC_CAT", "Renal_alerts_CAT", "Alert_type", "Alert_status", "Medical_Record_cat"
    ]
    df = convert_columns_to_category(df, columns_to_convert)

    # 3. Convert 'Age_cat' column to an ordered categorical type.
    age_order = ["< 1", "1-5", "6-10", "11-15", "16-18", "19-30", "31-44", "45-55", "56-64", "65-75", "76-85", "> 85"]
    df = convert_age_cat(df, 'Age_cat', age_order)

    # 4. Print summary before dropping missing AGE_num values.
    print_data_summary(df, "Data Summary before dropping missing AGE_num values:")

    # 5. Drop rows where 'AGE_num' is missing and print summary after.
    df = drop_missing_age(df, 'AGE_num')
    print_data_summary(df, "\nData Summary after dropping missing AGE_num values:")

    # 6. Print the categories for 'ResponseType_cat' and 'ShiftType_cat'.
    print_category_info(df, 'ResponseType_cat')
    print_category_info(df, 'ShiftType_cat')

    # 7. Compute the load index per shift.
    load_index = compute_load_index(df)

    # 8. Filter the DataFrame for active adult records.
    df_active_adult = filter_active_adult(df)
    print("\nSummary of df_main_active_adult:")
    print(df_active_adult.describe(include='all'))

    # 9. Update the categories of 'UnitName_cat'.
    new_categories = [
        "Internal", "Cardiology", "Emergency", "Geriatric", "Gynecology",
        "Hematology", "Internal-Covid19", "Nephrology", "Oncology", "Surgery"
    ]
    df_active_adult = update_unitname_category(df_active_adult, new_categories)
    print("\nHead of df_main_active_adult:")
    print(df_active_adult.head())

    df_active_adult = filter_rows_by_conditions(
        df_active_adult, conditions_dict = {
        'SectorText_EN_cat': lambda x: x != 'PARAMEDICAL'
        }
        )

    # 10. Save the filtered DataFrame.
    save_data(df_active_adult, output_file)

if __name__ == '__main__':
    main()