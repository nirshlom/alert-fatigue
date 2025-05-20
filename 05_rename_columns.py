import pandas as pd
from column_rename import rename_keep_dict, rename_keep_dict_false

def read_data(file_path: str) -> pd.DataFrame:
    """Read CSV file into a DataFrame."""
    print(f"Reading data from {file_path} ...")
    df = pd.read_csv(file_path)
    assert not df.empty, "DataFrame is empty after reading the file."
    print(f"Successfully read {len(df)} rows.")
    return df

def report_column_mapping(df: pd.DataFrame, rename_dict: dict) -> None:
    """
    Report which columns will be kept, dropped, or ignored based on the rename dictionary.
    """
    input_columns = set(df.columns)
    mapped_columns = set(rename_dict.keys())
    columns_to_keep = [col for col in input_columns if col in mapped_columns and rename_dict[col][1]]
    columns_to_drop = [col for col in input_columns if col in mapped_columns and not rename_dict[col][1]]
    columns_ignored = [col for col in input_columns if col not in mapped_columns]
    print(f"\nColumns to keep and rename: {len(columns_to_keep)}")
    print(f"Columns to drop: {len(columns_to_drop)}")
    print(f"Columns to ignore (not mapped): {len(columns_ignored)}")
    if columns_ignored:
        print(f"Ignored columns: {columns_ignored}")

def rename_and_filter_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame:
    """
    Rename columns based on the rename dictionary and filter out columns marked for dropping.
    Columns not in the dictionary are ignored (not included in the output).
    """
    # Only keep columns that are in the rename dictionary and marked True
    columns_to_keep = [col for col in df.columns if col in rename_dict and rename_dict[col][1]]
    rename_mapping = {col: rename_dict[col][0] for col in columns_to_keep}
    print(f"\nRenaming columns: {rename_mapping}")
    df = df[columns_to_keep].rename(columns=rename_mapping)
    print(f"Final columns in output: {list(df.columns)}")
    return df

def save_data(df: pd.DataFrame, file_path: str):
    """Save the DataFrame to a CSV file."""
    print(f"\nSaving renamed data to {file_path} ...")
    df.to_csv(file_path, index=False)
    print("Data saved successfully.")

def main():
    # File paths
    input_file = "alert_analysis/data/main_data_2022/df_main_active_adult_py_version.csv"
    output_file = "alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv"
    
    # Read data
    df = read_data(input_file)
    print("\nOriginal columns:", list(df.columns))
    
    # Combine both rename dictionaries
    combined_rename_dict = {**rename_keep_dict, **rename_keep_dict_false}
    
    # Report mapping
    report_column_mapping(df, combined_rename_dict)
    
    # Rename and filter columns
    df_renamed = rename_and_filter_columns(df, combined_rename_dict)
    print(f"\nNew columns: {list(df_renamed.columns)}")
    
    # Save the renamed DataFrame
    save_data(df_renamed, output_file)

if __name__ == "__main__":
    main() 