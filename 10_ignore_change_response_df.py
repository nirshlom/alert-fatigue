import pandas as pd
import os

# Get the script's directory (project root)
project_root = os.path.dirname(os.path.abspath(__file__))

#TODO: Load the data: df_main_active_adult_renamed_new_clean_sample_100pct.csv, this order level data only active adults with final data for alert analysis
input_path = os.path.join(project_root, "alert_analysis", "data", "main_data_2022", "df_main_active_adult_renamed_new_clean_sample_100pct.csv")
df_main_active_adult = pd.read_csv(input_path)
print(df_main_active_adult.shape)
df_main_active_adult.head(10)


# Filter for response type ignore change 
df_filtered_response_type = df_main_active_adult[df_main_active_adult['response_type_ud'].isin(["Change", "Ignore"])].copy()
print(df_filtered_response_type.shape)
df_filtered_response_type.head(10)

# verify the remaining values are only 0 and 1
df_filtered_response_type['response_type_ud'].value_counts(dropna=False)


# Crosstab: number of medication orders vs alert_status_binary
crosstab_table = pd.crosstab(
    df_filtered_response_type['response_type_ud'],
    "drug_order_count",
    values=df_filtered_response_type['drug_order_id'],
    aggfunc='nunique'
)

crosstab_table


# Save df_filtered_response_type_ignore_change.csv
output_path = os.path.join(project_root, "alert_analysis", "data", "main_data_2022", "response_type_ignore_change.csv")
df_filtered_response_type.to_csv(output_path, index=False)
print(f"Saved {len(df_filtered_response_type):,} rows")

