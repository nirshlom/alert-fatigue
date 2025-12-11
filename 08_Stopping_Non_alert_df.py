import pandas as pd

#TODO: Load the data: df_main_active_adult_renamed_new_clean_sample_100pct.csv, this order level data only active adults with final data for alert analysis
df_main_active_adult = pd.read_csv(r'C:\Users\hibaa\Documents\GitHub\alert-fatigue\alert_analysis\data\main_data_2022\df_main_active_adult_renamed_new_clean_sample_100pct.csv')
print(df_main_active_adult.shape)
df_main_active_adult.head(10)


# Filter for Stoping_alert and Non_alert
df_filtered_stopping_non_alert = df_main_active_adult[df_main_active_adult['alert_status_binary'].isin([0, 1])].copy()
print(df_filtered_stopping_non_alert.shape)
df_filtered_stopping_non_alert.head(10)

# verify the remaining values are only 0 and 1
df_filtered_stopping_non_alert['alert_status_binary'].value_counts(dropna=False)


# Crosstab: number of medication orders vs alert_status_binary
crosstab_table = pd.crosstab(
    df_filtered_stopping_non_alert['alert_status_binary'],
    "drug_order_count",
    values=df_filtered_stopping_non_alert['drug_order_id'],
    aggfunc='nunique'
)

crosstab_table


# Save alert_status_binary_stopping_non_alert.csv
df_filtered_stopping_non_alert.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/alert_status_binary_stopping_non_alert.csv", index=False)
print(f"Saved {len(df_filtered_stopping_non_alert):,} rows")

