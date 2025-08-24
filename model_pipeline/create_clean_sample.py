#!/usr/bin/env python3
import pandas as pd
import os
os.getcwd()
# Load data
df = pd.read_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv")

# Clean and sample
df = df[df['gender'] != 'gender']  # Remove header contamination
df['alert_status_binary'] = (df['alert_status'] == 'Stoping_alert').astype(int)
df_sample = df.sample(n=int(len(df) * 0.1), random_state=42)

# Save
df_sample.to_csv("C:/Users/hibaa/Documents/GitHub/alert-fatigue/alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv", index=False)
print(f"Saved {len(df_sample):,} rows")
