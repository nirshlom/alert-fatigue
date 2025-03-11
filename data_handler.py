import pandas as pd
from ydata_profiling import ProfileReport
import numpy as np
df = pd.read_csv('data/main_data_2022/df_main_flat.csv')
df.columns = map(str.lower, df.columns)
print(f'original df has {df.shape[0]} rows and {df.shape[1]} columns')
# original df has 3615984 rows and 66 columns

df_active_adult = pd.read_csv('data/df_main_active_adult.csv')
print(f'original df_active_adult has {df_active_adult.shape[0]} rows and {df_active_adult.shape[1]} columns')

#test for duplicated ids:
df['order_id_new_update'].duplicated().sum() == 0

df_sample = df.sample(10000)
df_sample.to_csv('data/df_sample_main_flat.csv', index=False)




profile = ProfileReport(df_active_adult, title="Data Profiling Report", explorative=True)
profile.to_file("df_active_adult_profiling.html") # open "1_data_profiling.html" file if you can't see the iframe
profile.to_notebook_iframe()