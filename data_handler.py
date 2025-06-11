import os
import pandas as pd
from ydata_profiling import ProfileReport


df_active_adult = pd.read_csv('alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv')
df_active_adult.shape
#TODO: create profile report for df_active_adult
profile = ProfileReport(df_active_adult, title="Data Profiling Report", explorative=True)
profile.to_file("df_active_adult_profiling_ud.html") # open "1_data_profiling.html" file if you can't see the iframe
profile.to_notebook_iframe()


# profiling to patient level data
src_tbl1_active_by_patient_gb = pd.read_csv('alert_analysis/data/main_data_2022/df_patients_level_data.csv')

##TODO: create profile report for src_tbl1_active_by_patient_gb
profile = ProfileReport(src_tbl1_active_by_patient_gb, title="Data Profiling Report", explorative=True)
profile.to_file("patient_level_data_profiling_ud.html")
profile.to_notebook_iframe()





#TODO: cretae a new flat column below/ exceeds dose, the full logic is in Hiba_project_fatigue_alert.rmd
# add Module_Alert_Rn and Alert_Message ###### Done ######
#add the logic below
# ifelse(data_distincted_active_mode_stoping$Module_Alert_Rn == "DRC - Frequency 1" & grepl("exceeds",
#                                                                                           data_distincted_active_mode_stoping$Alert_Message), "DRC - Frequency - exceeds ",
#
# ifelse(data_distincted_active_mode_stoping$Module_Alert_Rn == "DRC - Frequency 1" & grepl("below",
#                                                                                           data_distincted_active_mode_stoping$Alert_Message), "DRC - Frequency - below",

# Apply the same logic to "DRC - Single Dose 1"
# Apply the same logic to "DRC - Max Daily Dose 1"

#This is the main outcome to analyse
#df_active_adult['Alert_type'].value_counts()





#TODO: April 27
# 1. add to patient level data: number of diagnoses (diagnosisinreception, hospdiagnosis), add this to tableone analysis # TODO: Done
# 2. add unitname_cat to patient level data and tableone # TODO: Done   
# from the main data, remove PARAMEDICAL (column name = sectortext_en_cat) # TODO: Done
# create the alert table according to year-month # TODO: not done
# create pie chart- figure 1 (distribution of alert types)  - I should create a new column that combines all the features listed in the doc
# figure 3, answer_text_en
# figure 8, specified in the doc
# create table 8: general statistics of alert # TODO: nit Done


#TODO: May 11
#1. revised the dose column in 01_data_main_prep_ud.py for each modules_to_check = ["DRC - Frequency 1", "DRC - Single Dose 1", "DRC - Max Daily Dose 1"] will
# have the dose for itself (before data is flatten, order_id is not unique) so below/exceeds dose is not correct





