import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from charlson_index import calculate_cci  # Assumes you have this module available


def load_main_data():
    print("Loading main data from CSV...")
    df = pd.read_csv('alert_analysis/data_process/data_main_prep.csv')
    assert not df.empty, "Loaded main data is empty!"
    return df


def group_and_deduplicate(data):
    print("Step 1: Grouping and deduplication...")
    group_cols = [
        "Hospital_cat",
        "HospitalName_cat",
        "HospitalName_EN_cat",
        "UnitName_cat",
        "SeverityLevelToStopOrder_cat",
        "Medical_Record",
        "Admission_No",
        "AGE_num",
        "Gender_Text_cat",
        "Gender_Text_EN_cat",
        "SectorText_cat",
        "SectorText_EN_cat",
        "Order_ID_new",
        "Field",
        "IS_New_Order",
        "Drug_Header_cat",
        "Module_Alert_Rn",
        "Alert_Message",
        "Alert_Severity",
        "OrderOrigin",
        "ShiftType_cat",
        "DayHebrew_cat",
        "DayEN_cat",
        "HospAmount_new",
        "NumMedAmount",
        "Details",
        "ATC",
        "ATC_cln",
        "Basic_Name",
        "DiagnosisInReception",
        "HospDiagnosis",
        "id1",
        "id2",
        "Alert_Rn_Severity_cat",
        "Module_Severity_Rn"
    ]
    # Replace NaN in grouping columns with "NA"
    data[group_cols] = data[group_cols].fillna("NA")
    # Group by and take the first row per group
    data_grouped = data.groupby(group_cols, as_index=False, sort=False).first()
    print(f"Grouping completed. Rows after deduplication: {len(data_grouped)}")
    return data_grouped


def select_columns(data):
    print("Step 2: Selecting and reordering columns...")
    select_cols = [
        "Hospital_cat",
        "HospitalName_cat",
        "HospitalName_EN_cat",
        "UnitName_cat",
        "SeverityLevelToStopOrder_cat",
        "Medical_Record",
        "Admission_No",
        "AGE_num",
        "Gender_Text_cat",
        "Gender_Text_EN_cat",
        "SectorText_cat",
        "SectorText_EN_cat",
        "Order_ID_new",
        "Time_Prescribing_Order",  # additional column not used in grouping
        "Field",
        "IS_New_Order",
        "Drug_Header_cat",
        "Module_Alert_Rn",
        "Alert_Message",
        "Alert_Severity",
        "OrderOrigin",
        "ShiftType_cat",
        "DayHebrew_cat",
        "DayEN_cat",
        "HospAmount_new",
        "NumMedAmount",
        "Other_Text",  # additional column
        "ResponseType_cat",  # additional column
        "Details",
        "ATC",
        "ATC_cln",
        "Basic_Name",
        "DiagnosisInReception",
        "HospDiagnosis",
        "id1",
        "id2",
        "Alert_Rn_Severity_cat",
        "Response",  # additional column
        "Answer_Text_EN",  # additional column
        "Module_Severity_Rn",
        "Time_Mabat_Request_convert_ms_res",  # additional column
        "Time_Mabat_Response_convert_ms_res",  # additional column
        "diff_time_mabat_ms",  # additional column
        "DRC_SUB_GROUP",  # additional column
        "NeoDRC_SUB_GROUP"  # additional column
    ]
    missing_cols = [col for col in select_cols if col not in data.columns]
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    data_selected = data[select_cols]
    print("Columns selected and reordered.")
    return data_selected


def filter_atc_codes(data):
    print("Step 3: Creating ATC count column and filtering rows...")
    # Count the number of ATC codes by splitting on commas
    data['ATC_cnt'] = (
        data['ATC']
        .str.split(',')
        .apply(lambda codes: len([c for c in codes if c.strip() not in ['', 'NA']]))
    )
    # Remove rows with 3 or more ATC codes
    data = data[data['ATC_cnt'] < 3]
    # Keep rows where ATC_cnt equals 1 or (equals 2 and contains "B05XA03")
    mask = ((data['ATC_cnt'] == 2) & (data['ATC'].str.contains("B05XA03"))) | (data['ATC_cnt'] == 1)
    data = data[mask]
    print(f"Filtering completed. Rows remaining: {len(data)}")
    return data


def extract_first_word(text):
    """
    Ensure there is a space before '(' then split the string and return the first word in uppercase.
    """
    text = str(text)
    text = re.sub(r'\(', ' (', text)
    parts = text.split()
    return parts[0].upper() if parts else ''


def add_first_word_columns(data):
    print("Adding first word columns for Drug_Header_cat and Basic_Name...")
    data["Drug_Header_cat_1st_word"] = data["Drug_Header_cat"].apply(extract_first_word)
    data["Basic_Name_1st_word"] = data["Basic_Name"].apply(extract_first_word)
    return data


def process_basic_atc():
    print("Processing BASIC_NAME_ATC data...")
    basic_atc = pd.read_csv("alert_analysis/data/BASIC_NAME_ATC.csv")
    basic_atc = basic_atc[["ATC5", "BASIC NAME"]]
    basic_atc = basic_atc.drop_duplicates()
    basic_atc = basic_atc.rename(columns={"BASIC NAME": "BASIC_NAME_EXT"})
    basic_atc = basic_atc.dropna()
    basic_atc["BASIC_NAME_EXT_1st_word"] = basic_atc["BASIC_NAME_EXT"].apply(
        lambda x: x.split()[0].upper() if isinstance(x, str) and x.split() else ''
    )
    basic_atc_cln = basic_atc.groupby("BASIC_NAME_EXT_1st_word", as_index=False).first()[
        ["BASIC_NAME_EXT_1st_word", "ATC5"]]
    print("BASIC_NAME_ATC processing completed.")
    return basic_atc_cln


def merge_with_basic_atc(data, basic_atc_cln):
    print("Merging main data with BASIC_NAME_ATC data...")
    merged = pd.merge(data, basic_atc_cln, how="left",
                      left_on="Drug_Header_cat_1st_word", right_on="BASIC_NAME_EXT_1st_word")
    # First condition: if Drug_Header_cat_1st_word != Basic_Name_1st_word, use ATC5; otherwise ATC_cln.
    merged["ATC_NEW"] = np.where(
        merged["Drug_Header_cat_1st_word"] != merged["Basic_Name_1st_word"],
        merged["ATC5"],
        merged["ATC_cln"]
    )
    # Second condition: if (Drug_Header_cat_1st_word != Basic_Name_1st_word) and OrderOrigin is 'Chronic Meds', use ATC5.
    merged["ATC_NEW"] = np.where(
        ((merged["Drug_Header_cat_1st_word"] != merged["Basic_Name_1st_word"]) & (
                    merged["OrderOrigin"] == "Chronic Meds")),
        merged["ATC5"],
        merged["ATC_cln"]
    )
    # Third condition: if OrderOrigin is 'Chronic Meds' and ATC5 is missing, set ATC_NEW to ATC_cln.
    merged["ATC_NEW"] = np.where(
        ((merged["OrderOrigin"] == "Chronic Meds") & (merged["ATC5"].isna())),
        merged["ATC_cln"],
        merged["ATC_NEW"]
    )
    # Save intermediate result
    merged.to_csv('alert_analysis/data_process/data_distincted_main_new_raw_1.csv', index=False)
    print("Merge completed and intermediate CSV saved.")
    return merged


def update_order_id(data):
    print("Updating Order_ID_new with chronic id count...")
    data.sort_values(by=["Order_ID_new", "ATC_NEW"], inplace=True)
    data['cnt_chronic_id'] = data.groupby('Order_ID_new')['ATC_NEW'].transform(
        lambda s: (s != s.shift(1).fillna(s.iloc[0])).cumsum()
    )
    data['Order_ID_new_update'] = np.where(
        (data['OrderOrigin'] == 'Chronic Meds') & (data['ATC_NEW'].notna()),
        data['Order_ID_new'].astype(str) + "_" + data['cnt_chronic_id'].astype(str),
        data['Order_ID_new']
    )
    return data


# def plot_histogram(data, column, title, bins=400):
#     print(f"Plotting histogram for {column}...")
#     plt.hist(data[column], bins=bins)
#     plt.title(title)
#     plt.xlabel(column)
#     plt.ylabel("Frequency")
#     plt.show()


def calculate_medication_count(data):
    print("Calculating the number of medications per patient...")
    # First, count unique order groups per Medical_Record & Order_ID_new_update
    num_med_df = data.groupby(['Medical_Record', 'Order_ID_new_update']).size().reset_index(name='NumMedAmount_calc')
    # Then, count unique order groups per Medical_Record
    num_med_df_2 = num_med_df.groupby('Medical_Record').size().reset_index(name='NumMedAmount_calc')
    data = data.merge(num_med_df_2, on='Medical_Record', how='left')
    print("Medication count calculation completed.")
    return data


def process_diagnosis(data):
    print("Processing diagnosis information...")
    # Create num_of_diagnosis column
    data["num_of_diagnosis"] = np.where(
        data["HospDiagnosis"].isna() | (data["HospDiagnosis"] == "NA"),
        0,
        data["HospDiagnosis"].str.split(";").str.len()
    )
    # Create diseaseSplit column
    data["diseaseSplit"] = data["HospDiagnosis"].str.split(";")

    def fix_disease_split(ds):
        if isinstance(ds, list):
            if len(ds) == 1 and ds[0].strip().upper() in ['NANA', 'NA']:
                return None
            return [s.upper() for s in ds]
        return ds

    data["diseaseSplit"] = data["diseaseSplit"].apply(fix_disease_split)
    return data


def count_disease_keywords(data):
    print("Counting disease keywords...")
    my_list = ['LIVER', 'HEPATIC', 'CIRRHOSIS', 'HEPATITIS', 'PORTAL HYPERTENSION', 'JAUNDICE',
               'MYOCARDIAL', 'HEART FAILURE',
               'PERIPHERAL VASCULAR DISEASE', 'PVD',
               'CEREBROVASCULAR', 'STROKE', 'CEREBRAL INFARCTION', 'TIA', 'CVA',
               'DEMENTIA',
               'CHRONIC PULMONARY DISEASE', 'COPD', 'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
               'RHEUMATOLOGIC DISEASE', 'FIBROMYALGIA', 'GOUT', 'ARTHRITIS', 'ANKYLOSING SPONDYLITIS', 'SCLERODERMA',
               'DUODENAL ULCER', 'PEPTIC ULCER', 'GASTRIC ULCER',
               'DIABETES',
               'HEMIPLEGIA', 'HEMIPARESIS',
               'PARAPLEGIA',
               'RENAL', 'PYELONEPHRITIS', 'HEMODIALYSIS', 'KIDNEY', 'CKD', 'ARF', 'AKI',
               'MALIGNANCY', 'MALIGNANT', 'CARCINOMA', 'NEOPLASM', 'ADENOCARCINOMA',
               'LEUKEMIA', 'AML', 'CML',
               'LYMPHOMA',
               'HIV',
               'METASTATIC', 'METASTASIS']

    def count_keyword(ds, keyword):
        if not isinstance(ds, list):
            return 0
        return sum(1 for s in ds if isinstance(s, str) and keyword in s)

    for item in my_list:
        col_name = f"{item}_count"
        data[col_name] = data["diseaseSplit"].apply(lambda ds: count_keyword(ds, item))

    # Liver group
    liver_cols = ["LIVER_count", "HEPATIC_count", "JAUNDICE_count", "PORTAL HYPERTENSION_count", "CIRRHOSIS_count"]
    portal_cirrhosis = ["PORTAL HYPERTENSION_count", "CIRRHOSIS_count"]
    data["liver_group_cnt"] = np.where(
        data[liver_cols].fillna(0).gt(0).sum(axis=1) > 0,
        np.where(
            data[portal_cirrhosis].fillna(0).gt(0).sum(axis=1) > 0,
            3,
            1
        ),
        0
    )
    # PVD group
    data["PVD_group_cnt"] = np.where(
        data[["PERIPHERAL VASCULAR DISEASE_count", "PVD_count"]].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Cerebrovascular group
    data["CEREBROVASCULAR_group_cnt"] = np.where(
        data[['CEREBROVASCULAR_count', 'STROKE_count', 'CEREBRAL INFARCTION_count', 'TIA_count', 'CVA_count']].fillna(
            0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # COPD group
    data["COPD_group_cnt"] = np.where(
        data[['CHRONIC PULMONARY DISEASE_count', 'COPD_count', 'CHRONIC OBSTRUCTIVE PULMONARY DISEASE_count']].fillna(
            0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Gout group
    data["GOUT_group_cnt"] = np.where(
        data[['RHEUMATOLOGIC DISEASE_count', 'FIBROMYALGIA_count', 'GOUT_count', 'ARTHRITIS_count',
              'ANKYLOSING SPONDYLITIS_count', 'SCLERODERMA_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Ulcer group
    data["ULCER_group_cnt"] = np.where(
        data[['DUODENAL ULCER_count', 'PEPTIC ULCER_count', 'GASTRIC ULCER_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Hemiplegia group
    data["HEMIPLEGIA_group_cnt"] = np.where(
        data[['HEMIPLEGIA_count', 'HEMIPARESIS_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Renal group
    data["RENAL_group_cnt"] = np.where(
        data[['RENAL_count', 'PYELONEPHRITIS_count', 'HEMODIALYSIS_count', 'KIDNEY_count', 'CKD_count', 'ARF_count',
              'AKI_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Malignancy group
    data["MALIGNANCY_group_cnt"] = np.where(
        data[['MALIGNANCY_count', 'MALIGNANT_count', 'CARCINOMA_count', 'NEOPLASM_count',
              'ADENOCARCINOMA_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Leukemia group
    data["LEUKEMIA_group_cnt"] = np.where(
        data[['LEUKEMIA_count', 'AML_count', 'CML_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    # Metastatic group
    data["METASTATIC_group_cnt"] = np.where(
        data[['METASTATIC_count', 'METASTASIS_count']].fillna(0).gt(0).sum(axis=1) > 0,
        1,
        0
    )
    print("Disease keyword counting completed.")
    return data


def calculate_charlson(data):
    print("Calculating Charlson Comorbidity Index Scores...")
    data_with_cci = calculate_cci(data)
    print("Charlson index calculation completed.")
    return data_with_cci


def main():
    # Load and preprocess main data
    data_main = load_main_data()
    data_grouped = group_and_deduplicate(data_main)
    data_selected = select_columns(data_grouped)
    data_filtered = filter_atc_codes(data_selected)
    data_with_first_words = add_first_word_columns(data_filtered)

    # Process the BASIC_NAME_ATC data and merge with the main data
    basic_atc_cln = process_basic_atc()
    data_merged = merge_with_basic_atc(data_with_first_words, basic_atc_cln)

    # Update order IDs based on chronic medications
    data_updated_order = update_order_id(data_merged)

    # Optionally plot histogram for original NumMedAmount
    #plot_histogram(data_updated_order, "NumMedAmount", "Histogram of NumMedAmount")

    # Calculate and plot the number of medications per patient
    data_med_count = calculate_medication_count(data_updated_order)
    #plot_histogram(data_med_count, "NumMedAmount_calc", "Histogram of Calculated NumMedAmount")
    data_med_count.to_csv('alert_analysis/data_process/data_distincted_main_new_raw_3.csv', index=False)

    # Process diagnosis information and count disease keywords
    data_diagnosis = process_diagnosis(data_med_count)
    data_with_disease_counts = count_disease_keywords(data_diagnosis)

    # Calculate Charlson Comorbidity Index Scores
    data_with_cci = calculate_charlson(data_with_disease_counts)

    # Save final output
    output_filename = 'alert_analysis/data_process/data_distincted_main_new_with_cci.csv'
    data_with_cci.to_csv(output_filename, index=False)
    print(f"Final data with Charlson index saved to {output_filename}")


if __name__ == "__main__":
    main()