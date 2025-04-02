import pandas as pd
import numpy as np
import re


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and verify required columns."""
    print("Reading input data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded with shape: {df.shape}")
    assert 'AGE_num' in df.columns, "'AGE_num' column is missing."
    return df


def add_adult_child_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add adult/child categorization based on AGE_num."""
    df['adult_child_cat'] = np.where(df['AGE_num'] < 19, 'child', 'adult')
    print("Added 'adult_child_cat' column.")
    return df


def add_age_category(df: pd.DataFrame) -> pd.DataFrame:
    """Create age categories using defined bins and labels."""
    bins = [0, 1, 6, 11, 16, 19, 31, 45, 56, 65, 76, 86, float('inf')]
    labels = ['< 1', '1-5', '6-10', '11-15', '16-18', '19-30', '31-44', '45-55', '56-64', '65-75', '76-85', '> 85']
    df['Age_cat'] = pd.cut(df['AGE_num'], bins=bins, labels=labels, right=False)
    print("Added 'Age_cat' column.")
    # Ensure not all Age_cat values are null
    assert df['Age_cat'].isnull().sum() < df.shape[0], "All 'Age_cat' values are null."
    return df


def extract_capital_words(text):
    """Extract words that contain capital letters."""
    if pd.isna(text):
        return np.nan
    matches = re.findall(r"\b[A-Za-z]*[A-Z]+[A-Za-z]*\b", str(text))
    return ",".join(matches)


def extract_drugs_from_alert_message(df: pd.DataFrame) -> pd.DataFrame:
    """Extract drug names from alert messages based on severity flags."""
    print("Extracting drug names from Alert_Message...")
    # Verify necessary columns exist
    assert 'Alert_Rn_Severity_cat' in df.columns, "'Alert_Rn_Severity_cat' column missing."
    assert 'Alert_Message' in df.columns, "'Alert_Message' column missing."

    df['drugs_from_AlretMessage'] = np.where(
        df['Alert_Rn_Severity_cat'].str.contains("DDI", case=False) |
        df['Alert_Rn_Severity_cat'].str.contains("DT", case=False),
        df['Alert_Message'].apply(lambda x: re.sub(r"(?i)(.*)\bmay\b.*", r"\1", x) if pd.notna(x) else x),
        np.nan
    )

    print(f"Non-null 'drugs_from_AlretMessage': {df['drugs_from_AlretMessage'].notnull().sum()}")

    df['drugs_from_AlretMessage'] = df['drugs_from_AlretMessage'].apply(extract_capital_words)
    print("Extracted capital words from 'drugs_from_AlretMessage'.")
    return df


def remove_unwanted_words(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted words from the extracted drug names."""
    print("Removing unwanted words...")
    words_to_drop = ["The", "TEVA", "KWIK", "PEN", "BAYER", "SOLOSTAR", "FLASH", "PLUS", "RTH", "Duplicate",
                     "Therapy", "TURBUHALER", "CHILD", "ORAL", "INOVAMED", "CFC", "FREE", "NEW", "KERN",
                     "PHARMA", "STRAWB", "(EU)", "CREAM", "OINTMENT", "VELO", "INOVAMED", "UNIT", "RETARD",
                     "PENFILL", "novoRAPID", "APIDRA", "ROMPHAR", "ROMPHARM", "CLARIS", "FRESENIUS", "FLASH",
                     "DIASPORAL", ",GRA", "AVENIR", "MYLAN", "RATIO", "SALF", "LANTUS", "RESPIR", "FLEX",
                     "BASAGLAR", "CRT", "TOUJEO", "LIPURO", "HCL", "DECANOAS", "KALCEKS", "ODT", "AOV",
                     "TRIMA", "DEXCEL", "PANPHA", "ROTEXMEDICA", "ROTEX", "ROTEXMED", "FORTE", "HumuLIN",
                     "ADVANCE", "BANANA", "COFFEE", "FIBER", "VANILLA", "TREGLUDEC", "CHOCOLATE", "PENFILL",
                     "PWD", "RESPULES", "Drops", ",DRP"]
    drop_pattern = r"\b(" + "|".join(words_to_drop) + r")\b"
    df['drugs_from_AlretMessage'] = df['drugs_from_AlretMessage'].str.replace(drop_pattern, "", case=False, regex=True)
    print("Unwanted words removed.")
    return df


def fix_common_phrases(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known phrases in the drugs field."""
    print("Replacing known phrases...")
    replacements = [
        ('ACTILYSE,TPA', 'ACTILYSE-TPA'),
        ('V,DALGIN', 'V-DALGIN'),
        ('FOLIC,ACID', 'FOLIC-ACID'),
        ('COD,ACAMOL', 'COD-ACAMOL'),
        ('PROCTO,GLYVENOL', 'PROCTO-GLYVENOL'),
        ('Betacorten,G', 'Betacorten-G'),
        ('MICRO,KALIUM', 'MICRO-KALIUM'),
        ('VASODIP,COMBO', 'VASODIP-COMBO'),
        ('DEPO', 'DEPO-medrol'),
        ('VITA,CAL', 'VITA-CAL'),
        ('TAZO,PIP', 'TAZO-PIP'),
        ('Solu,CORTEF', 'Solu-CORTEF'),
        ('DEPALEPT,CHRONO', 'DEPALEPT-CHRONO'),
        ('PIPERACILLIN,TAZOBACTAM', 'PIPERACILLIN-TAZOBACTAM'),
        ('SOLU', 'SOLU-medrol'),
        ('SOPA,K', 'SOPA-K'),
        ('V,OPTIC', 'V-OPTIC'),
        ('SLOW,K', 'SLOW-K'),
        ('JARDIANCE DUO', 'JARDIANCE-DUO')
    ]
    for old, new in replacements:
        df['drugs_from_AlretMessage'] = df['drugs_from_AlretMessage'].str.replace(old, new, regex=False)
    print("Phrases replaced.")
    return df


def remove_duplicates_in_cell(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries within the drugs cell."""
    print("Removing duplicates within cell for 'drugs_from_AlretMessage'...")

    def remove_duplicates(text):
        if pd.isna(text):
            return np.nan
        items = text.split(",")
        items = [itm.strip() for itm in items if itm.strip() != ""]
        items_unique = list(dict.fromkeys(items))
        return ",".join(items_unique)

    df['drugs_from_AlretMessage'] = df['drugs_from_AlretMessage'].apply(remove_duplicates)
    print("Duplicates within cell removed.")
    df.loc[df['drugs_from_AlretMessage'] == 'NA', 'drugs_from_AlretMessage'] = np.nan
    return df


def create_test_dataset(df: pd.DataFrame):
    """Print a sample of the dataset for inspection."""
    print("\nSample output data:")
    print(df[['Order_ID_new_update', 'Alert_Rn_Severity_cat', 'Alert_Message', 'drugs_from_AlretMessage']].head())


def remove_duplicates_and_pivot(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Remove duplicates and pivot the dataset by alert severity."""
    print("Removing duplicates and pivoting data...")
    src_for_flat_cln = df.drop_duplicates()
    print(f"src_for_flat_cln shape after deduplication: {src_for_flat_cln.shape}")

    # Pivot by 'Alert_Rn_Severity_cat'
    print("Pivoting to wide format by 'Alert_Rn_Severity_cat'...")
    alert_counts = (
        pd.pivot_table(
            src_for_flat_cln,
            index=["Order_ID_new_update"],
            columns="Alert_Rn_Severity_cat",
            aggfunc="size",
            fill_value=0
        )
        .reset_index()
    )
    print(f"Alert counts shape: {alert_counts.shape}")

    flat_by_sevirity = pd.merge(
        src_for_flat_cln.drop_duplicates(subset=["Order_ID_new_update"]),
        alert_counts,
        on="Order_ID_new_update",
        how="left"
    )
    print(f"Shape after merge: {flat_by_sevirity.shape}")
    expected_rows = 3615043
    assert flat_by_sevirity.shape[
               0] == expected_rows, f"Expected rows {expected_rows}, found {flat_by_sevirity.shape[0]}"
    return flat_by_sevirity, src_for_flat_cln


def create_alert_sum_column(df: pd.DataFrame) -> pd.DataFrame:
    """Sum up specific alert columns to get total alerts per order."""
    alert_cols = ["DAM", "DDI-Contraindicated Drug Combination", "DDI-Moderate Interaction",
                  "DDI-Severe Interaction", "DRC", "Renal alerts", "Technical alert"]
    for col in alert_cols:
        assert col in df.columns, f"Missing alert column: {col}"
    df["num_of_alerts_per_order_id"] = df[alert_cols].sum(axis=1)
    return df


def create_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and create binary categorical flags for alerts."""
    print("Creating binary alert flags...")
    df.rename(
        columns={
            "Renal alerts": "Renal_alerts",
            "Technical alert": "Technical_alerts",
            "DDI-Severe Interaction": "DDI_Severe_Interaction",
            "DDI-Moderate Interaction": "DDI_Moderate_Interaction",
            "DDI-Contraindicated Drug Combination": "DDI_Contraindicated_Drug_Combination"
        },
        inplace=True
    )

    df["DAM_CAT"] = (df["DAM"] > 0).astype(int).astype("category")
    df["DDI_Contraindicated_Drug_Combination_CAT"] = (df["DDI_Contraindicated_Drug_Combination"] > 0).astype(
        int).astype("category")
    df["DDI_Moderate_Interaction_CAT"] = (df["DDI_Moderate_Interaction"] > 0).astype(int).astype("category")
    df["DDI_Severe_Interaction_CAT"] = (df["DDI_Severe_Interaction"] > 0).astype(int).astype("category")
    df["DRC_CAT"] = (df["DRC"] > 0).astype(int).astype("category")
    df['Technical_alerts_CAT'] = df['Technical_alerts'].apply(lambda x: 1 if x > 0 else 0).astype("category")
    df["Renal_alerts_CAT"] = df["Renal_alerts"].apply(lambda x: 1 if x > 0 else 0).astype("category")
    df["NeoDRC_CAT"] = df["NeoDRC"].apply(lambda x: 1 if x > 0 else 0).astype("category")
    print("Binary flags created.")
    return df


def calculate_chronic_num_calc(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the chronic medication count and add it to the dataframe."""
    print("Calculating chronic_num_calc...")
    chronic_counts = (
        df[df['OrderOrigin'] == "Chronic Meds"]
        .groupby(['Medical_Record', 'id1'])
        .size()
        .reset_index(name='chronic_count')
    )
    df = df.merge(chronic_counts, on=['Medical_Record', 'id1'], how='left')
    df['chronic_num_calc'] = np.where(
        df['OrderOrigin'] == "Chronic Meds",
        df['chronic_count'],
        0
    )
    df['chronic_num_calc'] = df['chronic_num_calc'].fillna(0).astype(int)
    df = df.drop(columns=['chronic_count'])
    print("chronic_num_calc done.")
    return df


def calculate_hospital_days(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate hospital days based on prescribing order dates."""
    print("Calculating hospital days...")
    df["date_time_prescribe"] = df["Time_Prescribing_Order"].str[:10]
    df["date_time_prescribe"] = pd.to_datetime(df["date_time_prescribe"], errors="coerce")
    df["hosp_days"] = (
        df.groupby(["Medical_Record", "id1"])["date_time_prescribe"]
        .transform(lambda x: (x.max() - x.min()).days + 1)
    )
    print("Hospital days added.")
    return df


def join_max_diff_time(df: pd.DataFrame, src_for_flat_cln: pd.DataFrame) -> pd.DataFrame:
    """Join the maximum diff_time_mabat_ms back into the main dataframe."""
    print("Joining max(diff_time_mabat_ms)...")
    df_diff_time_ms = (
        src_for_flat_cln.groupby("Order_ID_new_update", as_index=False)
        .agg({"diff_time_mabat_ms": "max"})
    )
    df = df.merge(df_diff_time_ms, on="Order_ID_new_update", how="left")
    print("diff_time_mabat_ms joined.")
    return df


def join_drc_sub_group(df: pd.DataFrame, src_for_flat_cln: pd.DataFrame) -> pd.DataFrame:
    """Join DRC_SUB_GROUP counts using a pivot table."""
    print("Joining DRC_SUB_GROUP...")
    src_for_flat_cln['DRC_SUB_GROUP'] = np.where(
        src_for_flat_cln['DRC_SUB_GROUP'].isna(),
        'NA',
        src_for_flat_cln['DRC_SUB_GROUP']
    )
    df_DRC_SUB_GROUP = (
        pd.pivot_table(
            src_for_flat_cln,
            index="Order_ID_new_update",
            columns="DRC_SUB_GROUP",
            aggfunc="size",
            fill_value=0
        )
        .reset_index()
    )
    if "NA" in df_DRC_SUB_GROUP.columns:
        df_DRC_SUB_GROUP.drop(columns=["NA"], inplace=True)
    df = df.merge(df_DRC_SUB_GROUP, on="Order_ID_new_update", how="left")
    print("DRC_SUB_GROUP joined.")
    return df


def join_neodrc_sub_group(df: pd.DataFrame, src_for_flat_cln: pd.DataFrame) -> pd.DataFrame:
    """Join NeoDRC_SUB_GROUP counts using a pivot table."""
    print("Joining NeoDRC_SUB_GROUP...")
    src_for_flat_cln['NeoDRC_SUB_GROUP'] = np.where(
        src_for_flat_cln['NeoDRC_SUB_GROUP'].isna(),
        'NA',
        src_for_flat_cln['NeoDRC_SUB_GROUP']
    )
    df_NeoDRC_SUB_GROUP = (
        pd.pivot_table(
            src_for_flat_cln,
            index="Order_ID_new_update",
            columns="NeoDRC_SUB_GROUP",
            aggfunc="size",
            fill_value=0
        )
        .reset_index()
    )
    if "NA" in df_NeoDRC_SUB_GROUP.columns:
        df_NeoDRC_SUB_GROUP.drop(columns=["NA"], inplace=True)
    df = df.merge(df_NeoDRC_SUB_GROUP, on="Order_ID_new_update", how="left")
    print("NeoDRC_SUB_GROUP joined.")
    return df


def determine_alert_type(row) -> str:
    """
    Determines the alert type based on various categorical alert flags.
    Returns one of: "Error_Alert", "Non_Error_alert", or "Non_alert".
    """
    required_cols = [
        "Renal_alerts_CAT", "DRC_CAT", "DAM_CAT", "NeoDRC_CAT",
        "DDI_Contraindicated_Drug_Combination_CAT",
        "Technical_alerts_CAT", "DDI_Moderate_Interaction_CAT", "DDI_Severe_Interaction_CAT"
    ]
    for col in required_cols:
        assert col in row.index, f"Missing column '{col}' in row."

    if (
            (row.get("Renal_alerts_CAT") == 1)
            or (row.get("DRC_CAT") == 1)
            or (row.get("DAM_CAT") == 1)
            or (row.get("NeoDRC_CAT") == 1)
            or (row.get("DDI_Contraindicated_Drug_Combination_CAT") == 1)
    ):
        return "Error_Alert"
    elif (
            (row.get("Technical_alerts_CAT") == 1)
            or (row.get("DDI_Moderate_Interaction_CAT") == 1)
            or (row.get("DDI_Severe_Interaction_CAT") == 1)
    ):
        return "Non_Error_alert"
    else:
        return "Non_alert"


def determine_alert_status(row) -> str:
    """
    Determines the alert status based on alert type and specific alert category flags.
    Returns one of: "Stoping_alert", "Non_stoping_alert", or "Non_alert".
    """
    required_cols = [
        "Alert_type", "Technical_alerts_CAT",
        "DDI_Moderate_Interaction_CAT", "DDI_Severe_Interaction_CAT"
    ]
    for col in required_cols:
        assert col in row.index, f"Missing column '{col}' in row."

    if row["Alert_type"] == "Error_Alert" or row.get("Technical_alerts_CAT") == 1:
        return "Stoping_alert"
    elif (row.get("DDI_Moderate_Interaction_CAT") == 1) or (row.get("DDI_Severe_Interaction_CAT") == 1):
        return "Non_stoping_alert"
    else:
        return "Non_alert"


def add_alert_type_and_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add Alert_type and Alert_status columns based on flag logic."""
    print("Determining Alert Types...")
    required_columns = [
        "Renal_alerts_CAT", "DRC_CAT", "DAM_CAT", "NeoDRC_CAT",
        "DDI_Contraindicated_Drug_Combination_CAT", "Technical_alerts_CAT",
        "DDI_Moderate_Interaction_CAT", "DDI_Severe_Interaction_CAT"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    df["Alert_type"] = df.apply(determine_alert_type, axis=1)
    assert df["Alert_type"].notnull().all(), "Some 'Alert_type' values are null."
    df["Alert_type"] = df["Alert_type"].astype("category")
    print("✅ 'Alert_type' column added with categories:", df["Alert_type"].cat.categories.tolist())
    print("\nAlert Type Distribution:")
    print(df["Alert_type"].value_counts())

    print("\nDetermining Alert Status...")
    required_columns_status = [
        "Alert_type", "Technical_alerts_CAT", "DDI_Moderate_Interaction_CAT", "DDI_Severe_Interaction_CAT"
    ]
    missing_columns_status = [col for col in required_columns_status if col not in df.columns]
    assert not missing_columns_status, f"Missing required columns: {missing_columns_status}"

    df["Alert_status"] = df.apply(determine_alert_status, axis=1)
    assert df["Alert_status"].notnull().all(), "Some 'Alert_status' values are null."
    df["Alert_status"] = df["Alert_status"].astype("category")
    print("✅ 'Alert_status' column added with categories:", df["Alert_status"].cat.categories.tolist())
    print("\nAlert Status Distribution:")
    print(df["Alert_status"].value_counts())

    return df


def filter_and_save_final(df: pd.DataFrame) -> None:
    """Filter the final dataset according to given conditions and save to CSV."""
    # Create response type based on alert type and status
    mask_non_alert = (
            (df["Alert_type"] == "Non_alert") &
            (df["Alert_status"] == "Non_alert")
    )
    df.loc[mask_non_alert, "ResponseType_cat"] = "Non_alert"

    mask_non_stoping_alert = (
            (df["Alert_type"] == "Non_Error_alert") &
            (df["Alert_status"] == "Non_stoping_alert")
    )
    df.loc[mask_non_stoping_alert, "ResponseType_cat"] = "Non_stoping_alert"

    rename_map = {
        "DRC - Frequency 1": "DRC_Frequency_1",
        "DRC - Single Dose 1": "DRC_Single_Dose_1",
        "DRC - Max Daily Dose 1": "DRC_Max_Daily_Dose_1"
    }
    df.rename(columns=rename_map, inplace=True)

    condition_filter = (
            ((df["DRC_Frequency_1"] < 2) | (df["DRC_Frequency_1"].isnull())) &
            ((df["DRC_Single_Dose_1"] < 2) | (df["DRC_Single_Dose_1"].isnull())) &
            ((df["DRC_Max_Daily_Dose_1"] < 2) | (df["DRC_Max_Daily_Dose_1"].isnull()))
    )

    df_final_filtered = df[condition_filter].copy()

    unique_VALIDATION = (
        df_final_filtered
        .groupby("Order_ID_new_update", as_index=False)
        .size()
        .rename(columns={"size": "total_count"})
    )
    unique_to_delete = unique_VALIDATION.loc[unique_VALIDATION["total_count"] == 2, "Order_ID_new_update"]

    test_final2 = df_final_filtered[
        (~df_final_filtered["Order_ID_new_update"].isin(unique_to_delete)) &
        (df_final_filtered["Order_ID_new_update"].notna())
        ].copy()

    test_hiba = test_final2[test_final2["ATC_NEW"].notna()]
    print(f'Final shape: {test_hiba.shape[0]}, {test_hiba.shape[1]}')

    # Save for inspection
    #output_path = "C:/Users/Keren/Desktop/Fatigue_alert/Data/Main_data_2022/df_main_flat_py_version.csv"
    output_path = "/Users/nirshlomo/Dropbox/PHD/alert-fatigue/alert_analysis/data/main_data_2022/df_main_flat_py_version.csv"
    test_hiba.to_csv(output_path, index=False)
    print(f"✅ Final flat file saved to {output_path}")


def main():
    # Step 1: Load data
    data_path = 'alert_analysis/data_process/data_distincted_main_new_with_cci.csv'
    df = load_data(data_path)

    # Step 2: Add adult-child and age categories
    df = add_adult_child_category(df)
    df = add_age_category(df)

    # Step 3: Extract and process drug names from alert messages
    df = extract_drugs_from_alert_message(df)
    df = remove_unwanted_words(df)
    df = fix_common_phrases(df)
    df = remove_duplicates_in_cell(df)

    # Step 4: Print sample output for inspection
    create_test_dataset(df)

    # Step 5: Remove duplicates and pivot the data
    flat_by_sevirity, src_for_flat_cln = remove_duplicates_and_pivot(df)

    # Step 6: Create alert sum column
    flat_by_sevirity = create_alert_sum_column(flat_by_sevirity)

    # Step 7: Create binary categorical flags for alerts
    flat_by_sevirity = create_categorical_flags(flat_by_sevirity)

    # Step 8: Calculate chronic count
    flat_by_sevirity_ud = calculate_chronic_num_calc(flat_by_sevirity)

    # Step 9: Calculate hospital days
    flat_by_sevirity_ud = calculate_hospital_days(flat_by_sevirity_ud)

    # Step 10: Join max(diff_time_mabat_ms)
    flat_by_sevirity_ud = join_max_diff_time(flat_by_sevirity_ud, src_for_flat_cln)

    # Step 11: Join DRC_SUB_GROUP and NeoDRC_SUB_GROUP
    flat_by_sevirity_ud = join_drc_sub_group(flat_by_sevirity_ud, src_for_flat_cln)
    flat_by_sevirity_ud = join_neodrc_sub_group(flat_by_sevirity_ud, src_for_flat_cln)

    print("Final dataset shape:", flat_by_sevirity_ud.shape)
    #print(flat_by_sevirity_ud.head(5))

    # Copy final dataset for further processing
    df_final = flat_by_sevirity_ud.copy()

    # Step 12: Determine Alert Type and Alert Status
    df_final = add_alert_type_and_status(df_final)

    # Verify final row count matches expected value
    expected_rows = 3615043
    assert df_final.shape[0] == expected_rows, f"Row count mismatch! Expected {expected_rows}, got {df_final.shape[0]}"
    print(f"\n✅ Final row count verified: {df_final.shape[0]}")

    # Step 13: Filter final dataset and save to CSV
    filter_and_save_final(df_final)


if __name__ == '__main__':
    main()