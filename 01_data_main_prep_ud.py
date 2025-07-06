import pandas as pd
import numpy as np
import re
from datetime import datetime
import os


# -------------------------------
# Helper Functions
# -------------------------------

def load_data(filepath):
    """Load CSV data and perform a basic validation."""
    data = pd.read_csv(filepath)
    assert not data.empty, f"Data loaded from {filepath} is empty!"
    print(f"Data loaded successfully with shape: {data.shape}")
    return data


def process_hospital_columns(data):
    """Process hospital columns: convert to categorical and map Hebrew names to English."""
    data['Hospital_cat'] = data['Hospital'].astype('category')
    data['HospitalName_cat'] = data['HospitalName'].astype('category')

    hospital_name_mapping = {
        'בילינסון': 'BELINSON',
        'בית רבקה': 'BET_RIVKA',
        'כרמל': 'CARMEL',
        'העמק': 'EMEK',
        'קפלן': 'KAPLAN',
        'לווינשטיין': 'LEVINSTIEN',
        'מאיר': 'MEIR',
        'שניידר': 'SCHIENDER',
        'השרון': 'SHARON',
        'סורוקה': 'SOROKA'
    }
    data['HospitalName_EN_cat'] = (
        data['HospitalName']
        .map(hospital_name_mapping)
        .fillna('YOSEFTAL')
        .astype('category')
    )
    print("Hospital columns processed.")


def categorize_unit(row):
    """
    Categorize the unit based on various conditions applied on 'UnitName'
    and 'HospitalName', mimicking the nested ifelse logic from R.
    """
    unit = row['UnitName']
    hospital = row['HospitalName']

    # Condition for 'Surgery'
    if (
            ("אורתופד" in unit) or ("אורטופד" in unit) or
            ("כירורגי" in unit) or ("אורולוג" in unit) or
            (unit == "מחלקת א.א.ג") or ("עיניים" in unit) or
            ("פלסטיק" in unit) or ("אוזן" in unit) or
            (unit == "אאג") or ("כלי דם" in unit) or
            (unit == "ניתוחי עמוד שדרה") or (unit == "ניתוחי חזה ולב - מחלקה")
    ):
        return "Surgery"

    # Condition for 'Emergency'
    elif (
            ("מיון" in unit) or (unit == "המחלקה לרפואה דחופה") or
            ("מלרד" in unit) or (unit == "רפואה דחופה זיהומים")
    ):
        return "Emergency"

    # Condition for 'Internal'
    elif (
            ("פנימית" in unit) or ("עצבים" in unit) or
            (unit == "מחלקת עור") or ("נוירולוגיה" in unit) or
            (unit == "מחלקת ריאות") or (unit == "מחלקת שיקום") or
            (unit == "מחלקה לעור ומין")
    ):
        return "Internal"

    # Condition for 'Pediatric'
    elif (
            ("ילדים" in unit) or ("ילודי" in unit) or
            (unit == "אשפוז בית" and hospital == "שניידר")
    ):
        return "Pediatric"

    # Condition for 'Internal-Covid19'
    elif "קורונה" in unit:
        return "Internal-Covid19"

    # Condition for 'Ambulatory'
    elif unit == "מרפאה פנימית ג":
        return "Ambulatory"

    # Condition for 'Cardiology'
    elif "קרדיולוגיה" in unit:
        return "Cardiology"

    # Condition for 'Geriatric'
    elif (hospital == "בית רבקה") or ("גריאטריה" in unit):
        return "Geriatric"

    # Additional condition for 'Pediatric' (for a specific hospital and unit)
    elif (hospital == "לווינשטיין") and (unit == "מחלקת ילדים"):
        return "Pediatric"

    # Condition for 'Rehabilitation' for the same hospital if unit is not 'מחלקת ילדים'
    elif (hospital == "לווינשטיין") and (unit != "מחלקת ילדים"):
        return "Rehabilitation"

    # Another condition for 'Cardiology' based on keyword 'לב'
    elif "לב" in unit:
        return "Cardiology"

    # Condition for 'Day_care'
    elif (
            ("אשפוז יום" in unit) or
            (unit == "גסטרואנטרולוגיה - מכון") or
            (unit == "מחלקת ריאות") or
            (unit == "הפרעות אכילה - מחלקה") or
            (unit == "טיפול יום פסיכיאטרי") or
            (unit == "מכון ריאות")
    ):
        return "Day_care "

    # Condition for 'Hematology'
    elif (
            (unit == "השתלות מוח עצם") or
            (unit == "מחלקה להמטואונקולוגיה") or
            (unit == "מחלקה להמטולוגיה")
    ):
        return "Hematology"

    # Condition for 'ICU'
    elif "טיפול נמרץ" in unit:
        return "ICU"

    # Condition for 'Nephrology'
    elif ("נפרולוג" in unit) or ("דיאליזה" in unit):
        return "Nephrology"

    # Condition for 'Oncology'
    elif unit == "המחלקה לאונקולוגיה":
        return "Oncology"

    # Default: 'Gynecology'
    else:
        return "Gynecology"


def process_unit(data):
    """Apply unit categorization and print a sample of the results."""
    data['UnitName_cat'] = data.apply(categorize_unit, axis=1).astype('category')
    test_UnitName = data[['HospitalName', 'UnitName', 'UnitName_cat']]
    print("Sample of unit categorization:\n", test_UnitName.head())


def process_severity(data):
    """Map and convert SeverityLevelToStopOrder to a categorical variable."""
    severity_mapping = {0: "Silence Mode", 1: "Active Mode"}
    data['SeverityLevelToStopOrder_cat'] = data['SeverityLevelToStopOrder'].map(severity_mapping)
    data['SeverityLevelToStopOrder_cat'] = pd.Categorical(
        data['SeverityLevelToStopOrder_cat'],
        categories=["Silence Mode", "Active Mode"],
        ordered=True
    )
    print("SeverityLevelToStopOrder categories:", data['SeverityLevelToStopOrder_cat'].cat.categories)


def convert_age(age_str):
    """Convert an AGE string into a numeric value based on its length and content."""
    if pd.isna(age_str):
        return np.nan
    s = str(age_str)
    n = len(s)

    if n == 6:
        try:
            return float(f"{s[0]}.{s[4]}")
        except ValueError:
            return np.nan

    if n == 7 and s[0].isdigit() and s[1].isdigit():
        try:
            return float(f"{s[:2]}.{s[5]}")
        except ValueError:
            return np.nan

    if n == 7 and s[0].isdigit() and (not s[1].isdigit()):
        try:
            return float(f"{s[0]}.{s[4:6]}")
        except ValueError:
            return np.nan

    if n == 8:
        try:
            return float(f"{s[:2]}.{s[5:7]}")
        except ValueError:
            return np.nan

    if n == 3 and (not s[1].isdigit()):
        try:
            return float(f"0.0{s[0]}")
        except ValueError:
            return np.nan

    if n == 3 and s[1].isdigit():
        try:
            return float(f"0.{s[:2]}")
        except ValueError:
            return np.nan

    if n == 4:
        try:
            return float(s[:2])
        except ValueError:
            return np.nan

    if n == 2 and (not s[1].isdigit()):
        try:
            return float(s[0])
        except ValueError:
            return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def handle_hundred(age_str, current_value):
    """Override conversion for three-digit numbers representing 100+."""
    if pd.isna(age_str):
        return current_value
    s = str(age_str)
    if len(s) == 3 and re.match(r'^\d{3}$', s):
        try:
            return float(s)
        except ValueError:
            return current_value
    return current_value


def process_age(data):
    """Convert AGE strings to numeric values."""
    data['AGE_num'] = data['AGE'].apply(convert_age)
    data['AGE_num'] = data.apply(lambda row: handle_hundred(row['AGE'], row['AGE_num']), axis=1)
    print("AGE conversion completed.")


def process_gender(data):
    """Convert Gender_Text to categorical and map to English labels."""
    data['Gender_Text_cat'] = data['Gender_Text'].astype('category')
    data['Gender_Text_EN_cat'] = data['Gender_Text'].apply(
        lambda x: 'MALE ' if x == 'זכר' else 'FEMALE'
    ).astype('category')
    print("Gender conversion completed.")


def process_sector(data):
    """Convert SectorText to categorical and map to English labels."""
    data['SectorText_cat'] = data['SectorText'].astype('category')
    data['SectorText_EN_cat'] = data['SectorText'].apply(
        lambda x: 'DOCTOR' if x == 'רופא' else ('NURSE' if x == 'סיעוד' else 'PARAMEDICAL')
    ).astype('category')
    print("Sector conversion completed.")


def process_order_id(data):
    """Create a new Order_ID by concatenating Hospital and Order_ID (removing spaces)."""
    data['Order_ID_new'] = (data['Hospital'].astype(str) + "_" + data['Order_ID'].astype(str)).str.replace(" ", "")
    print("Order_ID_new created.")


def determine_shift(time_str):
    """
    Determine the shift based on the time string (HH:MM:SS):
      - morning: 07:00:00 <= time < 15:00:00
      - afternoon: 15:01:00 <= time < 23:00:00
      - night: all other times
    """
    if "07:00:00" <= time_str < "15:00:00":
        return "morning"
    elif "15:01:00" <= time_str < "23:00:00":
        return "afternoon"
    else:
        return "night"


def process_shift(data):
    """Determine the shift type from Time_Prescribing_Order and convert it to a categorical type."""
    time_str = pd.to_datetime(data['Time_Prescribing_Order']).dt.strftime('%H:%M:%S')
    data['ShiftType_cat'] = time_str.apply(determine_shift).astype('category')
    print("Shift type processed.")


def process_day(data):
    """Convert DayHebrew to a categorical variable and map to English day names."""
    day_order = ["שבת", "שישי", "חמישי", "רביעי", "שלישי", "שני", "ראשון"]
    data['DayHebrew_cat'] = pd.Categorical(data['DayHebrew'], categories=day_order, ordered=True)
    print("DayHebrew categories:", data['DayHebrew_cat'].cat.categories)
    day_mapping = {
        "ראשון": "Sunday",
        "שני": "Monday",
        "שלישי": "Tuesday",
        "רביעי": "Wednesday ",
        "חמישי": "Thursday",
        "שישי": "Friday",
        "שבת": "Saturday"
    }
    data['DayEN_cat'] = data['DayHebrew_cat'].map(day_mapping).astype('category')
    print("Day mapping completed.")


def process_hosp_amount(data):
    """Set HospAmount_new to 0 when HospAmount is -1; otherwise, keep original value."""
    data['HospAmount_new'] = np.where(data['HospAmount'] == -1, 0, data['HospAmount'])
    print("HospAmount_new processed.")


def process_drug_header(data):
    """Convert Drug_Header to a categorical type."""
    data['Drug_Header_cat'] = data['Drug_Header'].astype('category')
    print("Drug_Header converted to categorical.")


def process_time_columns(data):
    """
    Convert Excel-style time columns to datetime values and calculate the difference in milliseconds.
    In R: as.POSIXct(x * 86400, origin='1899-12-30', tz="UTC")
    """
    data['Time_Mabat_Request_convert_ms_res'] = pd.to_datetime(
        data['Time_Mabat_Request'] * 86400, unit='s', origin='1899-12-30', utc=True
    )
    data['Time_Mabat_Response_convert_ms_res'] = pd.to_datetime(
        data['Time_Mabat_Response'] * 86400, unit='s', origin='1899-12-30', utc=True
    )
    data['diff_time_mabat_ms'] = (
                                         data['Time_Mabat_Response_convert_ms_res'] - data[
                                     'Time_Mabat_Request_convert_ms_res']
                                 ).dt.total_seconds() * 1000
    print("Time columns processed.")


def process_atc(data):
    """Create a new ATC_cln column from the ATC column and replace 'NA' strings with NaN."""
    data['ATC_cln'] = data['ATC'].astype(str).str[:7]
    data.loc[data['ATC_cln'] == 'NA', 'ATC_cln'] = np.nan
    print("ATC_cln column created.")


def rename_alert_message(data):
    """Rename the column 'Alert\\Message' to 'Alert_Message'."""
    data.rename(columns={"Alert\\Message": "Alert_Message"}, inplace=True)
    print("Column 'Alert\\Message' renamed to 'Alert_Message'.")


def process_answer_text(data):
    """Map Answer_Text to an English version using specified conditions."""
    conditions = [
        data['Answer_Text'].isin(['אחר - נא לפרט', 'אחר - נא פרט', 'אחר - פירוט', 'אחר-פרט', 'אחר - פרט']),
        data['Answer_Text'].isin(['הודעת מערכת (התראה טכנית)', 'הודעת מערכת (התרעה טכנית)']),
        data['Answer_Text'].isin(['התרופה אינה ממופה', 'התרופה לא ממופה',
                                  'התרופה לא ממופה – NOT MAPPED', 'התרופה לא ממופה Not Mapped']),
        data['Answer_Text'] == 'קיים מידע בספרות המקצועית הרלוונטית'
    ]
    choices = [
        'Other - detail',
        'Technical system alert',
        'Not Mapped',
        'Information exists in the relevant professional literature'
    ]
    data['Answer_Text_EN'] = np.select(conditions, choices, default=data['Answer_Text'])
    print("Answer_Text mapping completed.")


def process_other_text(data):
    """Replace 'NA' strings in Other_Text with actual NaN values."""
    data.loc[data["Other_Text"] == 'NA', "Other_Text"] = np.nan
    print("Other_Text cleaned.")


def process_alert_rn_severity(data):
    """
    Combine Module_Alert_Rn and Alert_Severity into Alert_Rn_Severity_cat using several conditions.
    """
    # Condition 1: DDI
    mask = data['Module_Alert_Rn'].str.contains("DDI", na=False)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "DDI-" + data.loc[mask, 'Alert_Severity'].astype(str)

    # Condition 2: DAM
    mask = data['Module_Alert_Rn'].str.contains("DAM", na=False)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "DAM"

    # Condition 3: DT
    mask = data['Module_Alert_Rn'].str.contains("DT", na=False)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "DT"

    # Condition 4: NeoDRC - Message 1 with specific alert texts
    mask = (data['Module_Alert_Rn'] == "NeoDRC - Message 1") & (
            data['Alert_Message'].str.contains("Renal adjustmen", na=False) |
            data['Alert_Message'].str.contains("Weight is required", na=False) |
            data['Alert_Message'].str.contains("Gestational age", na=False) |
            data['Alert_Message'].str.contains("A valid order is required", na=False)
    )
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

    # Condition 5: NeoDRC
    mask = data['Module_Alert_Rn'].str.contains("Neo", na=False)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "NeoDRC"

    # Condition 6: DRC types
    drc_types = ["DRC - Frequency 1", "DRC - Single Dose 1", "DRC - Single Dose 2", "DRC - Max Daily Dose 1"]
    mask = data['Module_Alert_Rn'].isin(drc_types)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "DRC"

    # Condition 7: Technical alert based on Alert_Message or specific Module_Alert_Rn values
    mask = (
            data['Alert_Message'].str.contains("Weight is required", na=False) |
            data['Alert_Message'].str.contains("Unable to convert units", na=False) |
            data['Alert_Message'].str.contains("The drug may be used short term", na=False) |
            data['Alert_Message'].str.contains("Frequency is required", na=False) |
            data['Alert_Message'].str.contains("amoxicillin/potassium clavulanate", na=False) |
            data['Alert_Message'].str.contains("Missing Data Information", na=False) |
            data['Alert_Message'].str.contains("Clinical route is required", na=False) |
            data['Alert_Message'].str.contains("fosfomycin trometamol", na=False) |
            data['Alert_Message'].str.contains("Unknown dose unit", na=False) |
            data['Module_Alert_Rn'].isin(["MA -  1", "MA -  2"]) |
            data['Alert_Message'].str.contains("Renal adjustment", na=False)
    )
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

    # Condition 8: Renal alerts based on Alert_Message
    mask = data['Alert_Message'].str.contains("A Creatinine Clearance Range from", na=False)
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "Renal alerts"

    # Condition 9: For remaining missing values with specific Module_Alert_Rn, assign Technical alert
    mask = (data['Module_Alert_Rn'] == "DRC - Message 1") & (data['Alert_Rn_Severity_cat'].isna())
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"
    mask = (data['Module_Alert_Rn'] == "NeoDRC - Message 1") & (data['Alert_Rn_Severity_cat'].isna())
    data.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

    data['Alert_Rn_Severity_cat'] = data['Alert_Rn_Severity_cat'].astype('category')
    print("Alert_Rn_Severity_cat processed.")


def process_below_exceed_dose(data):
    """
    Creates six new columns indicating dose direction (exceeds/below) for different alert types:
    1. dose_direction_DRC_Frequency_1
    2. dose_direction_DRC_Single_Dose_1
    3. dose_direction_DRC_Max_Daily_Dose_1
    4. dose_direction_NeoDRC_Frequency_1
    5. dose_direction_NeoDRC_Single_Dose_1
    6. dose_direction_NeoDRC_Max_Daily_Dose_1

    Each column will contain:
    - "exceeds" if the alert message contains "exceeds"
    - "below" if the alert message contains "below"
    - None otherwise

    Parameters:
        data (pd.DataFrame): Input DataFrame containing 'Module_Alert_Rn' and 'Alert_Message' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with six new dose direction columns.
    """
    # Ensure 'Alert_Message' column is of string type
    data['Alert_Message'] = data['Alert_Message'].astype(str)

    # Define the alert types and their corresponding columns
    alert_types = {
        'DRC - Frequency 1': 'dose_direction_DRC_Frequency_1',
        'DRC - Single Dose 1': 'dose_direction_DRC_Single_Dose_1',
        'DRC - Max Daily Dose 1': 'dose_direction_DRC_Max_Daily_Dose_1',
        'NeoDRC - Frequency 1': 'dose_direction_NeoDRC_Frequency_1',
        'NeoDRC - Single Dose 1': 'dose_direction_NeoDRC_Single_Dose_1',
        'NeoDRC - Max Daily Dose 1': 'dose_direction_NeoDRC_Max_Daily_Dose_1'
    }

    # Initialize all columns with None
    for col in alert_types.values():
        data[col] = None

    # Process each alert type
    for alert_type, col_name in alert_types.items():
        # Create mask for rows with this alert type
        mask = data['Module_Alert_Rn'] == alert_type
        
        # Set values based on Alert_Message content
        data.loc[mask & data['Alert_Message'].str.contains("exceeds", na=False), col_name] = "exceeds"
        data.loc[mask & data['Alert_Message'].str.contains("below", na=False), col_name] = "below"

    return data


def process_drc_subgroup(data):
    """Create DRC_SUB_GROUP and NeoDRC_SUB_GROUP columns based on Module_Alert_Rn."""
    drc_values = ["DRC - Duration 1", "DRC - Frequency 1", "DRC - Max Daily Dose 1", "DRC - Message 1",
                  "DRC - Single Dose 1"]
    data['DRC_SUB_GROUP'] = np.where(data['Module_Alert_Rn'].isin(drc_values), data['Module_Alert_Rn'], np.nan)
    data['DRC_SUB_GROUP'] = data['DRC_SUB_GROUP'].astype('category')

    neodrc_values = ["NeoDRC - Duration 1", "NeoDRC - Frequency 1", "NeoDRC - Max Daily Dose 1", "NeoDRC - Message 1",
                     "NeoDRC - Single Dose 1"]
    data['NeoDRC_SUB_GROUP'] = np.where(data['Module_Alert_Rn'].isin(neodrc_values), data['Module_Alert_Rn'], np.nan)
    data['NeoDRC_SUB_GROUP'] = data['NeoDRC_SUB_GROUP'].astype('category')
    print("DRC and NeoDRC subgroups created.")


def process_response_type(data):
    """
    Create ResponseType_cat:
      - If ResponseType contains "Change", set to "Change".
      - Else if ResponseType equals "Non Alert" and Alert_Message is not missing, set to "Non_stoping_alert".
      - Otherwise, keep the original ResponseType.
    """
    data['ResponseType_cat'] = np.where(
        data['ResponseType'].str.contains("Change", na=False),
        "Change",
        np.where(
            (data['ResponseType'] == "Non Alert") & (data['Alert_Message'].notna()),
            "Non_stoping_alert",
            data['ResponseType']
        )
    )
    data['ResponseType_cat'] = data['ResponseType_cat'].astype('category')
    print("ResponseType_cat processed.")


def save_data(data, output_path):
    """Save the processed DataFrame to a CSV file."""
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


# -------------------------------
# Main processing function
# -------------------------------

def main():
    input_file = os.path.join('alert_analysis', 'data', 'main_data_2022', 'emek.data - Copy.csv')
    output_file = os.path.join('alert_analysis', 'data_process', 'data_main_prep.csv')

    data = load_data(input_file)

    process_hospital_columns(data)
    process_unit(data)
    process_severity(data)
    process_age(data)
    process_gender(data)
    process_sector(data)
    process_order_id(data)
    process_shift(data)
    process_day(data)
    process_hosp_amount(data)
    process_drug_header(data)
    process_time_columns(data)
    process_atc(data)
    rename_alert_message(data)
    process_answer_text(data)
    process_other_text(data)
    process_alert_rn_severity(data)
    process_below_exceed_dose(data)
    process_drc_subgroup(data)
    process_response_type(data)

    print("Final data shape:", data.shape)
    save_data(data, output_file)


if __name__ == "__main__":
    main()