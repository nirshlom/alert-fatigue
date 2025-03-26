import pandas as pd
data_main = pd.read_csv('alert_analysis/data/main_data_2022/emek.data - Copy.csv')
data_main.shape
data_main.columns

#data_main = data_main.sample(100000)
# --------------------------------------------------------
# Convert 'Hospital' column to a categorical type and store in a new column
data_main['Hospital_cat'] = data_main['Hospital'].astype('category')

# Convert 'HospitalName' column to a categorical type and store in a new column
data_main['HospitalName_cat'] = data_main['HospitalName'].astype('category')

# Map Hebrew hospital names to their corresponding English names.
# If the hospital name is not found in the mapping, use 'YOSEFTAL' as the default.
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

# Use the mapping; if a hospital name is not in the dictionary, fill with 'YOSEFTAL'
data_main['HospitalName_EN_cat'] = (
    data_main['HospitalName']
    .map(hospital_name_mapping)
    .fillna('YOSEFTAL')
    .astype('category')
)


# --------------------------------------------------------
# Define a helper function to assign unit categories
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


# Apply the helper function to each row to create the 'UnitName_cat' column and convert it to a categorical type
data_main['UnitName_cat'] = data_main.apply(categorize_unit, axis=1).astype('category')

# --------------------------------------------------------
# Create a new DataFrame with only the selected columns
test_UnitName = data_main[['HospitalName', 'UnitName', 'UnitName_cat']]

# (Optional) Display the resulting DataFrame
print(test_UnitName.head())

# Define a mapping dictionary: 0 maps to "Silence Mode" and 1 maps to "Active Mode"
severity_mapping = {0: "Silence Mode", 1: "Active Mode"}

# Create a new column by mapping the 'SeverityLevelToStopOrder' values using the dictionary
data_main['SeverityLevelToStopOrder_cat'] = data_main['SeverityLevelToStopOrder'].map(severity_mapping)

# Convert the new column to a categorical type with the specified order of categories
data_main['SeverityLevelToStopOrder_cat'] = pd.Categorical(
    data_main['SeverityLevelToStopOrder_cat'],
    categories=["Silence Mode", "Active Mode"],
    ordered=True
)

# Print the levels (categories) of the new categorical column
print(data_main['SeverityLevelToStopOrder_cat'].cat.categories)

# Optional: Create a new DataFrame with only the relevant columns for testing
# test = data_main[['SeverityLevelToStopOrder', 'SeverityLevelToStopOrder_cat']]

import pandas as pd
import re
from datetime import datetime

# --------------------------------------------------------
# Convert AGE string to a numeric value using complex rules.
import numpy as np
# def convert_age(age):
#     """
#     Convert an AGE string into a numeric value based on its length and pattern.
#     For 3-digit fully numeric strings (e.g., "105"), return the number directly.
#     Otherwise, construct a decimal string based on specified positions only if
#     those positions are digits; if not, return np.nan.
#     """
#     s = str(age)
#     n = len(s)
#
#     # If it's a 3-digit fully numeric string, handle it as 100+ age.
#     if n == 3 and s.isdigit():
#         return float(s)
#
#     result = None
#     try:
#         if n == 6:
#             # Expect first and fifth characters to be digits.
#             if s[0].isdigit() and s[4].isdigit():
#                 result = f"{s[0]}.{s[4]}"
#         elif n == 7:
#             if s[0].isdigit() and s[1].isdigit():
#                 if s[5].isdigit():
#                     result = f"{s[0:2]}.{s[5]}"
#             elif s[0].isdigit() and not s[1].isdigit():
#                 if s[4:6].isdigit():
#                     result = f"{s[0]}.{s[4:6]}"
#         elif n == 8:
#             if s[0:2].isdigit() and s[5:7].isdigit():
#                 result = f"{s[0:2]}.{s[5:7]}"
#         elif n == 3:
#             # When not fully numeric, check the second character.
#             if not s[1].isdigit():
#                 if s[0].isdigit():
#                     result = f"0.0{s[0]}"
#             else:
#                 if s[0:2].isdigit():
#                     result = f"0.{s[0:2]}"
#         elif n == 4:
#             if s[0:2].isdigit():
#                 result = s[0:2]
#         elif n == 2:
#             if not s[1].isdigit():
#                 if s[0].isdigit():
#                     result = s[0]
#             else:
#                 if s.isdigit():
#                     result = s
#
#         # If no valid result was generated, return np.nan
#         if result is None:
#             return np.nan
#         return float(result)
#     except Exception:
#         return np.nan
#
#
# # Now apply the function to the AGE column:
# data_main['AGE_num'] = data_main['AGE'].apply(convert_age)

import pandas as pd
import numpy as np
import re

# Define a function to convert the AGE string to a numeric value based on its length and content.
def convert_age(age_str):
    # If the age value is missing, return NaN
    if pd.isna(age_str):
        return np.nan
    # Convert to string and get its length
    s = str(age_str)
    n = len(s)

    # 1. If the string has exactly 6 characters:
    #    - Take the first character and the 5th character (index 0 and 4),
    #      join them with a dot and convert to float.
    if n == 6:
        try:
            return float(f"{s[0]}.{s[4]}")
        except ValueError:
            return np.nan

    # 2. If the string has exactly 7 characters AND:
    #    - The first character is a digit AND the second character is also a digit,
    #      then take the first two characters and the 6th character (index 5).
    if n == 7 and s[0].isdigit() and s[1].isdigit():
        try:
            return float(f"{s[:2]}.{s[5]}")
        except ValueError:
            return np.nan

    # 3. If the string has exactly 7 characters AND:
    #    - The first character is a digit BUT the second character is NOT a digit,
    #      then take the first character and the 5th & 6th characters (indexes 4 and 5).
    if n == 7 and s[0].isdigit() and (not s[1].isdigit()):
        try:
            return float(f"{s[0]}.{s[4:6]}")
        except ValueError:
            return np.nan

    # 4. If the string has exactly 8 characters:
    #    - Take the first two characters and the 6th & 7th characters (indexes 5 and 6).
    if n == 8:
        try:
            return float(f"{s[:2]}.{s[5:7]}")
        except ValueError:
            return np.nan

    # 5. If the string has exactly 3 characters AND the second character is NOT a digit:
    #    - Concatenate "0", ".0", and the first character. (E.g., "A12" becomes "0.0A".)
    #      Note: In practice, the string should represent a number; this mirrors the R behavior.
    if n == 3 and (not s[1].isdigit()):
        try:
            return float(f"0.0{s[0]}")
        except ValueError:
            return np.nan

    # 6. If the string has exactly 3 characters AND the second character IS a digit:
    #    - Concatenate "0", ".", and the first two characters.
    if n == 3 and s[1].isdigit():
        try:
            return float(f"0.{s[:2]}")
        except ValueError:
            return np.nan

    # 7. If the string has exactly 4 characters:
    #    - Simply take the first two characters and convert to float.
    if n == 4:
        try:
            return float(s[:2])
        except ValueError:
            return np.nan

    # 8. If the string has exactly 2 characters AND the second character is not a digit:
    #    - Take the first character and convert it.
    if n == 2 and (not s[1].isdigit()):
        try:
            return float(s[0])
        except ValueError:
            return np.nan

    # Default: if none of the above conditions match, try converting the whole string to a float.
    try:
        return float(s)
    except ValueError:
        return np.nan


# Define a function to handle cases where the AGE string represents a three-digit number (100+).
def handle_hundred(age_str, current_value):
    # If the age value is missing, return the current converted value.
    if pd.isna(age_str):
        return current_value
    s = str(age_str)
    # If the string is exactly three digits (e.g., "101"), then convert the entire string directly.
    if len(s) == 3 and re.match(r'^\d{3}$', s):
        try:
            return float(s)
        except ValueError:
            return current_value
    # Otherwise, keep the value computed by convert_age.
    return current_value


# Assume data_main is your pandas DataFrame and it has an 'AGE' column.
# First, apply the conversion function to create a new column 'AGE_num'
data_main['AGE_num'] = data_main['AGE'].apply(convert_age)

# override the value for rows where AGE is a three-digit number (i.e. 100+)
data_main['AGE_num'] = data_main.apply(lambda row: handle_hundred(row['AGE'], row['AGE_num']), axis=1)

# --------------------------------------------------------
# Convert Gender_Text to categorical and map to English labels.
data_main['Gender_Text_cat'] = data_main['Gender_Text'].astype('category')
data_main['Gender_Text_EN_cat'] = data_main['Gender_Text'].apply(
    lambda x: 'MALE ' if x == 'זכר' else 'FEMALE'
).astype('category')

# --------------------------------------------------------
# Convert SectorText to categorical and map to English labels.
data_main['SectorText_cat'] = data_main['SectorText'].astype('category')
data_main['SectorText_EN_cat'] = data_main['SectorText'].apply(
    lambda x: 'DOCTOR' if x == 'רופא' else ('NURSE' if x == 'סיעוד' else 'PARAMEDICAL')
).astype('category')

# --------------------------------------------------------
# Create a new Order_ID by concatenating Hospital and Order_ID (removing any spaces).
data_main['Order_ID_new'] = (data_main['Hospital'].astype(str) + "_" + data_main['Order_ID'].astype(str)).str.replace(
    " ", "")


# --------------------------------------------------------
# Process Time_Prescribing_Order to determine the shift type.
def determine_shift(time_str):
    """
    Determine the shift based on the time string (HH:MM:SS):
      - morning: 07:00:00 <= time < 15:00:00
      - afternoon: 15:01:00 <= time < 23:00:00
      - night: all other times (including exactly 15:00:00 or times between 23:00:00 and 07:00:00)
    """
    if "07:00:00" <= time_str < "15:00:00":
        return "morning"
    elif "15:01:00" <= time_str < "23:00:00":
        return "afternoon"
    else:
        return "night"


# First, extract the time part from Time_Prescribing_Order (assumed to be a datetime string).
data_main['ShiftType_cat'] = pd.to_datetime(data_main['Time_Prescribing_Order']).dt.strftime('%H:%M:%S')
# Then, apply the shift determination function and convert the result to a categorical type.
data_main['ShiftType_cat'] = data_main['ShiftType_cat'].apply(determine_shift).astype('category')

# --------------------------------------------------------
# Convert DayHebrew to a categorical variable with a specified order.
day_order = ["שבת", "שישי", "חמישי", "רביעי", "שלישי", "שני", "ראשון"]
data_main['DayHebrew_cat'] = pd.Categorical(data_main['DayHebrew'], categories=day_order, ordered=True)
# Print the levels (categories)
print(data_main['DayHebrew_cat'].cat.categories)

# Map Hebrew day names to English day names.
day_mapping = {
    "ראשון": "Sunday",
    "שני": "Monday",
    "שלישי": "Tuesday",
    "רביעי": "Wednesday ",
    "חמישי": "Thursday",
    "שישי": "Friday",
    "שבת": "Saturday"
}
data_main['DayEN_cat'] = data_main['DayHebrew_cat'].map(day_mapping).astype('category')

import numpy as np
import pandas as pd

#--------------------------------------------------------
# If HospAmount is -1, set HospAmount_new to 0; otherwise, keep the original value.
data_main['HospAmount_new'] = np.where(data_main['HospAmount'] == -1, 0, data_main['HospAmount'])
# For validation you might display:
# print(data_main[['HospAmount', 'HospAmount_new']].head())

#--------------------------------------------------------
# IS_New_Order - no operations are defined for this field in the given chunk.

#--------------------------------------------------------
# Convert Drug_Header to a categorical type.
data_main['Drug_Header_cat'] = data_main['Drug_Header'].astype('category')

#--------------------------------------------------------
# Convert Excel-style time columns to proper datetime values.
# In R: as.POSIXct(x * 86400, origin='1899-12-30', tz="UTC")
# Here we multiply by 86400 to convert days to seconds and specify the origin date.
data_main['Time_Mabat_Request_convert_ms_res'] = pd.to_datetime(
    data_main['Time_Mabat_Request'] * 86400, unit='s', origin='1899-12-30', utc=True
)
data_main['Time_Mabat_Response_convert_ms_res'] = pd.to_datetime(
    data_main['Time_Mabat_Response'] * 86400, unit='s', origin='1899-12-30', utc=True
)

# Calculate the difference in time (in milliseconds) between response and request.
data_main['diff_time_mabat_ms'] = (
    data_main['Time_Mabat_Response_convert_ms_res'] - data_main['Time_Mabat_Request_convert_ms_res']
).dt.total_seconds() * 1000

#--------------------------------------------------------
# Create a new column ATC_cln by taking the first 7 characters of the ATC column.
data_main['ATC_cln'] = data_main['ATC'].astype(str).str[:7]
# Replace the string 'NA' with actual NaN values.
data_main.loc[data_main['ATC_cln'] == 'NA', 'ATC_cln'] = np.nan

#--------------------------------------------------------
# Rename the column "Alert\\Message" to "Alert_Message".
data_main.rename(columns={"Alert\\Message": "Alert_Message"}, inplace=True)

#--------------------------------------------------------
# Map Answer_Text to an English version using nested conditions.
# Define conditions for each mapping.
conditions = [
    data_main['Answer_Text'].isin([
        'אחר - נא לפרט', 'אחר - נא פרט', 'אחר - פירוט', 'אחר-פרט', 'אחר - פרט'
    ]),
    data_main['Answer_Text'].isin([
        'הודעת מערכת (התראה טכנית)', 'הודעת מערכת (התרעה טכנית)'
    ]),
    data_main['Answer_Text'].isin([
        'התרופה אינה ממופה', 'התרופה לא ממופה',
        'התרופה לא ממופה – NOT MAPPED', 'התרופה לא ממופה Not Mapped'
    ]),
    data_main['Answer_Text'] == 'קיים מידע בספרות המקצועית הרלוונטית'
]

choices = [
    'Other - detail',
    'Technical system alert',
    'Not Mapped',
    'Information exists in the relevant professional literature'
]

# Create a new column Answer_Text_EN based on the conditions;
# if none of the conditions are met, keep the original Answer_Text.
data_main['Answer_Text_EN'] = np.select(conditions, choices, default=data_main['Answer_Text'])

import numpy as np
import pandas as pd

#--------------------------------------------------------
# Replace 'NA' strings in Other_Text with actual NaN values.
data_main.loc[data_main["Other_Text"] == 'NA', "Other_Text"] = np.nan

#--------------------------------------------------------
# Combine Module_Alert_Rn and Alert_Severity into Alert_Rn_Severity_cat.
# This column is created using several nested conditions similar to R's ifelse and grepl.

# Condition 1: If "DDI" appears in Module_Alert_Rn, then use "DDI-" concatenated with Alert_Severity.
mask = data_main['Module_Alert_Rn'].str.contains("DDI", na=False)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "DDI-" + data_main.loc[mask, 'Alert_Severity'].astype(str)

# Condition 2: If "DAM" appears in Module_Alert_Rn, set to "DAM".
mask = data_main['Module_Alert_Rn'].str.contains("DAM", na=False)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "DAM"

# Condition 3: If "DT" appears in Module_Alert_Rn, set to "DT".
mask = data_main['Module_Alert_Rn'].str.contains("DT", na=False)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "DT"

# Condition 4: For Module_Alert_Rn equal to "NeoDRC - Message 1" and if Alert_Message contains any of:
# "Renal adjustmen", "Weight is required", "Gestational age", or "A valid order is required",
# then assign "Technical alert".
mask = (data_main['Module_Alert_Rn'] == "NeoDRC - Message 1") & (
    data_main['Alert_Message'].str.contains("Renal adjustmen", na=False) |
    data_main['Alert_Message'].str.contains("Weight is required", na=False) |
    data_main['Alert_Message'].str.contains("Gestational age", na=False) |
    data_main['Alert_Message'].str.contains("A valid order is required", na=False)
)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

# Condition 5: If "Neo" appears in Module_Alert_Rn, assign "NeoDRC".
mask = data_main['Module_Alert_Rn'].str.contains("Neo", na=False)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "NeoDRC"

# Condition 6: If Module_Alert_Rn is one of these DRC types, assign "DRC".
drc_types = ["DRC - Frequency 1", "DRC - Single Dose 1", "DRC - Single Dose 2", "DRC - Max Daily Dose 1"]
mask = data_main['Module_Alert_Rn'].isin(drc_types)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "DRC"

# Condition 7: If any of the following strings appear in Alert_Message or if Module_Alert_Rn is "MA -  1" or "MA -  2",
# assign "Technical alert". Also check for "Renal adjustment" in Alert_Message.
mask = (
    data_main['Alert_Message'].str.contains("Weight is required", na=False) |
    data_main['Alert_Message'].str.contains("Unable to convert units", na=False) |
    data_main['Alert_Message'].str.contains("The drug may be used short term", na=False) |
    data_main['Alert_Message'].str.contains("Frequency is required", na=False) |
    data_main['Alert_Message'].str.contains("amoxicillin/potassium clavulanate", na=False) |
    data_main['Alert_Message'].str.contains("Missing Data Information", na=False) |
    data_main['Alert_Message'].str.contains("Clinical route is required", na=False) |
    data_main['Alert_Message'].str.contains("fosfomycin trometamol", na=False) |
    data_main['Alert_Message'].str.contains("Unknown dose unit", na=False) |
    data_main['Module_Alert_Rn'].isin(["MA -  1", "MA -  2"]) |
    data_main['Alert_Message'].str.contains("Renal adjustment", na=False)
)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

# Condition 8: If Alert_Message contains "A Creatinine Clearance Range from", assign "Renal alerts".
mask = data_main['Alert_Message'].str.contains("A Creatinine Clearance Range from", na=False)
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "Renal alerts"

# For rows where Alert_Rn_Severity_cat is still missing and Module_Alert_Rn equals "DRC - Message 1" or "NeoDRC - Message 1",
# assign "Technical alert".
mask = (data_main['Module_Alert_Rn'] == "DRC - Message 1") & (data_main['Alert_Rn_Severity_cat'].isna())
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"
mask = (data_main['Module_Alert_Rn'] == "NeoDRC - Message 1") & (data_main['Alert_Rn_Severity_cat'].isna())
data_main.loc[mask, 'Alert_Rn_Severity_cat'] = "Technical alert"

# Convert the column to a categorical type.
data_main['Alert_Rn_Severity_cat'] = data_main['Alert_Rn_Severity_cat'].astype('category')

# For validation, you can extract the following columns:
test22 = data_main[['Module_Alert_Rn', 'Alert_Message', 'Alert_Rn_Severity_cat', 'Module_Severity_Rn']]

#--------------------------------------------------------
# Create DRC_SUB_GROUP: if Module_Alert_Rn is in a specified list, keep its value; otherwise, set to NaN.
drc_values = ["DRC - Duration 1", "DRC - Frequency 1", "DRC - Max Daily Dose 1", "DRC - Message 1", "DRC - Single Dose 1"]
data_main['DRC_SUB_GROUP'] = np.where(data_main['Module_Alert_Rn'].isin(drc_values), data_main['Module_Alert_Rn'], np.nan)
data_main['DRC_SUB_GROUP'] = data_main['DRC_SUB_GROUP'].astype('category')

# Create NeoDRC_SUB_GROUP similarly.
neodrc_values = ["NeoDRC - Duration 1", "NeoDRC - Frequency 1", "NeoDRC - Max Daily Dose 1", "NeoDRC - Message 1", "NeoDRC - Single Dose 1"]
data_main['NeoDRC_SUB_GROUP'] = np.where(data_main['Module_Alert_Rn'].isin(neodrc_values), data_main['Module_Alert_Rn'], np.nan)
data_main['NeoDRC_SUB_GROUP'] = data_main['NeoDRC_SUB_GROUP'].astype('category')

#--------------------------------------------------------
# Create ResponseType_cat:
# - If ResponseType contains "Change", set to "Change".
# - Else if ResponseType equals "Non Alert" and Alert_Message is not missing, set to "Non_stoping_alert".
# - Otherwise, keep the original ResponseType.
data_main['ResponseType_cat'] = np.where(
    data_main['ResponseType'].str.contains("Change", na=False),
    "Change",
    np.where(
        (data_main['ResponseType'] == "Non Alert") & (data_main['Alert_Message'].notna()),
        "Non_stoping_alert",
        data_main['ResponseType']
    )
)
data_main['ResponseType_cat'] = data_main['ResponseType_cat'].astype('category')

#--------------------------------------------------------
data_main.shape #(4730108, 72)
## END of 01_data_main_prep.py
data_main.to_csv('alert_analysis/data_process/data_main_prep.csv', index=False)