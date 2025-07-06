# Alert Fatigue Analysis

A comprehensive analysis system for studying alert fatigue in healthcare settings, focusing on medication alerts and their impact on healthcare providers.

## Project Structure

The project is organized into several key Python scripts that handle different aspects of the data processing and analysis pipeline:

### Data Preparation and Processing

1. **01_data_main_prep_ud.py**
   - Initial data preprocessing and cleaning
   - Handles hospital and unit categorization
   - Processes time columns and converts them to appropriate formats
   - Maps Hebrew text to English for consistency
   - Creates dose direction indicators for DRC and NeoDRC alerts
   - Key functions:
     - `process_hospital_columns()`: Maps hospital names and categories
     - `process_time_columns()`: Converts Excel-style time to datetime
     - `process_drc_subgroup()`: Creates DRC and NeoDRC subgroups
     - `process_below_exceed_dose()`: Creates dose direction indicators for DRC and NeoDRC alerts

2. **02_data_main_prep_ud.py**
   - Column selection and data transformation
   - Medication count calculations
   - Disease keyword analysis
   - Preserves dose direction columns through the pipeline
   - Key functions:
     - `select_columns()`: Manages column selection and ordering
     - `calculate_medication_count()`: Computes medication counts per patient
     - `count_disease_keywords()`: Analyzes disease mentions in diagnoses

3. **03_data_main_cln_prep_ud.py**
   - Data cleaning and final preparation
   - Handles alert type categorization
   - Processes time differences and response metrics
   - Key functions:
     - `join_max_diff_time()`: Calculates maximum time differences
     - `determine_alert_type()`: Categorizes different types of alerts

4. **04_main_adults_flat_ud.py**
   - Focuses on adult patient analysis
   - Computes load indices and shift patterns
   - Handles age categorization and filtering
   - Key functions:
     - `compute_load_index()`: Calculates workload metrics per shift
     - `convert_age_cat()`: Processes age categories

5. **07_ignore_others.py**
   - Processes ignored orders and their associated text
   - Merges main data with ignore orders data
   - Handles UTF-8 encoding for proper text processing
   - Key features:
     - Merges data based on drug_order_id
     - Preserves response reasons text
     - Outputs UTF-8 encoded CSV file
     - Returns merged DataFrame for further analysis

### Data Analysis

6. **main_analysis.ipynb**
   - Jupyter notebook for exploratory analysis
   - Contains visualizations and statistical summaries
   - Key analyses:
     - Patient characteristics by gender
     - Alert type distribution
     - Order analysis by ATC group
     - Dose direction analysis for DRC and NeoDRC alerts

## Key Features

- **Comprehensive Data Processing**: Handles complex healthcare data with multiple variables
- **Multilingual Support**: Processes both Hebrew and English text
- **Time Analysis**: Detailed analysis of alert response times
- **Patient Categorization**: Sophisticated patient grouping based on multiple factors
- **Alert Classification**: Detailed categorization of different alert types
- **Dose Direction Analysis**: Tracks whether doses exceed or fall below recommended ranges for DRC and NeoDRC alerts
- **Ignore Orders Processing**: Tracks and analyzes orders that were ignored, including their associated text

## Column Renaming

The project includes a comprehensive column renaming system (`column_rename.py`) that standardizes column names throughout the pipeline. Key renaming patterns include:

- Hospital and unit information: `Hospital_cat` → `hospital_code`
- Alert-related columns: `Alert_Message` → `alert_message`
- DRC and NeoDRC dose direction columns:
  - `dose_direction_DRC_Frequency_1` → `drc_frequency_direction`
  - `dose_direction_DRC_Single_Dose_1` → `drc_single_dose_direction`
  - `dose_direction_DRC_Max_Daily_Dose_1` → `drc_max_daily_dose_direction`
  - `dose_direction_NeoDRC_Frequency_1` → `neodrc_frequency_direction`
  - `dose_direction_NeoDRC_Single_Dose_1` → `neodrc_single_dose_direction`
  - `dose_direction_NeoDRC_Max_Daily_Dose_1` → `neodrc_max_daily_dose_direction`

## Example Usage

```python
# Example of processing hospital data
from data_handler import df_active_adult

# Load and process main data
data_main = load_main_data()
data_grouped = group_and_deduplicate(data_main)
data_selected = select_columns(data_grouped)

# Analyze alert patterns
alert_counts = src_active_patients_merged['alert_type'].value_counts()
print(alert_counts)

# Analyze dose direction patterns
dose_direction_counts = src_active_patients_merged['drc_frequency_direction'].value_counts()
print(dose_direction_counts)

# Process ignore orders
from ignore_others import main as process_ignore_orders
ignore_orders_data = process_ignore_orders()
```