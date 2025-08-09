# Alert Fatigue Analysis

A comprehensive analysis system for studying alert fatigue in healthcare settings, focusing on medication alerts and their impact on healthcare providers.

## Project Structure

The project is organized into a data processing pipeline with several key Python scripts that handle different aspects of the analysis:

### Data Processing Pipeline

**Main Pipeline (Run in Order):**

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
   - Filters out specific hospitals (243, 113, 29) and units (Day_care, ICU, Pediatric, Rehabilitation)
   - Key functions:
     - `compute_load_index()`: Calculates workload metrics per shift
     - `convert_age_cat()`: Processes age categories
     - `filter_active_adult()`: Filters for active adult patients

5. **05_rename_columns.py**
   - Standardizes column names throughout the pipeline
   - Uses comprehensive renaming dictionaries from `column_rename.py`
   - Filters and renames columns based on mapping rules
   - Key functions:
     - `rename_and_filter_columns()`: Applies column renaming and filtering
     - `report_column_mapping()`: Reports which columns are kept/dropped

6. **06_patients_level_data.py**
   - Aggregates data to patient level
   - Creates patient-level summaries and counts
   - Converts unit and alert counts to binary format
   - Handles disease-related boolean conversions
   - Key functions:
     - `group_and_save_patient_data()`: Main aggregation function
     - `convert_unit_cat()`: Converts unit counts to binary (0/1)
     - `get_count_columns()`: Identifies count columns for processing

### Standalone Scripts

**07_ignore_others.py** (Standalone - Not part of main pipeline)
   - Processes ignored orders and their associated text
   - Merges main data with ignore orders data
   - Handles UTF-8 encoding for proper Hebrew text processing
   - Key features:
     - Merges data based on drug_order_id
     - Preserves response reasons text
     - Outputs UTF-8 encoded CSV file
     - Returns merged DataFrame for further analysis

### Analysis and Utilities

- **main_analysis.ipynb**: Jupyter notebook for exploratory analysis
- **patient_descriptive_analysis.ipynb**: Patient-level descriptive statistics
- **column_rename.py**: Comprehensive column renaming dictionaries
- **charlson_index.py**: Charlson Comorbidity Index calculations
- **data_handler.py**: Utility functions for data handling

## Key Features

- **Comprehensive Data Processing**: Handles complex healthcare data with multiple variables
- **Multilingual Support**: Processes both Hebrew and English text
- **Time Analysis**: Detailed analysis of alert response times
- **Patient Categorization**: Sophisticated patient grouping based on multiple factors
- **Alert Classification**: Detailed categorization of different alert types
- **Dose Direction Analysis**: Tracks whether doses exceed or fall below recommended ranges for DRC and NeoDRC alerts
- **Hospital Filtering**: Automatically excludes specific hospitals (243, 113, 29) and units (Day_care, ICU, Pediatric, Rehabilitation)
- **Patient-Level Aggregation**: Creates comprehensive patient-level datasets with binary indicators

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

## Pipeline Flow

```
Raw Data → 01_data_main_prep_ud.py → 02_data_main_prep_ud.py → 03_data_main_cln_prep_ud.py 
→ 04_main_adults_flat_ud.py → 05_rename_columns.py → 06_patients_level_data.py → Final Analysis
```

## Example Usage

```python
# Run the complete pipeline
# 1. Initial data preparation
python 01_data_main_prep_ud.py

# 2. Column selection and transformation
python 02_data_main_prep_ud.py

# 3. Data cleaning and final preparation
python 03_data_main_cln_prep_ud.py

# 4. Adult patient filtering and analysis
python 04_main_adults_flat_ud.py

# 5. Column renaming
python 05_rename_columns.py

# 6. Patient-level aggregation
python 06_patients_level_data.py

# Standalone: Process ignore orders (if needed)
python 07_ignore_others.py
```

## Data Outputs

The pipeline produces several key outputs:
- `df_main_active_adult_renamed.csv`: Clean, renamed data for active adult patients
- `df_patients_level_data.csv`: Patient-level aggregated data with binary indicators
- `ignore_orders_with_text.csv`: Ignored orders with associated text (standalone script)

## Requirements

See `requirements.txt` for the complete list of Python dependencies.