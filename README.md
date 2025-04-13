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
   - Key functions:
     - `process_hospital_columns()`: Maps hospital names and categories
     - `process_time_columns()`: Converts Excel-style time to datetime
     - `process_drc_subgroup()`: Creates DRC and NeoDRC subgroups

2. **02_data_main_prep_ud.py**
   - Column selection and data transformation
   - Medication count calculations
   - Disease keyword analysis
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

### Data Analysis

5. **main_analysis.ipynb**
   - Jupyter notebook for exploratory analysis
   - Contains visualizations and statistical summaries
   - Key analyses:
     - Patient characteristics by gender
     - Alert type distribution
     - Order analysis by ATC group

## Key Features

- **Comprehensive Data Processing**: Handles complex healthcare data with multiple variables
- **Multilingual Support**: Processes both Hebrew and English text
- **Time Analysis**: Detailed analysis of alert response times
- **Patient Categorization**: Sophisticated patient grouping based on multiple factors
- **Alert Classification**: Detailed categorization of different alert types

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
```

## Data Flow

1. Raw data → 01_data_main_prep_ud.py (Initial processing)
2. → 02_data_main_prep_ud.py (Column selection and transformation)
3. → 03_data_main_cln_prep_ud.py (Cleaning and categorization)
4. → 04_main_adults_flat_ud.py (Adult patient analysis)
5. → main_analysis.ipynb (Final analysis and visualization)

## Requirements

- Python 3.x
- pandas
- numpy
- scipy
- statsmodels
- tableone
- plotly
- jupyter

### Managing Dependencies

The project uses a `requirements.txt` file to manage Python package dependencies. This file lists all the necessary packages with their specific versions to ensure reproducibility of the analysis. To create or update the requirements file, you can use:

```bash
pip freeze > requirements.txt
```

To install the exact versions of packages used in this project:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts in order (01-04)
4. Open main_analysis.ipynb for final analysis

## Contributing

Please follow the existing code structure and naming conventions when contributing to this project.
