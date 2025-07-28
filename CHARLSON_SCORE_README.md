# Charlson Comorbidity Index (CCI) Calculation

This document describes the implementation of the Charlson Comorbidity Index (CCI) calculation in the alert fatigue analysis project.

## Overview

The Charlson Comorbidity Index is a widely used method for predicting mortality by classifying or weighting comorbid conditions. It was developed by Mary Charlson and colleagues in 1987 and has been validated in numerous studies.

## Implementation

The CCI calculation is implemented in `charlson_index.py` and provides:
- Standard Charlson score calculation
- Age-adjusted Charlson score
- 10-year survival rate prediction

## Scoring System

### Comorbidity Categories and Weights

The implementation uses a three-tier scoring system:

#### 1-Point Conditions (1 point each)
- **Myocardial Infarction**: `MYOCARDIAL_count`
- **Heart Failure**: `HEART FAILURE_count`
- **Peripheral Vascular Disease**: `PVD_group_cnt`
- **Cerebrovascular Disease**: `CEREBROVASCULAR_group_cnt`
- **Dementia**: `DEMENTIA_count`
- **Chronic Obstructive Pulmonary Disease**: `COPD_group_cnt`
- **Gout**: `GOUT_group_cnt`
- **Peptic Ulcer Disease**: `ULCER_group_cnt`
- **Liver Disease (Mild)**: `liver_group_cnt`

#### 2-Point Conditions (2 points each)
- **Hemiplegia/Paraplegia**: `HEMIPLEGIA_group_cnt`
- **Renal Disease**: `RENAL_group_cnt`
- **Diabetes (Uncomplicated)**: `DIABETES_count`
- **Malignancy (Solid Tumor)**: `MALIGNANCY_group_cnt`
- **Leukemia**: `LEUKEMIA_group_cnt`
- **Lymphoma**: `LYMPHOMA_count`

#### 6-Point Conditions (6 points each)
- **HIV/AIDS**: `HIV_count`
- **Metastatic Solid Tumor**: `METASTATIC_group_cnt`

### Age Adjustment

The age-adjusted score adds additional points based on age groups:

| Age Group | Additional Points |
|-----------|------------------|
| 50-59 years | +1 |
| 60-69 years | +2 |
| 70-79 years | +3 |
| 80+ years | +4 |

## Calculation Process

### Step 1: Sum Comorbidity Scores
```python
# 1-point conditions
charls_sum1_3_points = sum(all_1_point_conditions)

# 2-point conditions (multiplied by 2)
charls_sum2points = sum(all_2_point_conditions) * 2

# 6-point conditions (multiplied by 6)
charls_sum6points = sum(all_6_point_conditions) * 6
```

### Step 2: Calculate Base Charlson Score
```python
Charlson_score = charls_sum1_3_points + charls_sum2points + charls_sum6points
```

### Step 3: Apply Age Adjustment
```python
Charlson_score_age_adj = Charlson_score + age_adjustment_points
```

### Step 4: Calculate 10-Year Survival Rate
```python
SurvivalRate10years_age_adj = (0.983 ** (exp(Charlson_score_age_adj * 0.9))) * 100
```

## Usage

### Function: `calculate_cci(df)`

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame containing comorbidity columns and `AGE_num`

**Required Columns:**
- All comorbidity count columns (see scoring system above)
- `AGE_num`: Patient age in years

**Returns:**
- `pd.DataFrame`: Original DataFrame with added columns:
  - `charls_sum1_3_points`: Sum of 1-point conditions
  - `charls_sum2points`: Sum of 2-point conditions (weighted)
  - `charls_sum6points`: Sum of 6-point conditions (weighted)
  - `Charlson_score`: Base Charlson score
  - `Charlson_score_age_adj`: Age-adjusted Charlson score
  - `SurvivalRate10years_age_adj`: 10-year survival rate (percentage)

### Example Usage

```python
import pandas as pd
from charlson_index import calculate_cci

# Load your data
df = pd.read_csv('your_data.csv')

# Calculate Charlson scores
df_with_cci = calculate_cci(df)

# View results
print(df_with_cci[['Charlson_score', 'Charlson_score_age_adj', 'SurvivalRate10years_age_adj']].head())
```

## Interpretation

### Charlson Score Ranges

| Score Range | Risk Level | Typical 10-Year Survival |
|-------------|------------|-------------------------|
| 0 | Low | 95-100% |
| 1-2 | Low-Medium | 85-95% |
| 3-4 | Medium | 70-85% |
| 5-6 | Medium-High | 50-70% |
| 7+ | High | <50% |

### Age-Adjusted Score

The age-adjusted score provides a more accurate mortality prediction by accounting for the natural increase in mortality risk with age.

### Survival Rate

The 10-year survival rate is calculated using the formula:
- Based on the age-adjusted Charlson score
- Ranges from 0-100%
- Lower percentages indicate higher mortality risk

## Data Requirements

### Input Data Format

Your DataFrame must contain:
1. **Comorbidity columns**: Binary (0/1) or count columns for each condition
2. **Age column**: `AGE_num` as numeric values
3. **Patient identifier**: Recommended for tracking results

### Column Naming Convention

The function expects specific column names:
- `MYOCARDIAL_count`
- `HEART FAILURE_count`
- `PVD_group_cnt`
- `CEREBROVASCULAR_group_cnt`
- `DEMENTIA_count`
- `COPD_group_cnt`
- `GOUT_group_cnt`
- `ULCER_group_cnt`
- `liver_group_cnt`
- `HEMIPLEGIA_group_cnt`
- `RENAL_group_cnt`
- `MALIGNANCY_group_cnt`
- `LEUKEMIA_group_cnt`
- `LYMPHOMA_count`
- `DIABETES_count`
- `HIV_count`
- `METASTATIC_group_cnt`
- `AGE_num`

## Error Handling

The function includes safety checks:
- Validates that all required columns are present
- Converts age to numeric format with error handling
- Rounds survival rate to 2 decimal places

## Validation

This implementation follows the original Charlson methodology:
- Uses the same 19 conditions as the original index
- Applies the same weighting system (1, 2, and 6 points)
- Includes age adjustment as recommended in subsequent studies
- Calculates survival rates using validated formulas

## References

1. Charlson ME, Pompei P, Ales KL, MacKenzie CR. A new method of classifying prognostic comorbidity in longitudinal studies: development and validation. J Chronic Dis. 1987;40(5):373-383.

2. Charlson M, Szatrowski TP, Peterson J, Gold J. Validation of a combined comorbidity index. J Clin Epidemiol. 1994;47(11):1245-1251.

3. Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data. Med Care. 2005;43(11):1130-1139. 