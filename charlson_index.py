import pandas as pd
import numpy as np

def calculate_cci(df):
    """
    Calculate Charlson Comorbidity Index (CCI), age-adjusted CCI, and 10-year survival rate.

    Args:
        df (pd.DataFrame): Input DataFrame with required comorbidity columns and AGE_num.

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - charls_sum1_3_points
            - charls_sum2points
            - charls_sum6points
            - Charlson_score
            - Charlson_score_age_adj
            - SurvivalRate10years_age_adj
    """
    # Column groups
    cols_sum1 = [
        'MYOCARDIAL_count',
        "HEART FAILURE_count",
        "PVD_group_cnt",
        "CEREBROVASCULAR_group_cnt",
        'DEMENTIA_count',
        "COPD_group_cnt",
        "GOUT_group_cnt",
        "ULCER_group_cnt",
        "liver_group_cnt"
    ]

    cols_sum2 = [
        "HEMIPLEGIA_group_cnt",
        "RENAL_group_cnt",
        "MALIGNANCY_group_cnt",
        "LEUKEMIA_group_cnt",
        "LYMPHOMA_count",
        'DIABETES_count'
    ]

    cols_sum6 = [
        'HIV_count',
        'METASTATIC_group_cnt'
    ]

    # Safety check
    all_cols = cols_sum1 + cols_sum2 + cols_sum6 + ['AGE_num']
    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Score calculations
    df['charls_sum1_3_points'] = df[cols_sum1].sum(axis=1)
    df['charls_sum2points'] = df[cols_sum2].sum(axis=1) * 2
    df['charls_sum6points'] = df[cols_sum6].sum(axis=1) * 6

    df['Charlson_score'] = (
        df['charls_sum1_3_points'] +
        df['charls_sum2points'] +
        df['charls_sum6points']
    )

    df['AGE_num'] = pd.to_numeric(df['AGE_num'], errors='coerce')

    # Age adjustment
    conditions = [
        (df['AGE_num'] >= 50) & (df['AGE_num'] <= 59),
        (df['AGE_num'] >= 60) & (df['AGE_num'] <= 69),
        (df['AGE_num'] >= 70) & (df['AGE_num'] <= 79),
        (df['AGE_num'] >= 80)
    ]
    choices = [1, 2, 3, 4]

    df['Charlson_score_age_adj'] = df['Charlson_score'] + np.select(conditions, choices, default=0)

    # Survival rate calculation
    df['SurvivalRate10years_age_adj'] = (
        0.983 ** (np.exp(df['Charlson_score_age_adj'] * 0.9))
    ) * 100

    # Optional: Round to 2 decimals
    df['SurvivalRate10years_age_adj'] = df['SurvivalRate10years_age_adj'].round(2)

    return df