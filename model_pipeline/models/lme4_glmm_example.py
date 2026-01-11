"""
Example usage of LME4GLMMModel

This file demonstrates how to use the LME4GLMMModel class for fitting
Generalized Linear Mixed Models using R's lme4 package.

Prerequisites:
1. Install rpy2: pip install rpy2
2. Install R: https://www.r-project.org/
3. Install lme4 in R: install.packages("lme4")
"""

import pandas as pd
import numpy as np
from model_pipeline.models.lme4_glmm import LME4GLMMModel

# Example 1: Simple random intercept model
# Assuming you have patient-level data with repeated measures
def example_random_intercept():
    """Example with random intercept by patient."""
    
    # Create sample data
    np.random.seed(42)
    n_patients = 100
    n_obs_per_patient = 5
    
    patient_ids = np.repeat(range(n_patients), n_obs_per_patient)
    gender = np.random.choice(['MALE', 'FEMALE'], size=len(patient_ids))
    unit = np.random.choice(['ICU', 'Internal', 'Surgery'], size=len(patient_ids))
    y = np.random.binomial(1, 0.3, size=len(patient_ids))
    
    X = pd.DataFrame({
        'gender': gender,
        'unit': unit
    })
    y_series = pd.Series(y, name='alert_status_binary')
    groups = pd.Series(patient_ids, name='patient_id')
    
    # Fit model with random intercept by patient
    model = LME4GLMMModel(random_effects="(1|patient_id)")
    model.fit(X, y_series, groups=groups)
    
    # Get coefficients
    coef_df = model.get_coefficients()
    print("Fixed Effects Coefficients:")
    print(coef_df)
    
    # Make predictions
    predictions = model.predict_proba(X, groups=groups)
    print(f"\nPredictions shape: {predictions.shape}")
    
    # Get model summary
    print("\nModel Summary:")
    print(model.get_model_summary())
    
    return model


# Example 2: Random intercept and slope
def example_random_slope():
    """Example with random intercept and slope."""
    
    # Create sample data with a continuous predictor
    np.random.seed(42)
    n_patients = 50
    n_obs_per_patient = 10
    
    patient_ids = np.repeat(range(n_patients), n_obs_per_patient)
    age = np.random.normal(65, 15, size=len(patient_ids))
    gender = np.random.choice(['MALE', 'FEMALE'], size=len(patient_ids))
    y = np.random.binomial(1, 0.3, size=len(patient_ids))
    
    X = pd.DataFrame({
        'age': age,
        'gender': gender
    })
    y_series = pd.Series(y, name='alert_status_binary')
    groups = pd.Series(patient_ids, name='patient_id')
    
    # Fit model with random intercept and slope for age
    model = LME4GLMMModel(random_effects="(1 + age|patient_id)")
    model.fit(X, y_series, groups=groups)
    
    # Get coefficients
    coef_df = model.get_coefficients()
    print("Fixed Effects Coefficients:")
    print(coef_df)
    
    return model


# Example 3: Multiple random effects
def example_multiple_random_effects():
    """Example with multiple grouping variables."""
    
    # Create sample data with patient and unit grouping
    np.random.seed(42)
    n_patients = 100
    n_units = 10
    n_obs_per_patient = 3
    
    patient_ids = np.repeat(range(n_patients), n_obs_per_patient)
    unit_ids = np.random.choice(range(n_units), size=len(patient_ids))
    gender = np.random.choice(['MALE', 'FEMALE'], size=len(patient_ids))
    y = np.random.binomial(1, 0.3, size=len(patient_ids))
    
    X = pd.DataFrame({
        'gender': gender
    })
    y_series = pd.Series(y, name='alert_status_binary')
    
    # Multiple grouping variables
    groups_df = pd.DataFrame({
        'patient_id': patient_ids,
        'unit_id': unit_ids
    })
    
    # Fit model with nested or crossed random effects
    # Nested: (1|unit_id/patient_id) or Crossed: (1|unit_id) + (1|patient_id)
    model = LME4GLMMModel(random_effects="(1|patient_id) + (1|unit_id)")
    model.fit(X, y_series, groups=groups_df)
    
    # Get coefficients
    coef_df = model.get_coefficients()
    print("Fixed Effects Coefficients:")
    print(coef_df)
    
    return model


# Example 4: Using with existing pipeline
def example_with_pipeline():
    """Example of integrating with existing pipeline."""
    
    # Load your data (example)
    # df = pd.read_csv("your_data.csv")
    
    # After preprocessing, you can use LME4GLMMModel instead of StatsmodelsLogitModel
    # In your pipeline code:
    """
    from model_pipeline.models.lme4_glmm import LME4GLMMModel
    
    # Assuming you have patient IDs in your data
    train_features = train_df[feature_columns]
    train_target = train_df[target_column]
    train_groups = train_df['patient_id']  # or whatever grouping variable
    
    # Fit GLMM model
    model = LME4GLMMModel(random_effects="(1|patient_id)")
    model.fit(train_features, train_target, groups=train_groups)
    
    # Use model just like StatsmodelsLogitModel
    eval_proba = model.predict_proba(eval_features, groups=eval_df['patient_id'])
    coef_df = model.get_coefficients()
    """
    
    pass


if __name__ == "__main__":
    print("Example 1: Random Intercept Model")
    print("=" * 50)
    model1 = example_random_intercept()
    
    print("\n\nExample 2: Random Intercept and Slope")
    print("=" * 50)
    # model2 = example_random_slope()
    
    print("\n\nExample 3: Multiple Random Effects")
    print("=" * 50)
    # model3 = example_multiple_random_effects()
