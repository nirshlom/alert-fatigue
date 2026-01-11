from __future__ import annotations

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Central configuration function for the Alert Fatigue Analysis pipeline.
    
    Modify the values in this function to change configuration across all scripts.
    All validation is performed automatically when this function is called.
    
    Returns:
        Dictionary with all configuration parameters
        
    Raises:
        ValueError: If any configuration validation fails
    """
    
    # ============================================================================
    # DATA CONFIGURATION
    # ============================================================================
    config = {
        # Input data
        'input_csv_path': "../alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv",
        'date_column': "time_prescribing_order",
        'target_column': "alert_status_binary",
        'feature_columns': ["age", "gender", "hospital_days", "charlson_score",
         "shift_type", "drug_atc", 'hospital_name', 'unit_category_ud'],

        # Optional: specify reference category per categorical feature
        # Example: {'gender': 'F', 'unit_category_ud': 'ICU'}
        'categorical_reference_levels': {'gender': 'FEMALE', 'unit_category_ud': 'Geriatric'},
        
        # Data splitting
        'train_frac': 0.7,
        'eval_frac': 0.15,
        'test_frac': 0.15,
        'ascending': True,
        'stratify': False,
        
        # Reproducibility
        'random_seed': 42,
        
        # Preprocessing options
        'impute_numeric': True,
        'scale_numeric': False,
        'rare_category_threshold': 0.01,
        # If True, use OneHotPreprocessor (one-hot encoding for multi-level categoricals)
        # If False, use Preprocessor (categorical encoding via statsmodels)
        'use_onehot_encoding': False,
        # If use_onehot_encoding=True, handle unseen categories (True) or raise error (False)
        'handle_unseen_categories': False,
        
        # Output settings
        'output_dir': os.path.normpath("model_pipeline/outputs"),
        'generate_profile': True,
        
        # Model options
        # Model type: 'statsmodels' for regular GLM/Logit, 'lme4' for GLMM
        'model_type': 'lme4',  # Options: 'statsmodels' or 'lme4'
        'use_glm': True,  # Only used for statsmodels model_type
        
        # GLMM-specific options (only used when model_type='lme4')
        # Random effects formula in R lme4 syntax (e.g., "(1|patient_id)" for random intercept)
        # This is the R FORMULA that specifies the STRUCTURE of random effects
        # Examples:
        #   "(1|patient_id)" - random intercept by patient
        #   "(1 + age|patient_id)" - random intercept and slope for age by patient
        #   "(1|patient_id) + (1|unit_id)" - multiple random intercepts
        # IMPORTANT: Variable names in this formula must match column names in grouping_columns!
        'random_effects': "(1|hospital_name) + (1|unit_category_ud)",  # R formula string
        
        # Grouping columns for random effects (column names in your data)
        # These are the ACTUAL COLUMN NAMES in your DataFrame that contain the grouping variable values
        # Python will extract these columns and pass them to R
        # IMPORTANT: Column names here must match variable names used in random_effects formula above!
        # Can be a single column name (str) or list of column names
        # If None and random_effects is specified, assumes grouping columns are in feature_columns
        # NOTE: If a column appears in both feature_columns and grouping_columns, it will be
        #       used as BOTH a fixed effect AND for random effects structure. This is valid but
        #       uncommon - typically use variables as EITHER fixed OR random effects.
        'grouping_columns': ['hospital_name', 'unit_category_ud'], 
        
        # GLMM distribution family and link function
        'glmm_family': 'binomial',  # Options: 'binomial', 'poisson', 'gaussian', etc.
        'glmm_link': 'logit',  # Options: 'logit', 'probit', 'identity', etc.
        
        # Optional GLMM control parameters (passed to lme4::glmerControl)
        # Example: {'optimizer': 'bobyqa'} or {'optimizer': 'Nelder_Mead'}
        'glmm_control': None
    }
    
    # ============================================================================
    # VALIDATION TESTS
    # ============================================================================
    
    # Validate data splitting fractions
    total_frac = config['train_frac'] + config['eval_frac'] + config['test_frac']
    if abs(total_frac - 1.0) > 1e-8:
        raise ValueError(
            f"Data split fractions must sum to 1.0; got {total_frac:.6f}. "
            f"Current: train={config['train_frac']}, eval={config['eval_frac']}, test={config['test_frac']}"
        )
    
    # Validate individual fractions
    for split_name, frac in [('train', config['train_frac']), ('eval', config['eval_frac']), ('test', config['test_frac'])]:
        if not (0 < frac < 1):
            raise ValueError(f"{split_name}_frac must be between 0 and 1, got {frac}")
    
    # Validate rare category threshold
    if not (0 <= config['rare_category_threshold'] <= 1):
        raise ValueError(f"rare_category_threshold must be between 0 and 1, got {config['rare_category_threshold']}")
    
    # Validate random seed
    if not isinstance(config['random_seed'], int) or config['random_seed'] < 0:
        raise ValueError(f"random_seed must be a non-negative integer, got {config['random_seed']}")
    
    # Validate file paths
    if not config['input_csv_path']:
        raise ValueError("input_csv_path cannot be empty")
    
    if not config['output_dir']:
        raise ValueError("output_dir cannot be empty")
    
    # Validate feature columns
    if not config['feature_columns'] or not isinstance(config['feature_columns'], list):
        raise ValueError("feature_columns must be a non-empty list")
    
    if len(config['feature_columns']) == 0:
        raise ValueError("feature_columns list cannot be empty")
    
    # Check for duplicate features
    if len(config['feature_columns']) != len(set(config['feature_columns'])):
        raise ValueError("feature_columns contains duplicate values")
    
    # Validate boolean flags
    boolean_fields = ['ascending', 'stratify', 'impute_numeric', 'scale_numeric', 'generate_profile', 'use_glm', 'use_onehot_encoding', 'handle_unseen_categories']
    for field in boolean_fields:
        if not isinstance(config[field], bool):
            raise ValueError(f"{field} must be a boolean, got {type(config[field])}")
    
    # Validate model type
    if config['model_type'] not in ['statsmodels', 'lme4']:
        raise ValueError(f"model_type must be 'statsmodels' or 'lme4', got {config['model_type']}")
    
    # Validate GLMM-specific options (only if model_type is 'lme4')
    if config['model_type'] == 'lme4':
        # Random effects should be specified for GLMM
        if config['random_effects'] is None:
            raise ValueError(
                "random_effects must be specified when model_type='lme4'. "
                "Example: random_effects='(1|patient_id)'"
            )
        
        if not isinstance(config['random_effects'], str) or not config['random_effects'].strip():
            raise ValueError("random_effects must be a non-empty string (R formula syntax)")
        
        # Validate grouping_columns if provided
        if config['grouping_columns'] is not None:
            if isinstance(config['grouping_columns'], str):
                # Single column name
                if not config['grouping_columns'].strip():
                    raise ValueError("grouping_columns cannot be an empty string")
            elif isinstance(config['grouping_columns'], list):
                # List of column names
                if len(config['grouping_columns']) == 0:
                    raise ValueError("grouping_columns list cannot be empty")
                for col in config['grouping_columns']:
                    if not isinstance(col, str) or not col.strip():
                        raise ValueError(f"All items in grouping_columns must be non-empty strings, got {col}")
            else:
                raise ValueError("grouping_columns must be a string, list of strings, or None")
        
        # Validate GLMM family
        valid_families = ['binomial', 'poisson', 'gaussian', 'Gamma', 'inverse.gaussian']
        if config['glmm_family'] not in valid_families:
            raise ValueError(f"glmm_family must be one of {valid_families}, got {config['glmm_family']}")
        
        # Validate GLMM link
        valid_links = ['logit', 'probit', 'identity', 'log', 'sqrt', 'inverse', '1/mu^2']
        if config['glmm_link'] not in valid_links:
            raise ValueError(f"glmm_link must be one of {valid_links}, got {config['glmm_link']}")
        
        # Validate glmm_control if provided
        if config['glmm_control'] is not None:
            if not isinstance(config['glmm_control'], dict):
                raise ValueError("glmm_control must be a dictionary or None")
    else:
        # For statsmodels, random_effects should be None
        if config['random_effects'] is not None:
            print("⚠️  WARNING: random_effects specified but model_type='statsmodels'. random_effects will be ignored.")
    
    # Validate numeric fields
    numeric_fields = ['train_frac', 'eval_frac', 'test_frac', 'rare_category_threshold']
    for field in numeric_fields:
        if not isinstance(config[field], (int, float)):
            raise ValueError(f"{field} must be numeric, got {type(config[field])}")
    
    # Validate string fields
    string_fields = ['input_csv_path', 'date_column', 'target_column', 'output_dir']
    for field in string_fields:
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a non-empty string, got {config[field]}")

    # Validate categorical_reference_levels (optional dict[str, str])
    if 'categorical_reference_levels' not in config or config['categorical_reference_levels'] is None:
        config['categorical_reference_levels'] = {}
    if not isinstance(config['categorical_reference_levels'], dict):
        raise ValueError("categorical_reference_levels must be a dict mapping feature -> reference category")
    # Keys must be features; values must be non-empty strings
    for feat, ref in config['categorical_reference_levels'].items():
        if feat not in config['feature_columns']:
            raise ValueError(f"categorical_reference_levels contains unknown feature '{feat}' not in feature_columns")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"Reference for feature '{feat}' must be a non-empty string; got {ref}")
    
    # ============================================================================
    # ADVANCED VALIDATION
    # ============================================================================
    
    # Check if input file exists (warning only, not error)
    if os.path.exists(config['input_csv_path']):
        file_size = os.path.getsize(config['input_csv_path'])
        if file_size == 0:
            raise ValueError(f"Input file exists but is empty: {config['input_csv_path']}")
    else:
        print(f"⚠️  WARNING: Input file not found: {config['input_csv_path']}")
        print("   This will cause an error when running the pipeline.")
    
    # Validate output directory can be created
    try:
        os.makedirs(config['output_dir'], exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory '{config['output_dir']}': {e}")
    
    # ============================================================================
    # CONFIGURATION SUMMARY
    # ============================================================================
    
    print("✓ Configuration validated successfully!")
    print(f"  - Input: {config['input_csv_path']}")
    print(f"  - Features: {len(config['feature_columns'])} columns")
    print(f"  - Data split: {config['train_frac']:.0%} train, {config['eval_frac']:.0%} eval, {config['test_frac']:.0%} test")
    print(f"  - Output: {config['output_dir']}")
    if config['categorical_reference_levels']:
        print(f"  - Categorical references: {len(config['categorical_reference_levels'])} specified")
    
    # Print model configuration
    print(f"  - Model type: {config['model_type']}")
    if config['model_type'] == 'lme4':
        print(f"    - Random effects: {config['random_effects']}")
        if config['grouping_columns']:
            grouping_str = config['grouping_columns'] if isinstance(config['grouping_columns'], str) else ', '.join(config['grouping_columns'])
            print(f"    - Grouping columns: {grouping_str}")
        print(f"    - Family: {config['glmm_family']} ({config['glmm_link']} link)")
    elif config['model_type'] == 'statsmodels':
        print(f"    - Use GLM: {config['use_glm']}")
    
    return config