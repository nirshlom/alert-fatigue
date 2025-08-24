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
        'input_csv_path': "alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv",
        'date_column': "time_prescribing_order",
        'target_column': "alert_status_binary",
        'feature_columns': ["age", "gender", "hospital_days", "charlson_score", "shift_type", "unit_category"],
        
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
        
        # Output settings
        'output_dir': os.path.normpath("model_pipeline/outputs"),
        'generate_profile': True,
        
        # Model options
        'use_glm': True
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
    boolean_fields = ['ascending', 'stratify', 'impute_numeric', 'scale_numeric', 'generate_profile', 'use_glm']
    for field in boolean_fields:
        if not isinstance(config[field], bool):
            raise ValueError(f"{field} must be a boolean, got {type(config[field])}")
    
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
    
    return config