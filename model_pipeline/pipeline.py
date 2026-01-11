from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from .data_loading import load_csv, select_columns
from .evaluation.metrics import (
    compute_pr_metrics, compute_roc_metrics, compute_summary_metrics,
    find_optimal_threshold, threshold_table
)
from .evaluation.plots import (
    plot_or_forest, plot_pr_curve, plot_roc_curve, plot_threshold_metrics
)
from .models.statsmodels_logit import StatsmodelsLogitModel
# Conditionally import LME4GLMMModel
try:
    from .models.lme4_glmm import LME4GLMMModel
    LME4_AVAILABLE = True
except ImportError:
    LME4_AVAILABLE = False
from .preprocess import Preprocessor
from .preprocess_onehot import OneHotPreprocessor
from .reporting.coefficients import (
    coefficients_to_or, create_coefficient_summary, filter_significant_coefficients,
    sort_coefficients_by_importance
)
from .reporting.profile import generate_profile_report
from .reporting.save import make_run_dir, write_json
import json
import numpy as np
import pandas as pd
from .split import predictive_time_split


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def safe_json_dump(obj, file_path):
    """Safely dump object to JSON file by converting numpy types."""
    converted_obj = convert_numpy_types(obj)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_obj, f, indent=2, ensure_ascii=False)


def run_preprocessing_only(config: Dict[str, Any], run_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Run only the preprocessing pipeline (data loading, splitting, preprocessing).
    
    Args:
        config: Training configuration dictionary
        run_dir: Optional output directory to use (if None, creates new one)
        
    Returns:
        Tuple of (train_df, eval_df, test_df, artifacts)
    """
    print("Running preprocessing pipeline...")
    
    # Load data
    print("Loading data...")
    df = load_csv(config['input_csv_path'])
    df = select_columns(df, config['feature_columns'] + [config['target_column'], config['date_column']])
    print(f"✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Split data
    print("\nSplitting data...")
    train_df, eval_df, test_df = predictive_time_split(
        df, config['date_column'], config['train_frac'], config['eval_frac'], config['test_frac'],
        ascending=config['ascending'], stratify=config['stratify'], target_column=config['target_column']
    )
    print(f"✓ Data split: Train={len(train_df):,}, Eval={len(eval_df):,}, Test={len(test_df):,}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    use_onehot = config.get('use_onehot_encoding', False)
    
    if use_onehot:
        preprocessor = OneHotPreprocessor(
            impute_numeric=config['impute_numeric'],
            scale_numeric=config['scale_numeric'],
            rare_category_threshold=config['rare_category_threshold'],
            categorical_reference_levels=config.get('categorical_reference_levels'),
            handle_unseen_categories=config.get('handle_unseen_categories', False)
        )
        print(f"  Using OneHotPreprocessor (one-hot encoding for multi-level categoricals)")
    else:
        preprocessor = Preprocessor(
            impute_numeric=config['impute_numeric'],
            scale_numeric=config['scale_numeric'],
            rare_category_threshold=config['rare_category_threshold'],
            categorical_reference_levels=config.get('categorical_reference_levels')
        )
        print(f"  Using Preprocessor (categorical encoding via statsmodels)")
    
    # Log preprocessing configuration
    print(f"  Preprocessing configuration:")
    print(f"    - Impute numeric: {config['impute_numeric']}")
    print(f"    - Scale numeric: {config['scale_numeric']}")
    print(f"    - Rare category threshold: {config['rare_category_threshold']}")
    if use_onehot:
        print(f"    - Handle unseen categories: {config.get('handle_unseen_categories', False)}")
    
    # Fit preprocessor on training data
    train_features = train_df[config['feature_columns']]
    train_target = train_df[config['target_column']]
    
    print(f"  Fitting preprocessor on training data...")
    preprocessor.fit(train_features)
    
    # Log preprocessing details
    print(f"  Preprocessing details:")
    print(f"    - Numeric columns: {len(preprocessor.numeric_columns)}")
    print(f"    - Categorical columns: {len(preprocessor.categorical_columns)}")
    
    if config['impute_numeric'] and preprocessor.numeric_columns:
        print(f"    - Numeric imputation applied to: {', '.join(preprocessor.numeric_columns)}")
        # Count missing values before imputation
        missing_before = train_features[preprocessor.numeric_columns].isnull().sum()
        missing_count = missing_before.sum()
        if missing_count > 0:
            print(f"    - Total missing values imputed: {missing_count:,}")
            for col in preprocessor.numeric_columns:
                if missing_before[col] > 0:
                    print(f"    - {col}: {missing_before[col]:,} missing values")
        else:
            print(f"    - No missing values found in numeric columns")
    
    if config['scale_numeric'] and preprocessor.numeric_columns:
        print(f"    - Numeric scaling applied to: {', '.join(preprocessor.numeric_columns)}")
    
    if preprocessor.categorical_columns:
        print(f"    - Categorical processing applied to: {', '.join(preprocessor.categorical_columns)}")
        for col in preprocessor.categorical_columns:
            if col in preprocessor.categorical_levels_:
                levels = preprocessor.categorical_levels_[col]
                print(f"      - {col}: {len(levels)} levels preserved")
    
    # Transform all splits
    print(f"  Transforming data...")
    train_processed = preprocessor.transform(train_features)
    eval_processed = preprocessor.transform(eval_df[config['feature_columns']])
    test_processed = preprocessor.transform(test_df[config['feature_columns']])
    
    # Handle feature column name changes after one-hot encoding
    updated_feature_columns = config['feature_columns'].copy()
    feature_mapping = None
    if use_onehot and hasattr(preprocessor, 'get_feature_mapping'):
        feature_mapping = preprocessor.get_feature_mapping()
        # Get the new feature column names
        updated_feature_columns = preprocessor.get_transformed_feature_names()
        print(f"  Feature columns after one-hot encoding: {len(updated_feature_columns)} columns")
        print(f"    Original: {config['feature_columns']}")
        print(f"    Transformed: {updated_feature_columns[:5]}{'...' if len(updated_feature_columns) > 5 else ''}")
    
    # Add target columns back
    train_processed[config['target_column']] = train_target
    eval_processed[config['target_column']] = eval_df[config['target_column']]
    test_processed[config['target_column']] = test_df[config['target_column']]
    
    # Log final shapes
    print(f"  Final processed shapes:")
    print(f"    - Train: {train_processed.shape}")
    print(f"    - Eval: {eval_processed.shape}")
    print(f"    - Test: {test_processed.shape}")
    
    # Generate profile report if requested
    artifacts = {
        'updated_feature_columns': updated_feature_columns,
        'original_feature_columns': config['feature_columns'],
        'feature_mapping': feature_mapping,
        'use_onehot_encoding': use_onehot
    }
    if config['generate_profile']:
        print("\nGenerating profile report...")
        # Use provided run_dir if available, otherwise create new one
        if run_dir is None:
            run_dir = make_run_dir(config['output_dir'])
        profile_path = os.path.join(run_dir, "train_profile_report.html")
        generate_profile_report(train_processed, profile_path)
        artifacts['profile_path'] = profile_path
        artifacts['run_directory'] = run_dir
        print(f"✓ Profile report saved to: {profile_path}")
    
    print("\n✓ Preprocessing completed successfully!")
    return train_processed, eval_processed, test_processed, artifacts


def run_full_training(config: Dict[str, Any]) -> Dict:
    """Run the complete model training pipeline.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Dictionary with all results and artifacts
    """
    print("Running full model training pipeline...")
    
    # Create run directory
    run_dir = make_run_dir(config['output_dir'])
    print(f"Run directory: {run_dir}")
    
    # Run preprocessing
    train_df, eval_df, test_df, preprocessing_artifacts = run_preprocessing_only(config, run_dir)
    
    # Use updated feature columns if one-hot encoding was used
    feature_columns_to_use = preprocessing_artifacts.get('updated_feature_columns', config['feature_columns'])
    train_features = train_df[feature_columns_to_use]
    train_target = train_df[config['target_column']]
    
    # Train model based on config
    model_type = config.get('model_type', 'statsmodels')
    
    if model_type == 'lme4':
        if not LME4_AVAILABLE:
            raise ImportError(
                "LME4GLMMModel is not available. Please install rpy2 and R with lme4 package.\n"
                "Install: pip install rpy2\n"
                "In R: install.packages('lme4')"
            )
        
        print("Training GLMM model (lme4)...")
        model = LME4GLMMModel(
            random_effects=config['random_effects'],
            family=config['glmm_family'],
            link=config['glmm_link'],
            control=config.get('glmm_control')
        )
        
        # Prepare grouping variables for random effects
        grouping_columns = config.get('grouping_columns')
        if grouping_columns is not None:
            if isinstance(grouping_columns, str):
                # Single grouping column
                grouping_columns_list = [grouping_columns]
            elif isinstance(grouping_columns, list):
                # Multiple grouping columns
                grouping_columns_list = grouping_columns
            else:
                raise ValueError(f"grouping_columns must be str or list, got {type(grouping_columns)}")
            
            # Check for overlap between feature_columns and grouping_columns
            feature_cols_set = set(feature_columns_to_use)
            grouping_cols_set = set(grouping_columns_list)
            overlap = feature_cols_set & grouping_cols_set
            
            if overlap:
                print(f"⚠️  NOTE: Columns {list(overlap)} appear in both feature_columns and grouping_columns.")
                print(f"    They will be used as BOTH fixed effects (from feature_columns) AND random effects (from grouping_columns).")
                print(f"    This is valid but uncommon. Typically, use them as EITHER fixed OR random effects.")
            
            train_groups = train_df[grouping_columns_list]
        else:
            # If no grouping columns specified, try to infer from random_effects formula
            # This is a simple heuristic - user should specify grouping_columns explicitly
            train_groups = None
            print("⚠️  WARNING: No grouping_columns specified. Make sure grouping variables are in your data.")
        
        model.fit(train_features, train_target, groups=train_groups)
        
    else:  # statsmodels
        print("Training logistic regression model (statsmodels)...")
        model = StatsmodelsLogitModel(use_glm=config['use_glm'])
        model.fit(train_features, train_target)
    
    # Get model summary
    model_summary = model.get_model_summary()
    aic_bic = model.get_aic_bic()
    
    # Get coefficients and convert to odds ratios
    print("Analyzing coefficients...")
    coef_df = model.get_coefficients()
    or_df = coefficients_to_or(coef_df)
    
    # Sort coefficients by importance
    or_df_sorted = sort_coefficients_by_importance(or_df, method='odds_ratio')
    
    # Filter significant coefficients
    significant_coef = filter_significant_coefficients(or_df)
    
    # Create coefficient summary
    coef_summary = create_coefficient_summary(or_df)
    
    # Make predictions
    print("Making predictions...")
    eval_features = eval_df[feature_columns_to_use]
    eval_target = eval_df[config['target_column']]
    
    # For GLMM, need to pass grouping variables if available
    if model_type == 'lme4' and config.get('grouping_columns') is not None:
        grouping_columns = config['grouping_columns']
        if isinstance(grouping_columns, str):
            eval_groups = eval_df[grouping_columns]
        elif isinstance(grouping_columns, list):
            eval_groups = eval_df[grouping_columns]
        else:
            eval_groups = None
        eval_proba = model.predict_proba(eval_features, groups=eval_groups)
    else:
        eval_proba = model.predict_proba(eval_features)
    
    eval_scores = eval_proba[:, 1]  # Probability of positive class
    
    # Calculate evaluation metrics
    print("Computing evaluation metrics...")
    pr_metrics = compute_pr_metrics(eval_target, eval_scores)
    roc_metrics = compute_roc_metrics(eval_target, eval_scores)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(eval_target, eval_scores, metric='f1')
    
    # Make predictions at optimal threshold
    eval_pred = model.predict(eval_features, threshold=optimal_threshold)
    
    # Compute summary metrics
    summary_metrics = compute_summary_metrics(eval_target, eval_pred, eval_scores)
    
    # Generate threshold table
    threshold_df = threshold_table(eval_target, eval_scores)
    
    # Generate plots
    print("Generating plots...")
    
    # Forest plot
    forest_plot_path = os.path.join(run_dir, "coefficients_forest.png")
    plot_or_forest(or_df_sorted, forest_plot_path, 
                   title="Forest Plot of Odds Ratios - Alert Fatigue Model")
    
    # PR curve
    pr_plot_path = os.path.join(run_dir, "pr_curve.png")
    plot_pr_curve(pr_metrics['precision'], pr_metrics['recall'], pr_plot_path,
                  title="Precision-Recall Curve - Alert Fatigue Model",
                  auc_pr=pr_metrics['auc_pr'])
    
    # ROC curve
    roc_plot_path = os.path.join(run_dir, "roc_curve.png")
    plot_roc_curve(roc_metrics['fpr'], roc_metrics['tpr'], roc_plot_path,
                   title="ROC Curve - Alert Fatigue Model",
                   auc_roc=roc_metrics['auc_roc'])
    
    # Threshold metrics plot
    threshold_plot_path = os.path.join(run_dir, "threshold_metrics.png")
    plot_threshold_metrics(threshold_df, threshold_plot_path,
                          title="Metrics by Threshold - Alert Fatigue Model")
    
    # Save results
    print("Saving results...")
    
    # Save coefficients
    coef_path = os.path.join(run_dir, "coefficients_or.csv")
    or_df_sorted.to_csv(coef_path, index=False)
    
    # Save significant coefficients
    significant_path = os.path.join(run_dir, "significant_coefficients.csv")
    significant_coef.to_csv(significant_path, index=False)
    
    # Save threshold metrics
    threshold_path = os.path.join(run_dir, "threshold_metrics.csv")
    threshold_df.to_csv(threshold_path, index=False)
    
    # Save evaluation predictions
    eval_pred_df = eval_df.copy()
    eval_pred_df['predicted_probability'] = eval_scores
    eval_pred_df['predicted_class'] = eval_pred
    eval_pred_path = os.path.join(run_dir, "eval_predictions.csv")
    eval_pred_df.to_csv(eval_pred_path, index=False)
    
    # Save configuration
    config_path = os.path.join(run_dir, "config_used.json")
    safe_json_dump(config, config_path)
    
    # Create run summary
    run_summary = {
        'run_timestamp': datetime.now().isoformat(),
        'run_directory': run_dir,
        'data_info': {
            'train_samples': len(train_df),
            'eval_samples': len(eval_df),
            'test_samples': len(test_df),
            'features': len(config['feature_columns'])
        },
        'model_info': {
            'model_type': config.get('model_type', 'statsmodels'),
            'use_glm': config.get('use_glm', True) if config.get('model_type', 'statsmodels') == 'statsmodels' else None,
            'random_effects': config.get('random_effects') if config.get('model_type') == 'lme4' else None,
            'glmm_family': config.get('glmm_family') if config.get('model_type') == 'lme4' else None,
            'glmm_link': config.get('glmm_link') if config.get('model_type') == 'lme4' else None,
            'aic': aic_bic['aic'],
            'bic': aic_bic['bic']
        },
        'coefficient_summary': coef_summary,
        'evaluation_metrics': summary_metrics,
        'pr_metrics': {
            'auc_pr': pr_metrics['auc_pr']
        },
        'roc_metrics': {
            'auc_roc': roc_metrics['auc_roc']
        },
        'optimal_threshold': {
            'threshold': optimal_threshold,
            'f1_score': optimal_f1
        },
        'output_files': {
            'coefficients': coef_path,
            'significant_coefficients': significant_path,
            'threshold_metrics': threshold_path,
            'eval_predictions': eval_pred_path,
            'forest_plot': forest_plot_path,
            'pr_curve': pr_plot_path,
            'roc_curve': roc_plot_path,
            'threshold_plot': threshold_plot_path
        },
        'profile_report': preprocessing_artifacts.get('profile_path') if config['generate_profile'] else None
    }
    
    # Save run summary
    summary_path = os.path.join(run_dir, "run_summary.json")
    safe_json_dump(run_summary, summary_path)
    
    print(f"Training pipeline completed! Results saved to: {run_dir}")
    
    return {
        'run_directory': run_dir,
        'run_summary': run_summary,
        'model': model,
        'train_df': train_df,
        'eval_df': eval_df,
        'test_df': test_df,
        'coefficients': or_df_sorted,
        'significant_coefficients': significant_coef,
        'evaluation_metrics': summary_metrics,
        'pr_metrics': pr_metrics,
        'roc_metrics': roc_metrics,
        'threshold_metrics': threshold_df,
        'optimal_threshold': optimal_threshold
    }


