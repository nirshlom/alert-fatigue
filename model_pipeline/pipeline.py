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
from .preprocess import Preprocessor
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
    preprocessor = Preprocessor(
        impute_numeric=config['impute_numeric'],
        scale_numeric=config['scale_numeric'],
        rare_category_threshold=config['rare_category_threshold']
    )
    
    # Log preprocessing configuration
    print(f"  Preprocessing configuration:")
    print(f"    - Impute numeric: {config['impute_numeric']}")
    print(f"    - Scale numeric: {config['scale_numeric']}")
    print(f"    - Rare category threshold: {config['rare_category_threshold']}")
    
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
    artifacts = {}
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
    
    # Train model
    print("Training logistic regression model...")
    model = StatsmodelsLogitModel(use_glm=config['use_glm'])
    
    train_features = train_df[config['feature_columns']]
    train_target = train_df[config['target_column']]
    
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
    eval_features = eval_df[config['feature_columns']]
    eval_target = eval_df[config['target_column']]
    
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
            'model_type': 'Logistic Regression (Statsmodels)',
            'use_glm': config['use_glm'],
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


