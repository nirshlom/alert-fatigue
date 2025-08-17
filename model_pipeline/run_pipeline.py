#!/usr/bin/env python3
"""
Run script for the Alert Fatigue Analysis pipeline.
This script runs the complete model training pipeline with the specified configuration.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from model_pipeline.config import TrainingConfig
from model_pipeline.pipeline import run_full_training, run_preprocessing_only


def main():
    """Main function to run the pipeline with the specified configuration."""
    
    # Define your configuration
    csv_path = "../alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv"
    date_col = "time_prescribing_order"
    target_col = "alert_status_binary"
    features = ["age", "gender", "hospital_days", "charlson_score", "shift_type", "unit_category"]
    
    print("Alert Fatigue Analysis Pipeline")
    print("=" * 50)
    print(f"Input CSV: {csv_path}")
    print(f"Date Column: {date_col}")
    print(f"Target Column: {target_col}")
    print(f"Features: {features}")
    print()
    
    # Check if input file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: Input file not found: {csv_path}")
        print("Please check the file path and ensure the file exists.")
        return
    
    # Create configuration
    config = TrainingConfig(
        input_csv_path=csv_path,
        date_column=date_col,
        target_column=target_col,
        feature_columns=features,
        train_frac=0.7,
        eval_frac=0.15,
        test_frac=0.15,
        ascending=True,
        stratify=False,
        random_seed=42,
        impute_numeric=True,
        scale_numeric=False,
        rare_category_threshold=0.01,
        output_dir="model_pipeline/outputs",
        generate_profile=True,
        use_glm=True
    )
    
    # Validate configuration
    try:
        config.validate()
        print("Configuration validated successfully!")
    except ValueError as e:
        print(f"Configuration error: {e}")
        return
    
    print("\nStarting pipeline execution...")
    print("-" * 30)
    
    try:
        # Run the full training pipeline
        results = run_full_training(config)
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results['run_directory']}")
        print(f"Run summary: {results['run_directory']}/run_summary.json")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to run just preprocessing to isolate the issue
        print("\nAttempting to run preprocessing only to isolate the issue...")
        try:
            train_df, eval_df, test_df, artifacts = run_preprocessing_only(config)
            print("Preprocessing completed successfully!")
            print(f"Train samples: {len(train_df)}")
            print(f"Eval samples: {len(eval_df)}")
            print(f"Test samples: {len(test_df)}")
        except Exception as preprocess_error:
            print(f"Preprocessing also failed: {preprocess_error}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
