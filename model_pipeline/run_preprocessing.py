#!/usr/bin/env python3
"""
Script to run only the preprocessing step of the Alert Fatigue Analysis pipeline.
This allows you to inspect the data after preprocessing before running the full training.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from model_pipeline.config import TrainingConfig
from model_pipeline.pipeline import run_preprocessing_only

def main():
    """Run only the preprocessing step."""
    
    print("Alert Fatigue Analysis - Preprocessing Only")
    print("=" * 50)
    
    # Define your configuration
    csv_path = "../alert_analysis/data/main_data_2022/df_main_active_adult_renamed_clean_sample_10pct.csv"
    date_col = "time_prescribing_order"
    target_col = "alert_status_binary"
    features = ["age", "gender", "hospital_days", "charlson_score", "shift_type", "unit_category"]

    
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
        print("✓ Configuration validated successfully!")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return
    
    print("\nStarting preprocessing pipeline...")
    print("-" * 30)
    
    try:
        # Run preprocessing only
        train_df, eval_df, test_df, artifacts = run_preprocessing_only(config)
        
        print(f"\n✓ Preprocessing completed successfully!")
        print(f"✓ Data ready for training:")
        print(f"  - Training set: {len(train_df):,} samples")
        print(f"  - Evaluation set: {len(eval_df):,} samples")
        print(f"  - Test set: {len(test_df):,} samples")
        
        if artifacts.get('run_directory'):
            print(f"✓ Profile report saved to: {artifacts['run_directory']}")
        
        print(f"\nNext steps:")
        print(f"  1. Review the preprocessing logs above")
        print(f"  2. Check the profile report if generated")
        print(f"  3. Run the full training pipeline when ready")
        print(f"  4. Or use the preprocessed data for custom analysis")
        
    except Exception as e:
        print(f"\n✗ ERROR during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
