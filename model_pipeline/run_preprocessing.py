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

from model_pipeline.config import get_config, TrainingConfig
from model_pipeline.pipeline import run_preprocessing_only

def main():
    """Run only the preprocessing step."""
    
    print("Alert Fatigue Analysis - Preprocessing Only")
    print("=" * 50)
    
    # Get centralized configuration
    config_dict = get_config()
    
    # Display configuration
    print(f"Input CSV: {config_dict['input_csv_path']}")
    print(f"Date Column: {config_dict['date_column']}")
    print(f"Target Column: {config_dict['target_column']}")
    print(f"Features: {config_dict['feature_columns']}")
    print()
    
    # Check if input file exists
    if not os.path.exists(config_dict['input_csv_path']):
        print(f"ERROR: Input file not found: {config_dict['input_csv_path']}")
        print("Please check the file path and ensure the file exists.")
        return
    
    # Create configuration object from centralized config
    config = TrainingConfig(**config_dict)
    
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
