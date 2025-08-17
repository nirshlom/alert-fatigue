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

from model_pipeline.config import get_config
from model_pipeline.pipeline import run_full_training, run_preprocessing_only


def main():
    """Main function to run the pipeline with the specified configuration."""
    
    print("Alert Fatigue Analysis Pipeline")
    print("=" * 50)
    
    # Get centralized configuration
    config = get_config()
    
    # Display configuration
    print(f"Input CSV: {config['input_csv_path']}")
    print(f"Date Column: {config['date_column']}")
    print(f"Target Column: {config['target_column']}")
    print(f"Features: {config['feature_columns']}")
    print()
    
    # Check if input file exists
    if not os.path.exists(config['input_csv_path']):
        print(f"ERROR: Input file not found: {config['input_csv_path']}")
        print("Please check the file path and ensure the file exists.")
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
