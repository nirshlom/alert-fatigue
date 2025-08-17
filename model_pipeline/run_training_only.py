#!/usr/bin/env python3
"""
Script to run only the training step of the Alert Fatigue Analysis pipeline.
This assumes preprocessing has already been completed and you want to focus on model training.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from model_pipeline.config import get_config
from model_pipeline.pipeline import run_full_training

def main():
    """Run only the training step (with preprocessing if needed)."""
    
    print("Alert Fatigue Analysis - Training Only")
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
    
    # Override generate_profile for faster training iterations
    config['generate_profile'] = False
    
    print("\nStarting training pipeline...")
    print("-" * 30)
    
    try:
        # Run the full training pipeline
        results = run_full_training(config)
        
        print(f"\n✓ Training pipeline completed successfully!")
        print(f"✓ Results saved to: {results['run_directory']}")
        print(f"✓ Run summary: {results['run_directory']}/run_summary.json")
        
        # Show key results
        if 'evaluation_metrics' in results:
            metrics = results['evaluation_metrics']
            print(f"\nKey Results:")
            print(f"  - Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            print(f"  - Precision: {metrics.get('precision', 'N/A'):.3f}")
            print(f"  - Recall: {metrics.get('recall', 'N/A'):.3f}")
            print(f"  - F1 Score: {metrics.get('f1', 'N/A'):.3f}")
        
        if 'optimal_threshold' in results:
            opt = results['optimal_threshold']
            print(f"  - Optimal Threshold: {opt.get('threshold', 'N/A'):.3f}")
            print(f"  - Optimal F1: {opt.get('f1_score', 'N/A'):.3f}")
        
        print(f"\nOutput files created:")
        if 'output_files' in results.get('run_summary', {}):
            for file_type, file_path in results['run_summary']['output_files'].items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  - {file_type}: {file_size:,} bytes")
        
    except Exception as e:
        print(f"\n✗ ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
