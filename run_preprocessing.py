#!/usr/bin/env python3
"""
Wrapper script to run preprocessing from the root alert-fatigue directory.
"""

import subprocess
import sys
import os

def main():
    """Run the preprocessing script from the model_pipeline directory."""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_pipeline_dir = os.path.join(script_dir, "model_pipeline")
    
    # Check if model_pipeline directory exists
    if not os.path.exists(model_pipeline_dir):
        print(f"ERROR: model_pipeline directory not found at: {model_pipeline_dir}")
        return
    
    # Change to model_pipeline directory and run the script
    os.chdir(model_pipeline_dir)
    
    # Run the preprocessing script
    script_path = "run_preprocessing.py"
    if not os.path.exists(script_path):
        print(f"ERROR: run_preprocessing.py not found in {model_pipeline_dir}")
        return
    
    print(f"Running preprocessing from: {os.getcwd()}")
    print("=" * 50)
    
    # Execute the script
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n✓ Preprocessing completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Preprocessing failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"✗ Python executable not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
