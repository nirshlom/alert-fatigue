#!/usr/bin/env python3
"""
Test file to verify pandas and numpy functionality in hiba_env virtual environment
"""

import sys
import pandas as pd
import numpy as np

def test_pandas_numpy():
    """Test basic pandas and numpy operations"""
    
    print("=" * 50)
    print("PANDAS AND NUMPY TEST")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
    print()
    
    # Test numpy operations
    print("NUMPY TESTS:")
    print("-" * 20)
    
    # Create a simple array
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Original array: {arr}")
    print(f"Array shape: {arr.shape}")
    print(f"Array mean: {np.mean(arr)}")
    print(f"Array sum: {np.sum(arr)}")
    
    # Create a 2D array
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D array:\n{arr_2d}")
    print(f"2D array shape: {arr_2d.shape}")
    print()
    
    # Test pandas operations
    print("PANDAS TESTS:")
    print("-" * 20)
    
    # Create a simple DataFrame
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Boston']
    }
    
    df = pd.DataFrame(data)
    print("Sample DataFrame:")
    print(df)
    print()
    
    print("DataFrame info:")
    print(df.info())
    print()
    
    print("DataFrame statistics:")
    print(df.describe())
    print()
    
    # Test basic operations
    print("Basic operations:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    print(f"Average age: {df['Age'].mean()}")
    
    print()
    print("=" * 50)
    print("ALL TESTS PASSED! ✅")
    print("Pandas and Numpy are working correctly.")
    print("=" * 50)

if __name__ == "__main__":
    test_pandas_numpy()
