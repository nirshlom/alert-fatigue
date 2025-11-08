"""Test script for OneHotPreprocessor"""

import pandas as pd
import numpy as np
from preprocess_onehot import OneHotPreprocessor

# Create sample training data
print("=" * 80)
print("Creating sample training data...")
print("=" * 80)

train_data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],  # Binary (2 levels)
    'unit_category': ['ICU', 'Geriatric', 'Emergency', 'ICU', 'Surgery', 'Emergency', 'ICU', 'Geriatric'],  # Multi-level (>2)
    'score': [10, 20, 15, 25, 30, 35, 40, 45]
})

print("\nTraining data:")
print(train_data)
print(f"\nTraining data shape: {train_data.shape}")

# Test 1: Basic functionality with strict mode (default)
print("\n" + "=" * 80)
print("TEST 1: Basic functionality with strict mode (handle_unseen_categories=False)")
print("=" * 80)

preprocessor_strict = OneHotPreprocessor(
    impute_numeric=True,
    scale_numeric=False,
    rare_category_threshold=0.01,
    handle_unseen_categories=False
)

print("\nFitting preprocessor on training data...")
preprocessor_strict.fit(train_data)

print("\nFeature mapping:")
feature_mapping = preprocessor_strict.get_feature_mapping()
for orig, new_cols in feature_mapping.items():
    print(f"  {orig} -> {new_cols}")

print("\nTransformed feature names:")
transformed_names = preprocessor_strict.get_transformed_feature_names()
print(f"  {transformed_names}")

print("\nTransforming training data...")
train_transformed = preprocessor_strict.transform(train_data)
print("\nTransformed training data:")
print(train_transformed)
print(f"\nTransformed shape: {train_transformed.shape}")

# Test 2: Transform with eval data (same categories)
print("\n" + "=" * 80)
print("TEST 2: Transform eval data with same categories (should work)")
print("=" * 80)

eval_data = pd.DataFrame({
    'age': [28, 32, 38],
    'gender': ['M', 'F', 'M'],
    'unit_category': ['ICU', 'Emergency', 'Surgery'],
    'score': [12, 22, 18]
})

print("\nEval data:")
print(eval_data)

eval_transformed = preprocessor_strict.transform(eval_data)
print("\nTransformed eval data:")
print(eval_transformed)

# Test 3: Transform with unseen categories (should raise error in strict mode)
print("\n" + "=" * 80)
print("TEST 3: Transform with unseen categories (should raise error in strict mode)")
print("=" * 80)

eval_with_unseen = pd.DataFrame({
    'age': [28, 32],
    'gender': ['M', 'F'],
    'unit_category': ['ICU', 'Pediatric'],  # 'Pediatric' is unseen
    'score': [12, 22]
})

print("\nEval data with unseen category:")
print(eval_with_unseen)

try:
    eval_transformed_unseen = preprocessor_strict.transform(eval_with_unseen)
    print("\nERROR: Should have raised ValueError!")
except ValueError as e:
    print(f"\n✓ Correctly raised ValueError: {e}")

# Test 4: Lenient mode (handle_unseen_categories=True)
print("\n" + "=" * 80)
print("TEST 4: Lenient mode (handle_unseen_categories=True)")
print("=" * 80)

preprocessor_lenient = OneHotPreprocessor(
    impute_numeric=True,
    scale_numeric=False,
    rare_category_threshold=0.01,
    handle_unseen_categories=True
)

print("\nFitting preprocessor in lenient mode...")
preprocessor_lenient.fit(train_data)

print("\nTransforming eval data with unseen category (lenient mode)...")
eval_transformed_lenient = preprocessor_lenient.transform(eval_with_unseen)
print("\nTransformed eval data (unseen mapped to 'Other'):")
print(eval_transformed_lenient)

# Check if 'Other' column exists
if 'unit_category_Other' in eval_transformed_lenient.columns:
    print("\n✓ 'Other' column created for unseen categories")
    print(f"  unit_category_Other values: {eval_transformed_lenient['unit_category_Other'].values}")

# Test 5: Binary categorical handling
print("\n" + "=" * 80)
print("TEST 5: Binary categorical handling (gender should be 0/1)")
print("=" * 80)

print("\nOriginal gender values:")
print(train_data['gender'].value_counts())

print("\nTransformed gender column (should be 0/1):")
print(train_transformed['gender'].value_counts())
print(f"\nGender column dtype: {train_transformed['gender'].dtype}")
print(f"Gender unique values: {sorted(train_transformed['gender'].unique())}")

# Test 6: Multi-level categorical one-hot encoding
print("\n" + "=" * 80)
print("TEST 6: Multi-level categorical one-hot encoding")
print("=" * 80)

print("\nOriginal unit_category values:")
print(train_data['unit_category'].value_counts())

print("\nOne-hot encoded columns for unit_category:")
unit_cols = [col for col in train_transformed.columns if col.startswith('unit_category_')]
print(f"  Columns: {unit_cols}")
for col in unit_cols:
    print(f"  {col}: {train_transformed[col].sum()} occurrences")

# Test 7: Feature mapping details
print("\n" + "=" * 80)
print("TEST 7: Feature mapping details")
print("=" * 80)

print("\nDetailed feature mapping:")
for orig_col in train_data.columns:
    if orig_col in feature_mapping:
        new_cols = feature_mapping[orig_col]
        if len(new_cols) == 1 and new_cols[0] == orig_col:
            print(f"  {orig_col}: Kept as-is ({'binary' if orig_col == 'gender' else 'numeric'})")
        else:
            print(f"  {orig_col}: One-hot encoded into {len(new_cols)} columns")
            for new_col in new_cols:
                print(f"    - {new_col}")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)

