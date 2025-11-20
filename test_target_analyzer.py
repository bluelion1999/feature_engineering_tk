"""
Quick test script for TargetAnalyzer class
"""

import pandas as pd
import numpy as np
from data_analysis import TargetAnalyzer

# Test 1: Classification task
print("=" * 80)
print("TEST 1: CLASSIFICATION TASK")
print("=" * 80)

np.random.seed(42)
df_class = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
})

analyzer_class = TargetAnalyzer(df_class, target_column='target', task='auto')
print(f"\nDetected task: {analyzer_class.task}")
print(f"Task info: {analyzer_class.get_task_info()}")

print("\nClass Distribution:")
print(analyzer_class.analyze_class_distribution())

print("\nClass Imbalance Info:")
imbalance = analyzer_class.get_class_imbalance_info()
for key, value in imbalance.items():
    print(f"  {key}: {value}")

print("\n" + analyzer_class.generate_summary_report())

# Test 2: Regression task
print("\n\n")
print("=" * 80)
print("TEST 2: REGRESSION TASK")
print("=" * 80)

df_reg = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.randn(1000) * 10 + 50
})

analyzer_reg = TargetAnalyzer(df_reg, target_column='target', task='auto')
print(f"\nDetected task: {analyzer_reg.task}")
print(f"Task info: {analyzer_reg.get_task_info()}")

print("\nTarget Distribution:")
dist = analyzer_reg.analyze_target_distribution()
for key, value in dist.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

print("\n" + analyzer_reg.generate_summary_report())

print("\n" + "=" * 80)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 80)
