"""
Benchmark suite for measuring performance of critical operations.

This module provides baseline measurements and regression testing for
performance-critical operations in the Feature Engineering Toolkit.
"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering_tk import (
    DataPreprocessor,
    FeatureEngineer,
    DataAnalyzer,
    TargetAnalyzer,
    FeatureSelector
)


class BenchmarkSuite:
    """Suite for benchmarking toolkit operations."""

    def __init__(self):
        self.results = {}

    def create_test_data(self, rows: int = 100000, cols: int = 50) -> pd.DataFrame:
        """Create test DataFrame with mixed data types."""
        np.random.seed(42)

        data = {}

        # Numeric columns
        for i in range(cols // 2):
            data[f'num_{i}'] = np.random.randn(rows)

        # Categorical columns
        for i in range(cols // 4):
            data[f'cat_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], rows)

        # String columns
        for i in range(cols // 4):
            data[f'str_{i}'] = [f'text_{np.random.randint(0, 1000)}' for _ in range(rows)]

        # Target column
        data['target'] = np.random.choice([0, 1], rows)

        return pd.DataFrame(data)

    def benchmark(self, name: str, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Benchmark a function and return timing results."""
        # Warm up
        func(*args, **kwargs)

        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        result = {
            'name': name,
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min(times),
            'max_time': max(times),
        }

        self.results[name] = result
        return result

    def benchmark_initialization(self):
        """Benchmark class initialization (Critical Issue #1)."""
        print("\n=== Benchmarking DataFrame Initialization ===")

        # Small DataFrame
        df_small = self.create_test_data(rows=1000, cols=10)
        result = self.benchmark(
            "DataPreprocessor.__init__ (1K rows)",
            lambda: DataPreprocessor(df_small)
        )
        print(f"  1K rows: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

        # Medium DataFrame
        df_medium = self.create_test_data(rows=100000, cols=50)
        result = self.benchmark(
            "DataPreprocessor.__init__ (100K rows)",
            lambda: DataPreprocessor(df_medium)
        )
        print(f"  100K rows: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

        # Large DataFrame
        df_large = self.create_test_data(rows=1000000, cols=50)
        result = self.benchmark(
            "DataPreprocessor.__init__ (1M rows)",
            lambda: DataPreprocessor(df_large)
        )
        print(f"  1M rows: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

    def benchmark_string_validation(self):
        """Benchmark string column validation (Critical Issue #3)."""
        print("\n=== Benchmarking String Column Validation ===")

        df = self.create_test_data(rows=100000, cols=50)
        string_cols = [col for col in df.columns if col.startswith('str_')]

        from feature_engineering_tk.utils import get_string_columns

        result = self.benchmark(
            "get_string_columns (12 columns)",
            get_string_columns,
            df, string_cols
        )
        print(f"  12 string columns: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

    def benchmark_statistics_computation(self):
        """Benchmark class-wise statistics (Critical Issue #4)."""
        print("\n=== Benchmarking Class-wise Statistics ===")

        df = self.create_test_data(rows=100000, cols=50)
        analyzer = TargetAnalyzer(df, 'target', task='classification')

        result = self.benchmark(
            "TargetAnalyzer.analyze_class_wise_statistics",
            analyzer.analyze_class_wise_statistics
        )
        print(f"  100K rows, 25 features: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

    def benchmark_mean_calculations(self):
        """Benchmark mean calculations in fillna (Critical Issue #5)."""
        print("\n=== Benchmarking Mean Calculations ===")

        df = self.create_test_data(rows=100000, cols=50)
        # Add some missing values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df.loc[np.random.choice(df.index, 5000), col] = np.nan

        preprocessor = DataPreprocessor(df)
        numeric_cols = [col for col in df.columns if col.startswith('num_')]

        result = self.benchmark(
            "handle_missing_values (mean strategy)",
            preprocessor.handle_missing_values,
            columns=numeric_cols,
            strategy='mean',
            inplace=False
        )
        print(f"  25 numeric columns: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

    def benchmark_outlier_detection(self):
        """Benchmark outlier detection."""
        print("\n=== Benchmarking Outlier Detection ===")

        df = self.create_test_data(rows=100000, cols=50)
        preprocessor = DataPreprocessor(df)
        numeric_cols = [col for col in df.columns if col.startswith('num_')][:5]

        result = self.benchmark(
            "handle_outliers (5 columns)",
            preprocessor.handle_outliers,
            columns=numeric_cols,
            method='iqr',
            action='remove',
            inplace=False
        )
        print(f"  5 columns: {result['avg_time']*1000:.2f}ms ± {result['std_time']*1000:.2f}ms")

    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("=" * 70)
        print("Feature Engineering Toolkit - Performance Baseline Benchmarks")
        print("=" * 70)

        self.benchmark_initialization()
        self.benchmark_string_validation()
        self.benchmark_statistics_computation()
        self.benchmark_mean_calculations()
        self.benchmark_outlier_detection()

        print("\n" + "=" * 70)
        print("Benchmark Summary")
        print("=" * 70)

        for name, result in self.results.items():
            print(f"{name}: {result['avg_time']*1000:.2f}ms")

        return self.results

    def save_baseline(self, filename: str = "baseline_results.json"):
        """Save benchmark results as baseline."""
        import json

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nBaseline saved to {filename}")


if __name__ == '__main__':
    suite = BenchmarkSuite()
    results = suite.run_all_benchmarks()
    suite.save_baseline()
