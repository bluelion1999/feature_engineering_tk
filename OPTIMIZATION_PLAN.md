# Critical Optimizations Implementation Plan

## Baseline Performance Metrics

| Operation | Current Performance | Target Performance | Impact |
|-----------|-------------------|-------------------|---------|
| DataFrame init (1M rows) | 231ms | <25ms | **10x** |
| Class-wise statistics | 969ms | <100ms | **10x** |
| Mean calculations | 75ms | <25ms | **3x** |
| Outlier detection (5 cols) | 221ms | <100ms | **2x** |
| String validation | 0.03ms | <0.01ms | **3x** |

---

## Critical Optimization #1: Copy-on-Write Pattern

### Current Issue
Every class initialization performs a full `df.copy()` operation:
```python
def __init__(self, df):
    self.df = validate_and_copy_dataframe(df)  # Always copies!
```

**Measured Impact**: 231ms for 1M row DataFrame

### Solution: Lazy Copy Pattern
Only copy when first modification occurs:

```python
class FeatureEngineeringBase:
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            logger.warning("Initializing with empty DataFrame")

        self._df_original = df
        self._df_modified = None
        self._is_modified = False

    @property
    def df(self):
        """Get DataFrame (returns modified version if it exists)."""
        return self._df_modified if self._is_modified else self._df_original

    @df.setter
    def df(self, value):
        """Set DataFrame (marks as modified)."""
        self._df_modified = value
        self._is_modified = True

    def _ensure_copy(self):
        """Ensure we have a copy before modifying."""
        if not self._is_modified:
            self._df_modified = self._df_original.copy()
            self._is_modified = True
```

**Implementation Steps**:
1. Update `FeatureEngineeringBase.__init__()` to use lazy pattern
2. Add `_ensure_copy()` method
3. Update all inplace operations to call `_ensure_copy()` first
4. Run full test suite to ensure no regressions

**Expected Improvement**: 10-50x for read-only operations

**Files to Modify**:
- `feature_engineering_tk/base.py`
- `feature_engineering_tk/utils.py` (remove immediate copy)

**Testing**:
- Verify read-only operations don't trigger copy
- Verify inplace operations do trigger copy
- Benchmark before/after

**Risk**: LOW - Well-established pattern, easy to test

---

## Critical Optimization #2: Vectorize String Column Validation

### Current Issue
Python loop iterating through columns one by one:
```python
for col in columns:
    if col not in df.columns:
        logger.warning(...)
        continue
    if df[col].dtype != 'object':
        logger.warning(...)
        continue
    string_cols.append(col)
```

**Measured Impact**: 0.03ms (low absolute time but scales poorly)

### Solution: Vectorized Pandas Operations
```python
def get_string_columns(df, columns=None):
    if columns is None:
        return df.select_dtypes(include=['object']).columns.tolist()

    # Vectorized validation
    columns_set = set(df.columns)
    valid_cols = [col for col in columns if col in columns_set]
    missing_cols = set(columns) - columns_set

    if missing_cols:
        logger.warning(f"Columns not found: {missing_cols}")

    # Vectorized dtype check
    string_cols = df[valid_cols].select_dtypes(include=['object']).columns.tolist()
    non_string = set(valid_cols) - set(string_cols)

    if non_string:
        logger.warning(f"Non-string columns: {non_string}")

    return string_cols
```

**Implementation Steps**:
1. Update `get_string_columns()` in utils.py
2. Apply same pattern to `validate_numeric_columns()`
3. Run tests
4. Benchmark

**Expected Improvement**: 10-20x for many columns

**Files to Modify**:
- `feature_engineering_tk/utils.py`

**Risk**: VERY LOW - Simple refactoring

---

## Critical Optimization #3: Fix N+1 Query Pattern

### Current Issue
Filtering DataFrame repeatedly in nested loops:
```python
for feature in feature_columns:
    for cls in classes:
        class_data = self.df[self.df[self.target_column] == cls][feature]
        # Filters ENTIRE DataFrame for each class!
```

**Measured Impact**: 969ms for 100K rows, 25 features, 2 classes

### Solution: Single GroupBy Operation
```python
# Single groupby operation
grouped = self.df.groupby(self.target_column)[feature_columns]

results = {}
for feature in feature_columns:
    stats = grouped[feature].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    results[feature] = {
        cls: {
            'count': row['count'],
            'mean': row['mean'],
            'median': row['median'],
            'std': row['std'],
            'min': row['min'],
            'max': row['max']
        }
        for cls, row in stats.iterrows()
    }
```

**Implementation Steps**:
1. Identify all N+1 patterns in `data_analysis.py`
2. Replace with groupby operations
3. Update return format if needed
4. Run tests
5. Benchmark

**Expected Improvement**: 10-50x

**Files to Modify**:
- `feature_engineering_tk/data_analysis.py` (TargetAnalyzer)

**Locations**:
- `analyze_class_wise_statistics()` (line ~1020)
- `analyze_feature_target_relationship()` (line ~922)

**Risk**: LOW - groupby is well-tested, maintain output format

---

## Critical Optimization #4: Pre-compute Means

### Current Issue
Computing means during fillna operation:
```python
df_result[numeric_cols] = df_result[numeric_cols].fillna(
    df_result[numeric_cols].mean()  # Computes mean for each column
)
```

**Measured Impact**: 75ms for 25 columns

### Solution: Pre-compute Once
```python
elif strategy == 'mean':
    numeric_cols = get_numeric_columns(df_result, columns)
    if len(numeric_cols) > 0:
        means = df_result[numeric_cols].mean()  # Compute once
        df_result[numeric_cols] = df_result[numeric_cols].fillna(means)
```

**Implementation Steps**:
1. Update `handle_missing_values()` in preprocessing.py
2. Apply to median strategy as well
3. Run tests
4. Benchmark

**Expected Improvement**: 2-5x

**Files to Modify**:
- `feature_engineering_tk/preprocessing.py`

**Risk**: VERY LOW - Simple change

---

## Critical Optimization #5: Fix Outlier Detection Warning

### Current Issue
Boolean Series key reindexing warning in `handle_outliers()`:
```python
df_result = df_result[~outlier_mask]  # Index mismatch after loop
```

**Fix**: Reset index or use .loc properly
```python
# Inside loop, use proper indexing
if action == 'remove':
    df_result = df_result[~df_result.index.isin(outlier_mask[outlier_mask].index)]
# OR accumulate mask and apply once at end
```

**Risk**: LOW

---

## Implementation Order

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Create baseline benchmarks
2. ⬜ Optimize #4: Pre-compute means
3. ⬜ Optimize #2: Vectorize string validation
4. ⬜ Fix #5: Outlier detection warning

### Phase 2: Medium Complexity (2-3 hours)
5. ⬜ Optimize #3: Fix N+1 query pattern

### Phase 3: Architecture Change (3-4 hours)
6. ⬜ Optimize #1: Copy-on-write pattern

---

## Testing Strategy

After each optimization:
1. Run full test suite (182 tests must pass)
2. Run benchmark suite
3. Compare before/after metrics
4. Commit if successful

---

## Success Criteria

| Optimization | Target Improvement | Status |
|-------------|-------------------|--------|
| Copy-on-write | 10x for read-only ops | ⬜ |
| String validation | 10-20x for many cols | ⬜ |
| N+1 pattern | 10x | ⬜ |
| Mean pre-compute | 3x | ⬜ |
| Outlier warning | Fixed | ⬜ |

**Overall Target**: 5-10x end-to-end performance improvement for typical workflows
