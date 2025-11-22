"""
Custom exceptions for feature_engineering_tk.

Provides specific exception types for better error handling and debugging.
"""


class MLToolkitError(Exception):
    """Base exception for all MLToolkit errors."""
    pass


class ValidationError(MLToolkitError):
    """Raised when input validation fails."""
    pass


class ColumnNotFoundError(ValidationError):
    """Raised when a specified column is not found in the dataframe."""

    def __init__(self, column_name: str, available_columns: list = None):
        self.column_name = column_name
        self.available_columns = available_columns

        message = f"Column '{column_name}' not found in dataframe"
        if available_columns:
            message += f". Available columns: {available_columns[:10]}"
            if len(available_columns) > 10:
                message += f"... ({len(available_columns)} total)"

        super().__init__(message)


class InvalidStrategyError(ValidationError):
    """Raised when an invalid strategy is specified."""

    def __init__(self, strategy: str, valid_strategies: list):
        self.strategy = strategy
        self.valid_strategies = valid_strategies
        message = f"Invalid strategy '{strategy}'. Valid strategies: {valid_strategies}"
        super().__init__(message)


class InvalidMethodError(ValidationError):
    """Raised when an invalid method is specified."""

    def __init__(self, method: str, valid_methods: list):
        self.method = method
        self.valid_methods = valid_methods
        message = f"Invalid method '{method}'. Valid methods: {valid_methods}"
        super().__init__(message)


class DataTypeError(ValidationError):
    """Raised when a column has an unexpected data type."""

    def __init__(self, column_name: str, expected_type: str, actual_type: str):
        self.column_name = column_name
        self.expected_type = expected_type
        self.actual_type = actual_type
        message = f"Column '{column_name}' has type '{actual_type}', expected '{expected_type}'"
        super().__init__(message)


class EmptyDataFrameError(ValidationError):
    """Raised when an operation cannot be performed on an empty dataframe."""
    pass


class InsufficientDataError(ValidationError):
    """Raised when there is insufficient data for an operation."""

    def __init__(self, operation: str, required: int, actual: int):
        self.operation = operation
        self.required = required
        self.actual = actual
        message = f"Insufficient data for {operation}: requires {required} rows, got {actual}"
        super().__init__(message)


class TransformerNotFittedError(MLToolkitError):
    """Raised when trying to use a transformer that hasn't been fitted."""

    def __init__(self, transformer_type: str):
        self.transformer_type = transformer_type
        message = f"No {transformer_type} transformers have been fitted. Call a fit method first."
        super().__init__(message)


class ConstantColumnError(ValidationError):
    """Raised when an operation cannot be performed on a constant column."""

    def __init__(self, column_name: str, operation: str):
        self.column_name = column_name
        self.operation = operation
        message = f"Cannot perform {operation} on constant column '{column_name}' (zero variance)"
        super().__init__(message)
