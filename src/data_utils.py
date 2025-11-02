"""
Data loading and preprocessing utilities for flower species classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Optional
import config


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        df: DataFrame to explore
    """
    print("\n=== Data Exploration ===")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    
    if config.TARGET_COLUMN in df.columns:
        print(f"\nTarget variable distribution:\n{df[config.TARGET_COLUMN].value_counts()}")


def preprocess_data(df: pd.DataFrame, 
                   feature_columns: Optional[list] = None,
                   target_column: Optional[str] = None,
                   scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, Optional[StandardScaler]]:
    """
    Preprocess the data by handling missing values and scaling features.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of the target column
        scale_features: Whether to scale features using StandardScaler
        
    Returns:
        Tuple of (features DataFrame, target Series, fitted scaler or None)
    """
    df = df.copy()
    
    # Use config defaults if not provided
    if feature_columns is None:
        feature_columns = config.FEATURE_COLUMNS
    if target_column is None:
        target_column = config.TARGET_COLUMN
    
    # Handle missing values (simple strategy: drop rows with missing values)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values")
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Scale features if requested
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        print("Features scaled using StandardScaler")
    
    return X, y, scaler


def split_data(X: pd.DataFrame, 
               y: pd.Series, 
               test_size: float = None,
               random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if test_size is None:
        test_size = config.TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split completed:")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def create_sample_data(output_path: str) -> None:
    """
    Create a sample iris dataset CSV file.
    
    Args:
        output_path: Path where to save the CSV file
    """
    from sklearn.datasets import load_iris
    
    # Load iris dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = iris.target
    
    # Map target numbers to species names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_map)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample iris dataset created at {output_path}")
