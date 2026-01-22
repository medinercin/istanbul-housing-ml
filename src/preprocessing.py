"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
import src.config as config


def map_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names to standard names"""
    df = df.copy()
    column_mapping = {}
    
    for standard_name, possible_names in config.EXPECTED_COLUMNS.items():
        for col in df.columns:
            if col.lower() in [name.lower() for name in possible_names]:
                if col != standard_name:
                    column_mapping[col] = standard_name
                    break
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
        print(f"Column mapping applied: {column_mapping}")
    
    # Check for missing columns
    required_cols = list(config.EXPECTED_COLUMNS.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Missing columns: {missing_cols}")
    
    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values and return summary"""
    missing_summary = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percent': (df.isnull().sum() / len(df)) * 100,
        'dtype': df.dtypes
    })
    missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_count', ascending=False)
    return missing_summary


def clean_data(df: pd.DataFrame, 
                min_area: Optional[float] = None,
                min_price: Optional[float] = None,
                trim_outliers: bool = True) -> pd.DataFrame:
    """Clean data: remove invalid values and trim outliers"""
    df = df.copy()
    initial_len = len(df)
    
    # Use config defaults if not provided
    if min_area is None:
        min_area = config.PREPROCESSING['min_area']
    if min_price is None:
        min_price = config.PREPROCESSING['min_price']
    
    # Basic filtering
    if 'area' in df.columns:
        df = df[df['area'] > min_area]
        print(f"Filtered by area > {min_area}: {initial_len - len(df)} rows removed")
    
    if 'price' in df.columns:
        df = df[df['price'] > min_price]
        print(f"Filtered by price > {min_price}: {initial_len - len(df)} rows removed")
    
    # Trim outliers
    if trim_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_percentile = config.PREPROCESSING['outlier_trim_percentile_low']
        high_percentile = config.PREPROCESSING['outlier_trim_percentile_high']
        
        for col in numeric_cols:
            if col in ['area', 'price', 'age', 'floor', 'room', 'living_room']:
                low_threshold = df[col].quantile(low_percentile)
                high_threshold = df[col].quantile(high_percentile)
                before = len(df)
                df = df[(df[col] >= low_threshold) & (df[col] <= high_threshold)]
                removed = before - len(df)
                if removed > 0:
                    print(f"Trimmed outliers in {col}: {removed} rows removed ({low_percentile*100}%-{high_percentile*100}%)")
    
    print(f"Final data size: {len(df)} rows (removed {initial_len - len(df)} rows)")
    return df.reset_index(drop=True)


def prepare_target(df: pd.DataFrame, use_log: bool = None) -> Tuple[pd.DataFrame, bool]:
    """Prepare target variable with optional log transformation"""
    df = df.copy()
    
    if use_log is None:
        use_log = config.PREPROCESSING['use_log_transform']
    
    if 'price' not in df.columns:
        raise ValueError("'price' column not found")
    
    if use_log:
        df['price_log'] = np.log1p(df['price'])
        print("Applied log1p transformation to price")
        return df, True
    else:
        return df, False


def prepare_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare categorical features for encoding"""
    df = df.copy()
    
    # Ensure categorical columns are strings
    for col in ['district', 'neighborhood']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


def get_feature_columns(df: pd.DataFrame, exclude_price: bool = True) -> list:
    """Get list of feature columns for modeling"""
    exclude_cols = ['price']
    if 'price_log' in df.columns:
        exclude_cols.append('price_log')
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def split_features_target(df: pd.DataFrame, target_col: str = 'price_log') -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target"""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = df.drop(columns=[target_col, 'price'] if 'price' in df.columns and target_col != 'price' else [target_col])
    y = df[target_col]
    
    return X, y

