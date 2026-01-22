"""Feature engineering utilities"""

import pandas as pd
import numpy as np
from typing import Dict, List
import src.config as config


def create_derived_features(df: pd.DataFrame, for_model: bool = False) -> pd.DataFrame:
    """
    Create derived features.
    If for_model=False, creates price_per_m2 for EDA (leakage risk).
    If for_model=True, only creates safe features without target leakage.
    """
    df = df.copy()
    
    # Price per m2 (EDA only - has leakage if used in model)
    if not for_model and config.FEATURES['create_price_per_m2'] and 'price' in df.columns and 'area' in df.columns:
        df['price_per_m2'] = df['price'] / df['area']
        print("Created price_per_m2 feature (EDA only)")
    
    # Safe features for modeling
    if for_model:
        # Total rooms
        if 'room' in df.columns and 'living_room' in df.columns:
            df['total_rooms'] = df['room'] + df['living_room']
        
        # Area per room
        if 'area' in df.columns and 'room' in df.columns:
            df['area_per_room'] = df['area'] / (df['room'] + 1)  # +1 to avoid division by zero
        
        # Age categories (optional)
        if 'age' in df.columns:
            df['age_category'] = pd.cut(df['age'], bins=[-1, 0, 5, 10, 20, 100], 
                                       labels=['Yeni', '0-5', '6-10', '11-20', '20+'])
    
    return df


def create_aggregation_features(df: pd.DataFrame, 
                                groupby_cols: List[str],
                                agg_cols: List[str] = None) -> pd.DataFrame:
    """Create aggregation features at district/neighborhood level"""
    df = df.copy()
    
    if agg_cols is None:
        agg_cols = ['price', 'area', 'age']
        agg_cols = [col for col in agg_cols if col in df.columns]
    
    aggregation_stats = {}
    for col in agg_cols:
        if col in df.columns:
            aggregation_stats[col] = ['mean', 'median', 'std', 'count']
    
    if not aggregation_stats:
        return df
    
    # Create aggregations for each groupby level
    for group_col in groupby_cols:
        if group_col not in df.columns:
            continue
        
        agg_df = df.groupby(group_col)[list(aggregation_stats.keys())].agg(aggregation_stats)
        agg_df.columns = [f'{group_col}_{col}_{stat}' for col, stats in aggregation_stats.items() for stat in stats]
        
        # Merge back
        for col_name in agg_df.columns:
            df[col_name] = df[group_col].map(agg_df[col_name])
    
    print(f"Created aggregation features for: {groupby_cols}")
    return df


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for model input"""
    df = create_derived_features(df, for_model=True)
    
    # Add aggregation features if needed
    if config.FEATURES.get('aggregation_levels'):
        df = create_aggregation_features(df, config.FEATURES['aggregation_levels'])
    
    return df


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical columns"""
    cat_cols = []
    for col in ['district', 'neighborhood', 'age_category']:
        if col in df.columns:
            cat_cols.append(col)
    return cat_cols


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric columns (excluding target)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['price', 'price_log', 'price_per_m2']
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    return numeric_cols

