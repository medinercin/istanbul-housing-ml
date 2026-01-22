"""Volatility analysis for rental prices"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import src.config as config
from src.io import save_dataframe


def calculate_volatility_metrics(df: pd.DataFrame, 
                                 groupby_cols: list = None) -> pd.DataFrame:
    """Calculate volatility metrics (CV, IQR, std) by district/neighborhood"""
    if groupby_cols is None:
        groupby_cols = ['district', 'neighborhood']
    
    if 'price' not in df.columns:
        raise ValueError("'price' column required for volatility analysis")
    
    volatility_results = []
    
    for group_col in groupby_cols:
        if group_col not in df.columns:
            continue
        
        group_stats = df.groupby(group_col)['price'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('min', 'min'),
            ('max', 'max')
        ]).reset_index()
        
        # Calculate volatility metrics
        group_stats['IQR'] = group_stats['q75'] - group_stats['q25']
        group_stats['CV'] = (group_stats['std'] / group_stats['mean']) * 100  # Coefficient of Variation (%)
        group_stats['range'] = group_stats['max'] - group_stats['min']
        
        # Rename for clarity
        group_stats = group_stats.rename(columns={group_col: 'location'})
        group_stats['level'] = group_col
        
        volatility_results.append(group_stats)
    
    if volatility_results:
        result_df = pd.concat(volatility_results, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()


def plot_volatility_analysis(volatility_df: pd.DataFrame, save_path, level: str = 'neighborhood', top_n: int = 20):
    """Plot volatility analysis"""
    level_df = volatility_df[volatility_df['level'] == level].copy()
    
    if len(level_df) == 0:
        print(f"No data for level: {level}")
        return
    
    # Sort by CV (coefficient of variation)
    level_df = level_df.sort_values('CV', ascending=False).head(top_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top volatile by CV
    axes[0, 0].barh(range(len(level_df)), level_df['CV'].values)
    axes[0, 0].set_yticks(range(len(level_df)))
    axes[0, 0].set_yticklabels(level_df['location'].values)
    axes[0, 0].set_xlabel('Coefficient of Variation (%)')
    axes[0, 0].set_title(f'Top {top_n} Most Volatile {level.capitalize()}s (by CV)')
    axes[0, 0].invert_yaxis()
    
    # IQR comparison
    axes[0, 1].barh(range(len(level_df)), level_df['IQR'].values, color='orange')
    axes[0, 1].set_yticks(range(len(level_df)))
    axes[0, 1].set_yticklabels(level_df['location'].values)
    axes[0, 1].set_xlabel('Interquartile Range (IQR)')
    axes[0, 1].set_title(f'Top {top_n} {level.capitalize()}s by IQR')
    axes[0, 1].invert_yaxis()
    
    # CV vs Mean scatter
    axes[1, 0].scatter(level_df['mean'], level_df['CV'], alpha=0.6)
    axes[1, 0].set_xlabel('Mean Price (TL)')
    axes[1, 0].set_ylabel('Coefficient of Variation (%)')
    axes[1, 0].set_title('Volatility vs Mean Price')
    for idx, row in level_df.head(10).iterrows():
        axes[1, 0].annotate(row['location'], (row['mean'], row['CV']), fontsize=8)
    
    # Count vs CV
    axes[1, 1].scatter(level_df['count'], level_df['CV'], alpha=0.6, color='green')
    axes[1, 1].set_xlabel('Number of Listings')
    axes[1, 1].set_ylabel('Coefficient of Variation (%)')
    axes[1, 1].set_title('Volatility vs Sample Size')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved volatility analysis to: {save_path}")


def generate_volatility_report(df: pd.DataFrame):
    """Generate volatility analysis report"""
    print("\n=== Generating Volatility Analysis ===")
    
    volatility_df = calculate_volatility_metrics(df)
    
    if len(volatility_df) > 0:
        # Save metrics table
        save_dataframe(volatility_df, "volatility_metrics.csv")
        
        # Plot district-level volatility
        plot_volatility_analysis(volatility_df, 
                                config.FIGURES_DIR / "07_volatility_district.png",
                                level='district', top_n=20)
        
        # Plot neighborhood-level volatility
        plot_volatility_analysis(volatility_df,
                                config.FIGURES_DIR / "08_volatility_neighborhood.png",
                                level='neighborhood', top_n=30)
        
        print("=== Volatility Analysis Complete ===\n")
        return volatility_df
    else:
        print("No volatility data generated")
        return pd.DataFrame()

