# Keşifsel Veri Analizi (EDA) fonksiyonları
# Veri görselleştirme ve analiz grafikleri

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import src.config as config


def setup_style():
    # Matplotlib stilini ayarla
    plt.style.use(config.VISUALIZATION['style'])
    sns.set_palette("husl")


def plot_missing_values(missing_df: pd.DataFrame, save_path: Path):
    # Eksik değerleri görselleştir
    if len(missing_df) == 0:
        print("No missing values found")
        return
    
    fig, ax = plt.subplots(figsize=config.VISUALIZATION['figsize'])
    missing_df.plot(x='column', y='missing_percent', kind='barh', ax=ax)
    ax.set_xlabel('Missing Percentage (%)')
    ax.set_title('Missing Values Analysis')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved missing values plot to: {save_path}")


def plot_target_distribution(df: pd.DataFrame, save_path: Path):
    # Hedef değişkenin dağılımını çiz (fiyat)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original price
    if 'price' in df.columns:
        axes[0, 0].hist(df['price'], bins=50, edgecolor='black')
        axes[0, 0].set_title('Price Distribution (Original)')
        axes[0, 0].set_xlabel('Price (TL)')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].boxplot(df['price'])
        axes[0, 1].set_title('Price Boxplot (Original)')
        axes[0, 1].set_ylabel('Price (TL)')
    
    # Log transformed price
    if 'price_log' in df.columns:
        axes[1, 0].hist(df['price_log'], bins=50, edgecolor='black')
        axes[1, 0].set_title('Price Distribution (Log Transformed)')
        axes[1, 0].set_xlabel('Log(Price + 1)')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].boxplot(df['price_log'])
        axes[1, 1].set_title('Price Boxplot (Log Transformed)')
        axes[1, 1].set_ylabel('Log(Price + 1)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved target distribution plot to: {save_path}")


def plot_feature_distributions(df: pd.DataFrame, save_path: Path):
    # Sayısal özelliklerin dağılımlarını çiz
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['price', 'price_log', 'price_per_m2']
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    n_cols = min(4, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved feature distributions plot to: {save_path}")


def plot_correlation_matrix(df: pd.DataFrame, save_path: Path):
    # Korelasyon matrisini çiz
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['price_log']  # Keep price for correlation
    numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation matrix")
        return
    
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix to: {save_path}")


def plot_district_price_analysis(df: pd.DataFrame, save_path: Path, top_n: int = 20):
    # İlçe bazında fiyat analizi
    if 'district' not in df.columns or 'price' not in df.columns:
        return
    
    district_stats = df.groupby('district')['price'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
    top_districts = district_stats.head(top_n)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Mean price by district
    top_districts['mean'].plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title(f'Top {top_n} Districts by Mean Price')
    axes[0].set_xlabel('Mean Price (TL)')
    axes[0].invert_yaxis()
    
    # Boxplot for top districts
    top_district_names = top_districts.index.tolist()
    top_data = df[df['district'].isin(top_district_names)]
    sns.boxplot(data=top_data, x='district', y='price', ax=axes[1])
    axes[1].set_title(f'Price Distribution for Top {top_n} Districts')
    axes[1].set_xlabel('District')
    axes[1].set_ylabel('Price (TL)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved district price analysis to: {save_path}")


def plot_spatial_heatmap(df: pd.DataFrame, save_path: Path):
    # İlçe x mahalle pivot tablosu ile heatmap oluştur
    if 'district' not in df.columns or 'neighborhood' not in df.columns or 'price' not in df.columns:
        return
    
    # Create pivot table
    pivot = df.pivot_table(values='price', index='neighborhood', columns='district', aggfunc='mean')
    
    # Select top neighborhoods and districts for readability
    top_neighborhoods = df.groupby('neighborhood')['price'].mean().nlargest(30).index
    top_districts = df.groupby('district')['price'].mean().nlargest(15).index
    
    pivot_filtered = pivot.loc[pivot.index.isin(top_neighborhoods), pivot.columns.isin(top_districts)]
    
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(pivot_filtered, annot=False, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Mean Price (TL)'})
    ax.set_title('Price Heatmap: District x Neighborhood (Top Areas)')
    ax.set_xlabel('District')
    ax.set_ylabel('Neighborhood')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved spatial heatmap to: {save_path}")


def generate_eda_report(df: pd.DataFrame):
    # Tüm EDA görselleştirmelerini oluştur
    setup_style()
    
    print("\n=== Generating EDA Visualizations ===")
    
    # Missing values
    from src.preprocessing import analyze_missing_values
    missing_df = analyze_missing_values(df)
    if len(missing_df) > 0:
        plot_missing_values(missing_df, config.FIGURES_DIR / "01_missing_values.png")
        missing_df.to_csv(config.TABLES_DIR / "missing_values_summary.csv", index=False)
    
    # Target distribution
    plot_target_distribution(df, config.FIGURES_DIR / "02_target_distribution.png")
    
    # Feature distributions
    plot_feature_distributions(df, config.FIGURES_DIR / "03_feature_distributions.png")
    
    # Correlation matrix
    plot_correlation_matrix(df, config.FIGURES_DIR / "04_correlation_matrix.png")
    
    # District analysis
    plot_district_price_analysis(df, config.FIGURES_DIR / "05_district_price_analysis.png")
    
    # Spatial heatmap
    plot_spatial_heatmap(df, config.FIGURES_DIR / "06_spatial_heatmap.png")
    
    print("=== EDA Visualizations Complete ===\n")

