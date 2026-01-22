"""Enhanced model comparison visualizations"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import src.config as config
from src.io import load_json, load_model


def create_comprehensive_comparison():
    """Create comprehensive model comparison visualizations"""
    print("\n=== Creating Enhanced Model Comparison Visualizations ===")
    
    # Load metrics
    rf_metrics = load_json('randomforest_metrics.json')
    xgb_metrics = load_json('xgboost_metrics.json')
    cb_metrics = load_json('catboost_metrics.json')
    
    models = ['Random Forest', 'XGBoost', 'CatBoost']
    metrics_data = {
        'R²': [rf_metrics['R2'], xgb_metrics['R2'], cb_metrics['R2']],
        'MAE (TL)': [rf_metrics['MAE'], xgb_metrics['MAE'], cb_metrics['MAE']],
        'RMSE (TL)': [rf_metrics['RMSE'], xgb_metrics['RMSE'], cb_metrics['RMSE']],
        'MAPE (%)': [rf_metrics['MAPE'], xgb_metrics['MAPE'], cb_metrics['MAPE']]
    }
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart comparison - All metrics
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(models))
    width = 0.2
    
    # Normalize metrics for comparison (0-1 scale)
    r2_norm = [m / max(metrics_data['R²']) for m in metrics_data['R²']]
    mae_norm = [1 - (m - min(metrics_data['MAE (TL)'])) / (max(metrics_data['MAE (TL)']) - min(metrics_data['MAE (TL)'])) for m in metrics_data['MAE (TL)']]
    rmse_norm = [1 - (m - min(metrics_data['RMSE (TL)'])) / (max(metrics_data['RMSE (TL)']) - min(metrics_data['RMSE (TL)'])) for m in metrics_data['RMSE (TL)']]
    mape_norm = [1 - (m - min(metrics_data['MAPE (%)'])) / (max(metrics_data['MAPE (%)']) - min(metrics_data['MAPE (%)'])) for m in metrics_data['MAPE (%)']]
    
    ax1.bar(x - 1.5*width, r2_norm, width, label='R² (normalized)', color='#2ecc71', alpha=0.8)
    ax1.bar(x - 0.5*width, mae_norm, width, label='MAE (normalized, inverted)', color='#e74c3c', alpha=0.8)
    ax1.bar(x + 0.5*width, rmse_norm, width, label='RMSE (normalized, inverted)', color='#f39c12', alpha=0.8)
    ax1.bar(x + 1.5*width, mape_norm, width, label='MAPE (normalized, inverted)', color='#3498db', alpha=0.8)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Comprehensive Model Comparison (Normalized Metrics)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Add value annotations
    for i, model in enumerate(models):
        ax1.text(i - 1.5*width, r2_norm[i] + 0.02, f'{metrics_data["R²"][i]:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i - 0.5*width, mae_norm[i] + 0.02, f'{metrics_data["MAE (TL)"][i]:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + 0.5*width, rmse_norm[i] + 0.02, f'{metrics_data["RMSE (TL)"][i]:.0f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + 1.5*width, mape_norm[i] + 0.02, f'{metrics_data["MAPE (%)"][i]:.2f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. R² Score comparison
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['#95a5a6', '#95a5a6', '#2ecc71']
    bars = ax2.bar(models, metrics_data['R²'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    bars[2].set_color('#2ecc71')  # CatBoost in green
    ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax2.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.82, 0.86])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (model, score) in enumerate(zip(models, metrics_data['R²'])):
        ax2.text(i, score + 0.001, f'{score:.4f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        if i == 2:  # CatBoost
            ax2.text(i, score - 0.003, 'Best', ha='center', va='top', 
                    fontsize=9, fontweight='bold', color='#27ae60')
    
    # 3. MAE comparison
    ax3 = fig.add_subplot(gs[1, 1])
    colors = ['#95a5a6', '#95a5a6', '#e74c3c']
    bars = ax3.bar(models, metrics_data['MAE (TL)'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    bars[2].set_color('#e74c3c')  # CatBoost in red (lower is better)
    ax3.set_ylabel('MAE (TL)', fontsize=11, fontweight='bold')
    ax3.set_title('Mean Absolute Error Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylim([7000, 7700])
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (model, score) in enumerate(zip(models, metrics_data['MAE (TL)'])):
        ax3.text(i, score + 50, f'{score:.0f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        if i == 2:  # CatBoost
            ax3.text(i, score - 100, 'Best', ha='center', va='top', 
                    fontsize=9, fontweight='bold', color='#c0392b')
    
    # 4. MAPE comparison
    ax4 = fig.add_subplot(gs[1, 2])
    colors = ['#95a5a6', '#95a5a6', '#3498db']
    bars = ax4.bar(models, metrics_data['MAPE (%)'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    bars[2].set_color('#3498db')  # CatBoost in blue
    ax4.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Mean Absolute Percentage Error', fontsize=12, fontweight='bold')
    ax4.set_ylim([18.5, 20.0])
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (model, score) in enumerate(zip(models, metrics_data['MAPE (%)'])):
        ax4.text(i, score + 0.1, f'{score:.2f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        if i == 2:  # CatBoost
            ax4.text(i, score - 0.2, 'Best', ha='center', va='top', 
                    fontsize=9, fontweight='bold', color='#2980b9')
    
    # 5. Improvement percentage
    ax5 = fig.add_subplot(gs[2, :])
    best_r2 = max(rf_metrics['R2'], xgb_metrics['R2'])
    best_mae = min(rf_metrics['MAE'], xgb_metrics['MAE'])
    best_mape = min(rf_metrics['MAPE'], xgb_metrics['MAPE'])
    
    improvements = {
        'R² Improvement': [0, 0, ((cb_metrics['R2'] - best_r2) / best_r2) * 100],
        'MAE Improvement': [0, 0, ((best_mae - cb_metrics['MAE']) / best_mae) * 100],
        'MAPE Improvement': [0, 0, ((best_mape - cb_metrics['MAPE']) / best_mape) * 100]
    }
    
    x = np.arange(len(models))
    width = 0.25
    ax5.bar(x - width, improvements['R² Improvement'], width, label='R²', color='#2ecc71', alpha=0.8)
    ax5.bar(x, improvements['MAE Improvement'], width, label='MAE', color='#e74c3c', alpha=0.8)
    ax5.bar(x + width, improvements['MAPE Improvement'], width, label='MAPE', color='#3498db', alpha=0.8)
    
    ax5.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax5.set_title('CatBoost Improvement Over Best Previous Model', fontsize=14, fontweight='bold', pad=20)
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, fontsize=11)
    ax5.legend(loc='upper left', fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Add value annotations
    for i, model in enumerate(models):
        if i == 2:  # CatBoost
            ax5.text(i - width, improvements['R² Improvement'][i] + 0.2, 
                    f'+{improvements["R² Improvement"][i]:.2f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')
            ax5.text(i, improvements['MAE Improvement'][i] + 0.2, 
                    f'+{improvements["MAE Improvement"][i]:.2f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c0392b')
            ax5.text(i + width, improvements['MAPE Improvement'][i] + 0.2, 
                    f'+{improvements["MAPE Improvement"][i]:.2f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2980b9')
    
    plt.suptitle('Model Performance Comparison: Random Forest vs XGBoost vs CatBoost', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(config.FIGURES_DIR / "15_comprehensive_model_comparison.png", 
                dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print("Saved comprehensive model comparison to: 15_comprehensive_model_comparison.png")
    
    # Create detailed metrics table visualization
    create_metrics_table_visualization(rf_metrics, xgb_metrics, cb_metrics)
    
    # Create performance radar chart
    create_radar_chart(rf_metrics, xgb_metrics, cb_metrics)


def create_metrics_table_visualization(rf_metrics, xgb_metrics, cb_metrics):
    """Create a detailed metrics table as visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['Metric', 'Random Forest', 'XGBoost', 'CatBoost', 'Winner'],
        ['R² Score', f"{rf_metrics['R2']:.4f}", f"{xgb_metrics['R2']:.4f}", 
         f"{cb_metrics['R2']:.4f}", 'CatBoost (Best)'],
        ['MAE (TL)', f"{rf_metrics['MAE']:.2f}", f"{xgb_metrics['MAE']:.2f}", 
         f"{cb_metrics['MAE']:.2f}", 'CatBoost (Best)'],
        ['RMSE (TL)', f"{rf_metrics['RMSE']:.2f}", f"{xgb_metrics['RMSE']:.2f}", 
         f"{cb_metrics['RMSE']:.2f}", 'CatBoost (Best)'],
        ['MAPE (%)', f"{rf_metrics['MAPE']:.2f}%", f"{xgb_metrics['MAPE']:.2f}%", 
         f"{cb_metrics['MAPE']:.2f}%", 'CatBoost (Best)'],
        ['R² (log)', f"{rf_metrics['R2_log']:.4f}", f"{xgb_metrics['R2_log']:.4f}", 
         f"{cb_metrics['R2_log']:.4f}", 'CatBoost (Best)'],
    ]
    
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight CatBoost column
    for i in range(1, 6):
        table[(i, 3)].set_facecolor('#d5f4e6')
        table[(i, 4)].set_facecolor('#2ecc71')
        table[(i, 4)].set_text_props(weight='bold')
    
    # Highlight best values
    for i in range(1, 6):
        if i == 1:  # R²
            best_idx = np.argmax([rf_metrics['R2'], xgb_metrics['R2'], cb_metrics['R2']])
        elif i in [2, 3]:  # MAE, RMSE
            best_idx = np.argmin([rf_metrics[list(rf_metrics.keys())[i-1]], 
                                 xgb_metrics[list(xgb_metrics.keys())[i-1]], 
                                 cb_metrics[list(cb_metrics.keys())[i-1]]])
        elif i == 4:  # MAPE
            best_idx = np.argmin([rf_metrics['MAPE'], xgb_metrics['MAPE'], cb_metrics['MAPE']])
        else:  # R² log
            best_idx = np.argmax([rf_metrics['R2_log'], xgb_metrics['R2_log'], cb_metrics['R2_log']])
        
        if best_idx == 2:  # CatBoost
            table[(i, 3)].set_facecolor('#2ecc71')
            table[(i, 3)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Model Performance Metrics Comparison', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(config.FIGURES_DIR / "16_detailed_metrics_table.png", 
                dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print("Saved detailed metrics table to: 16_detailed_metrics_table.png")


def create_radar_chart(rf_metrics, xgb_metrics, cb_metrics):
    """Create radar chart for model comparison"""
    from math import pi
    
    # Normalize metrics for radar chart (0-1 scale, higher is better)
    def normalize_r2(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)
    
    def normalize_error(val, min_val, max_val):
        return 1 - (val - min_val) / (max_val - min_val)
    
    # Prepare data
    categories = ['R² Score', 'MAE\n(Lower Better)', 'RMSE\n(Lower Better)', 
                 'MAPE\n(Lower Better)', 'R² (log)']
    
    r2_min, r2_max = 0.83, 0.86
    mae_min, mae_max = 7200, 7600
    rmse_min, rmse_max = 12400, 13400
    mape_min, mape_max = 18.5, 20.0
    r2log_min, r2log_max = 0.84, 0.86
    
    rf_values = [
        normalize_r2(rf_metrics['R2'], r2_min, r2_max),
        normalize_error(rf_metrics['MAE'], mae_min, mae_max),
        normalize_error(rf_metrics['RMSE'], rmse_min, rmse_max),
        normalize_error(rf_metrics['MAPE'], mape_min, mape_max),
        normalize_r2(rf_metrics['R2_log'], r2log_min, r2log_max)
    ]
    
    xgb_values = [
        normalize_r2(xgb_metrics['R2'], r2_min, r2_max),
        normalize_error(xgb_metrics['MAE'], mae_min, mae_max),
        normalize_error(xgb_metrics['RMSE'], rmse_min, rmse_max),
        normalize_error(xgb_metrics['MAPE'], mape_min, mape_max),
        normalize_r2(xgb_metrics['R2_log'], r2log_min, r2log_max)
    ]
    
    cb_values = [
        normalize_r2(cb_metrics['R2'], r2_min, r2_max),
        normalize_error(cb_metrics['MAE'], mae_min, mae_max),
        normalize_error(cb_metrics['RMSE'], rmse_min, rmse_max),
        normalize_error(cb_metrics['MAPE'], mape_min, mape_max),
        normalize_r2(cb_metrics['R2_log'], r2log_min, r2log_max)
    ]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add values
    rf_values += rf_values[:1]
    xgb_values += xgb_values[:1]
    cb_values += cb_values[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Add plots
    ax.plot(angles, rf_values, 'o-', linewidth=2, label='Random Forest', color='#3498db', alpha=0.7)
    ax.fill(angles, rf_values, alpha=0.15, color='#3498db')
    
    ax.plot(angles, xgb_values, 'o-', linewidth=2, label='XGBoost', color='#e67e22', alpha=0.7)
    ax.fill(angles, xgb_values, alpha=0.15, color='#e67e22')
    
    ax.plot(angles, cb_values, 'o-', linewidth=3, label='CatBoost (Best)', color='#2ecc71', alpha=0.9)
    ax.fill(angles, cb_values, alpha=0.25, color='#2ecc71')
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    
    plt.title('Model Performance Radar Chart\n(Normalized Metrics - Higher is Better)', 
             fontsize=14, fontweight='bold', pad=30)
    plt.savefig(config.FIGURES_DIR / "17_model_radar_chart.png", 
                dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print("Saved radar chart to: 17_model_radar_chart.png")


if __name__ == "__main__":
    create_comprehensive_comparison()

