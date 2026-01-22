"""Model evaluation and metrics"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import src.config as config
from src.io import save_json, save_dataframe


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     use_log: bool = False) -> dict:
    """Calculate regression metrics"""
    # If log transformed, convert back
    if use_log:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    
    # Log space metrics (if applicable)
    if use_log:
        mae_log = mean_absolute_error(y_true, y_pred)
        rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
        r2_log = r2_score(y_true, y_pred)
    else:
        mae_log = None
        rmse_log = None
        r2_log = None
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape),
        'MAE_log': float(mae_log) if mae_log is not None else None,
        'RMSE_log': float(rmse_log) if rmse_log is not None else None,
        'R2_log': float(r2_log) if r2_log is not None else None
    }


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    model_name: str, save_path, use_log: bool = False):
    """Plot actual vs predicted values"""
    if use_log:
        y_true_plot = np.expm1(y_true)
        y_pred_plot = np.expm1(y_pred)
        ylabel = 'Price (TL) - Original Scale'
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        ylabel = 'Price (TL)'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Actual vs Predicted scatter
    axes[0, 0].scatter(y_true_plot, y_pred_plot, alpha=0.5)
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price (TL)')
    axes[0, 0].set_ylabel('Predicted Price (TL)')
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true_plot - y_pred_plot
    axes[0, 1].scatter(y_pred_plot, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price (TL)')
    axes[0, 1].set_ylabel('Residuals (TL)')
    axes[0, 1].set_title(f'{model_name}: Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1, 0].hist(residuals, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals (TL)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{model_name}: Residual Distribution')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    
    # Error distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=50, edgecolor='black', color='orange')
    axes[1, 1].set_xlabel('Absolute Error (TL)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'{model_name}: Absolute Error Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plots to: {save_path}")


def compare_models(metrics_dict: dict, save_path):
    """Compare multiple models (includes CatBoost if available)"""
    from src.io import load_json
    
    # Try to load CatBoost metrics if not in dict
    models_list = list(metrics_dict.keys())
    if 'CatBoost' not in models_list:
        try:
            cb_metrics = load_json('catboost_metrics.json')
            metrics_dict['CatBoost'] = cb_metrics
            print("CatBoost metrics loaded for comparison")
        except:
            pass  # CatBoost metrics not available
    
    models = list(metrics_dict.keys())
    # Ensure order: Random Forest, XGBoost, CatBoost
    preferred_order = ['RandomForest', 'Random Forest', 'XGBoost', 'CatBoost']
    models_sorted = []
    for pref in preferred_order:
        for model in models:
            if pref.lower() in model.lower() and model not in models_sorted:
                models_sorted.append(model)
    # Add any remaining models
    for model in models:
        if model not in models_sorted:
            models_sorted.append(model)
    models = models_sorted
    
    metric_names = ['MAE', 'RMSE', 'R2', 'MAPE']
    colors = {'RandomForest': '#3498db', 'Random Forest': '#3498db', 
              'XGBoost': '#e67e22', 'CatBoost': '#2ecc71'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        bar_colors = [colors.get(model, '#95a5a6') for model in models]
        bars = axes[idx].bar(models, values, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Highlight CatBoost if present
        for i, model in enumerate(models):
            if 'CatBoost' in model or 'catboost' in model.lower():
                bars[i].set_color('#2ecc71')
                bars[i].set_alpha(1.0)
                bars[i].set_linewidth(2.5)
        
        axes[idx].set_ylabel(metric, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Model Comparison: {metric}', fontsize=12, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=15 if len(models) > 2 else 0)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels with better formatting
        for i, (model, v) in enumerate(zip(models, values)):
            if metric == 'R2':
                label = f'{v:.4f}'
            elif metric == 'MAPE':
                label = f'{v:.2f}%'
            else:
                label = f'{v:.0f}' if v >= 1000 else f'{v:.2f}'
            
            axes[idx].text(i, v, label, ha='center', va='bottom', 
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.suptitle('Model Performance Comparison: Random Forest vs XGBoost vs CatBoost', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to: {save_path}")


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                   use_log: bool = False) -> dict:
    """Evaluate all models"""
    print("\n=== Evaluating Models ===")
    
    all_metrics = {}
    all_predictions = {}
    
    for model_name, pipeline in models.items():
        print(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        all_predictions[model_name] = y_pred
        
        # Metrics
        metrics = calculate_metrics(y_test.values, y_pred, use_log=use_log)
        all_metrics[model_name] = metrics
        
        # Save metrics
        save_json(metrics, f'{model_name.lower()}_metrics.json')
        
        # Plot
        plot_predictions(y_test.values, y_pred, model_name,
                        config.FIGURES_DIR / f"12_{model_name.lower()}_evaluation.png",
                        use_log=use_log)
        
        print(f"{model_name} - MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, R2: {metrics['R2']:.3f}")
    
    # Compare models
    compare_models(all_metrics, config.FIGURES_DIR / "13_model_comparison.png")
    
    # Save comparison table
    comparison_df = pd.DataFrame(all_metrics).T
    save_dataframe(comparison_df, "model_comparison.csv")
    
    print("=== Model Evaluation Complete ===\n")
    return all_metrics, all_predictions

