"""Model interpretability analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import src.config as config
from src.io import save_json, save_dataframe


def calculate_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series,
                                    n_repeats: int = 10, random_state: int = None) -> dict:
    """Calculate permutation importance"""
    if random_state is None:
        random_state = config.MODEL['random_state']
    
    print("Calculating permutation importance...")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    
    return result


def get_feature_names(preprocessor, original_features: list) -> list:
    """Get feature names after preprocessing"""
    try:
        # Get numeric feature names
        numeric_transformer = preprocessor.named_transformers_['num']
        if hasattr(numeric_transformer, 'get_feature_names_out'):
            numeric_features = numeric_transformer.get_feature_names_out()
        else:
            # StandardScaler doesn't have get_feature_names_out in older versions
            numeric_cols = [col for col in original_features if col not in preprocessor.named_transformers_['cat'].feature_names_in_]
            numeric_features = numeric_cols
        
        if isinstance(numeric_features, np.ndarray):
            numeric_features = numeric_features.tolist()
        
        # Get categorical feature names
        cat_transformer = preprocessor.named_transformers_['cat']
        if hasattr(cat_transformer, 'get_feature_names_out'):
            cat_features = cat_transformer.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            cat_cols = [col for col in original_features if col in preprocessor.named_transformers_['cat'].feature_names_in_]
            # Estimate number of categories (rough approximation)
            cat_features = [f'{col}_cat_{i}' for col in cat_cols for i in range(10)]  # Rough estimate
        
        if isinstance(cat_features, np.ndarray):
            cat_features = cat_features.tolist()
        
        # Combine
        all_features = list(numeric_features) + list(cat_features)
        return all_features
    except Exception as e:
        # Fallback: use indices based on transformed shape
        try:
            # Try to get shape from a sample transform
            n_features = len(original_features) * 2  # Rough estimate
            return [f'feature_{i}' for i in range(n_features)]
        except:
            return [f'feature_{i}' for i in range(100)]  # Default fallback


def plot_feature_importance(importance_result, feature_names: list, 
                           model_name: str, save_path, top_n: int = 20):
    """Plot feature importance"""
    importances = importance_result.importances_mean
    std = importance_result.importances_std
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=config.VISUALIZATION['figsize'])
    
    ax.barh(range(len(indices)), importances[indices], xerr=std[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Permutation Importance')
    ax.set_title(f'{model_name}: Top {top_n} Feature Importance')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to: {save_path}")


def generate_importance_table(importance_result, feature_names: list) -> pd.DataFrame:
    """Generate feature importance table"""
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importance_result.importances_mean,
        'importance_std': importance_result.importances_std
    })
    df = df.sort_values('importance_mean', ascending=False)
    return df


def analyze_interpretability(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                             preprocessor, original_features: list):
    """Analyze interpretability for all models"""
    print("\n=== Interpretability Analysis ===")
    
    # Get feature names - try to infer from preprocessor
    try:
        # Fit preprocessor if not already fitted (should be fitted from training)
        if not hasattr(preprocessor, 'transformers_'):
            # Create a dummy fit
            from sklearn.compose import ColumnTransformer
            dummy_X = X_test.head(1)
            preprocessor.fit(dummy_X)
        
        # Get feature names after transformation
        feature_names = get_feature_names(preprocessor, original_features)
        
        # If we got generic names, try to get actual count
        if len(feature_names) < 10:  # Likely wrong
            # Transform a sample to get actual feature count
            sample_transformed = preprocessor.transform(X_test.head(1))
            n_features = sample_transformed.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
    except Exception as e:
        print(f"Warning: Could not get feature names, using indices: {e}")
        # Fallback: use number of features from transformed data
        try:
            sample_transformed = preprocessor.transform(X_test.head(1))
            n_features = sample_transformed.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
        except:
            feature_names = [f'feature_{i}' for i in range(100)]
    
    all_importances = {}
    
    for model_name, pipeline in models.items():
        print(f"Analyzing {model_name}...")
        
        try:
            # Calculate permutation importance
            importance_result = calculate_permutation_importance(
                pipeline, X_test, y_test, n_repeats=10
            )
            
            # Adjust feature names if count doesn't match
            if len(importance_result.importances_mean) != len(feature_names):
                feature_names = [f'feature_{i}' for i in range(len(importance_result.importances_mean))]
            
            # Plot
            plot_feature_importance(
                importance_result, feature_names, model_name,
                config.FIGURES_DIR / f"14_{model_name.lower()}_feature_importance.png",
                top_n=20
            )
            
            # Save table
            importance_df = generate_importance_table(importance_result, feature_names)
            save_dataframe(importance_df, f"{model_name.lower()}_feature_importance.csv")
            save_json(
                {
                    'top_10_features': importance_df.head(10).to_dict('records')
                },
                f"{model_name.lower()}_top_features.json"
            )
            
            all_importances[model_name] = importance_df
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue
    
    print("=== Interpretability Analysis Complete ===\n")
    return all_importances

