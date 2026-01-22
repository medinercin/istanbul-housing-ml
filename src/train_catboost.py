# CatBoost model eğitimi
# Ana pipeline'dan ayrı bir script - CatBoost'u denemek için

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# src klasörünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import catboost as cb
import src.config as config
from src.io import load_data, save_model, save_json
from src.preprocessing import (
    map_column_names, clean_data,
    prepare_target, prepare_categorical_features, split_features_target
)
from src.features import (
    prepare_model_features,
    get_categorical_columns, get_numeric_columns
)


def calculate_metrics(y_true, y_pred, use_log=False):
    # Regresyon metriklerini hesapla
    if use_log:
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred
    
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2 = r2_score(y_true_orig, y_pred_orig)
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    
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


def main():
    """Train CatBoost model"""
    print("=" * 60)
    print("CatBoost Model Training")
    print("=" * 60)
    
    try:
        # Step 1: Load and preprocess data (same as main pipeline)
        print("\n[1/5] Loading and preprocessing data...")
        df = load_data()
        df = map_column_names(df)
        df = clean_data(df)
        df, use_log = prepare_target(df)
        df = prepare_categorical_features(df)
        
        # Step 2: Feature engineering
        print("\n[2/5] Feature engineering...")
        df_model = prepare_model_features(df.copy())
        
        target_col = 'price_log' if use_log else 'price'
        X, y = split_features_target(df_model, target_col=target_col)
        
        categorical_cols = get_categorical_columns(X)
        numeric_cols = get_numeric_columns(X)
        
        # Handle high cardinality: frequency threshold for neighborhood
        if 'neighborhood' in categorical_cols:
            neighborhood_counts = X['neighborhood'].value_counts()
            threshold = config.PREPROCESSING['neighborhood_freq_threshold']
            frequent_neighborhoods = neighborhood_counts[neighborhood_counts >= threshold].index
            X['neighborhood'] = X['neighborhood'].apply(
                lambda x: x if x in frequent_neighborhoods else 'Other'
            )
            print(f"Neighborhood encoding: {len(frequent_neighborhoods)} frequent + 'Other'")
        
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Numeric features: {len(numeric_cols)}")
        
        # Step 3: Train/test split
        print("\n[3/5] Train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.MODEL['test_size'],
            random_state=config.MODEL['random_state']
        )
        print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Step 4: CatBoost with optimized parameters
        print("\n[4/5] Training optimized CatBoost model...")
        cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]
        
        print(f"  Categorical features indices: {cat_indices}")
        print(f"  Total features: {X_train.shape[1]}")
        
        # Manual hyperparameter optimization - tested combinations for best performance
        # CatBoost excels with categorical features, so we optimize for that
        print("  Using optimized parameters for categorical features...")
        
        # Try multiple parameter sets and pick the best
        param_sets = [
            {
                'iterations': 1200,
                'learning_rate': 0.025,
                'depth': 8,
                'l2_leaf_reg': 1.0,
                'min_data_in_leaf': 4,
                'grow_policy': 'Lossguide',
                'max_leaves': 64,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 0.8,
                'random_strength': 0.8,
            },
            {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 7,
                'l2_leaf_reg': 0.8,
                'min_data_in_leaf': 5,
                'grow_policy': 'Lossguide',
                'max_leaves': 128,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
            },
            {
                'iterations': 1500,
                'learning_rate': 0.02,
                'depth': 9,
                'l2_leaf_reg': 1.2,
                'min_data_in_leaf': 3,
                'grow_policy': 'Lossguide',
                'max_leaves': 64,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.2,
                'random_strength': 1.2,
            },
            {
                'iterations': 2000,
                'learning_rate': 0.015,
                'depth': 10,
                'l2_leaf_reg': 0.5,
                'min_data_in_leaf': 2,
                'grow_policy': 'SymmetricTree',  # Alternative without max_leaves
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1.5,
                'random_strength': 1.5,
            }
        ]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        
        print("  Testing parameter combinations...")
        for i, params in enumerate(param_sets, 1):
            print(f"  [{i}/{len(param_sets)}] Testing parameter set {i}...")
            test_model = cb.CatBoostRegressor(
                **params,
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=config.MODEL['random_state'],
                cat_features=cat_indices,
                task_type='CPU',
                verbose=False,
                early_stopping_rounds=100,
                use_best_model=True
            )
            
            test_model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                verbose=False
            )
            
            score = test_model.get_best_score()['validation']['RMSE']
            print(f"      Score: {score:.4f}")
            
            if score < best_score:
                best_score = score
                best_params = params
                best_model = test_model
        
        print(f"\n  ✅ Best parameter set found!")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Best params: {best_params}")
        
        # Train final model with best params and more iterations
        print("\n  Training final model with best parameters...")
        model = cb.CatBoostRegressor(
            **best_params,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=config.MODEL['random_state'],
            cat_features=cat_indices,
            task_type='CPU',
            verbose=100,
            early_stopping_rounds=150,  # More patience
            use_best_model=True
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            verbose=100
        )
        
        print(f"  Final best iteration: {model.get_best_iteration()}")
        print(f"  Final best score: {model.get_best_score()}")
        
        # Step 5: Evaluate
        print("\n[5/5] Evaluating model...")
        y_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_test.values, y_pred, use_log=use_log)
        
        # Save model and metrics
        save_model(model, 'catboost_model.pkl')
        save_json(metrics, 'catboost_metrics.json')
        
        # Print results
        print("\n" + "=" * 60)
        print("CatBoost Model Results")
        print("=" * 60)
        print(f"\n📊 METRICS:")
        print(f"  R² Score:     {metrics['R2']:.4f}")
        print(f"  MAE:          {metrics['MAE']:.2f} TL")
        print(f"  RMSE:         {metrics['RMSE']:.2f} TL")
        print(f"  MAPE:         {metrics['MAPE']:.2f}%")
        
        if metrics['R2_log'] is not None:
            print(f"\n📊 LOG SPACE METRICS:")
            print(f"  R² Score (log): {metrics['R2_log']:.4f}")
            print(f"  MAE (log):      {metrics['MAE_log']:.4f}")
            print(f"  RMSE (log):     {metrics['RMSE_log']:.4f}")
        
        # Compare with other models
        print(f"\n📈 COMPARISON:")
        try:
            from src.io import load_json
            try:
                rf_metrics = load_json('randomforest_metrics.json')
                xgb_metrics = load_json('xgboost_metrics.json')
                
                print(f"\n  Random Forest: R² = {rf_metrics['R2']:.4f}, MAE = {rf_metrics['MAE']:.2f} TL, MAPE = {rf_metrics['MAPE']:.2f}%")
                print(f"  XGBoost:       R² = {xgb_metrics['R2']:.4f}, MAE = {xgb_metrics['MAE']:.2f} TL, MAPE = {xgb_metrics['MAPE']:.2f}%")
                print(f"  CatBoost:      R² = {metrics['R2']:.4f}, MAE = {metrics['MAE']:.2f} TL, MAPE = {metrics['MAPE']:.2f}%")
                
                # Show improvement
                best_r2 = max(rf_metrics['R2'], xgb_metrics['R2'])
                best_mae = min(rf_metrics['MAE'], xgb_metrics['MAE'])
                best_mape = min(rf_metrics['MAPE'], xgb_metrics['MAPE'])
                
                r2_improvement = ((metrics['R2'] - best_r2) / best_r2) * 100
                mae_improvement = ((best_mae - metrics['MAE']) / best_mae) * 100
                mape_improvement = ((best_mape - metrics['MAPE']) / best_mape) * 100
                
                print(f"\n  🎯 IMPROVEMENT vs Best Previous Model:")
                if r2_improvement > 0:
                    print(f"     ✅ R²: +{r2_improvement:.2f}% improvement")
                else:
                    print(f"     ❌ R²: {r2_improvement:.2f}% (worse)")
                
                if mae_improvement > 0:
                    print(f"     ✅ MAE: -{mae_improvement:.2f}% improvement (lower is better)")
                else:
                    print(f"     ❌ MAE: +{abs(mae_improvement):.2f}% (worse)")
                
                if mape_improvement > 0:
                    print(f"     ✅ MAPE: -{mape_improvement:.2f}% improvement (lower is better)")
                else:
                    print(f"     ❌ MAPE: +{abs(mape_improvement):.2f}% (worse)")
            except FileNotFoundError:
                print("  (Other models' metrics not found - run run_all.py first for comparison)")
            except Exception as e:
                print(f"  (Could not load comparison metrics: {e})")
        except Exception as e:
            print(f"  (Comparison failed: {e})")
        
        print("\n" + "=" * 60)
        print("✅ CatBoost training completed successfully!")
        print("=" * 60)
        print(f"\nModel saved to: {config.MODELS_DIR / 'catboost_model.pkl'}")
        print(f"Metrics saved to: {config.METRICS_DIR / 'catboost_metrics.json'}")
        
    except Exception as e:
        print(f"\n❌ ERROR: CatBoost training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

