"""Hyperparameter tuning"""

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import xgboost as xgb
from scipy.stats import randint, uniform
import src.config as config
from src.io import save_json


def tune_random_forest(X_train, y_train, preprocessor, n_iter: int = 20, cv: int = 3):
    """Tune Random Forest hyperparameters"""
    print("Tuning Random Forest...")
    
    param_distributions = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': randint(2, 20),
        'model__min_samples_leaf': randint(1, 10)
    }
    
    base_model = RandomForestRegressor(random_state=config.MODEL['random_state'], n_jobs=-1)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=config.MODEL['random_state'],
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    results = {
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'best_model': random_search.best_estimator_
    }
    
    # Save results
    save_json({
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_)
    }, 'random_forest_tuning_results.json')
    
    print(f"Best Random Forest score: {random_search.best_score_:.4f}")
    print(f"Best parameters: {random_search.best_params_}")
    
    return results


def tune_xgboost(X_train, y_train, preprocessor, n_iter: int = 20, cv: int = 3):
    """Tune XGBoost hyperparameters"""
    print("Tuning XGBoost...")
    
    param_distributions = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': randint(3, 10),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__subsample': uniform(0.6, 0.4),
        'model__colsample_bytree': uniform(0.6, 0.4)
    }
    
    base_model = xgb.XGBRegressor(random_state=config.MODEL['random_state'], n_jobs=-1)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=config.MODEL['random_state'],
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    results = {
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_),
        'best_model': random_search.best_estimator_
    }
    
    # Save results
    save_json({
        'best_params': random_search.best_params_,
        'best_score': float(random_search.best_score_)
    }, 'xgboost_tuning_results.json')
    
    print(f"Best XGBoost score: {random_search.best_score_:.4f}")
    print(f"Best parameters: {random_search.best_params_}")
    
    return results


def tune_models(X_train, y_train, preprocessor, model_name: str = 'XGBoost'):
    """Tune hyperparameters for specified model"""
    print(f"\n=== Hyperparameter Tuning for {model_name} ===")
    
    if model_name == 'RandomForest':
        return tune_random_forest(X_train, y_train, preprocessor)
    elif model_name == 'XGBoost':
        return tune_xgboost(X_train, y_train, preprocessor)
    else:
        print(f"Unknown model: {model_name}")
        return None

