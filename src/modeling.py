# Model eğitimi ve pipeline fonksiyonları
# Random Forest ve XGBoost modellerini eğitiyoruz

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import src.config as config
from src.io import save_model


def create_preprocessing_pipeline(categorical_cols: list, numeric_cols: list):
    # Preprocessing pipeline oluştur
    # Numeric: StandardScaler, Categorical: OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),  # sayısal özellikleri normalize et
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)  # kategorik özellikleri encode et
        ],
        remainder='passthrough'
    )
    return preprocessor


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, 
                       preprocessor: ColumnTransformer,
                       n_estimators: int = 100,
                       max_depth: int = None,
                       random_state: int = None) -> Pipeline:
    # Random Forest modelini eğit
    if random_state is None:
        random_state = config.MODEL['random_state']
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,  # ağaç sayısı
        max_depth=max_depth,  # maksimum derinlik
        random_state=random_state,
        n_jobs=-1  # tüm CPU'ları kullan
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 preprocessor: ColumnTransformer,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = None) -> Pipeline:
    # XGBoost modelini eğit
    if random_state is None:
        random_state = config.MODEL['random_state']
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,  # boosting round sayısı
        max_depth=max_depth,  # ağaç derinliği
        learning_rate=learning_rate,  # öğrenme oranı
        random_state=random_state,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'neg_mean_absolute_error') -> dict:
    # Cross-validation ile modeli değerlendir
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series,
                X_test: pd.DataFrame, y_test: pd.Series,
                categorical_cols: list, numeric_cols: list) -> dict:
    # Birden fazla modeli eğit
    print("\n=== Training Models ===")
    
    # Preprocessing pipeline oluştur
    preprocessor = create_preprocessing_pipeline(categorical_cols, numeric_cols)
    
    models = {}
    
    # Random Forest eğit
    if 'RandomForest' in config.MODEL['models_to_train']:
        print("Training Random Forest...")
        rf_pipeline = train_random_forest(X_train, y_train, preprocessor)
        models['RandomForest'] = rf_pipeline
        save_model(rf_pipeline, 'random_forest_model.pkl')
        print("Random Forest trained and saved")
    
    # XGBoost eğit
    if 'XGBoost' in config.MODEL['models_to_train']:
        print("Training XGBoost...")
        xgb_pipeline = train_xgboost(X_train, y_train, preprocessor)
        models['XGBoost'] = xgb_pipeline
        save_model(xgb_pipeline, 'xgboost_model.pkl')
        print("XGBoost trained and saved")
    
    print("=== Model Training Complete ===\n")
    return models

