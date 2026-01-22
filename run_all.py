"""Main script to run the entire pipeline"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import project modules
import src.config as config
from src.io import load_data, save_dataframe
from src.preprocessing import (
    map_column_names, analyze_missing_values, clean_data,
    prepare_target, prepare_categorical_features, split_features_target
)
from src.features import (
    create_derived_features, prepare_model_features,
    get_categorical_columns, get_numeric_columns
)
from src.eda import generate_eda_report
from src.volatility import generate_volatility_report
from src.clustering import generate_clustering_report
from src.modeling import train_models, create_preprocessing_pipeline
from src.tuning import tune_models
from src.evaluation import evaluate_models
from src.interpretability import analyze_interpretability
from src.anomaly import generate_anomaly_report
from src.pipeline_diagram import generate_pipeline_diagrams
from src.report import generate_analysis_md, save_analysis_report


def main():
    """Main pipeline execution"""
    print("=" * 60)
    print("Istanbul Rent Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        print("\n[1/12] Loading data...")
        df = load_data()
        print(f"Initial data shape: {df.shape}")
        
        # Step 2: Map column names
        print("\n[2/12] Mapping column names...")
        df = map_column_names(df)
        
        # Step 3: Preprocessing
        print("\n[3/12] Preprocessing data...")
        df = clean_data(df)
        df, use_log = prepare_target(df)
        df = prepare_categorical_features(df)
        
        # Step 4: Feature Engineering (EDA features)
        print("\n[4/12] Feature engineering (EDA)...")
        df_eda = create_derived_features(df, for_model=False)
        
        # Step 5: EDA
        print("\n[5/12] Generating EDA visualizations...")
        generate_eda_report(df_eda)
        
        # Step 6: Volatility Analysis
        print("\n[6/12] Volatility analysis...")
        volatility_df = generate_volatility_report(df_eda)
        
        # Step 7: Clustering
        print("\n[7/12] Clustering analysis...")
        neighborhood_features, cluster_summary = generate_clustering_report(df_eda)
        
        # Step 8: Prepare features for modeling
        print("\n[8/12] Preparing features for modeling...")
        df_model = prepare_model_features(df.copy())
        
        # Get feature columns
        target_col = 'price_log' if use_log else 'price'
        X, y = split_features_target(df_model, target_col=target_col)
        
        categorical_cols = get_categorical_columns(X)
        numeric_cols = get_numeric_columns(X)
        
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"Numeric features: {len(numeric_cols)}")
        
        # Handle high cardinality: frequency threshold for neighborhood
        if 'neighborhood' in categorical_cols:
            neighborhood_counts = X['neighborhood'].value_counts()
            threshold = config.PREPROCESSING['neighborhood_freq_threshold']
            frequent_neighborhoods = neighborhood_counts[neighborhood_counts >= threshold].index
            X['neighborhood'] = X['neighborhood'].apply(
                lambda x: x if x in frequent_neighborhoods else 'Other'
            )
            print(f"Neighborhood encoding: {len(frequent_neighborhoods)} frequent + 'Other'")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.MODEL['test_size'],
            random_state=config.MODEL['random_state']
        )
        print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Get test indices for anomaly detection (mapped to original df_eda indices)
        # X_test.index contains the original indices from X, which match df_eda
        test_indices = X_test.index.values
        
        # Step 9: Train models
        print("\n[9/12] Training models...")
        models = train_models(X_train, y_train, X_test, y_test,
                             categorical_cols, numeric_cols)
        
        # Get preprocessor from trained model (it's already fitted)
        if models:
            # Get preprocessor from first model's pipeline
            first_model = list(models.values())[0]
            preprocessor = first_model.named_steps['preprocessor']
        else:
            # Fallback: create new preprocessor
            preprocessor = create_preprocessing_pipeline(categorical_cols, numeric_cols)
            preprocessor.fit(X_train)
        
        # Step 10: Hyperparameter tuning (for best model)
        print("\n[10/12] Hyperparameter tuning...")
        best_model_name = 'XGBoost' if 'XGBoost' in models else list(models.keys())[0]
        try:
            tuning_results = tune_models(X_train, y_train, preprocessor, model_name=best_model_name)
        except Exception as e:
            print(f"Tuning failed: {e}")
            tuning_results = None
        
        # Step 11: Evaluate models
        print("\n[11/12] Evaluating models...")
        metrics_dict, predictions_dict = evaluate_models(
            models, X_test, y_test, use_log=use_log
        )
        
        # Step 12: Interpretability
        print("\n[12/12] Interpretability analysis...")
        try:
            interpretability_results = analyze_interpretability(
                models, X_test, y_test, preprocessor, list(X.columns)
            )
        except Exception as e:
            print(f"Interpretability analysis failed: {e}")
            interpretability_results = None
        
        # Step 13: Anomaly detection
        print("\n[13/14] Anomaly detection...")
        y_test_orig = np.expm1(y_test) if use_log else y_test
        y_pred_best = predictions_dict.get(best_model_name)
        if y_pred_best is not None:
            y_pred_orig = np.expm1(y_pred_best) if use_log else y_pred_best
        else:
            y_pred_orig = None
        
        anomaly_flags, df_anomalies = generate_anomaly_report(
            df_eda, 
            y_test_orig.values if y_pred_orig is not None else None,
            y_pred_orig if y_pred_orig is not None else None,
            test_indices=test_indices if 'test_indices' in locals() else None
        )
        anomaly_count = len(df_anomalies) if df_anomalies is not None and len(df_anomalies) > 0 else 0
        
        # Step 14: Generate pipeline diagrams
        print("\n[14/14] Generating pipeline diagrams...")
        generate_pipeline_diagrams()
        
        # Step 15: Generate report
        print("\n[15/15] Generating analysis report...")
        report_content = generate_analysis_md(
            metrics_dict,
            volatility_df=volatility_df,
            cluster_summary=cluster_summary,
            anomaly_count=anomaly_count,
            use_log=use_log
        )
        save_analysis_report(report_content)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
        print(f"Analysis report: {config.REPORTS_DIR / 'analysis.md'}")
        print("\nSummary:")
        for model_name, metrics in metrics_dict.items():
            print(f"  {model_name}: R² = {metrics['R2']:.3f}, MAE = {metrics['MAE']:.2f} TL")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

