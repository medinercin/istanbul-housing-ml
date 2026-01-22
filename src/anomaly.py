# Kira ilanları için anomaly detection
# Anormal değerleri tespit ediyoruz

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import src.config as config
from src.io import save_dataframe


def detect_residual_anomalies(y_true: np.ndarray, y_pred: np.ndarray, 
                             threshold_percentile: float = 95) -> np.ndarray:
    # Tahmin residual'larına göre anomalileri tespit et
    residuals = np.abs(y_true - y_pred)
    threshold = np.percentile(residuals, threshold_percentile)
    anomalies = residuals > threshold
    return anomalies


def detect_zscore_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> np.ndarray:
    # Z-score kullanarak anomalileri tespit et (mahalle içinde)
    if 'neighborhood' not in df.columns or 'price' not in df.columns:
        return np.zeros(len(df), dtype=bool)
    
    anomalies = np.zeros(len(df), dtype=bool)
    
    for neighborhood in df['neighborhood'].unique():
        mask = df['neighborhood'] == neighborhood
        neighborhood_prices = df.loc[mask, 'price']
        
        if len(neighborhood_prices) > 1:
            mean_price = neighborhood_prices.mean()
            std_price = neighborhood_prices.std()
            
            if std_price > 0:
                z_scores = np.abs((neighborhood_prices - mean_price) / std_price)
                anomalies[mask] = z_scores > z_threshold
    
    return anomalies


def detect_isolation_forest_anomalies(df: pd.DataFrame, 
                                     contamination: float = 0.1,
                                     features: list = None) -> np.ndarray:
    # Isolation Forest kullanarak anomalileri tespit et
    if features is None:
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['price', 'price_log', 'price_per_m2']
        features = [col for col in numeric_cols if col not in exclude]
        
        # Include district as numeric if possible
        if 'district' in df.columns:
            # Use district as categorical, encode it
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_encoded = df.copy()
            df_encoded['district_encoded'] = le.fit_transform(df['district'])
            if 'district_encoded' not in features:
                features.append('district_encoded')
    
    # Filter features that exist
    features = [f for f in features if f in df.columns]
    
    if len(features) < 2:
        print("Not enough features for Isolation Forest")
        return np.zeros(len(df), dtype=bool)
    
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(contamination=contamination, random_state=config.MODEL['random_state'])
    predictions = iso_forest.fit_predict(X_scaled)
    anomalies = predictions == -1
    
    return anomalies


def plot_anomaly_detection(df: pd.DataFrame, anomaly_flags: dict, save_path):
    # Anomaly detection sonuçlarını görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Area vs Price scatter with anomalies
    if 'area' in df.columns and 'price' in df.columns:
        for method, anomalies in anomaly_flags.items():
            if method == 'residual':
                ax = axes[0, 0]
            elif method == 'zscore':
                ax = axes[0, 1]
            elif method == 'isolation_forest':
                ax = axes[1, 0]
            else:
                continue
            
            normal_mask = ~anomalies
            ax.scatter(df.loc[normal_mask, 'area'], df.loc[normal_mask, 'price'], 
                      alpha=0.3, s=20, label='Normal', color='blue')
            ax.scatter(df.loc[anomalies, 'area'], df.loc[anomalies, 'price'], 
                      alpha=0.7, s=50, label='Anomaly', color='red', marker='x')
            ax.set_xlabel('Area (m²)')
            ax.set_ylabel('Price (TL)')
            ax.set_title(f'Anomalies: {method.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Anomaly count comparison
    ax = axes[1, 1]
    methods = list(anomaly_flags.keys())
    counts = [anomaly_flags[m].sum() for m in methods]
    ax.bar(methods, counts, color=['red', 'orange', 'green'])
    ax.set_ylabel('Number of Anomalies')
    ax.set_title('Anomaly Count by Method')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved anomaly detection plot to: {save_path}")


def generate_anomaly_report(df: pd.DataFrame, y_true: np.ndarray = None, 
                            y_pred: np.ndarray = None, test_indices: np.ndarray = None):
    # Anomaly detection raporunu oluştur
    print("\n=== Generating Anomaly Detection ===")
    
    n_samples = len(df)
    anomaly_flags = {}
    
    # Residual-based anomalies
    if y_true is not None and y_pred is not None:
        residual_anomalies = detect_residual_anomalies(
            y_true, y_pred, 
            threshold_percentile=config.ANOMALY['residual_threshold_percentile']
        )
        # Create full-length array for residual anomalies
        if test_indices is not None:
            residual_full = np.zeros(n_samples, dtype=bool)
            residual_full[test_indices] = residual_anomalies
        else:
            # If no test_indices, assume y_true/y_pred match df length
            residual_full = residual_anomalies if len(residual_anomalies) == n_samples else np.zeros(n_samples, dtype=bool)
        anomaly_flags['residual'] = residual_full
        print(f"Residual-based anomalies: {residual_full.sum()}")
    
    # Z-score anomalies
    zscore_anomalies = detect_zscore_anomalies(df, z_threshold=config.ANOMALY['z_score_threshold'])
    anomaly_flags['zscore'] = zscore_anomalies
    print(f"Z-score anomalies: {zscore_anomalies.sum()}")
    
    # Isolation Forest anomalies
    iso_anomalies = detect_isolation_forest_anomalies(
        df, 
        contamination=config.ANOMALY['isolation_forest_contamination']
    )
    anomaly_flags['isolation_forest'] = iso_anomalies
    print(f"Isolation Forest anomalies: {iso_anomalies.sum()}")
    
    # Ensure all flags have the same length
    for method, flags in anomaly_flags.items():
        if len(flags) != n_samples:
            print(f"Warning: {method} anomalies length ({len(flags)}) doesn't match df length ({n_samples})")
            # Pad or truncate to match
            if len(flags) < n_samples:
                padded = np.zeros(n_samples, dtype=bool)
                padded[:len(flags)] = flags
                anomaly_flags[method] = padded
            else:
                anomaly_flags[method] = flags[:n_samples]
    
    # Combine all anomalies
    flag_arrays = [flags for flags in anomaly_flags.values()]
    if len(flag_arrays) > 0:
        all_anomalies = np.any(flag_arrays, axis=0)
    else:
        all_anomalies = np.zeros(n_samples, dtype=bool)
    
    df_anomalies = df[all_anomalies].copy() if all_anomalies.sum() > 0 else pd.DataFrame()
    if len(df_anomalies) > 0:
        df_anomalies['anomaly_method'] = ''
        
        # Get the indices of anomaly rows in the original dataframe
        anomaly_indices = df.index[all_anomalies]
        
        for method, flags in anomaly_flags.items():
            # Create mask for anomaly rows only (matching df_anomalies indices)
            method_mask = flags[all_anomalies]  # Filter flags to only anomaly rows
            if method_mask.sum() > 0:
                df_anomalies.loc[method_mask, 'anomaly_method'] += method + ';'
    
    # Save top anomalies
    if len(df_anomalies) > 0:
        top_anomalies = df_anomalies.head(20)
        save_dataframe(top_anomalies, "top_anomalies.csv")
        
        # Plot
        plot_anomaly_detection(df, anomaly_flags, config.FIGURES_DIR / "11_anomaly_detection.png")
    
    print("=== Anomaly Detection Complete ===\n")
    return anomaly_flags, df_anomalies

