"""Clustering analysis for neighborhoods"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import src.config as config
from src.io import save_dataframe


def prepare_clustering_features(df: pd.DataFrame) -> tuple:
    """Prepare features for clustering"""
    # Aggregate neighborhood-level features
    neighborhood_features = df.groupby('neighborhood').agg({
        'price': ['mean', 'median', 'std'],
        'area': ['mean', 'median'],
        'age': ['mean', 'median'],
        'room': ['mean'],
        'floor': ['mean']
    }).reset_index()
    
    neighborhood_features.columns = ['neighborhood'] + [
        f'{col[0]}_{col[1]}' for col in neighborhood_features.columns[1:]
    ]
    
    # Calculate additional metrics
    neighborhood_features['price_cv'] = (
        neighborhood_features['price_std'] / neighborhood_features['price_mean']
    ) * 100
    
    # Select numeric features for clustering
    feature_cols = [col for col in neighborhood_features.columns 
                   if col != 'neighborhood' and pd.api.types.is_numeric_dtype(neighborhood_features[col])]
    
    X = neighborhood_features[feature_cols].fillna(0)
    
    return neighborhood_features, X, feature_cols


def find_optimal_clusters(X: pd.DataFrame, max_k: int = 10) -> dict:
    """Find optimal number of clusters using elbow and silhouette methods"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=config.CLUSTERING['random_state'], n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def plot_cluster_selection(metrics: dict, save_path):
    """Plot elbow and silhouette plots for cluster selection"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(metrics['k_range'], metrics['inertias'], marker='o')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal k')
    axes[0].grid(True)
    
    # Silhouette plot
    axes[1].plot(metrics['k_range'], metrics['silhouette_scores'], marker='o', color='orange')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score for Optimal k')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved cluster selection plot to: {save_path}")


def perform_clustering(X: pd.DataFrame, n_clusters: int = 5) -> tuple:
    """Perform KMeans clustering"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.CLUSTERING['random_state'], n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    return labels, kmeans, scaler


def plot_cluster_pca(X: pd.DataFrame, labels: np.ndarray, neighborhoods: pd.Series, save_path):
    """Plot clusters in 2D PCA space"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=config.CLUSTERING['random_state'])
    X_pca = pca.fit_transform(X_scaled)
    
    # Create plot
    fig, ax = plt.subplots(figsize=config.VISUALIZATION['figsize'])
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('Neighborhood Clusters (PCA Visualization)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Annotate some neighborhoods
    for idx in range(min(20, len(neighborhoods))):
        ax.annotate(neighborhoods.iloc[idx], (X_pca[idx, 0], X_pca[idx, 1]), fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.VISUALIZATION['dpi'], bbox_inches='tight')
    plt.close()
    print(f"Saved cluster PCA plot to: {save_path}")


def generate_cluster_summary(neighborhood_features: pd.DataFrame, labels: np.ndarray, 
                             feature_cols: list) -> pd.DataFrame:
    """Generate cluster summary statistics"""
    neighborhood_features = neighborhood_features.copy()
    neighborhood_features['cluster'] = labels
    
    # Calculate cluster means
    cluster_summary = neighborhood_features.groupby('cluster')[feature_cols].mean()
    cluster_summary['count'] = neighborhood_features.groupby('cluster').size()
    
    return cluster_summary


def generate_clustering_report(df: pd.DataFrame):
    """Generate clustering analysis report"""
    print("\n=== Generating Clustering Analysis ===")
    
    neighborhood_features, X, feature_cols = prepare_clustering_features(df)
    
    if len(X) < 2:
        print("Not enough neighborhoods for clustering")
        return None, None
    
    # Find optimal clusters
    metrics = find_optimal_clusters(X, max_k=10)
    plot_cluster_selection(metrics, config.FIGURES_DIR / "09_cluster_selection.png")
    
    # Select optimal k (highest silhouette score)
    optimal_k = metrics['k_range'][np.argmax(metrics['silhouette_scores'])]
    print(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(metrics['silhouette_scores']):.3f})")
    
    # Perform clustering
    labels, kmeans, scaler = perform_clustering(X, n_clusters=optimal_k)
    neighborhood_features['cluster'] = labels
    
    # Plot PCA visualization
    plot_cluster_pca(X, labels, neighborhood_features['neighborhood'], 
                    config.FIGURES_DIR / "10_cluster_pca.png")
    
    # Generate cluster summary
    cluster_summary = generate_cluster_summary(neighborhood_features, labels, feature_cols)
    save_dataframe(cluster_summary, "cluster_summary.csv")
    save_dataframe(neighborhood_features[['neighborhood', 'cluster']], "neighborhood_clusters.csv")
    
    print("=== Clustering Analysis Complete ===\n")
    return neighborhood_features, cluster_summary

