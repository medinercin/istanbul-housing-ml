"""Configuration settings for the project"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data paths
DATA_DIR = BASE_DIR / "dataset"
DATA_FILE = DATA_DIR / "istanbulApartmentForRent_cleaned.csv"

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create output directories if they don't exist
for dir_path in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Expected column names and mappings
EXPECTED_COLUMNS = {
    'district': ['district', 'ilce', 'ilçe', 'district_name'],
    'neighborhood': ['neighborhood', 'mahalle', 'semt', 'neighborhood_name'],
    'room': ['room', 'rooms', 'bedroom', 'bedrooms', 'oda'],
    'living_room': ['living room', 'livingroom', 'salon', 'living_room'],
    'area': ['area (m2)', 'area', 'area_m2', 'm2', 'square_meters', 'metrekare'],
    'age': ['age', 'building_age', 'yas', 'bina_yasi'],
    'floor': ['floor', 'kat', 'floor_number'],
    'price': ['price', 'rent', 'kira', 'rent_price', 'price_tl']
}

# Preprocessing parameters
PREPROCESSING = {
    'min_area': 10,
    'min_price': 0,
    'outlier_trim_percentile_low': 0.01,
    'outlier_trim_percentile_high': 0.99,
    'use_log_transform': True,
    'neighborhood_freq_threshold': 5  # Minimum frequency for neighborhood encoding
}

# Model parameters
MODEL = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models_to_train': ['RandomForest', 'XGBoost']
}

# Feature engineering
FEATURES = {
    'create_price_per_m2': True,  # For EDA only, not for model input
    'aggregation_levels': ['district', 'neighborhood']
}

# Clustering
CLUSTERING = {
    'n_clusters_range': range(2, 11),
    'random_state': 42
}

# Anomaly detection
ANOMALY = {
    'residual_threshold_percentile': 95,
    'z_score_threshold': 3,
    'isolation_forest_contamination': 0.1
}

# Visualization
VISUALIZATION = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

