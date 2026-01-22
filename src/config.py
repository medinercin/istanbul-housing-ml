# Proje ayarları ve konfigürasyon dosyası
# Tüm path'ler ve parametreler burada tanımlı

import os
from pathlib import Path

# Ana dizin
BASE_DIR = Path(__file__).parent.parent

# Veri dosyalarının yolu
DATA_DIR = BASE_DIR / "dataset"
DATA_FILE = DATA_DIR / "istanbulApartmentForRent_cleaned.csv"

# Çıktı klasörleri
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Çıktı klasörlerini oluştur (yoksa)
for dir_path in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Beklenen sütun isimleri ve eşleştirmeleri
# Farklı veri setlerinde farklı isimler olabilir, hepsini buraya ekledim
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

# Ön işleme parametreleri
PREPROCESSING = {
    'min_area': 10,  # minimum alan (m2)
    'min_price': 0,  # minimum fiyat
    'outlier_trim_percentile_low': 0.01,  # alt outlier eşiği
    'outlier_trim_percentile_high': 0.99,  # üst outlier eşiği
    'use_log_transform': True,  # log transformasyonu kullan
    'neighborhood_freq_threshold': 5  # mahalle encoding için minimum frekans
}

# Model parametreleri
MODEL = {
    'test_size': 0.2,  # test set oranı
    'random_state': 42,  # rastgelelik için seed
    'cv_folds': 5,  # cross validation fold sayısı
    'models_to_train': ['RandomForest', 'XGBoost']  # eğitilecek modeller
}

# Feature engineering ayarları
FEATURES = {
    'create_price_per_m2': True,  # EDA için kullanılacak, modelde leakage riski var
    'aggregation_levels': ['district', 'neighborhood']  # aggregation seviyeleri
}

# Clustering ayarları
CLUSTERING = {
    'n_clusters_range': range(2, 11),  # cluster sayısı aralığı
    'random_state': 42
}

# Anomaly detection ayarları
ANOMALY = {
    'residual_threshold_percentile': 95,  # residual için eşik
    'z_score_threshold': 3,  # z-score eşiği
    'isolation_forest_contamination': 0.1  # contamination oranı
}

# Görselleştirme ayarları
VISUALIZATION = {
    'figsize': (12, 8),  # figür boyutu
    'dpi': 300,  # çözünürlük
    'style': 'seaborn-v0_8'  # matplotlib stili
}

