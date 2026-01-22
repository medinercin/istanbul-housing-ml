# Veri okuma/yazma fonksiyonları
# CSV, JSON, model dosyalarını yönetir

import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import src.config as config


def load_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    # CSV veri dosyasını yükle
    if file_path is None:
        file_path = config.DATA_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_dataframe(df: pd.DataFrame, filename: str, subdir: str = "tables") -> Path:
    # DataFrame'i CSV olarak kaydet
    if subdir == "tables":
        output_path = config.TABLES_DIR / filename
    else:
        output_path = config.OUTPUT_DIR / subdir / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataframe to: {output_path}")
    return output_path


def save_json(data: Dict[str, Any], filename: str, subdir: str = "metrics") -> Path:
    # Dictionary'yi JSON olarak kaydet
    if subdir == "metrics":
        output_path = config.METRICS_DIR / filename
    else:
        output_path = config.OUTPUT_DIR / subdir / filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to: {output_path}")
    return output_path


def load_json(filename: str, subdir: str = "metrics") -> Dict[str, Any]:
    # JSON dosyasını yükle
    if subdir == "metrics":
        input_path = config.METRICS_DIR / filename
    else:
        input_path = config.OUTPUT_DIR / subdir / filename
    
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model(model: Any, filename: str) -> Path:
    # Eğitilmiş modeli pickle ile kaydet
    output_path = config.MODELS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to: {output_path}")
    return output_path


def load_model(filename: str) -> Any:
    # Pickle ile kaydedilmiş modeli yükle
    input_path = config.MODELS_DIR / filename
    with open(input_path, 'rb') as f:
        return pickle.load(f)

