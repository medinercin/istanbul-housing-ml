# Analiz raporu oluşturma
# Markdown formatında rapor üretiyoruz

import pandas as pd
from pathlib import Path
import json
import src.config as config
from src.io import load_json


def generate_analysis_md(metrics_dict: dict, volatility_df: pd.DataFrame = None,
                        cluster_summary: pd.DataFrame = None,
                        anomaly_count: int = 0,
                        use_log: bool = False) -> str:
    # analysis.md markdown raporunu oluştur
    
    report = []
    report.append("# Istanbul Kiralık Konut Fiyat Tahmini - Analiz Raporu\n")
    report.append("## 1. Problem Tanımı\n")
    report.append("Bu proje, İstanbul'un 39 ilçesi ve mahalle/semt bazındaki kiralık konut ilan verilerini kullanarak:")
    report.append("- Kira fiyat tahmini (regression)")
    report.append("- Bölgesel farkların analizi (district + neighborhood)")
    report.append("- Mahalle/ilçe bazında fiyat volatilitesi (oynaklık) analizi")
    report.append("- Aykırı/anormal fiyatlı ilanların tespiti (anomaly detection)")
    report.append("- Bölgelerin benzer profillere göre kümelenmesi (clustering)")
    report.append("- Yorumlanabilirlik (feature importance)")
    report.append("- Tüm sonuçları görsel ve metriklerle raporlama\n")
    
    report.append("## 2. Veri Özeti\n")
    report.append("### 2.1 Veri Kaynağı\n")
    report.append(f"- Veri dosyası: `{config.DATA_FILE.name}`")
    report.append("- Veri lokal CSV dosyasından okunmuştur.\n")
    
    report.append("### 2.2 Özellikler\n")
    report.append("- **district**: İlçe bilgisi")
    report.append("- **neighborhood**: Mahalle/semt bilgisi")
    report.append("- **room**: Oda sayısı")
    report.append("- **living_room**: Salon sayısı")
    report.append("- **area**: Metrekare (m²)")
    report.append("- **age**: Bina yaşı")
    report.append("- **floor**: Kat bilgisi")
    report.append("- **price**: Kira fiyatı (TL) - Hedef değişken\n")
    
    report.append("### 2.3 Veri Ön İşleme\n")
    report.append(f"- Minimum alan filtresi: {config.PREPROCESSING['min_area']} m²")
    report.append(f"- Minimum fiyat filtresi: {config.PREPROCESSING['min_price']} TL")
    report.append(f"- Aykırı değer kırpma: %{config.PREPROCESSING['outlier_trim_percentile_low']*100} - %{config.PREPROCESSING['outlier_trim_percentile_high']*100}")
    if use_log:
        report.append("- Hedef değişken dönüşümü: log1p(price) uygulanmıştır.\n")
    else:
        report.append("- Hedef değişken dönüşümü: Uygulanmamıştır.\n")
    
    report.append("## 3. EDA Bulguları\n")
    report.append("### 3.1 Görselleştirmeler\n")
    report.append("Aşağıdaki görselleştirmeler `outputs/figures/` dizininde bulunmaktadır:\n")
    report.append("- `01_missing_values.png`: Eksik değer analizi")
    report.append("- `02_target_distribution.png`: Hedef değişken dağılımı")
    report.append("- `03_feature_distributions.png`: Özellik dağılımları")
    report.append("- `04_correlation_matrix.png`: Korelasyon matrisi")
    report.append("- `05_district_price_analysis.png`: İlçe bazında fiyat analizi")
    report.append("- `06_spatial_heatmap.png`: Bölgesel fiyat haritası\n")
    
    report.append("## 4. Modeller ve Metrikler\n")
    report.append("### 4.1 Model Seçimi\n")
    report.append("Aşağıdaki regresyon modelleri eğitilmiştir:\n")
    for model_name in metrics_dict.keys():
        report.append(f"- **{model_name}**")
    report.append("")
    
    report.append("### 4.2 Model Metrikleri\n")
    report.append("| Model | MAE | RMSE | R² | MAPE (%) |\n")
    report.append("|-------|-----|------|----|----------|\n")
    
    for model_name, metrics in metrics_dict.items():
        mae = metrics.get('MAE', 0)
        rmse = metrics.get('RMSE', 0)
        r2 = metrics.get('R2', 0)
        mape = metrics.get('MAPE', 0)
        report.append(f"| {model_name} | {mae:.2f} | {rmse:.2f} | {r2:.3f} | {mape:.2f} |\n")
    
    report.append("\n### 4.3 Model Değerlendirme Görselleri\n")
    report.append("- `12_randomforest_evaluation.png`: Random Forest değerlendirme grafikleri")
    report.append("- `12_xgboost_evaluation.png`: XGBoost değerlendirme grafikleri")
    report.append("- `13_model_comparison.png`: Model karşılaştırma grafikleri\n")
    
    report.append("## 5. Volatilite Analizi\n")
    if volatility_df is not None and len(volatility_df) > 0:
        report.append("### 5.1 Volatilite Metrikleri\n")
        report.append("Volatilite analizi için aşağıdaki metrikler hesaplanmıştır:\n")
        report.append("- **CV (Coefficient of Variation)**: Standart sapmanın ortalamaya oranı (%). Yüksek CV değeri, fiyatların daha oynak olduğunu gösterir.")
        report.append("- **IQR (Interquartile Range)**: Q3 - Q1. Fiyat dağılımının genişliğini gösterir.")
        report.append("- **Std (Standard Deviation)**: Standart sapma.\n")
        
        report.append("### 5.2 Volatilite Görselleri\n")
        report.append("- `07_volatility_district.png`: İlçe bazında volatilite analizi")
        report.append("- `08_volatility_neighborhood.png`: Mahalle bazında volatilite analizi\n")
        
        # Top volatile districts
        district_vol = volatility_df[volatility_df['level'] == 'district'].sort_values('CV', ascending=False).head(5)
        if len(district_vol) > 0:
            report.append("### 5.3 En Oynak İlçeler (CV'ye göre)\n")
            for idx, row in district_vol.iterrows():
                report.append(f"- **{row['location']}**: CV = {row['CV']:.2f}%, IQR = {row['IQR']:.0f} TL\n")
    else:
        report.append("Volatilite analizi verisi mevcut değil.\n")
    
    report.append("## 6. Kümeleme Analizi\n")
    if cluster_summary is not None and len(cluster_summary) > 0:
        report.append("### 6.1 Kümeleme Yöntemi\n")
        report.append("- **Algoritma**: K-Means Clustering")
        report.append("- **Özellikler**: Mahalle bazında fiyat, alan, yaş, oda sayısı gibi metriklerin ortalamaları")
        report.append("- **Ölçeklendirme**: StandardScaler ile normalize edilmiştir.\n")
        
        report.append("### 6.2 Küme Özeti\n")
        report.append(f"Toplam {len(cluster_summary)} küme oluşturulmuştur.\n")
        report.append("| Küme | Örnek Sayısı | Ortalama Fiyat | Ortalama Alan |\n")
        report.append("|------|--------------|----------------|---------------|\n")
        
        for cluster_id in cluster_summary.index:
            count = cluster_summary.loc[cluster_id, 'count']
            price_col = [c for c in cluster_summary.columns if 'price_mean' in c]
            area_col = [c for c in cluster_summary.columns if 'area_mean' in c]
            price_val = cluster_summary.loc[cluster_id, price_col[0]] if price_col else 'N/A'
            area_val = cluster_summary.loc[cluster_id, area_col[0]] if area_col else 'N/A'
            if isinstance(price_val, (int, float)):
                price_val = f"{price_val:.0f}"
            if isinstance(area_val, (int, float)):
                area_val = f"{area_val:.1f}"
            report.append(f"| {cluster_id} | {count} | {price_val} | {area_val} |\n")
        
        report.append("\n### 6.3 Kümeleme Görselleri\n")
        report.append("- `09_cluster_selection.png`: Optimal küme sayısı seçimi (Elbow + Silhouette)")
        report.append("- `10_cluster_pca.png`: Kümeleme sonuçları (PCA görselleştirmesi)\n")
    else:
        report.append("Kümeleme analizi verisi mevcut değil.\n")
    
    report.append("## 7. Anomali Tespiti\n")
    report.append("### 7.1 Anomali Tespit Yöntemleri\n")
    report.append("Üç farklı yöntem kullanılmıştır:\n")
    report.append("1. **Residual-based**: Model tahmin hatası büyük olan ilanlar")
    report.append("2. **Z-score**: Mahalle içinde fiyat z-skoru |z| >= 3 olan ilanlar")
    report.append("3. **Isolation Forest**: Özellik uzayında anomali tespiti\n")
    
    report.append(f"### 7.2 Anomali Sayısı\n")
    report.append(f"Toplam {anomaly_count} anomali tespit edilmiştir.\n")
    report.append("### 7.3 Anomali Görselleri\n")
    report.append("- `11_anomaly_detection.png`: Anomali tespit görselleri\n")
    
    report.append("## 8. Yorumlanabilirlik Analizi\n")
    report.append("### 8.1 Feature Importance\n")
    report.append("Permutation importance yöntemi kullanılarak özellik önemleri hesaplanmıştır.\n")
    report.append("### 8.2 Feature Importance Görselleri\n")
    for model_name in metrics_dict.keys():
        report.append(f"- `14_{model_name.lower()}_feature_importance.png`: {model_name} özellik önemleri\n")
    
    report.append("## 9. Kısıtlar ve Gelecekte İyileştirmeler\n")
    report.append("### 9.1 Mevcut Kısıtlar\n")
    report.append("- Veri lokal CSV dosyasından okunmaktadır; gerçek zamanlı veri akışı yoktur.")
    report.append("- Coğrafi koordinat bilgisi olmadığı için gerçek uzamsal analiz yapılamamıştır.")
    report.append("- Kategorik özellikler için OneHotEncoder kullanılmıştır; yüksek kardinaliteli özellikler için alternatifler düşünülebilir.\n")
    
    report.append("### 9.2 Önerilen İyileştirmeler\n")
    report.append("1. **Veri Toplama**: Gerçek zamanlı veri akışı entegrasyonu")
    report.append("2. **Özellik Mühendisliği**:")
    report.append("   - Coğrafi koordinatlar eklenmesi")
    report.append("   - Ulaşım ağı mesafeleri (metro, otobüs durakları)")
    report.append("   - Okul, hastane gibi yakınlık özellikleri")
    report.append("   - Yüksek kardinaliteli kategorikler için target encoding veya hashing encoder")
    report.append("3. **Model İyileştirmeleri**:")
    report.append("   - Daha fazla model denemesi (LightGBM, CatBoost, Neural Networks)")
    report.append("   - Ensemble yöntemleri")
    report.append("   - SHAP değerleri ile daha detaylı yorumlanabilirlik")
    report.append("4. **Monitoring ve Updating**:")
    report.append("   - Model performans izleme sistemi")
    report.append("   - Periyodik yeniden eğitim pipeline'ı")
    report.append("   - A/B test altyapısı")
    report.append("   - Otomatik anomali uyarı sistemi\n")
    
    report.append("## 10. Sonuç\n")
    report.append("Bu proje, İstanbul kiralık konut verileri üzerinde kapsamlı bir analiz ve tahmin modeli geliştirmiştir. ")
    report.append("Elde edilen modeller, bölgesel farkları, volatiliteyi ve anomali durumlarını tespit edebilmektedir. ")
    report.append("Gelecekteki iyileştirmelerle daha güçlü ve gerçek zamanlı bir sistem geliştirilebilir.\n")
    
    report.append("---\n")
    report.append("*Bu rapor otomatik olarak oluşturulmuştur.*\n")
    
    return "\n".join(report)


def save_analysis_report(content: str):
    """Save analysis report to file"""
    output_path = config.REPORTS_DIR / "analysis.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Saved analysis report to: {output_path}")

