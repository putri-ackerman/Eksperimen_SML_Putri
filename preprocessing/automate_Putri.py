
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def automate_preprocessing(input_path, output_folder=None):
    
    # 1. LOAD DATA
    df = pd.read_csv(input_path)
    
    # 2. HANDLE MISSING VALUES
    missing_rules = {
        'House_Price': 'drop',
        'Square_Footage': 'median',
        'Num_Bedrooms': 'mode',
        'Num_Bathrooms': 'mode',
        'Lot_Size': 'median',
        'Garage_Size': 'mode',
        'Year_Built': 'mode',
        'Neighborhood_Quality': 'mode'
    }
    
    for col, strategy in missing_rules.items():
        if col in df.columns and df[col].isnull().any():
            if strategy == 'drop':
                df = df.dropna(subset=[col])
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode[0])
    
    # 3. HAPUS DUPLIKAT
    df = df.drop_duplicates()
    
    # 4. FEATURE ENGINEERING
    df['House_Age'] = 2024 - df['Year_Built']
    df = df.drop('Year_Built', axis=1)
    
    # 5. OUTLIER HANDLING
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return np.where(series > upper, upper, 
                       np.where(series < lower, lower, series))
    
    outlier_cols = ['Square_Footage', 'Lot_Size', 'Garage_Size']
    for col in outlier_cols:
        if col in df.columns:
            df[col] = cap_outliers(df[col])
    
    # 6. STANDARDIZATION
    X = df.drop('House_Price', axis=1)
    y = df['House_Price']
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    df_scaled = pd.concat([X_scaled, y], axis=1)
    
    # 7. SIMPAN JIKA DIMINTA
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(f'{output_folder}/house_price_clean.csv', index=False)
        df_scaled.to_csv(f'{output_folder}/house_price_scaled.csv', index=False)
        joblib.dump(scaler, f'{output_folder}/scaler.pkl')
    
    return df, df_scaled, scaler

if __name__ == "__main__":
    import os
    
    # Cek folder saat ini agar kita tahu GitHub sedang berada di mana
    print(f"Direktori saat ini: {os.getcwd()}")
    print(f"Isi direktori: {os.listdir('.')}")

    # PATH YANG BENAR UNTUK GITHUB (Tanpa nama Repo di depan)
    input_file = "house_price_raw/house_price_regression_dataset.csv"
    output_dir = "preprocessing/house_price_preprocessing"
    
    # Validasi: Cek apakah filenya beneran ada sebelum diproses
    if os.path.exists(input_file):
        print(f"✅ File ditemukan: {input_file}")
        df_clean, df_scaled, scaler = automate_preprocessing(input_file, output_dir)
        print(f"✅ Preprocessing Selesai!")
    else:
        print(f"❌ ERROR: File TIDAK ditemukan di {input_file}")
        # Cek isi folder raw untuk memastikan nama filenya benar
        if os.path.exists("house_price_raw"):
            print(f"Isi folder house_price_raw: {os.listdir('house_price_raw')}")
