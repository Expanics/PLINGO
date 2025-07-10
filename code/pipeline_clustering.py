import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import warnings
warnings.filterwarnings("ignore")

# --- 1. CLEANING FUNCTION
def clean_data(df):
    # Drop kolom tidak penting
    drop_cols = [
        "Video Width", "Video Height", "Video URL", "Video Cover",
        "Author ID", "Author Avatar", "Music ID", "Music Cover",
        "Music Duration", "Music URL"
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Drop duplikat
    if "id" in df.columns:
        df = df.drop_duplicates(subset="id")

    # Konversi kolom datetime
    df["Video Create Time"] = pd.to_datetime(df["Video Create Time"], unit="s", errors="coerce")

    # Format numerik
    numeric_cols = [
        "Video View Count", "Video Like Count",
        "Video Comment Count", "Video Share Count",
        "Video Duration"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bersihkan karakter aneh (emoji)
    def bersihkan_karakter_aneh(text):
        if pd.isna(text):
            return text
        return re.sub(r'[^\\x00-\\x7F]+', '', str(text))
    
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(bersihkan_karakter_aneh).str.strip()

    # Drop akun yang tidak diverifikasi (iklan)
    df = df[df["Author Verified"] != "Not Available"]

    # Ganti string kosong ke NaN
    df.replace(r'^\\s*$', np.nan, regex=True, inplace=True)

    # Imputasi kolom penting
    df["Video Description"] = df["Video Description"].fillna("missing_description")
    df["Author Nickname"] = df["Author Nickname"].fillna(df["Author Username"])
    df["Author Bio"] = df["Author Bio"].fillna("No bio yet")
    df["Music Title"] = df["Music Title"].fillna("unknown")
    df["Music Author"] = df["Music Author"].fillna("unknown")

    # Ganti string "Not Available" ke default
    df["Video Description"] = df["Video Description"].replace("Not Available", "missing_description")
    df["Video Category Type"] = df["Video Category Type"].replace("Not Available", "unknown")
    df["Music Author"] = df["Music Author"].replace("not available", "unknown")
    df["Music Title"] = df["Music Title"].replace("not available", "unknown")

    # Lowercase semua kolom string kecuali URL
    exclude_cols = ["Video Page URL"]
    for col in df.select_dtypes(include='object').columns:
        if col not in exclude_cols:
            df[col] = df[col].str.lower().str.strip()

    return df

# --- 2. FEATURE ENGINEERING FUNCTION
def feature_engineer(df):
    df["Upload Hour"] = df["Video Create Time"].dt.hour
    df["Upload Day"] = df["Video Create Time"].dt.dayofweek
    df["Desc Length"] = df["Video Description"].str.len()
    df["Bio Length"] = df["Author Bio"].str.len()
    return df

# --- 3. VECTORIZATION
def vectorize_data(df):
    selected_features = [
        "Video View Count", "Video Like Count", "Video Comment Count", "Video Share Count",
        "Upload Hour", "Upload Day", "Desc Length", "Bio Length", "Video Duration"
    ]
    X = df[selected_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# --- 4. CLUSTERING
def run_clustering(X, method='kmeans', n_clusters=5):
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        model = DBSCAN(eps=1.5, min_samples=5)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10)
    else:
        raise ValueError("method harus salah satu dari 'kmeans', 'dbscan', 'hdbscan'")
    
    labels = model.fit_predict(X)
    return labels

# --- 5. PIPELINE UTAMA
def clustering_pipeline(raw_df, method='kmeans', n_clusters=5):
    df_clean = clean_data(raw_df)
    df_fe = feature_engineer(df_clean)
    X_scaled = vectorize_data(df_fe)
    labels = run_clustering(X_scaled, method=method, n_clusters=n_clusters)
    df_fe["Cluster"] = labels
    return df_fe
