# 🎯 Analisa Segmentasi Audiens Konten Viral di TikTok Menggunakan Clustering Berbasis Engagement

Proyek ini bertujuan untuk mengelompokkan konten viral TikTok berdasarkan caption dan tingkat engagement menggunakan pendekatan unsupervised learning. 
Dengan menerapkan algoritma **K-Means**, **DBSCAN**, dan **HDBSCAN**, proyek ini membantu mengidentifikasi pola konten yang relevan untuk strategi digital marketing dan
pengembangan konten kreatif.


## 📁 Struktur Direktori

```bash
.
├── code/                  # Berisi script Python untuk preprocessing, training, dan visualisasi clustering
├── data/                  # Berisi file dataset mentah dan hasil pembersihan
├── model/                 # Berisi model clustering yang sudah disimpan
├── cleaned_data.csv       # Dataset akhir setelah preprocessing
├── merged_raw_data.csv    # Dataset gabungan hasil scraping mentah

'''

## ⚙️ Tools & Library yang Digunakan

- Python 3.10+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- HDBSCAN
- NLTK
- TF-IDF Vectorizer

---

## 🔍 Metodologi

1. **Data Collection:**  
   Menggunakan [Nifty TikTok Scraper](https://nifty.codes/e/tiktok-scraper) untuk mengekstrak data publik TikTok (caption, views, likes, shares, comments).

2. **Preprocessing:**  
   - Cleaning teks caption  
   - Handling missing values  
   - Stopword removal  
   - TF-IDF Vectorization

3. **Clustering Algorithms:**  
   - K-Means (baseline)  
   - DBSCAN (density-based)  
   - HDBSCAN (adaptive density-based)

4. **Evaluation:**  
   Menggunakan metrik seperti Silhouette Score, Calinski-Harabasz Index, dan visualisasi scatter plot 2D (PCA / t-SNE).

---

## 📊 Hasil Akhir

| Kriteria             | KMeans            | HDBSCAN                 | DBSCAN (eps=1.5)         |
|----------------------|-------------------|--------------------------|---------------------------|
| Cluster 0            | Satisfying / tips | Creative Tutorials       | Creative / Storytelling   |
| Cluster 1            | Creative Tutorials| F&B / Lifestyle          | F&B / Lifestyle           |
| Cluster 2            | F&B / Street Food | Satisfying               | Satisfying / Visual Tips  |
| Noise (-1)           | -                 | Brand / TikTok Event     | Brand / TikTok Event      |
| View Tertinggi       | 212M (Satisfying) | 731M (Event)             | 731M (Event)              |
| Outlier Separation   | ❌ Tercampur       | ✅ Sangat jelas           | ✅ Sangat jelas           |

---

## 🧠 Insight

- Konten **satisfying dan F&B** merupakan topik viral utama.
- **Konten brand/event** meski dianggap outlier, justru punya view tertinggi.
- DBSCAN & HDBSCAN jauh lebih efektif dalam pemisahan cluster dibanding K-Means.

---

## 🧩 Author

- **Akmal Dwi Putra Mahardika** – akmal.mahardika@binus.ac.id  
- **Muhammad Iqbal Saputra** – muhammad.saputra02@binus.ac.id  
- **Muhammad Reza Alghifari** – muhammad.alghifari011@binus.ac.id  

---

## 📜 Lisensi

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
