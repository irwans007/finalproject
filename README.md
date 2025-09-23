# Saudi Arabia Used Car Price Prediction  

Oleh: Eric Cen- Irwan Setio- Muhammad Bagaskara  (Gamma)

📑 [Tableu](https://public.tableau.com/app/profile/muhammad.bagaskara7905/viz/FinalprojectTableau_17586298692610/Dashboard3) | 🌐 [Streamlit Demo](https://gamma-finalprojek.streamlit.app/)  


---

## Introduction  
Proyek ini bertujuan membangun **model machine learning** untuk memprediksi harga mobil bekas Saudi Arabia syarah.com . Proyek ini mencari model terbaik dimana model ini membantu mendukung tim appraiser dalam menilai harga secara konsisten, serta memberi calon penjual estimasi harga awal sebelum listing.  

Daftar Isi
1. Business Problem Understanding
2. Data Understanding
3. Data Pre-Processing
4. Modelling
5. Evaluation
6. Conclusion
7. Recommendation

---

## 🔑 1. Business Problem  

 - **Context**  
Sebuah dealer mobil saudi arabia ingin menenentukan harga jual mobil bekas dengan menggunakan data analytics dan model Machine Learning untuk diimplementasikan kepada sales dealer tersebut. Pengunaan model ini bertujuan untuk menghindari kesalahan dalam memnberikan harga jual pada konsumen dan untuk juga meningkatkan pendapatan dealer & performa sales.  

- **Problem Statement** 
Pendapatan dealer bisa berkurang karena permasalahan harga yang tidak tepat & salah. sehingga dealer ingin meningkatkan pendapatan & mencari harga yang optimal. karena tidak didasarkan data yang tidak informatif seperti data dan cara tradisional menentukan harga mobil bekas.Jika penentuan harga tidak optimal, dealer akan terus mengalami kerugian. 

- **Stakeholders**
 
1. Calon Penjual Mobil
   - Masalah: Penjual bingung harga yang cocok untuk harga mobilnya 
   - Dampak: Mobil susah laku 

2. Tim Appraiser  
   - Masalah: Jumlah mobil yang akan dijual membutuhkan banyak pegawai untuk analisa harga
   - Dampak: Beban kerja tinggi  

3. Dealer  
   - Masalah: Perusahaan membutuhkan pengeluaran untuk hire banyak pegawai dalam penentuan harga  
   - Dampak: Pendapatan berkurang  

- **Goals** 
Berdasarkan permasalahan tersebut, model ini dapat digunakan calon penjual untuk mendapatkan perkiraan harga jual serta pegawai appraiser juga dapat menentukan harga mobil setelah inspeksi.


Target metric:
| Metric | Target |  
|--------|---------|  
| MAE    | < 18,000 SAR |  
| MAPE   | < 40 |  

- **Analytic Approach**
1. Analisis data untuk memahami pola antar fitur yang memengaruhi harga mobil.  
2. Membangun model untuk memprediksi harga mobil bekas.  
3. Evaluasi model menggunakan metrik error (MAE, MAPE, RMSE, R²).  
  
- **📏 Metric Evaluation**

- **MAE (Primary)** → Stabil terhadap outlier, hasil dalam satuan asli (SAR).  
- **MAPE (Secondary)** → Menunjukkan error dalam bentuk persentase.  
- **R² (Goodness of Fit)** → Mengukur seberapa baik model menjelaskan data.  
- **RMSE** → Memberi penalti lebih besar pada error ekstrem (outlier).  

👉 Semakin kecil nilai **MAE** & **MAPE**, semakin akurat model dalam memprediksi harga mobil sesuai keterbatasan fitur dataset.  

---
## 📊 2. Data Understanding  


  - **Sumber** : https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset
  - **Jumlah data:** 8035 baris, 13 kolom  
  - **Feature Breakdown**
**Numerical Features**:

Year: Tahun produksi mobil. Ini adalah fitur numerik kontinu, namun juga bisa dianggap sebagai fitur ordinal.
Engine_Size: Ukuran mesin mobil, kemungkinan dalam satuan liter.
Mileage: Total kilometer yang telah ditempuh oleh mobil. Fitur ini mungkin mengandung outlier, seperti nilai yang sangat besar dalam cuplikan data.
Price: Harga jual mobil. Ini adalah variabel target untuk model prediksi. Nilainya berupa angka kontinu, namun terdapat nilai 0 yang kemungkinan besar merupakan placeholder untuk harga yang belum dicantumkan atau hilang.

**Categorical Features:**
Make: Merek atau produsen mobil (misalnya Toyota, Ford, Hyundai).
Type: Model spesifik dari mobil (misalnya Corolla, Explorer, Sonata).
Origin: Negara asal mobil, dengan nilai seperti 'Saudi' atau 'Other'.
Color: Warna eksterior mobil. Terdapat nilai seperti "Another Color" yang menunjukkan perlunya pembersihan atau konsolidasi kategori.
Options: Tingkat kelengkapan fitur mobil (misalnya 'Full', 'Semi Full', 'Standard').
Fuel_Type: Jenis bahan bakar yang digunakan (misalnya 'Gas').
Gear_Type: Jenis transmisi (misalnya 'Automatic').
Region: Wilayah geografis atau kota tempat mobil dijual (misalnya 'Riyadh', 'Makkah').
Negotiable: Fitur biner yang menunjukkan apakah harga bisa dinegosiasikan. Dataset menggunakan nilai boolean (TRUE/FALSE), yang telah dikonversi menjadi nilai numerik (1/0) dalam kode. 



---

## ⚙️3.  Data Preprocessing  

- Hapus duplikat & data dengan `Price=0`  
- Pakai StandardScaler, RobustScaler 

---

## 🤖 4. Modelling  

Model yang digunakan dan diuji:  
- Linear Regression 
- Ridge 
- Lasso 
- K-Neighbors Regressor
- Decision Tree Regressor  
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- **CatBoost Regressor (Best)**  

---

## 5. Evaluation  

**🔥 Model Terbaik: CatBoost Regressor**  

| Metric | Score   |  
|--------|---------|  
| MAE    | 17743 |  
| MAPE   | 38% |  



---

## 📌 6. Conclusion  

1. Berdasarkan eksperimen dengan berbagai algoritma regresi, **CatBoost Regressor** terbukti menjadi model terbaik dengan hasil:  
   - MAE ≈ **17,743 SAR**  
   - MAPE ≈ **38%**  
   - Performanya lebih stabil dibandingkan sebagian besar model lain.  

2. **Fitur utama yang paling memengaruhi harga mobil** adalah:  
   - **Engine Size**  
   - **Year (Tahun Produksi)**  
   - **Mileage (Jarak Tempuh)**  

3. Model mampu memberikan estimasi harga mobil bekas yang dapat membantu:  
   - **Calon penjual** → mendapatkan perkiraan harga awal sebelum listing.  
   - **Tim Appraiser** → sebagai acuan dalam menilai harga secara konsisten.  
   - **Dealer/Perusahaan** → mengurangi ketergantungan pada penilaian manual dan menekan biaya appraisal.  

4. Meski demikian, tingkat error relatif (**MAPE 38%**) menunjukkan bahwa prediksi masih cukup meleset. Hal ini disebabkan keterbatasan dataset, terutama tidak adanya fitur yang merepresentasikan **kondisi fisik mobil** (riwayat servis, kerusakan, tabrakan, kondisi interior/eksterior, dll.).  

👉 Secara keseluruhan, project ini berhasil membuktikan bahwa **machine learning regression** dapat digunakan untuk membantu dealer mobil bekas di Saudi Arabia dalam **otomatisasi penentuan harga**, meskipun masih perlu pengembangan lebih lanjut agar akurasi semakin tinggi.  

---

## 💡7. Recommendation 

- **For Model Development** 
1. **Feature Engineering lebih kaya**  
   - Tambahkan fitur yang lebih menggambarkan kondisi mobil, misalnya riwayat servis, riwayat tabrakan, kondisi eksterior/interior, jumlah pemilik, dll.  
   - Gunakan data eksternal (misalnya harga rata-rata dari dealer lain) untuk memperkaya dataset.  

2. **Data Augmentation**  
   - Perbanyak jumlah dataset karena data setelah cleaning relatif kecil (~5800 baris).  
   - Lakukan balancing untuk mengurangi bias pada jenis mobil tertentu yang terlalu dominan.  

 -  **For Business** 
1. **Implementasi ke Platform**  
   - Integrasikan model ke website/app Syarah.com sehingga penjual bisa langsung mendapat estimasi harga otomatis saat upload mobil.    

3. **Efisiensi Biaya**  
   - Dengan model, perusahaan dapat menghemat biaya appraisal manual 
   - Hemat biaya bisa dialokasikan ke promosi atau ekspansi bisnis.  

4. **Ekspansi Use Case**  
   - Selain prediksi harga, model serupa bisa dikembangkan untuk:  
     - Prediksi *mobil paling laku* (demand forecasting).  
     - Rekomendasi mobil sesuai preferensi pelanggan.  
     - Analisis tren harga mobil per merek/tipe.  

---







