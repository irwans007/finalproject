# 🚗 Saudi Arabia Used Car Price Prediction  

 
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-brightgreen.svg)]([https://mobil-bekas-saudi.streamlit.app](https://gamma-finalprojek.streamlit.app/))  
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-orange.svg)]([https://public.tableau.com/app/profile/](https://public.tableau.com/app/profile/muhammad.bagaskara7905/viz/FinalprojectTableau_17586298692610/Dashboard3))


**👨‍💻 Author: Eric Cen- Irwan Setio- Muhammad Bagaskara **  

📑 [Presentation Deck](#) | 🌐 [Streamlit Demo](#)  

---

## 🎯 Tentang Proyek  
Proyek ini membangun **model regresi machine learning** untuk memprediksi harga **mobil bekas di Saudi Arabia** (platform [Syarah.com](https://syarah.com)).  
Model ini diharapkan membantu:  

- 🏢 **Perusahaan Syarah.com** → menghemat biaya appraisal manual  
- 👨‍🔧 **Team Appraiser** → sebagai referensi harga mobil  
- 🚘 **Calon Penjual** → mendapat estimasi harga sebelum listing  

---

## 🔑 Business Problem  

- ❌ Penentuan harga manual → tidak efisien & mahal  
- ❌ Workload tinggi jika mobil bertambah  
- ✅ Solusi: **Model Machine Learning** prediksi harga otomatis  

**🎯 Target Metrics**:  
- MAE `< 10,000 SAR`  
- MAPE `< 20%`  
- R² `> 0.70`  

---

## 📊 Dataset  

- **Jumlah data:** 5.624 baris, 11 kolom  
- **Sumber:** Syarah.com  
- **Fitur utama:**  
  - Type, Region, Make, Gear Type, Origin  
  - Options (Standard / Semi-Full / Full)  
  - Year, Engine Size, Mileage  
  - Price (Target)  

📌 Setelah preprocessing → **3.192 data usable**  

---

## ⚙️ Data Preprocessing  

- Hapus duplikat & data dengan `Price=0`  
- Handle **outlier**: Price (15k–182k), Year (2003–2022), Mileage (<376k)  
- Encoding: Ordinal, Binary, One-Hot  
- Scaling: Robust Scaler  

---

## 🤖 Modelling  

Model yang diuji:  
- KNN Regressor  
- Decision Tree  
- Linear Regression  
- XGB Regressor  
- Stacking (Linear, Decision Tree, KNN, XGB)  
- **CatBoost Regressor (Best)**  

---

## 🏆 Hasil  

**🔥 Model Terbaik: CatBoost Regressor**  

| Metric | Score   |  
|--------|---------|  
| MAE    | 9,837.29 |  
| MAPE   | 18% |  
| R²     | 0.84 |  
| RMSE   | 14,460.97 |  

✅ Rata-rata prediksi meleset ±18% dari harga sebenarnya.  

---

## 📌 Kesimpulan  

- Fitur paling berpengaruh: **Engine Size, Year, Mileage**  
- CatBoost dengan tuning memberikan performa terbaik  
- Dataset masih terbatas (tidak ada fitur kondisi fisik mobil)  

---

## 💡 Rekomendasi  

- Tambahkan fitur detail kondisi mobil (interior, tabrakan, dll.)  
- Gunakan gambar mobil → Computer Vision  
- Segmentasi prediksi berdasarkan range harga mobil  
- Perbanyak dataset agar prediksi lebih akurat  

---

## 🛠️ Tech Stack  

- Python (Pandas, NumPy, Scikit-learn)  
- CatBoost, XGBoost  
- Matplotlib, Seaborn  
- Jupyter Notebook  
- Streamlit (Deployment)  

---

## ✨ Impact Bisnis  

- 💰 Hemat biaya appraisal → dari **SAR 80,000/bln** ➝ hanya **SAR 12,000/bln**  
- 🚀 Skalabilitas tinggi: ribuan mobil bisa diprediksi otomatis  
- 🙌 Membantu penjual & perusahaan menentukan harga lebih cepat  

---



---
