# pages/01_Data_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle  # untuk load model .sav


st.set_page_config(page_title="📈 Data Overview", page_icon="📈", layout="wide")
st.title("📈 Data Overview (Only)")

st.write("Upload CSV → lihat ringkasan & rata-rata **Price** per kategori. "
         "Halaman ini **tidak** memuat prediksi maupun SHAP.")

# ==== KONFIGURASI ====
FEATURE_COLUMNS = [
    "Type","Region","Make","Gear_Type","Origin","Options",
    "Color","Fuel_Type",
    "Year","Engine_Size","Mileage",
]
CATEGORICAL = ["Type","Region","Make","Gear_Type","Origin","Options","Color","Fuel_Type"]

# ==== DATA SOURCE ====
st.sidebar.header("Upload CSV (untuk halaman ini)")
uploaded = st.sidebar.file_uploader("Choose file (.csv)", type=["csv"])

# Kamu juga bisa pakai data dari halaman lain via session_state:
# st.session_state["data_df"] = df
df_session = st.session_state.get("data_df")

if uploaded:
    try:
        data = pd.read_csv(uploaded)
        st.success(f"File `{uploaded.name}` loaded. Rows: {len(data):,}")
    except Exception as e:
        st.error(f"Gagal baca CSV: {e}")
        st.stop()
elif df_session is not None:
    data = df_session.copy()
    st.info("Pakai data dari session (halaman lain).")
else:
    st.info("👈 Upload CSV dulu atau kirimkan data lewat `st.session_state['data_df']` dari halaman utama.")
    st.stop()

# ==== VALIDASI MINIMAL ====
if "Price" not in data.columns:
    st.error("Kolom **Price** wajib ada untuk membuat ringkasan.")
    st.stop()

available_cats = [c for c in CATEGORICAL if c in data.columns]
if not available_cats:
    st.warning("Tidak ada kolom kategori standar yang ditemukan. "
               "Tambahkan salah satu dari: " + ", ".join(CATEGORICAL))
else:
    st.caption("Kolom kategori terdeteksi: " + ", ".join(available_cats))

# ==== PREVIEW ====
st.markdown("### Preview (first 12 rows)")
st.dataframe(data.head(12), height=260, use_container_width=True)
st.markdown("---")

# ==== OVERVIEW: METRICS CEPAT ====
st.subheader("📊 Quick Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(data):,}")
col2.metric("Mean Price", f"{data['Price'].mean():,.0f}")
col3.metric("Median Price", f"{data['Price'].median():,.0f}")
col4.metric("Std Price", f"{data['Price'].std():,.0f}")

# ==== PILIH KATEGORI & TAMPILKAN RINGKASAN ====
st.markdown("---")
st.subheader("🏷️ Category Averages (Price)")

if available_cats:
    opt = st.selectbox("Choose a category:", available_cats, index=0)

    # rata-rata Price per kategori
    avg_df = (
        data.groupby(opt, dropna=True)[["Price"]]
        .mean()
        .reset_index()
        .sort_values("Price", ascending=False)
    )

    # jumlah data per kategori (opsional untuk konteks)
    cnt_df = (
        data.groupby(opt, dropna=True)[["Price"]]
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    c1, c2 = st.columns([2,1])
    with c1:
        st.write(f"**Average Price by {opt}**")
        st.bar_chart(data=avg_df, x=opt, y="Price", use_container_width=True)
    with c2:
        st.write(f"**Top {opt} by Count**")
        st.dataframe(cnt_df.head(20), use_container_width=True, height=360)

    # tabel lengkap rata-rata
    st.write("### Tabel Rata-rata Price per Kategori")
    st.dataframe(avg_df, use_container_width=True, height=360)

    st.download_button(
        "💾 Download Average per Kategori (CSV)",
        avg_df.to_csv(index=False),
        file_name=f"avg_price_by_{opt}.csv",
        mime="text/csv",
    )
else:
    st.info("Tambahkan kolom kategori (mis. Type/Region/Make/…) untuk melihat ringkasan per kategori.")
