# pages/01_Data_Overview.py
import io
import streamlit as st
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# Call this before any other Streamlit commands
st.set_page_config(page_title="📈 Data Overview", page_icon="📈", layout="wide")
st.title("📈 Data Overview (Only)")

st.write(
    "Upload CSV → lihat ringkasan & rata-rata **Price** per kategori. "
    )

# ────────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
FEATURE_COLUMNS = [
    "Type", "Region", "Make", "Gear_Type", "Origin", "Options",
    "Color", "Fuel_Type",
    "Year", "Engine_Size", "Mileage",
]
CATEGORICAL = [
    "Type", "Region", "Make", "Gear_Type", "Origin", "Options",
    "Color", "Fuel_Type",
]

# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
@st.cache_data(show_spinner=False)
def read_csv_safely(file) -> pd.DataFrame:
    """Read CSV robustly, normalize headers, coerce Price to numeric."""
    if isinstance(file, (bytes, bytearray)):
        buffer = io.BytesIO(file)
    else:
        buffer = file

    df = pd.read_csv(buffer, low_memory=False)

    # Normalisasi nama kolom (trim spasi)
    df.columns = [c.strip() for c in df.columns]

    # Harga → numeric (handle koma pemisah ribuan/desimal)
    if "Price" in df.columns:
        # Hilangkan karakter non-digit kecuali tanda titik/koma
        # Lalu ganti koma menjadi titik jika tampak seperti desimal
        s = df["Price"].astype(str).str.replace(r"[^0-9,.-]", "", regex=True)
        # Jika ada koma dan titik, coba heuristik: ubah pemisah ribuan
        # Contoh: 1.234.567,89 → 1234567.89
        s = s.str.replace(r"\.(?=\d{3}(\D|$))", "", regex=True)
        s = s.str.replace(",", ".", regex=False)
        df["Price"] = pd.to_numeric(s, errors="coerce")

    return df


def summarize_by_category(df: pd.DataFrame, cat_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    avg_df = (
        df.groupby(cat_col, dropna=True)[["Price"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("Price", ascending=False)
    )
    cnt_df = (
        df.groupby(cat_col, dropna=True)[["Price"]]
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    return avg_df, cnt_df

# ────────────────────────────────────────────────────────────────────────────────
# DATA SOURCE
st.sidebar.header("Upload CSV (untuk halaman ini)")
uploaded = st.sidebar.file_uploader("Choose file (.csv)", type=["csv"])

# Kamu juga bisa pakai data dari halaman lain via session_state:
# st.session_state["data_df"] = df
_df_session = st.session_state.get("data_df")

if uploaded is not None:
    try:
        data = read_csv_safely(uploaded)
        st.success(f"File `{uploaded.name}` loaded. Rows: {len(data):,}")
    except Exception as e:
        st.error(f"Gagal baca CSV: {e}")
        st.stop()
elif _df_session is not None:
    data = _df_session.copy()
    st.info("Pakai data dari session (halaman lain).")
else:
    st.info("👈 Upload CSV dulu ")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# VALIDASI MINIMAL
if "Price" not in data.columns:
    st.error("Kolom **Price** wajib ada untuk membuat ringkasan.")
    st.stop()

# Drop baris tanpa Price atau Price non-positif (opsional)
data = data.copy()
price_na = data["Price"].isna().sum()
if price_na:
    st.warning(f"{price_na:,} baris memiliki Price tidak valid dan akan diabaikan.")

data = data.loc[data["Price"].notna()]

available_cats = [c for c in CATEGORICAL if c in data.columns]
if not available_cats:
    st.warning(
        "Tidak ada kolom kategori standar yang ditemukan. "
        "Tambahkan salah satu dari: " + ", ".join(CATEGORICAL)
    )
else:
    st.caption("Kolom kategori terdeteksi: " + ", ".join(available_cats))

# ────────────────────────────────────────────────────────────────────────────────
# PREVIEW
st.markdown("### Preview (first 12 rows)")
st.dataframe(data.head(12), height=260, use_container_width=True)
st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────────
# OVERVIEW: METRICS CEPAT
st.subheader("📊 Quick Stats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(data):,}")
col2.metric("Mean Price", f"{data['Price'].mean():,.0f}")
col3.metric("Median Price", f"{data['Price'].median():,.0f}")
std_val = float(np.nan_to_num(data["Price"].std(ddof=1)))
col4.metric("Std Price", f"{std_val:,.0f}")

# ────────────────────────────────────────────────────────────────────────────────
# PILIH KATEGORI & TAMPILKAN RINGKASAN
st.markdown("---")
st.subheader("🏷️ Category Averages (Price)")

if available_cats:
    opt = st.selectbox("Choose a category:", available_cats, index=0)

    # Batasi jumlah kategori yang divisualisasikan agar chart tetap ringan
    nunique = data[opt].nunique(dropna=True)
    if nunique > 200:
        st.info(
            f"Kolom **{opt}** memiliki {nunique:,} kategori. "
            "Grafik akan menampilkan 200 kategori teratas berdasarkan rata-rata Price."
        )

    avg_df, cnt_df = summarize_by_category(data, opt)

    # Untuk chart, batasi top-N agar tetap terbaca
    top_n = 200 if len(avg_df) > 200 else len(avg_df)
    avg_for_chart = avg_df.head(top_n)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(f"**Average Price by {opt}**")
        st.bar_chart(data=avg_for_chart, x=opt, y="Price", use_container_width=True)
    with c2:
        st.write(f"**Top {opt} by Count**")
        st.dataframe(cnt_df.head(20), use_container_width=True, height=360)

    # Tabel lengkap rata-rata
    st.write("### Tabel Rata-rata Price per Kategori")
    st.dataframe(avg_df, use_container_width=True, height=360)

    # Download CSV (pastikan bytes, bukan string)
    csv_bytes = avg_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Average per Kategori (CSV)",
        csv_bytes,
        file_name=f"avg_price_by_{opt}.csv",
        mime="text/csv",
    )
else:
    st.info("Tambahkan kolom kategori (mis. Type/Region/Make/…) untuk melihat ringkasan per kategori.")

