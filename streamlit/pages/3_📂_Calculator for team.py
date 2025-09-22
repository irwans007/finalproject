import io
import pandas as pd
import streamlit as st
import joblib
import requests

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="https://raw.githubusercontent.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/refs/heads/main/Streamlit/favicon.ico",
    layout="wide"
)

st.title("🚘 Predict Used Car Price for Batch Data")

# ====== KONFIGURASI ======
MODEL_URL = "https://raw.githubusercontent.com/irwans007/finalproject/main/best_catboost_pipeline.joblib"

# Urutan fitur FINAL yang dipakai model (sesuai CSV kamu)
FEATURE_ORDER = [
    "Make","Type","Year","Origin","Color","Options",
    "Engine_Size","Fuel_Type","Gear_Type","Mileage","Region"
]
CAT_COLS = ["Make","Type","Origin","Color","Options","Fuel_Type","Gear_Type","Region"]
NUM_COLS = ["Year","Engine_Size","Mileage"]

# ====== UTIL ======
@st.cache_resource(show_spinner="📥 Downloading model from GitHub (RAW)...")
def _download_model_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    head = r.content[:200].lower()
    if head.startswith(b"<!doctype html") or b"<html" in head:
        raise RuntimeError("URL tampaknya bukan RAW. Pakai raw.githubusercontent.com atau tambah ?raw=1.")
    return r.content

@st.cache_resource(show_spinner="🧠 Loading model into memory...")
def _load_model_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))

def _prepare_df_exact(df: pd.DataFrame) -> pd.DataFrame:
    # Ambil hanya kolom fitur yang diperlukan & urutkan persis
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset kurang kolom: {', '.join(missing)}")

    X = df[FEATURE_ORDER].copy()

    # Set tipe konsisten
    for c in CAT_COLS:
        X[c] = X[c].astype("string")
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    return X

def _predict_any(model, X: pd.DataFrame):
    # 1) coba pipeline sklearn (jika ada preprocessing di dalam model)
    try:
        return model.predict(X)
    except Exception:
        pass
    # 2) fallback: CatBoost murni, gunakan Pool dengan NAMA kolom kategori
    try:
        from catboost import Pool
    except ImportError as ie:
        raise RuntimeError("Model kemungkinan CatBoost murni. Install: pip install catboost") from ie
    pool = Pool(X, cat_features=[c for c in X.columns if c in CAT_COLS],
                feature_names=list(X.columns))
    return model.predict(pool)

# ====== UI UPLOAD ======
uploaded_file = st.sidebar.file_uploader(
    label="Upload your file",
    type=["csv"],
    help="Upload file format .csv only."
)

if uploaded_file is None:
    st.info("👈 Please upload your CSV file to start.")
    st.stop()

st.sidebar.success(f"File '{uploaded_file.name}' successfully uploaded!")
st.write("Here are the preview of your data:")

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading file: {e}")
    st.stop()

st.dataframe(data.head(10), height=220)

# Validasi kolom minimal
missing_cols = [c for c in FEATURE_ORDER if c not in data.columns]
if missing_cols:
    st.error(f"⚠️ Your dataset is missing required columns: {', '.join(missing_cols)}")
    st.stop()
else:
    st.success(f"✅ Dataset valid! Total rows: {data.shape[0]}")

# ====== PREDICT ======
if st.button("🚀 Predict the Price"):
    try:
        model_bytes = _download_model_bytes(MODEL_URL)
        model = _load_model_from_bytes(model_bytes)

        # (Opsional) tampilkan info model
        with st.expander("🔎 Model details"):
            st.write("Model class:", type(model).__name__)
            st.write("Feature order (used for predict):", FEATURE_ORDER)

        X = _prepare_df_exact(data)
        preds = _predict_any(model, X)

        out = data.copy()
        out["Prediction"] = pd.Series(preds).round().astype(int)

        st.subheader("🔮 Prediction Result")
        st.dataframe(out, height=320)

        csv = out.to_csv(index=False)
        st.download_button(
            label="💾 Download Prediction Result",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
