import json, io, hashlib
from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import requests  # perlu untuk download dari GitHub

st.title(" Predict Used Car Price")

# ================== KONFIG ==================
# Raw URL model & CSV di GitHub (public)
MODEL_URL = "https://raw.githubusercontent.com/irwans007/finalproject/main/best_catboost_pipeline.joblib"
CSV_URL   = "https://raw.githubusercontent.com/irwans007/finalproject/refs/heads/main/streamlit/UsedCarsSA_Clean_EN.csv"

# (Opsional) verifikasi integritas file model; kosongkan jika tidak dipakai
EXPECTED_SHA256 = ""  # contoh: "3b5d5c3712955042212316173ccf37be..."

# ====== metadata fitur ======
NUM_COLS = ["Mileage", "Engine_Size", "Year"]
CAT_COLS = ["Make", "Type", "Origin", "Color", "Options", "Fuel_Type", "Gear_Type", "Region"]
ALL_COLS = CAT_COLS + NUM_COLS

FEATURES_JSON = Path("feature_names.json")
CATCOLS_JSON  = Path("cat_cols.json")

# ---------- UTIL HASH ----------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---------- CACHE: DATASET & MODEL ----------
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset_from_github(csv_url: str) -> pd.DataFrame:
    # pd.read_csv bisa langsung dari URL raw
    return pd.read_csv(csv_url)

@st.cache_resource(show_spinner="Downloading model from GitHub…")
def download_model_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    # Cek konten: pastikan bukan HTML (berarti salah URL)
    ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype.lower() and not url.endswith("?raw=1"):
        raise RuntimeError("URL bukan raw file. Gunakan raw.githubusercontent.com atau tambahkan ?raw=1.")
    return r.content

@st.cache_resource(show_spinner="Loading model into memory…")
def load_joblib_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))

# ---------- SCHEMA & PREDIKSI ----------
def _ensure_dataframe_schema(df_in: pd.DataFrame, feature_names, cat_cols):
    df_pred = df_in.reindex(columns=feature_names)
    for c in cat_cols:
        if c in df_pred.columns:
            df_pred[c] = df_pred[c].astype("string")
    for c in NUM_COLS:
        if c in df_pred.columns:
            df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce")
    return df_pred

def _predict_safely(model_obj, df_pred, feature_names, cat_cols):
    try:
        return model_obj.predict(df_pred)
    except Exception:
        # fallback CatBoost murni
        try:
            from catboost import Pool
        except ImportError as ie:
            raise RuntimeError("Model kemungkinan CatBoost. Install dulu: pip install catboost") from ie
        cat_in_df = [c for c in cat_cols if c in df_pred.columns]
        pool = Pool(df_pred, cat_features=cat_in_df, feature_names=feature_names)
        return model_obj.predict(pool)

# -------- UI: FORM INPUT --------
def user_input_features(df: pd.DataFrame) -> pd.DataFrame:
    # Make & Type (dependent)
    list_brand, list_type = [], []
    for brand in sorted(df["Make"].dropna().unique()):
        types = sorted(df.loc[df["Make"] == brand, "Type"].dropna().unique().tolist())
        list_brand.append(brand); list_type.append(types)
    df_brand_type = pd.DataFrame({"Make": list_brand, "Type": list_type})

    Make = st.selectbox("Select Make (Brand of Car)", options=df_brand_type["Make"].tolist())
    Type_allowed_values = df_brand_type[df_brand_type["Make"] == Make]["Type"].tolist()[0]
    Type = st.selectbox("Select Type", options=Type_allowed_values)

    # Origin & Region (dependent)
    list_Origin, list_Region = [], []
    for origin in sorted(df["Origin"].dropna().unique()):
        regions = sorted(df.loc[df["Origin"] == origin, "Region"].dropna().unique().tolist())
        list_Origin.append(origin); list_Region.append(regions)
    df_origin_region = pd.DataFrame({"Origin": list_Origin, "Region": list_Region})

    Origin = st.selectbox("Select Origin", options=df_origin_region["Origin"].tolist())
    Region_allowed_values = df_origin_region[df_origin_region["Origin"] == Origin]["Region"].tolist()[0]
    Region = st.selectbox("Select Region", options=Region_allowed_values)

    Gear_Type = st.radio("Choose Gear Type:", sorted(df["Gear_Type"].dropna().unique().tolist()), horizontal=True)
    Options   = st.radio("Choose Options:",   sorted(df["Options"].dropna().unique().tolist()),   horizontal=True)

    # Color & Fuel
    color_vals = sorted(df["Color"].dropna().unique().tolist())
    Color = st.selectbox("Select Color", color_vals)

    fuel_vals = sorted(df["Fuel_Type"].dropna().unique().tolist())
    Fuel_Type = st.selectbox("Select Fuel Type", fuel_vals)

    Engine_Size = st.number_input("Fill Engine Size", min_value=1.0, max_value=9.0, step=0.1, value=5.0)
    Year        = st.number_input("Fill Year (2003 - 2021)", min_value=2003, max_value=2021, step=1, value=2010)
    Mileage     = st.number_input("Fill Mileage (in KM per hour)", min_value=0, max_value=376000, step=100, value=0)

    df_new = pd.DataFrame({
        "Make": [Make], "Type": [Type], "Year": [Year], "Origin": [Origin],
        "Color": [Color], "Options": [Options], "Engine_Size": [Engine_Size],
        "Fuel_Type": [Fuel_Type], "Gear_Type": [Gear_Type], "Mileage": [Mileage],
        "Region": [Region],
    })
    return df_new

# ====== layout (3 kolom) ======
col1, col2, col3 = st.columns([10, 1, 4])

with col1:
    st.write("Fill the Detail")
    # Load dataset sekali, cached
    df_full = load_dataset_from_github(CSV_URL)
    df_customer = user_input_features(df_full)
    do_predict = st.button("Predict")

    price_val = None
    err_msg = None

# ====== PREDIKSI ======
if do_predict:
    try:
        # 1) Unduh bytes model (cached)
        model_bytes = download_model_bytes(MODEL_URL)

        # 2) (opsional) verifikasi SHA256
        if EXPECTED_SHA256:
            got = _sha256_bytes(model_bytes)
            if got.lower() != EXPECTED_SHA256.lower():
                raise RuntimeError(f"Hash mismatch. expected={EXPECTED_SHA256} got={got}")

        # 3) Load model dari bytes
        model_loaded = load_joblib_from_bytes(model_bytes)

        # 4) Baca metadata fitur jika ada; jika tidak, gunakan default
        feature_names = json.loads(FEATURES_JSON.read_text()) if FEATURES_JSON.exists() else ALL_COLS
        cat_cols      = json.loads(CATCOLS_JSON.read_text())  if CATCOLS_JSON.exists()  else CAT_COLS

        # 5) Siapkan data & prediksi
        df_pred = _ensure_dataframe_schema(df_customer, feature_names, cat_cols)
        price = _predict_safely(model_loaded, df_pred, feature_names, cat_cols)

        price_val = float(price[0])
        st.success("Model loaded from GitHub")

    except Exception as e:
        err_msg = f"Gagal memuat model atau melakukan prediksi: {e}"

with col2:
    st.write("")

with col3:
    st.write("Final Prediction")
    if do_predict:
        if err_msg:
            st.error(err_msg)
        elif price_val is not None:
            range_error = 18
            price_formated = f"{price_val:,.0f}"
            price_down = f"{price_val * (1 - range_error/100):,.0f}"
            price_up   = f"{price_val * (1 + range_error/100):,.0f}"

            st.title("SAR " + price_formated)
            st.markdown("---")
            st.write(f"Estimation (±{range_error}%)")
            st.write(f"SAR {price_down} - {price_up}")
        else:
            st.info("Silakan isi form lalu tekan tombol Predict.")
    else:
        st.caption("Isi form di kiri, lalu klik **Predict** untuk melihat hasil.")

