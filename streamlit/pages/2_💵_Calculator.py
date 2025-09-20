import pandas as pd
import streamlit as st
import pickle
import requests

st.title(" Predict Used Car Price")

# -------- form builder --------
def user_input_features():
    df = pd.read_csv('UsedCarsSA_Clean_EN.csv')

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

     # === Color & Fuel
    color_vals = sorted(df["Color"].dropna().unique().tolist())
    Color = st.selectbox("Select Color", color_vals)

    fuel_vals = sorted(df["Fuel_Type"].dropna().unique().tolist())
    Fuel_Type = st.selectbox("Select Fuel Type", fuel_vals)

    Engine_Size = st.number_input("Fill Engine Size", min_value=1.0, max_value=9.0, step=0.1, value=5.0)
    Year        = st.number_input("Fill Year (2003 - 2021)", min_value=2003, max_value=2021, step=1, value=2010)
    Mileage     = st.number_input("Fill Mileage (in KM per hour)", min_value=0, max_value=376000, step=100, value=0)

    df_new = pd.DataFrame({
        "Make": [Make],
        "Type": [Type],
        "Year": [Year],
        "Origin": [Origin],
        "Color": [Color],
        "Options": [Options],
        "Engine_Size": [Engine_Size],
        "Fuel_Type": [Fuel_Type],
        "Gear_Type": [Gear_Type],
        "Mileage": [Mileage],
        "Region": [Region],
    })
    return df_new

# ====== layout  (3 kolom), ======
col1, col2, col3 = st.columns([10, 1, 4])

with col1:
    st.write("Fill the Detail")
    df_customer = user_input_features()

    # tombol predict 
    do_predict = st.button("Predict")

    price_val = None
    err_msg = None

    if do_predict:
        try:
            # Load model dari GitHub (JANGAN load file lokal .sav)
            url_model = "https://github.com/irwans007/finalproject//raw/refs/heads/main/model1.sav"
            resp = requests.get(url_model, timeout=30)
            resp.raise_for_status()
            model_loaded = pickle.loads(resp.content)

            # Prediksi
            price = model_loaded.predict(df_customer)
            price_val = float(price[0])

        except Exception as e:
            err_msg = f"Gagal memuat model atau melakukan prediksi: {e}"

with col2:
    st.write("")

with col3:
    st.write("Final Prediction")
    if "do_predict" in locals() and do_predict:
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
