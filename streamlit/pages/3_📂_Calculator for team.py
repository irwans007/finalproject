import pandas as pd
import streamlit as st
import pickle
import requests

st.set_page_config(
    page_title="Syarah.com Car Price Machine Learning",
    page_icon="https://raw.githubusercontent.com/glenvj-j/Saudi-Arabia-Used-Car-Regression-Prediction/refs/heads/main/Streamlit/favicon.ico",
    layout="wide")

# Judul
st.title("🚘 Predict Used Car Price for Batch Data")

# Upload file
uploaded_file = st.sidebar.file_uploader(
    label="Upload your file", 
    type=["csv"],  
    help="Upload file format .csv only."
)

if uploaded_file is not None:
    st.sidebar.success(f"File '{uploaded_file.name}' successfully uploaded!")
    st.write("Here are the preview of your data:")
    
    try:
        # Baca CSV apa adanya (tanpa slicing kolom)
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head(10), height=200)  # preview
        
        # Kolom wajib
        required_columns = ['Type','Region','Make','Gear_Type','Origin','Options','Year','Engine_Size','Mileage']
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            st.error(f"⚠️ Your dataset is missing required columns: {', '.join(missing_cols)}")
        else:
            st.success(f"✅ Dataset valid! Total rows: {data.shape[0]}")

            if st.button("🚀 Predict the Price"):
                # Load model ML dari GitHub
                url = "https://github.com/irwans007/finalproject//raw/refs/heads/main/model1.sav"
                response = requests.get(url)
                model = pickle.loads(response.content)

                # Prediksi
                predictions = model.predict(data[required_columns])
                data['Prediction'] = predictions.round().astype(int)

                st.subheader("🔮 Prediction Result")
                st.dataframe(data, height=300)  # tampilkan semua data + prediksi

                # Download hasil
                csv = data.to_csv(index=False)
                st.download_button(
                    label="💾 Download Prediction Result",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload your CSV file to start.")
