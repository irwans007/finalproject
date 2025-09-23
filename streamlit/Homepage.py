import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Syarah.com Machine Learning",
    page_icon="🚗"
)


# Override CSS bawaan supaya konten nempel ke kiri
st.markdown("""
    <style>
        /* Atur lebar container */
        .block-container {
            max-width: 100%;
            padding-left: 2rem;   /* bisa diganti 0rem kalau mau nempel banget */
            padding-right: 2rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# Judul utama
st.title("Welcome to Saudi Arabian Used Car's Price Prediction tools")
st.sidebar.success("Select a page above.")

# Deskripsi singkat
st.write("In this page you can predict a price for used car by filling the specification of the car.")


# --- Bagian Fitur ditampilkan secara baris ---
st.subheader("💡 Informed Decisions")
st.write("Make every decision with confidence using clear, data-backed insights designed for the Saudi Arabian car market.")

st.subheader("🚗 Car Price Analysis")
st.write("Gain a deeper understanding of your car’s market value with clear, data-driven insights. We analyze multiple factors.")

st.subheader("⚙️ Powered by CatBoost")
st.write("Our tool learns from thousands of data points to deliver precise and reliable pricing recommendations.")

st.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)


# Link tambahan
st.markdown("For detail how the model works you can visit : [Click Here](https://github.com/irwans007/finalproject/)

