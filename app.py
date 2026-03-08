import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
st.write("Daftar file yang terdeteksi di server:", os.listdir())

# 1. Load Model dan Scaler
model = joblib.load('diamond_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Judul Aplikasi
st.title("💎 Diamond Price Predictor")
st.write("Aplikasi ini memprediksi harga berlian berdasarkan karakteristik fisiknya.")

# 2. Input dari User (Sidebar atau Kolom)
st.header("Masukkan Karakteristik Berlian:")

col1, col2, col3 = st.columns(3)

with col1:
    carat = st.number_input("Carat (Berat)", min_value=0.1, max_value=5.0, value=0.7)
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.0)
    table = st.number_input("Table %", min_value=40.0, max_value=80.0, value=57.0)

with col2:
    # Menggunakan pilihan kategori (Label Encoding harus sesuai dengan urutan saat training)
    # Catatan: Urutan LabelEncoder biasanya alfabetis
    cut = st.selectbox("Cut (Kualitas Potongan)", ['Fair', 'Good', 'Ideal', 'Premium', 'Very Good'])
    color = st.selectbox("Color (Warna)", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox("Clarity (Kejernihan)", ['I1', 'IF', 'SI1', 'SI2', 'VS1', 'VS2', 'VVS1', 'VVS2'])

with col3:
    x = st.number_input("Panjang (x) mm", min_value=0.1, max_value=12.0, value=5.5)
    y = st.number_input("Lebar (y) mm", min_value=0.1, max_value=12.0, value=5.5)
    z = st.number_input("Tinggi (z) mm", min_value=0.1, max_value=10.0, value=3.5)

# 3. Mapping Kategori ke Angka (Sesuai LabelEncoder di Colab)
cut_map = {'Fair': 0, 'Good': 1, 'Ideal': 2, 'Premium': 3, 'Very Good': 4}
color_map = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
clarity_map = {'I1': 0, 'IF': 1, 'SI1': 2, 'SI2': 3, 'VS1': 4, 'VS2': 5, 'VVS1': 6, 'VVS2': 7}

# Konversi input ke format dataframe
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut_map[cut]],
    'color': [color_map[color]],
    'clarity': [clarity_map[clarity]],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z]
})

# 4. Prediksi
if st.button("Prediksi Harga"):
    # Scaling input data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    # Tampilkan Hasil
    st.success(f"Estimasi Harga Berlian adalah: **${prediction[0]:,.2f}**")


st.info("Catatan: Prediksi didasarkan pada model Machine Learning yang telah dilatih.")
