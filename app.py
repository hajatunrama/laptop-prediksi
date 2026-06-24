import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import pickle

# Konfigurasi Halaman Web
st.set_page_config(page_title="IUL Store ML Dashboard", page_icon="💻", layout="wide")

@st.cache_resource
def load_data():
    pipe = pickle.load(open('pipe.pkl', 'rb'))
 #   df = pickle.load(open('df.pkl', 'rb'))
    return pipe, df

pipe, df = load_data()

# ====== MEMBUAT SISTEM NAVIGASI TABS ======
tab1, tab2 = st.tabs(["💻 Kalkulator Harga AI", "📊 Analisis & Visualisasi Data"])

# ==========================================================
#                   TAB 1: KALKULATOR UTAMA
# ==========================================================
with tab1:
    st.title("🤖 Prediksi Harga Laptop & PC")
    st.markdown("Masukkan spesifikasi incaranmu, AI kami akan menebak harga pasarnya di Indonesia.")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox('Pilih Brand / Merek', sorted(df['Brand'].unique()))
        ram_size = st.selectbox('Kapasitas RAM (GB)', sorted(df['RAM_Size'].unique()))
        storage = st.selectbox('Kapasitas Storage / SSD (GB)', sorted(df['Storage_Capacity'].unique()))

    with col2:
        processor_speed = st.slider('Kecepatan Prosesor (GHz)', 1.0, 5.0, 2.5, 0.1)
        screen_size = st.slider('Ukuran Layar (Inch)', 10.0, 18.0, 14.0, 0.1)
        weight = st.slider('Berat Laptop (Kg)', 1.0, 6.0, 2.0, 0.1)

    if st.button('🚀 Hitung Estimasi Harga', use_container_width=True):
        query = pd.DataFrame([[brand, processor_speed, ram_size, storage, screen_size, weight]], 
                             columns=['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight'])
        
        prediction = pipe.predict(query)[0]
        harga_rupiah = prediction * 750  # Jalur Kalibrasi Pasar Lokal Indonesia
        harga_format = f"Rp {harga_rupiah:,.0f}".replace(',', '.')
        
        st.success("Perhitungan Selesai!")
        st.metric(label="Estimasi Harga di Pasaran:", value=harga_format)

# ==========================================================
#               TAB 2: DASHBOARD VISUALISASI
# ==========================================================
with tab2:
    st.title("📊 Dashboard Business Intelligence")
    st.markdown("Grafik di bawah ini di-render secara *real-time* dari dataset **Laptop_price.csv** yang dipelajari oleh model AI.")

    # --- GRAFIK 1: BARPLOT BRAND VS HARGA ---
    st.subheader("1. Persepsi Harga Pasar Berdasarkan Merek (Brand)")
    
    fig_brand, ax_brand = plt.subplots(figsize=(10, 4))
    # Kita kalikan Price dengan 750 agar grafik bar-nya langsung berwujud nominal Rupiah
    harga_df_rupiah = df['Price'] * 750 
    
    sns.barplot(x=df['Brand'], y=harga_df_rupiah, palette='viridis', ax=ax_brand)
    ax_brand.set_ylabel("Harga (Rupiah)")
    ax_brand.set_xlabel("Merek Laptop")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    st.pyplot(fig_brand) # Perintah krusial memunculkan grafik matplotlib ke web Streamlit

    st.divider() # Garis pembatas horizontal yang cantik

    # --- GRAFIK 2: HEATMAP KORELASI ---
    st.subheader("2. Matriks Korelasi (Faktor Penentu Utama Mahalnya Laptop)")
    st.markdown("*Semakin angka di dalam kotak mendekati **1.0**, artinya spesifikasi tersebut semakin membuat harga laptop melambung tinggi.*")
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    kolom_numerik = df.select_dtypes(include=[np.number])
    
    sns.heatmap(kolom_numerik.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax_corr)
    
    st.pyplot(fig_corr)
