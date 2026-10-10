import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import time

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Retail Demand & Inventory Predictor",
    page_icon="📦",
    layout="centered"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main-header {
        font-size: 24px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 13px;
        color: #6B7280;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model XGBoost (.json) dan Kolom Training (.joblib)
@st.cache_resource
def load_model_and_columns():
    model = xgb.XGBRegressor()
    model.load_model('xgboost_retail_model.json')
    
    if os.path.exists('train_columns.joblib'):
        columns = joblib.load('train_columns.joblib')
    else:
        columns = None
    return model, columns

try:
    best_model, train_columns = load_model_and_columns()
except Exception as e:
    st.error(f"Gagal memuat file model. Pastikan file 'xgboost_retail_model.json' ada di folder yang sama! Error: {e}")
    st.stop()

# ==========================================
# PEMBUATAN SIDEBAR DI KIRI
# ==========================================
with st.sidebar:
    st.title("📌 Navigasi & Info")
    st.markdown("---")
    
    st.markdown("### ℹ️ Tentang Sistem")
    st.info(
        "Aplikasi ini menggunakan algoritma **XGBoost Machine Learning** "
        "untuk memprediksi permintaan produk ritel dan mendeteksi risiko "
        "kehabisan stok secara akurat."
    )
    
    st.markdown("### 💡 Panduan Input")
    st.markdown("""
    - **Units Sold:** Jumlah barang yang terjual kemarin.
    - **Inventory Level:** Sisa stok fisik nyata di rak/gudang saat ini.
    - **Units Ordered:** Pesanan baru ke supplier.
    """)
    
    st.markdown("---")
    st.markdown("🛠️ *Minimarket Retail Inventory System*")

# ==========================================
# KONTEN UTAMA HALAMAN (MAIN APP)
# ==========================================
st.markdown('<div class="main-header">📦 Sistem Prediksi Permintaan & Risiko Stok</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dashboard Analisis Operasional Ritel Berbasis XGBoost</div>', unsafe_allow_html=True)

st.markdown("### 📝 Masukkan Data Operasional Harian")

with st.form("main_form"):
    date_input = st.date_input("📅 1. Pilih Tanggal Transaksi")
    
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("📦 2. Kategori Produk", ['Groceries', 'Furniture', 'Electronics', 'Toys', 'F&B', 'Personal Care', 'Household'])
        units_sold_input = st.number_input("📉 3. Units Sold (Terjual Kemarin)", min_value=0, value=None, step=1, format="%d", placeholder="Contoh: 10")
        inventory_level_input = st.number_input("🏬 4. Inventory Level (Sisa Stok Saat Ini)", min_value=0, value=None, step=1, format="%d", placeholder="Contoh:50")
        
    with col2:
        units_ordered_input = st.number_input("📥 5. Units Ordered (Pesanan Masuk)", min_value=0, value=None, step=1, format="%d", placeholder="Contoh: 25")
        price_input = st.number_input("🏷️ 6. Harga Satuan (Rp)", min_value=0, value=None, step=500, format="%d", placeholder="Contoh: 50000")
        discount = st.slider("🔖 7. Besaran Diskon", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.2f")

    submitted = st.form_submit_button("🚀 Jalankan Analisis Prediksi & Risiko Stok", use_container_width=True)

# Logika Setelah Tombol Ditekan
if submitted:
    units_sold = float(units_sold_input) if units_sold_input is not None else 0.0
    inventory_level = float(inventory_level_input) if inventory_level_input is not None else 0.0
    units_ordered = float(units_ordered_input) if units_ordered_input is not None else 0.0
    price = float(price_input) if price_input is not None else 0.0

    with st.spinner('🔄 Menganalisis parameter model dan menghitung risiko stok...'):
        time.sleep(0.6)

        dt = pd.to_datetime(date_input)

        # Hitung Stock_Change secara otomatis di balik layar untuk model XGBoost
        # Rumus: Stock_Change = Inventory_Level - Units_Sold
        stock_change = inventory_level - units_sold
        sold_percent = units_sold / (units_ordered + 1e-5)

        # Bungkus ke DataFrame sesuai struktur fitur model
        input_data = pd.DataFrame({
            'Date': [dt],
            'Category': [category],
            'Price': [price],
            'Units Sold': [units_sold],
            'Stock_Change': [stock_change],
            'Inventory Level': [inventory_level],
            'Units Ordered': [units_ordered],
            'Sold_Percent': [sold_percent],
            'Discount': [discount],
            'Demand_Lag_1': [units_sold],
            'Demand_Lag_7': [units_sold],
            'Demand_Rolling_Mean_7': [units_sold]
        })

        # Fitur pendukung otomatis
        input_data['Revenue'] = input_data['Units Sold'] * input_data['Price']
        input_data['Is_Discounted'] = np.where(input_data['Discount'] > 0, 1, 0)
        input_data['DayOfWeek'] = input_data['Date'].dt.dayofweek
        input_data['Month'] = input_data['Date'].dt.month
        input_data['Is_Weekend'] = input_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # One-Hot Encoding & Penyelarasan Kolom
        input_encoded = pd.get_dummies(input_data.drop(columns=['Date']), drop_first=True)
        
        if train_columns is not None:
            input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0).astype(float)
        
        X_input_array = input_encoded.to_numpy()

        # Prediksi Demand
        raw_prediction = best_model.predict(X_input_array)[0]
        predicted_demand = round(max(0, raw_prediction))

        # Logika Risiko Stok
        risk_status = 1 if inventory_level < predicted_demand else 0

    # ==========================================
    # TAMPILKAN HASIL DASHBOARD
    # ==========================================
    st.divider()
    st.markdown("### 📊 Hasil Analisis & Keputusan Inventaris")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="🎯 Prediksi Permintaan (Demand)", value=f"{predicted_demand} Unit")
    with col_m2:
        st.metric(label="🏬 Sisa Stok Saat Ini", value=f"{inventory_level:.0f} Unit")

    with st.expander("🔍 Lihat Detail Parameter Input"):
        st.markdown(f"""
        - **Tanggal Transaksi:** `{date_input.strftime('%Y-%m-%d')}`
        - **Kategori Produk:** `{category}`
        - **Harga Satuan:** `Rp {price:,.0f}`
        - **Diskon Aktif:** `{discount * 100:.1f}%`
        - **Units Sold:** `{units_sold:.0f}`
        - **Inventory Level:** `{inventory_level:.0f}`
        - **Stock_Change (Dihitung Otomatis):** `{stock_change:.0f}`
        - **Units Ordered:** `{units_ordered:.0f}`
        """)

    st.markdown("---")
    
    if risk_status == 1:
        st.error("🚨 **WARNING: Potensi Kehabisan Stok!**\n\nStok gudang saat ini lebih rendah daripada prediksi permintaan konsumen. **Segera lakukan restock ke supplier**.")
    else:
        st.success("✅ **SAFE: Stok Aman!**\n\nKetersediaan inventaris di gudang saat ini masih mencukupi untuk memenuhi perkiraan permintaan konsumen.")