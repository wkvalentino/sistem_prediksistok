# 📦 Sistem Prediksi Stok

Aplikasi prediksi stok barang berbasis **Machine Learning** yang dikembangkan menggunakan Python dan Streamlit. Proyek ini bertujuan membantu analisis persediaan barang melalui model prediksi berbasis data retail.

## 🚀 Live Demo

### [🌐 Buka Sistem Prediksi Stok](https://sistemprediksistok-anob5kqwjmpfncdrb5ejek.streamlit.app/)

Aplikasi dapat diakses secara online melalui browser tanpa perlu instalasi lokal.

## 🎯 Tujuan Proyek

- Menerapkan Machine Learning untuk analisis dan prediksi stok barang.
- Membantu memahami pola persediaan berdasarkan data retail.
- Menyajikan hasil prediksi melalui antarmuka web interaktif.

## ✨ Fitur

- Antarmuka aplikasi berbasis Streamlit.
- Pemrosesan input untuk kebutuhan prediksi stok.
- Pemanfaatan model Machine Learning berbasis XGBoost.
- Pemuatan model dan konfigurasi kolom fitur dari file lokal.

## 🛠️ Teknologi

- **Python** — bahasa pemrograman.
- **Streamlit** — framework aplikasi web.
- **XGBoost** — algoritma Machine Learning.
- **Joblib** — penyimpanan dan pemuatan model.
- **JSON** — format penyimpanan data model.

## 📂 Struktur Proyek

```text
sistem_prediksistok/
├── app.py
├── requirements.txt
├── train_columns.joblib
├── xgboost_retail_model.joblib
└── xgboost_retail_model.json
```

### 📄 Deskripsi File

| File | Keterangan |
|---|---|
| `app.py` | Kode utama aplikasi Streamlit. |
| `requirements.txt` | Daftar dependensi Python. |
| `train_columns.joblib` | Penyimpanan informasi kolom fitur pelatihan. |
| `xgboost_retail_model.joblib` | File model yang disimpan menggunakan Joblib. |
| `xgboost_retail_model.json` | File model dalam format JSON yang didukung XGBoost. |

## ⚙️ Instalasi dan Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/USERNAME/NAMA-REPOSITORY.git
```

### 2. Masuk ke direktori proyek

```bash
cd sistem_prediksistok
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan di alamat lokal yang ditampilkan oleh Streamlit.

## ☁️ Deployment

Aplikasi di-deploy menggunakan Streamlit Community Cloud.

**Link aplikasi:**  
https://sistemprediksistok-anob5kqwjmpfncdrb5ejek.streamlit.app/

## 👨‍💻 Pengembangan

Proyek ini merupakan implementasi pembelajaran Machine Learning dan pengembangan aplikasi web untuk mendukung analisis persediaan barang.

---

⭐ Terima kasih telah mengunjungi proyek ini!
