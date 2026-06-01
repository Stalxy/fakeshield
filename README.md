# FakeShield Dashboard

## Dashboard Analisis dan Deteksi Berita Hoaks Berbahasa Indonesia

FakeShield Dashboard aplikasi berbasis Streamlit yang digunakan untuk menganalisis karakteristik berita berbahasa Indonesia serta melakukan klasifikasi berita ke dalam kategori **Hoaks** atau **Valid** menggunakan model Deep Learning **BiLSTM dengan Bahdanau Attention**.

---

## Struktur Proyek

```text
FakeShield/
├── data/
│   └── dataset_fakeshield_v2_CLEANED_BALANCED 37K.csv
│
├── models/
│   ├── tokenizer (5).json
│   ├── scaler (2).pkl
│   └── fakeshield_model (6).keras/
│       ├── config.json
│       ├── metadata.json
│       └── model.weights.h5
│
├── app.py
├── requirements.txt
└── README.md

```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan Dataset Tersedia

```text
data/
└── dataset_fakeshield_v2_CLEANED_BALANCED 37K.csv
```

### 3. Pastikan File Model Tersedia

Struktur folder `models/` harus :

```text
models/
├── tokenizer (5).json
├── scaler (2).pkl
│
└── fakeshield_model (6).keras/
    ├── config.json
    ├── metadata.json
    └── model.weights.h5
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan berjalan pada:

```text
http://localhost:8501
```

## Fitur Utama

### Dashboard Analisis Dataset

Dashboard interaktif untuk mengeksplorasi dataset berita Indonesia yang telah melalui proses pembersihan dan penyeimbangan data.

Visualisasi yang tersedia:

- Distribusi berita Hoaks dan Valid
- Analisis kategori berita
- Pengaruh panjang teks terhadap klasifikasi
- Analisis kata kunci menggunakan TF-IDF
- Tren publikasi berita periode 2024–2026

### Demo Prediksi

Pengguna dapat memasukkan teks berita secara langsung dan sistem akan:

- Mengklasifikasikan berita sebagai Hoaks atau Valid
- Menampilkan confidence score
- Menampilkan kata-kata yang paling berpengaruh terhadap prediksi menggunakan mekanisme Bahdanau Attention

---

## Arsitektur Model

Model yang digunakan:

- Embedding Layer
- BiLSTM
- Bahdanau Attention
- Dense Layer
- Sigmoid Output Layer

Model memanfaatkan kombinasi:

### Fitur Teks

- Isi berita

### Fitur Numerik

- Jumlah kata
- Rasio huruf kapital
- Jumlah tanda seru
- Jumlah tanda tanya