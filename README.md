# FakeShield Dashboard

> Aplikasi analisis dan deteksi berita hoaks berbahasa Indonesia berbasis Deep Learning

FakeShield mengklasifikasikan berita Indonesia sebagai **Hoaks** atau **Valid** menggunakan model **BiLSTM + Bahdanau Attention**, dilengkapi dashboard eksplorasi dataset interaktif.

**Live Demo:** [fakeshield.streamlit.app](https://fakeshield.streamlit.app)

---

## Fitur

### Dashboard Analisis Dataset
Eksplorasi interaktif dataset berita Indonesia yang telah dibersihkan dan diseimbangkan:
- Distribusi berita Hoaks vs Valid
- Analisis kategori berita
- Pengaruh panjang teks terhadap klasifikasi
- Analisis kata kunci dengan TF-IDF
- Tren publikasi berita periode 2024–2026

### Demo Prediksi
Input teks berita dan dapatkan:
- Klasifikasi **Hoaks** / **Valid**
- Confidence score
- Visualisasi kata berpengaruh via Bahdanau Attention

---

## Struktur Proyek

```
FakeShield/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset_fakeshield_v2_CLEANED_BALANCED 37K.csv
└── models/
    ├── tokenizer (5).json
    ├── scaler (2).pkl
    └── fakeshield_model (6).keras/
        ├── config.json
        ├── metadata.json
        └── model.weights.h5
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka di browser: `http://localhost:8501`

> **Catatan:** Pastikan folder `models/` dan `data/` sudah ada sesuai struktur di atas sebelum menjalankan aplikasi.

---

## Arsitektur Model

| Layer | Detail |
|---|---|
| Embedding | 50.000 vocab, dim 256 |
| Encoder | BiLSTM (128 units per arah) |
| Attention | Bahdanau Additive Attention |
| Output | Dense + Sigmoid (biner) |

**Fitur input:**
- Teks berita (tokenized)
- Jumlah kata
- Rasio huruf kapital
- Jumlah tanda seru & tanda tanya

---

## Dataset

| Properti | Nilai |
|---|---|
| Total artikel | 37.402 |
| Periode | 2024–2026 |
| Bahasa | Indonesia |
| Label | Hoaks & Valid (balanced) |