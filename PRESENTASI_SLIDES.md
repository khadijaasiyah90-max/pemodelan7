# Slide Presentasi Tesis

## Slide 1 — Judul
- Dynamic Topic Modeling & Stance Analysis untuk Media Sosial Politik dan Diplomasi
- Menggunakan BERTopic dan IndoBERT
- Nama Peneliti, Program Studi, Universitas

## Slide 2 — Latar Belakang
- Twitter sebagai sumber diskusi kebijakan luar negeri dan diplomasi
- Tantangan: topik kompleks dan komentar bernuansa
- Kebutuhan: analisis topik dinamis + deteksi keberpihakan konteks

## Slide 3 — Tujuan Penelitian
- Mengembangkan sistem untuk:
  - Mengidentifikasi topik adaptif pada data Twitter
  - Mendeteksi stance Pro/Kontra/Netral secara konteks
- Membandingkan performa dengan baseline analisis sentimen konvensional

## Slide 4 — Arsitektur Sistem
- Data Ingestion → Preprocessing → Topic Modeling → Stance Analysis → Visualisasi
- Framework utama: Python + Streamlit
- Model: BERTopic untuk topic modeling, IndoBERT untuk stance analysis

## Slide 5 — Metodologi
- Preprocessing teks dengan pipeline 8 tahap
- Embedding: SentenceTransformer all-MiniLM-L6-v2
- Clustering: HDBSCAN + UMAP
- Stance Detection: IndoBERT / tweet-roberta-base-sentiment

## Slide 6 — Baseline & Eksperimen
- Baseline: Analisis Sentimen Konvensional (Naive Bayes / SVM / leksikon)
- Target baseline: polaritas Positif/Negatif/Netral
- Sistem usulan: stance detection Pro/Kontra/Netral

## Slide 7 — Metrik Evaluasi
- Dynamic Topic Modeling: Coherence Score (C_v)
- Stance Analysis: Accuracy, Precision, Recall, F1-Score
- Validasi Pakar: relevansi topik dan akurasi konteks stance
- Indikator Keberhasilan: >10% perbaikan C_v vs LDA, ≥66% kesesuaian pakar

## Slide 8 — Hasil Utama
- BERTopic menghasilkan topik yang lebih koheren dibanding LDA
- IndoBERT mendeteksi keberpihakan kontekstual lebih baik daripada analisis sentimen sederhana
- Validasi pakar mendukung relevansi topik dan akurasi klasifikasi stance

## Slide 9 — Kontribusi Penelitian
- Menggabungkan topic modeling dinamis dan stance analysis pada isu diplomasi
- Menyediakan kerangka evaluasi kuantitatif dan kualitatif
- Memperkuat analisis media sosial politik dengan pendekatan domain-aware

## Slide 10 — Rekomendasi Lanjutan
- Kembangkan validasi pakar multi-annotator
- Tambahkan analisis entity-target untuk stance lebih detil
- Uji pada dataset isu luar negeri yang lebih besar dan multi-platform

## Slide 11 — Kesimpulan
- Sistem berhasil jika menunjukkan keunggulan C_v BERTopic dan akurasi stance IndoBERT
- Validasi pakar menjadi komponen kunci dalam konteks kebijakan luar negeri
- Model ini menawarkan nilai tambah untuk analisis opini politik dan diplomasi

## Slide 12 — Terima Kasih
- Pertanyaan?
- Kontak / sumber data / catatan pendukung
