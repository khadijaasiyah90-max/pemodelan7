# Bab 7: Evaluasi Sistem dan Indikator Keberhasilan

## 7.1 Pendahuluan

Bab ini menguraikan kerangka evaluasi untuk sistem Dynamic Topic Modeling (BERTopic) dan Stance Analysis (IndoBERT) pada dataset Twitter bertema politik dan kebijakan luar negeri. Evaluasi dirancang untuk membandingkan kinerja metode yang diusulkan dengan baseline konvensional, sekaligus menilai validitas hasil menggunakan penilaian pakar.

## 7.2 Skenario Baseline

### 7.2.1 Model Baseline
Model baseline yang digunakan adalah Analisis Sentimen Konvensional berbasis kamus leksikon atau algoritma klasik seperti Naive Bayes/SVM.

### 7.2.2 Target Baseline
Baseline hanya mengklasifikasikan teks secara polaritas dasar:
- Positif
- Negatif
- Netral

### 7.2.3 Tujuan Perbandingan
Tujuan utama adalah membuktikan bahwa Stance Analysis yang dikembangkan dengan IndoBERT mampu mendeteksi keberpihakan terhadap entitas atau topik tertentu (Pro / Kontra / Netral) lebih unggul dibandingkan analisis polaritas sentimen sederhana.

## 7.3 Metrik Kuantitatif

### 7.3.1 Dynamic Topic Modeling (BERTopic)
- **Coherence Score (C_v)**: Mengukur koherensi topik berdasarkan keterkaitan kata-kata kunci di dalam setiap klaster topik.
- Rentang skor: **0 sampai 1**.
- Interpretasi: semakin mendekati 1, semakin baik topik tersebut mudah dipahami secara manusiawi.

### 7.3.2 Stance Analysis (IndoBERT)
Pengukuran berdasarkan dataset uji komentar/reply:
- **Accuracy**: Persentase total prediksi kelas Pro/Kontra/Netral yang benar.
- **Precision**: Kemampuan model untuk menghindari kesalahan pelabelan, misalnya tidak memberi label "Pro" pada komentar netral.
- **Recall**: Kemampuan model menemukan semua komentar yang benar-benar termasuk dalam kelas tertentu.
- **F1-Score**: Rata-rata harmonis dari precision dan recall; metrik penting saat dataset tidak seimbang.

> F1-Score menjadi nilai utama ketika distribusi kelas komentar Twitter tidak seimbang, misalnya kelas "Kontra" lebih dominan dibanding "Pro".

## 7.4 Metrik Kualitatif (Validasi Pakar)

Karena data mengandung isu diplomasi dan kebijakan luar negeri yang membutuhkan konteks domain, evaluasi kualitatif oleh pakar sangat penting.

### 7.4.1 Validasi Relevansi Topik
Pakar kebijakan luar negeri menilai apakah topik yang dibentuk BERTopic merepresentasikan isu nyata, seperti:
- Kinerja Menteri Luar Negeri
- Krisis lingkungan internasional
- Diplomasi multilateral
- Hubungan politik bilateral

### 7.4.2 Validasi Akurasi Klasifikasi Stance
Pakar melakukan anotasi manual pada sampel komentar/reply untuk menilai apakah pelabelan IndoBERT sudah tepat secara konteks, terutama ketika sentimen sebenarnya bersifat:
- Pro terhadap tujuan tertentu
- Kontra terhadap kebijakan tertentu
- Netral dengan alasan atau kondisi spesifik

## 7.5 Indikator Keberhasilan Sistem

Sistem tesis ini dinyatakan berhasil apabila memenuhi ketiga indikator berikut:

### 7.5.1 Keunggulan Topic Modeling
- Coherence Score (C_v) BERTopic meningkat **≥ 10%** dibandingkan LDA pada dataset unggahan yang sama.
- Peningkatan ini menunjukkan bahwa model BERTopic menghasilkan topik yang lebih koheren dan relevan daripada metode generatif klasik.

### 7.5.2 Keandalan Analisis Kepakaran
- Kesesuaian antara prediksi IndoBERT dan anotasi pakar mencapai **≥ 66%**.
- Pengukuran dapat dilakukan dengan metrik Accuracy atau Fleiss’ Kappa jika ada lebih dari satu pakar.

### 7.5.3 Superioritas Deteksi Konteks/Ambivalensi
- IndoBERT mampu mendeteksi kasus stance yang bersifat ambivalen, sarkastik, atau bercabang secara lebih akurat dibandingkan Sentiment Analysis konvensional.
- Evaluasi difokuskan pada contoh komentar yang memiliki elemen pro dan kontra sekaligus atau netral beralasan.

## 7.6 Implikasi Hasil

Jika semua indikator terpenuhi, maka sistem ini menunjukkan:
- Kelompok topik yang lebih relevan dan kontekstual untuk isu politik luar negeri.
- Kapasitas deteksi stance yang lebih dalam dibandingkan analisis polaritas sederhana.
- Validitas hasil yang didukung oleh pakar domain diplomasi.

## 7.7 Rekomendasi Lanjutan

Untuk penelitian lebih lanjut, disarankan:
- Menguji model dengan dataset yang lebih besar dan lebih beragam topik kebijakan luar negeri.
- Menambahkan analisis entitas target (entity-target stance) untuk meningkatkan akurasi klasifikasi stance.
- Menggunakan validasi pakar multi-level untuk menghitung kepastian annotasi dan Fleiss’ Kappa.
- Menerapkan model adaptif khusus bahasa Indonesia untuk menangani kode campur dan slang politik.
