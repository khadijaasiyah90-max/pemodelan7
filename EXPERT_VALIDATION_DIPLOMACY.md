# Validasi Expert Diplomasi / Hubungan Luar Negeri

Dokumen ini menjelaskan pendekatan validasi oleh ahli diplomasi dan foreign policy expert untuk sistem Topic Modeling dan Stance Analysis di proyek DTM.

## Tujuan

- Memastikan topik yang dihasilkan relevan dengan kebijakan luar negeri dan diplomasi.
- Mengukur kualitas stance classification untuk komentar terkait isu diplomasi.
- Menghasilkan ground truth domain-spesifik yang dapat digunakan untuk evaluasi dan fine-tuning.
- Mengidentifikasi kelemahan model yang berhubungan dengan nuansa diplomasi, konteks internasional, dan retorika politik luar negeri.

## 1. Panel Expert Diplomasi

### Komposisi

| Peran | Keahlian | Tugas |
|------|-----------|-------|
| **Diplomat / Foreign Policy Expert** | Kebijakan luar negeri, hubungan bilateral/multilateral, diplomasi publik | Validasi topik, interpretasi isu, verifikasi stance dalam konteks LN |
| **Analis Politik Luar Negeri** | Analisis kebijakan, geopolitik, retorika diplomatik | Review coherence topik dan akurasi stance di isu LN |
| **Linguist Bahasa Indonesia** | Nuansa bahasa, istilah diplomatik, frasa diplomasi | Verifikasi preprocessing, ambiguitas, dan pengaruh wording |
| **Subject Matter Expert (Hubungan Internasional)** | Isu keamanan, perdagangan internasional, organisasi internasional | Validasi kesesuaian label dan saran perbaikan domain-spesifik |

## 2. Prinsip Validasi Diplomasi

1. **Relevansi Kebijakan LN**: Topik harus mencerminkan isu-isu nyata dalam domain hubungan luar negeri.
2. **Konteks Internasional**: Penilaian harus mempertimbangkan konteks bilateral, multilateral, dan organisasi internasional.
3. **Nuansa Retorika**: Prediksi stance harus menyadari frasa diplomatik, kata-kata tersirat, dan bahasa netral.
4. **Akurasi Label**: Expert harus membandingkan hasil model dengan interpretasi kebijakan luar negeri yang benar.
5. **Keberlanjutan Data**: Hasil validasi harus menjadi bagian dari ground truth yang dapat digunakan untuk pelatihan ulang.

## 3. Rubrik Validasi Topik

### Dimensi Penilaian

| Dimensi | Skala | Kriteria Khusus Diplomasi |
|---------|-------|--------------------------|
| **Relevance** | 1-5 | Seberapa baik topik merepresentasikan isu politik luar negeri? |
| **Coherence** | 1-5 | Seberapa koheren kata dan frasa dalam topik terkait diplomasi? |
| **Policy Alignment** | 1-5 | Apakah topik selaras dengan kebijakan LN yang sedang dibahas? |
| **International Context** | 1-5 | Seberapa jelas konteks internasional/multilateralnya? |
| **Interpretability** | 1-5 | Seberapa mudah topik dimengerti oleh ahli diplomasi? |
| **Completeness** | 1-5 | Apakah aspek penting (misal ASEAN, PBB, perjanjian bilateral) tercakup? |

### Status Validasi

- **VALID**: Topik jelas, relevan, dan cukup spesifik.
- **NEEDS REVISION**: Topik memiliki campuran konsep atau kurang jelas konteks LN.
- **INVALID**: Topik tidak relevan dengan isu luar negeri atau terlalu kabur.

### Contoh Form Topik

```markdown
Topic ID: [ID]
Top Words: [word1, word2, word3, ...]

Rating:
- Relevance: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Coherence: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Policy Alignment: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- International Context: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Interpretability: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]
- Completeness: [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ]

Status: [ ] VALID  [ ] NEEDS REVISION  [ ] INVALID

Suggested Label/Interpretation:
[Expert's label and explanation]

Expert Notes:
[Detail notes, references to policy, institution, atau peristiwa internasional]

Expert Name: _________
Date: _________
```

## 4. Rubrik Validasi Stance

### Dimensi Penilaian

| Dimensi | Kriteria Diplomasi |
|--------|-------------------|
| **Accuracy** | Apakah stance model tepat terkait dukungan/penolakan kebijakan LN? |
| **Context Sensitivity** | Apakah model memahami konteks isu internasional? |
| **Nuance** | Seberapa baik model menangkap nada diplomatik atau ambigu? |
| **Confidence** | Seberapa yakin expert dengan label yang diberikan? |
| **Disagreement Reason** | Alasan jika model salah: misinterpretasi istilah, ironi, atau negasi. |

### Kategori Stance

- **POSITIVE**: Mendukung kebijakan luar negeri, kerjasama, diplomasi positif.
- **NEGATIVE**: Menolak kebijakan, khawatir dampak geopolitik, menolak kerja sama.
- **NEUTRAL**: Pernyataan netral, informatif, atau tidak jelas sikapnya.

### Form Stance

```markdown
Comment ID: [ID]
Original Text: [Original comment]
Preprocessed Text: [Preprocessed comment]

Model Prediction: [POSITIVE/NEGATIVE/NEUTRAL] (Confidence: [X%])

Expert Judgment:
- Stance: [ ] POSITIVE  [ ] NEGATIVE  [ ] NEUTRAL
- Confidence: [ ] Very Confident  [ ] Confident  [ ] Somewhat  [ ] Low  [ ] Ambiguous

Explanation:
[Why this stance tepat dalam konteks kebijakan LN]

If different from model:
- Disagreement Reason: [Negation, nuance, missed context, slang, sarcasm, domain term]

Expert Notes:
[ tambahan instruksi atau referensi kebijakan ]

Expert Name: _________
Date: _________
```

## 5. Dataset Ground Truth Diplomasi

### Struktur File

**ground_truth_diplomacy_topics.csv**

- `topic_id`
- `top_words`
- `expert_label`
- `relevance_score`
- `coherence_score`
- `policy_alignment_score`
- `international_context_score`
- `status`
- `expert_notes`
- `expert_name`
- `created_at`

**ground_truth_diplomacy_stance.csv**

- `comment_id`
- `text_original`
- `text_preprocessed`
- `model_prediction`
- `model_confidence`
- `expert_stance`
- `expert_confidence`
- `agreement`
- `error_type`
- `expert_notes`
- `expert_name`
- `created_at`

### Contoh Data

```csv
topic_id,top_words,expert_label,relevance_score,coherence_score,policy_alignment_score,international_context_score,status,expert_notes,expert_name,created_at
0,"diplomasi,bilateral,negara,kerjasama,penting","Kerjasama Bilateral Indonesia-China",5,5,5,4,VALID,"Topik relevan dengan MoU bilateral dan investasi.",Dr. A,2026-04-25
```

```csv
comment_id,text_original,text_preprocessed,model_prediction,model_confidence,expert_stance,expert_confidence,agreement,error_type,expert_notes,expert_name,created_at
c001,"Setuju kerjasama ini akan memperkuat posisi regional","setuju kerjasama ini akan memperkuat posisi regional",POSITIVE,0.88,POSITIVE,5,TRUE,-,"Prediksi sesuai konteks diplomasi ekonomi.",Dr. B,2026-04-25
```

## 6. Metrik Evaluasi untuk Expert Diplomasi

### Metrik Utama

- **Accuracy** terhadap expert judgment
- **Precision / Recall / F1 Score** per kelas stance
- **Agreement Rate** untuk validasi topik
- **Cohen's Kappa** untuk inter-rater reliability
- **Average Relevance / Policy Alignment / Context Score** untuk topic modeling

### Contoh Output Evaluasi

```text
Stance Evaluation:
- Accuracy: 0.82
- Precision (macro): 0.80
- Recall (macro): 0.79
- F1 Score (macro): 0.79
- Precision (weighted): 0.81
- Recall (weighted): 0.82
- F1 Score (weighted): 0.81
```

## 7. Proses Validasi Diplomasi

### Langkah Pelaksanaan

1. Pilih sample data yang representatif dari isu luar negeri.
2. Minta expert membaca konteks dan menilai topik serta stance.
3. Simpan hasil validasi dalam format CSV.
4. Hitung metrik evaluasi.
5. Tinjau kembali kesalahan ke model dan revisi guideline.

### Contoh Alur Validasi

1. **Topic Validation**: Expert menilai 20-30 topic utama, khusus diplomasi.
2. **Stance Validation**: Expert menilai 100-150 komentar yang diklasifikasikan.
3. **Inter-rater Check**: Minimal dua expert independen untuk subset data.
4. **Consensus**: Diskusikan perbedaan dan finalisasi ground truth.

## 8. Rekomendasi Implementasi

- Gunakan form Google/Excel jika belum ada UI Streamlit.
- Simpan raw annotation ke folder `validation/raw_annotations/`.
- Gunakan `model_evaluation.py` untuk menghitung F1 score dan metrik lain.
- Tambahkan kolom `expert_stance` pada dataset untuk memudahkan evaluasi otomatis.
- Sertakan komentar ahli yang menjelaskan istilah diplomasi atau kebijakan khusus.

## 9. Integrasi dengan Streamlit App

Jika ingin mengintegrasikan ke `streamlit_app.py`:

- Tambahkan upload dataset `expert_stance`.
- Tampilkan metrik F1 dan classification report jika expert labels tersedia.
- Tambahkan panel validasi topik untuk input expert menggunakan form.

## 10. Template Ringkas Validasi Diplomasi

- Topik: validasi relevansi LN, konteks internasional, kelengkapan.
- Stance: validasi dukungan/penolakan netralitas kebijakan luar negeri.
- Output: ground truth, metrik F1, dan rekomendasi perbaikan model.
