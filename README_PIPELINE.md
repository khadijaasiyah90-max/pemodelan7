# Dynamic Topic Modeling + Stance Analysis

Program Python untuk analisis media sosial Twitter/X menggunakan dataset yang sudah tersedia. Program ini mengimplementasikan Dynamic Topic Modeling terhadap postingan dan Stance Analysis terhadap komentar.

## Arsitektur Sistem

1. **Dynamic Topic Modeling** terhadap postingan (`full_text`)
2. Pengelompokan postingan ke topik
3. **Stance Analysis** komentar (`full_text_comments`) terhadap postingan utama (`full_text`)
4. Penggabungan hasil topic dan stance
5. Analisis distribusi stance pada setiap topik

## Struktur Dataset

Dataset harus dalam format CSV/XLSX dengan kolom:

| Kolom | Keterangan |
|-------|------------|
| post_id | ID postingan |
| full_text | Isi postingan utama |
| full_text_comments | Isi komentar/reply |
| clean_text | Hasil preprocessing postingan |
| clean_comments | Hasil preprocessing komentar |
| created_at | Tanggal posting |
| comment_id | ID komentar |

## Struktur Program

Program terdiri dari beberapa modul:

- `load_data.py` - Memuat dan mempersiapkan dataset
- `topic_modeling.py` - Implementasi BERTopic untuk topic modeling
- `stance_analysis.py` - Analisis stance menggunakan transformers
- `merge_analysis.py` - Penggabungan hasil topic dan stance
- `visualization.py` - Pembuatan visualisasi
- `main.py` - Entry point utama

## Instalasi Dependencies

```bash
pip install -r requirements.txt
```

## Penggunaan

### Jalankan Analisis Lengkap

```bash
python main.py path/to/dataset.csv --results-dir results --batch-size 32
```

### Parameter

- `dataset_path`: Path ke file dataset CSV/XLSX
- `--results-dir`: Folder output (default: `results`)
- `--batch-size`: Batch size untuk stance analysis (default: 32)

## Output Program

Program akan menghasilkan:

### File CSV
- `posts_with_topics.csv` - Postingan dengan hasil topic modeling
- `comments_with_stance.csv` - Komentar dengan hasil stance analysis
- `merged_topic_stance.csv` - Gabungan topic dan stance
- `stance_distribution_by_topic.csv` - Distribusi stance per topik

### Visualisasi
- `topic_distribution.png` - Distribusi topik
- `stance_distribution.png` - Distribusi stance
- `topic_stance_heatmap.png` - Heatmap topic vs stance
- `topic_over_time.html` - Topic over time (Plotly)
- `posts_per_topic.png` - Jumlah postingan per topik

## Teknologi yang Digunakan

- **Topic Modeling**: BERTopic dengan Sentence Transformers, UMAP, HDBSCAN
- **Stance Analysis**: Transformers (IndoBERT/Twitter RoBERTa)
- **Visualisasi**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Pandas, NumPy

## Contoh Output

### Distribusi Stance per Topik

| Topik | Favor | Against | Neutral |
|-------|-------|---------|---------|
| Palestina | 70% | 20% | 10% |
| Diplomasi | 40% | 30% | 30% |
| ASEAN | 50% | 25% | 25% |

## Lisensi

MIT License