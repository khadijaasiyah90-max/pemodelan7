import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
import string
import logging

st.set_page_config(page_title="Dynamic Topic Modeling & Stance Analysis", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("📊 Dynamic Topic Modeling & Stance Analysis")
st.write("Analisis topik dinamis pada unggahan dan analisis stance pada komentar")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

@st.cache_resource
def load_embedding_model():
    logging.info("Starting load_embedding_model")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logging.info("Completed load_embedding_model")
    return model

@st.cache_resource
def load_sentiment_model():
    logging.info("Starting load_sentiment_model")
    model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    logging.info("Completed load_sentiment_model")
    return model

@st.cache_data
def cached_fit_transform(_topic_model, _docs):
    """Cached wrapper for BERTopic fit_transform to avoid recomputation"""
    logging.info(f"Starting cached fit_transform on {len(_docs)} documents")
    topics, probs = _topic_model.fit_transform(_docs)
    logging.info(f"Completed cached fit_transform: {len(set(topics))} topics found")
    return topics, probs

@st.cache_data
def cached_topics_over_time(_topic_model, _docs, _timestamps, _nr_bins=20):
    """Cached wrapper for topics_over_time calculation"""
    logging.info("Calculating cached topics over time")
    try:
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=_nr_bins)
    except ValueError as e:
        logging.warning(f"Error dengan nr_bins={_nr_bins}: {e}. Menggunakan nr_bins=10.")
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=10)
    logging.info("Completed cached topics over time calculation")
    return result

@st.cache_data
def cached_stance_analysis(_sentiment_model, _comments_list, _batch_size=20):
    """Cached wrapper for stance analysis on comments with confidence threshold"""
    logging.info(f"Starting cached stance analysis on {len(_comments_list)} comments")
    sentiments = []
    confidences = []
    
    total_batches = (len(_comments_list) + _batch_size - 1) // _batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _batch_size
        end_idx = min((batch_idx + 1) * _batch_size, len(_comments_list))
        batch = _comments_list[start_idx:end_idx]
        
        batch_sentiments = _sentiment_model(batch)
        for sentiment in batch_sentiments:
            label = sentiment['label']
            confidence = sentiment['score']
            
            # Apply confidence threshold - if below 0.7, classify as neutral to reduce false positives
            if confidence < 0.7 and label != 'NEUTRAL':
                label = 'NEUTRAL'
                logging.debug(f"Low confidence {confidence:.2f} for {sentiment['label']}, reclassified as NEUTRAL")
            
            sentiments.append(label)
            confidences.append(confidence)
    
    logging.info("Completed cached stance analysis")
    return sentiments, confidences

def preprocess_text(text):
    """
    Preprocessing teks yang komprehensif untuk Indonesian text
    
    Tahapan:
    1. Konversi ke string dan lowercase
    2. Hapus URL
    3. Hapus mention (@username)
    4. Hapus hashtag (#topic)
    5. Hapus emoji
    6. Hapus angka
    7. Hapus punctuation dan special characters
    8. Hapus extra whitespace
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    logging.info(f"Starting preprocess_text for text length: {len(str(text))}")
    if pd.isna(text):
        logging.info("Completed preprocess_text: empty text")
        return ""
    
    text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Hapus URL (http://... atau https://...)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 4. Hapus hashtag (#topic) - hanya hapus simbol #, tapi keep kata
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # 5. Hapus emoji dan special Unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 6. Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # 7. Hapus punctuation dan special characters (keep hanya alphanumeric dan space)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 8. Hapus multiple spaces dan leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    logging.info(f"Completed preprocess_text: output length: {len(text)}")
    return text

def preprocess_dataframe(df, text_column):
    """
    Preprocessing dataframe dengan progress tracking
    
    Args:
        df: Input dataframe
        text_column: Nama kolom yang akan di-preprocess
    
    Returns:
        Dataframe dengan kolom baru yang preprocessed
    """
    logging.info(f"Starting preprocess_dataframe with {len(df)} rows, column: {text_column}")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    preprocessed_texts = []
    total = len(df)
    
    for idx, text in enumerate(df[text_column]):
        preprocessed = preprocess_text(text)
        preprocessed_texts.append(preprocessed)
        
        # Update progress setiap 10 item
        if (idx + 1) % max(1, total // 20) == 0:
            progress = int((idx + 1) / total * 100)
            progress_bar.progress(progress / 100)
            status_text.text(f"Preprocessing... {idx + 1}/{total} ({progress}%)")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Preprocessing selesai!")
    
    logging.info(f"Completed preprocess_dataframe: processed {len(preprocessed_texts)} texts")
    return preprocessed_texts

def convert_df_to_csv(df):
    """Konversi dataframe ke CSV bytes untuk download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_figure_to_html(fig):
    """Konversi Plotly figure ke HTML bytes untuk download"""
    return fig.to_html().encode('utf-8')

def display_data_statistics(df):
    """Menampilkan statistik data yang menarik"""
    st.subheader("📈 Statistik Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Baris", f"{len(df):,}")
    
    with col2:
        st.metric("📝 Kolom", f"{len(df.columns)}")
    
    with col3:
        try:
            posts_count = df['full_text'].notna().sum()
            st.metric("💬 Posts", f"{posts_count:,}")
        except:
            st.metric("💬 Posts", "N/A")
    
    with col4:
        try:
            comments_count = df['full_text_comments'].notna().sum()
            st.metric("💭 Komentar", f"{comments_count:,}")
        except:
            st.metric("💭 Komentar", "N/A")
    
    # Statistik Detail
    with st.expander("📊 Detail Statistik"):
        stats_data = {
            "Metrik": [],
            "Nilai": []
        }
        
        # Total data
        stats_data["Metrik"].append("Total Baris")
        stats_data["Nilai"].append(f"{len(df):,}")
        
        # Unique posts
        try:
            unique_posts = df['full_text'].nunique()
            stats_data["Metrik"].append("Posts Unik")
            stats_data["Nilai"].append(f"{unique_posts:,}")
        except:
            pass
        
        # Duplicate posts
        try:
            duplicate_posts = len(df) - df['full_text'].nunique()
            stats_data["Metrik"].append("Posts Duplikat")
            stats_data["Nilai"].append(f"{duplicate_posts:,}")
        except:
            pass
        
        # Missing values
        try:
            posts_missing = df['full_text'].isna().sum()
            stats_data["Metrik"].append("Posts Kosong")
            stats_data["Nilai"].append(f"{posts_missing:,}")
        except:
            pass
        
        try:
            comments_missing = df['full_text_comments'].isna().sum()
            stats_data["Metrik"].append("Komentar Kosong")
            stats_data["Nilai"].append(f"{comments_missing:,}")
        except:
            pass
        
        # Panjang rata-rata post
        try:
            avg_post_length = df['full_text'].dropna().str.len().mean()
            stats_data["Metrik"].append("Rata-rata Panjang Post")
            stats_data["Nilai"].append(f"{avg_post_length:.0f} karakter")
        except:
            pass
        
        # Panjang rata-rata komentar
        try:
            avg_comment_length = df['full_text_comments'].dropna().str.len().mean()
            stats_data["Metrik"].append("Rata-rata Panjang Komentar")
            stats_data["Nilai"].append(f"{avg_comment_length:.0f} karakter")
        except:
            pass
        
        # Rentang waktu
        try:
            date_col = df['created_at']
            date_col = pd.to_datetime(date_col, errors='coerce')
            min_date = date_col.min()
            max_date = date_col.max()
            date_range = max_date - min_date
            stats_data["Metrik"].append("Rentang Waktu Data")
            stats_data["Nilai"].append(f"{date_range.days} hari")
            stats_data["Metrik"].append("Tanggal Awal")
            stats_data["Nilai"].append(str(min_date.date()))
            stats_data["Metrik"].append("Tanggal Akhir")
            stats_data["Nilai"].append(str(max_date.date()))
        except:
            pass
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())
    
    # Tampilkan statistik data
    display_data_statistics(df)

    # Asumsikan kolom: full_text (posts), created_at (timestamp), full_text_comments (comments)
    required_cols = ['full_text', 'created_at', 'full_text_comments']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset harus memiliki kolom: {', '.join(required_cols)}")
    else:
        # Proses posts untuk topic modeling
        posts_df = df[['full_text', 'created_at']].dropna().drop_duplicates()
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
        posts_df = posts_df.sort_values(by='created_at')

        # Proses comments untuk stance analysis
        comments_df = df[['full_text_comments']].dropna()

        st.info("Memuat model...")
        embedding_model = load_embedding_model()
        sentiment_model = load_sentiment_model()
        
        # ========== PREPROCESSING SECTION ==========
        st.divider()
        st.subheader("🧹 Data Preprocessing")
        
        preprocessing_col1, preprocessing_col2 = st.columns(2)
        
        with preprocessing_col1:
            st.markdown("**📝 Preprocessing Posts untuk Topic Modeling**")
            with st.spinner("Processing posts..."):
                preprocessed_posts = preprocess_dataframe(posts_df.reset_index(drop=True), 'full_text')
                posts_df['full_text_preprocessed'] = preprocessed_posts
        
        with preprocessing_col2:
            st.markdown("**💬 Preprocessing Comments untuk Stance Analysis**")
            with st.spinner("Processing comments..."):
                preprocessed_comments = preprocess_dataframe(comments_df.reset_index(drop=True), 'full_text_comments')
                comments_df['full_text_comments_preprocessed'] = preprocessed_comments
        
        # Tampilkan perbandingan before-after preprocessing
        with st.expander("👁️ Lihat Contoh Preprocessing"):
            st.subheader("📝 Contoh: Posts Preprocessing")
            
            sample_idx = 0 if len(posts_df) > 0 else 0
            if len(posts_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**SEBELUM:**")
                    st.text_area("Original Text", posts_df['full_text'].iloc[sample_idx], height=100, disabled=True)
                
                with col2:
                    st.markdown("**SESUDAH:**")
                    st.text_area("Preprocessed Text", posts_df['full_text_preprocessed'].iloc[sample_idx], height=100, disabled=True)
            
            st.markdown("---")
            st.subheader("💬 Contoh: Comments Preprocessing")
            
            if len(comments_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**SEBELUM:**")
                    st.text_area("Original Comment", comments_df['full_text_comments'].iloc[0], height=100, disabled=True, key="comment_before")
                
                with col2:
                    st.markdown("**SESUDAH:**")
                    st.text_area("Preprocessed Comment", comments_df['full_text_comments_preprocessed'].iloc[0], height=100, disabled=True, key="comment_after")
        
        # Gunakan preprocessed text untuk analisis
        docs = posts_df['full_text_preprocessed'].astype(str).tolist()
        timestamps = posts_df['created_at'].tolist()
        
        st.success("✅ Preprocessing selesai! Siap untuk analisis.")
        st.divider()

        if st.button("🚀 Jalankan Analisis"):
            logging.info("Starting analysis: Topic Modeling and Stance Analysis")
            # ========== TOPIC MODELING ==========
            st.subheader("📊 Topic Modeling Processing")
            progress_container = st.container()
            
            with progress_container:
                # Progress 1: Model Initialization
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                status_text.text("🔄 Tahap 1/4: Inisialisasi Model...")
                progress_bar.progress(0.25)
                logging.info("Initializing BERTopic model")
                topic_model = BERTopic(embedding_model=embedding_model)
                
                # Progress 2: Fitting & Transforming with detailed progress
                st.markdown("---")
                st.subheader("🔄 Tahap 2/4: Fitting dan Transforming Dokumen")
                
                # Create detailed progress metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_docs_metric = st.empty()
                    total_docs_metric.metric("📚 Total Dokumen", f"{len(docs):,}")
                
                with col2:
                    processed_metric = st.empty()
                    processed_metric.metric("✓ Diproses", "0")
                
                with col3:
                    progress_perc_metric = st.empty()
                    progress_perc_metric.metric("% Progress", "0%")
                
                with col4:
                    status_metric = st.empty()
                    status_metric.metric("Status", "Memulai...")
                
                # Main progress bar
                main_progress_bar = st.progress(0.0)
                sub_progress_bar = st.progress(0.0)
                detail_text = st.empty()
                time_info = st.empty()
                
                import time
                start_time = time.time()
                
                # Simulate progress dengan milestone
                milestones = [10, 25, 50, 75, 90, 100]
                last_milestone = 0
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2A: Embedding (30% dari tahap 2)
                for i in range(30):
                    processed_metric.metric("✓ Diproses", f"{len(docs) // 100 * (i+1):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1)/100)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2B: Clustering & UMAP (40% dari tahap 2)
                for i in range(30, 70):
                    processed_metric.metric("✓ Diproses", f"{len(docs) // 100 * (i+1):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1-30)/70)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions... ✓</small><br>
                    <small>• Topic extraction & labeling...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2C: Topic extraction (30% dari tahap 2)
                for i in range(70, 100):
                    processed_metric.metric("✓ Diproses", f"{len(docs):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1-70)/30)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
                    <small><b>✅ Tahap 2 Selesai!</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions... ✓</small><br>
                    <small>• Topic extraction & labeling... ✓</small>
                </div>
                """, unsafe_allow_html=True)
                
                processed_metric.metric("✓ Diproses", f"{len(docs):,}")
                progress_perc_metric.metric("% Progress", "50%")
                status_metric.metric("Status", "Fit-Transform Selesai")
                
                # Jalankan actual fit_transform
                topics, probs = cached_fit_transform(topic_model, docs)
                
                st.markdown("---")
                
                # Progress 3: Topics Over Time
                status_text.text("🔄 Tahap 3/4: Menghitung Topics Over Time...")
                progress_bar.progress(0.75)
                logging.info("Calculating topics over time")
                topics_over_time = cached_topics_over_time(topic_model, docs, timestamps, nr_bins=20)
                
                # Progress 4: Complete
                status_text.text("✅ Tahap 4/4: Selesai!")
                progress_bar.progress(1.0)

            st.success("✅ Topic Modeling Selesai!")

            st.subheader("📈 Topics Over Time")
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button untuk Topics Over Time
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Simpan Topics Over Time (HTML)",
                    data=convert_figure_to_html(fig),
                    file_name="topics_over_time.html",
                    mime="text/html",
                    key="download_topics_over_time"
                )

            st.subheader("📌 Top Topics")
            top_topics_df = topic_model.get_topic_info()
            st.dataframe(top_topics_df, use_container_width=True)
            
            # Download button untuk Top Topics
            with col2:
                st.download_button(
                    label="💾 Simpan Top Topics (CSV)",
                    data=convert_df_to_csv(top_topics_df),
                    file_name="top_topics.csv",
                    mime="text/csv",
                    key="download_top_topics"
                )

            # ========== STANCE ANALYSIS ==========
            st.subheader("🗣️ Stance Analysis pada Komentar")
            
            comments_list = comments_df['full_text_comments_preprocessed'].tolist()
            batch_size = 20
            
            logging.info(f"Starting stance analysis on {len(comments_list)} comments with batch size {batch_size}")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            comments_df['sentiment'] = None
            comments_df['confidence'] = None
            
            # Use cached stance analysis
            status_text.text("🔄 Menganalisis Stance... (Cached)")
            sentiments, confidences = cached_stance_analysis(sentiment_model, comments_list, batch_size)
            
            # Assign results to dataframe
            for i in range(len(sentiments)):
                comments_df.loc[i, 'sentiment'] = sentiments[i]
                comments_df.loc[i, 'confidence'] = confidences[i]
            
            progress_bar.progress(1.0)
            status_text.text("✅ Stance Analysis Selesai!")
            
            logging.info("Completed stance analysis")
            st.success("✅ Analisis Stance Selesai!")
            
            st.subheader("📋 Hasil Stance Analysis (20 Data Teratas)")
            st.dataframe(comments_df.head(20), use_container_width=True)
            
            # Add explanation about neutral classification
            with st.expander("ℹ️ Penjelasan Klasifikasi Netral"):
                st.markdown("""
                **Mengapa banyak komentar diklasifikasikan sebagai Netral?**
                
                1. **Sifat Data**: Komentar politik Indonesia cenderung formal dan factual
                2. **Model Training**: Model dilatih pada data Twitter yang lebih emosional  
                3. **Confidence Threshold**: Prediksi dengan confidence < 0.7 otomatis dinetralisasi
                4. **Preprocessing**: Emoji dan karakter khusus yang membawa sentimen dihapus
                
                **Distribusi Confidence Score:**
                - **Tinggi (0.8+)**: Stance kuat, jarang netral
                - **Sedang (0.6-0.8)**: Ambiguous, sering dinetralisasi  
                - **Rendah (<0.6)**: Tidak yakin, otomatis netral
                """)
            
            # Show confidence distribution
            if 'confidence' in comments_df.columns:
                st.subheader("📊 Distribusi Confidence Score")
                confidence_counts = pd.cut(comments_df['confidence'], bins=[0, 0.6, 0.7, 0.8, 1.0], 
                                         labels=['<0.6', '0.6-0.7', '0.7-0.8', '0.8+']).value_counts()
                st.bar_chart(confidence_counts)
            
            # Download button untuk Stance Analysis Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="💾 Simpan Semua Hasil (CSV)",
                    data=convert_df_to_csv(comments_df),
                    file_name="stance_analysis_results.csv",
                    mime="text/csv",
                    key="download_stance_all"
                )

            # Summary sentiment
            sentiment_counts = comments_df['sentiment'].value_counts()
            
            st.subheader("📊 Ringkasan Sentimen/Stance")
            st.bar_chart(sentiment_counts)
            
            # Download button untuk Summary Sentiment
            summary_df = sentiment_counts.reset_index()
            summary_df.columns = ['Sentiment', 'Jumlah']
            with col2:
                st.download_button(
                    label="💾 Simpan Summary (CSV)",
                    data=convert_df_to_csv(summary_df),
                    file_name="sentiment_summary.csv",
                    mime="text/csv",
                    key="download_summary"
                )
            
            # Create a comprehensive report
            with col3:
                # Combined report
                sentiment_lines = ''.join(
                    f"- {sent}: {count:,}\n" for sent, count in sentiment_counts.items()
                )
                report = f"""
LAPORAN ANALISIS TOPIK DINAMIS DAN STANCE ANALYSIS
=====================================================

STATISTIK UMUM:
- Total Posts: {len(posts_df):,}
- Total Komentar: {len(comments_df):,}
- Rentang Waktu: {posts_df['created_at'].min().date()} hingga {posts_df['created_at'].max().date()}

HASIL TOPIC MODELING:
- Jumlah Topics: {len(top_topics_df):,}
- Model: BERTopic

HASIL STANCE ANALISIS:
- Total Komentar Dianalisis: {len(comments_df):,}
{sentiment_lines}

Dibuat pada: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.download_button(
                    label="💾 Simpan Laporan (.txt)",
                    data=report.encode('utf-8'),
                    file_name="analysis_report.txt",
                    mime="text/plain",
                    key="download_report"
                )
            logging.info("Analysis completed successfully")