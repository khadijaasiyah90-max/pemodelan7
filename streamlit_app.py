import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import pipeline
from datetime import datetime
import re
import string
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import numpy as np
import os

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
    """Cached wrapper for BERTopic topics_over_time calculation"""
    logging.info("Calculating cached topics over time")
    try:
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=_nr_bins)
    except ValueError as e:
        logging.warning(f"Error with nr_bins={_nr_bins}: {e}. Retrying with nr_bins=10.")
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=10)
    logging.info("Completed cached topics over time calculation")
    return result

@st.cache_data
def calculate_topic_coherence(_topic_model, _docs, coherence_type='c_v'):
    """
    Calculate topic coherence using gensim CoherenceModel
    
    Args:
        _topic_model: Fitted BERTopic model
        _docs: List of preprocessed documents
        coherence_type: Type of coherence measure ('c_v', 'c_uci', 'c_npmi', 'u_mass')
    
    Returns:
        dict: Coherence scores for each topic and overall average
    """
    logging.info(f"Calculating topic coherence with {coherence_type} measure")
    
    try:
        # Get topics from BERTopic model
        topics = _topic_model.get_topics()
        
        # Filter out outlier topic (-1)
        topics = {k: v for k, v in topics.items() if k != -1}
        
        if not topics:
            logging.warning("No valid topics found for coherence calculation")
            return {"error": "No valid topics found"}
        
        # Prepare documents for coherence calculation
        # Tokenize documents
        tokenized_docs = [doc.split() for doc in _docs if doc.strip()]
        
        # Create dictionary
        dictionary = Dictionary(tokenized_docs)
        
        # Filter out words that appear in less than 5 documents or more than 50% of documents
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # Create corpus
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        # Prepare topic words for coherence calculation
        topic_words = []
        for topic_id in sorted(topics.keys()):
            # Get top 10 words for each topic
            words = [word for word, _ in topics[topic_id][:10]]
            topic_words.append(words)
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence=coherence_type
        )
        
        # Get coherence score for each topic
        topic_coherences = coherence_model.get_coherence_per_topic()
        
        # Calculate overall coherence
        overall_coherence = coherence_model.get_coherence()
        
        # Prepare results
        results = {
            'overall_coherence': overall_coherence,
            'topic_coherences': topic_coherences,
            'coherence_type': coherence_type,
            'num_topics': len(topic_words)
        }
        
        logging.info(f"Coherence calculation completed: {overall_coherence:.4f}")
        return results
        
    except Exception as e:
        logging.error(f"Error calculating coherence: {e}")
        return {"error": str(e)}

@st.cache_data
def calculate_topic_metrics(_topic_model, _docs):
    """
    Calculate additional topic modeling evaluation metrics
    
    Args:
        _topic_model: Fitted BERTopic model
        _docs: List of preprocessed documents
    
    Returns:
        dict: Various topic modeling metrics
    """
    logging.info("Calculating additional topic metrics")
    
    try:
        # Get topics info
        topics_info = _topic_model.get_topic_info()
        topics = _topic_model.get_topics()
        
        # Filter out outlier topic
        topics_info = topics_info[topics_info['Topic'] != -1]
        topics = {k: v for k, v in topics.items() if k != -1}
        
        if topics_info.empty:
            return {"error": "No valid topics found"}
        
        # Calculate topic diversity (unique words per topic)
        topic_diversities = []
        all_words = set()
        
        for topic_id, words_weights in topics.items():
            topic_words = [word for word, _ in words_weights[:10]]
            topic_diversities.append(len(set(topic_words)))
            all_words.update(topic_words)
        
        avg_topic_diversity = np.mean(topic_diversities)
        total_unique_words = len(all_words)
        
        # Calculate topic size distribution
        topic_sizes = topics_info['Count'].values
        topic_size_std = np.std(topic_sizes)
        topic_size_cv = topic_size_std / np.mean(topic_sizes) if np.mean(topic_sizes) > 0 else 0
        
        # Calculate document coverage
        total_docs = len(_docs)
        covered_docs = topics_info['Count'].sum()
        doc_coverage = covered_docs / total_docs if total_docs > 0 else 0
        
        # Calculate average topic size
        avg_topic_size = np.mean(topic_sizes)
        
        results = {
            'num_topics': len(topics),
            'avg_topic_diversity': avg_topic_diversity,
            'total_unique_words': total_unique_words,
            'topic_size_std': topic_size_std,
            'topic_size_cv': topic_size_cv,  # Coefficient of variation
            'doc_coverage': doc_coverage,
            'avg_topic_size': avg_topic_size,
            'topic_sizes': topic_sizes.tolist(),
            'topic_diversities': topic_diversities
        }
        
        logging.info(f"Topic metrics calculated: {len(topics)} topics, diversity: {avg_topic_diversity:.2f}")
        return results
        
    except Exception as e:
        logging.error(f"Error calculating topic metrics: {e}")
        return {"error": str(e)}

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


def initialize_expert_validation_state():
    if 'expert_stance_annotations' not in st.session_state:
        st.session_state['expert_stance_annotations'] = []
    if 'expert_topic_annotations' not in st.session_state:
        st.session_state['expert_topic_annotations'] = []
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False


def _export_validation_to_csv(data):
    if not data:
        return None
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')


def render_expert_validation_ui():
    initialize_expert_validation_state()

    st.header("🧑‍💼 Validasi Ahli Diplomasi")
    st.write("Gunakan antarmuka ini untuk menyimpan penilaian expert terhadap stance model dan topik diplomasi.")

    expert_name = st.text_input("Nama Expert:", key="expert_name")
    expert_org = st.text_input("Organisasi/Institusi:", key="expert_org")

    if not st.session_state['analysis_done']:
        st.warning("Jalankan analisis terlebih dahulu di mode Analisis agar hasil model tersedia untuk divalidasi.")
        return

    validation_type = st.selectbox(
        "Pilih Jenis Validasi",
        ["Validasi Stance", "Validasi Topik"],
        key="validation_type"
    )

    if validation_type == "Validasi Stance":
        comments_df = st.session_state['comments_df']
        comments_df = comments_df.reset_index(drop=False)
        selected_index = st.selectbox(
            "Pilih komentar untuk divalidasi:",
            comments_df.index,
            format_func=lambda idx: f"Komentar #{idx + 1}"
        )
        row = comments_df.loc[selected_index]

        st.subheader("Komentar untuk divalidasi")
        st.write(f"**Original Text:** {row['full_text_comments']}")
        st.write(f"**Preprocessed Text:** {row['full_text_comments_preprocessed']}")
        st.write(f"**Prediksi Model:** {row.get('sentiment', 'N/A')} (Confidence: {row.get('confidence', 0):.2f})")

        expert_stance = st.radio(
            "Expert Stance:",
            ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            horizontal=True,
            key="expert_stance_selection"
        )
        expert_confidence = st.select_slider(
            "Expert Confidence:",
            options=["Very Confident", "Confident", "Somewhat", "Low", "Ambiguous"],
            key="expert_confidence_selection"
        )
        agreement = st.checkbox("Setuju dengan prediksi model", key="expert_agreement")
        disagreement_reason = ""
        if not agreement:
            disagreement_reason = st.text_area("Alasan perbedaan:", key="expert_disagreement_reason")
        expert_notes = st.text_area("Catatan ahli:", height=120, key="expert_notes")

        if st.button("💾 Simpan Validasi Komentar", key="save_comment_validation"):
            annotation = {
                'comment_row': int(row['index']),
                'original_text': row['full_text_comments'],
                'preprocessed_text': row['full_text_comments_preprocessed'],
                'model_prediction': row.get('sentiment', ''),
                'model_confidence': float(row.get('confidence', 0) or 0),
                'expert_stance': expert_stance,
                'expert_confidence': expert_confidence,
                'agreement': agreement,
                'disagreement_reason': disagreement_reason,
                'expert_notes': expert_notes,
                'expert_name': expert_name,
                'expert_org': expert_org,
                'saved_at': datetime.now().isoformat()
            }
            st.session_state['expert_stance_annotations'].append(annotation)
            st.success("Validasi komentar berhasil disimpan.")
            
            # Save to CSV
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame(st.session_state['expert_stance_annotations']).to_csv(os.path.join(results_dir, f"expert_stance_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
            st.success("Validasi komentar berhasil disimpan dan diekspor ke CSV.")

        if st.session_state['expert_stance_annotations']:
            st.subheader("Hasil Validasi Komentar")
            st.dataframe(pd.DataFrame(st.session_state['expert_stance_annotations']))
            csv_data = _export_validation_to_csv(st.session_state['expert_stance_annotations'])
            if csv_data is not None:
                st.download_button(
                    label="💾 Unduh Validasi Komentar (CSV)",
                    data=csv_data,
                    file_name="expert_comment_validation.csv",
                    mime="text/csv"
                )

    else:
        topic_df = st.session_state.get('topic_validation_df', pd.DataFrame())
        topic_docs_mapping = st.session_state.get('topic_docs_mapping', {})

        if topic_df.empty:
            st.warning("Data topik tidak tersedia. Jalankan analisis terlebih dahulu.")
            return

        selected_topic = st.selectbox(
            "Pilih Topic ID untuk divalidasi:",
            topic_df['Topic'].tolist(),
            key="selected_topic_id"
        )
        topic_row = topic_df[topic_df['Topic'] == selected_topic].iloc[0]

        st.subheader(f"Topik {selected_topic}")
        st.write(f"**Top Words:** {topic_row['Top Words']}")
        st.write(f"**Topic Title:** {topic_row.get('Name', 'N/A')}")

        sample_docs = topic_docs_mapping.get(int(selected_topic), [])
        if sample_docs:
            with st.expander("Contoh dokumen topik"):
                for idx, sample in enumerate(sample_docs, 1):
                    st.write(f"{idx}. {sample}")

        relevance = st.slider("Relevance terhadap kebijakan luar negeri:", 1, 5, 3, key="topic_relevance")
        coherence = st.slider("Coherence topik:", 1, 5, 3, key="topic_coherence")
        policy_alignment = st.slider("Policy Alignment:", 1, 5, 3, key="topic_policy_alignment")
        international_context = st.slider("International Context:", 1, 5, 3, key="topic_international_context")
        interpretability = st.slider("Interpretability:", 1, 5, 3, key="topic_interpretability")
        completeness = st.slider("Completeness:", 1, 5, 3, key="topic_completeness")

        status = st.radio(
            "Status Validasi:",
            ["VALID", "NEEDS REVISION", "INVALID"],
            horizontal=True,
            key="topic_validation_status"
        )
        expert_label = st.text_input("Suggested label/topik interpretasi:", key="topic_expert_label")
        topic_notes = st.text_area("Catatan ahli:", height=120, key="topic_expert_notes")

        if st.button("💾 Simpan Validasi Topik", key="save_topic_validation"):
            topic_annotation = {
                'topic_id': int(selected_topic),
                'top_words': topic_row['Top Words'],
                'topic_name': topic_row.get('Name', ''),
                'relevance': relevance,
                'coherence': coherence,
                'policy_alignment': policy_alignment,
                'international_context': international_context,
                'interpretability': interpretability,
                'completeness': completeness,
                'status': status,
                'expert_label': expert_label,
                'expert_notes': topic_notes,
                'expert_name': expert_name,
                'expert_org': expert_org,
                'saved_at': datetime.now().isoformat()
            }
            st.session_state['expert_topic_annotations'].append(topic_annotation)
            st.success("Validasi topik berhasil disimpan.")
            
            # Save to CSV
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame(st.session_state['expert_topic_annotations']).to_csv(os.path.join(results_dir, f"expert_topic_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
            st.success("Validasi topik berhasil disimpan dan diekspor ke CSV.")

        if st.session_state['expert_topic_annotations']:
            st.subheader("Hasil Validasi Topik")
            st.dataframe(pd.DataFrame(st.session_state['expert_topic_annotations']))
            csv_data = _export_validation_to_csv(st.session_state['expert_topic_annotations'])
            if csv_data is not None:
                st.download_button(
                    label="💾 Unduh Validasi Topik (CSV)",
                    data=csv_data,
                    file_name="expert_topic_validation.csv",
                    mime="text/csv"
                )


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

    app_mode = st.sidebar.selectbox(
        "Mode Aplikasi",
        ["Analisis", "Validasi Ahli Diplomasi"]
    )
    st.sidebar.info("Mode Validasi Ahli Diplomasi digunakan setelah analisis selesai.")

    # ========== SIDEBAR FILTERS (Available after analysis) ==========
    if st.session_state.get('analysis_done', False) and app_mode == "Analisis":
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Data Filters")
        
        # Get available topics and stances
        posts_df = st.session_state.get('posts_df', pd.DataFrame())
        comments_df = st.session_state.get('comments_df', pd.DataFrame())
        
        if not posts_df.empty and not comments_df.empty:
            # Topic Filter
            available_topics = sorted([t for t in posts_df['Topik'].unique() if t != -1])
            selected_topics = st.sidebar.multiselect(
                "Filter by Topics",
                options=available_topics,
                default=available_topics,
                key="topic_filter",
                help="Select topics to include in visualizations"
            )
            
            # Stance Filter
            if 'sentiment' in comments_df.columns:
                available_stances = sorted(comments_df['sentiment'].dropna().unique())
                selected_stances = st.sidebar.multiselect(
                    "Filter by Stance",
                    options=available_stances,
                    default=available_stances,
                    key="stance_filter",
                    help="Select stances to include in visualizations"
                )
            else:
                st.sidebar.warning("⚠️ Stance analysis belum selesai. Jalankan analisis terlebih dahulu.")
                available_stances = []
                selected_stances = []
            
            # Confidence Threshold
            min_confidence = st.sidebar.slider(
                "Minimum Confidence Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="confidence_filter",
                help="Filter comments by minimum confidence score"
            )
            
            # Apply Filters Button
            if st.sidebar.button("🔄 Apply Filters", use_container_width=True):
                st.session_state['filters_applied'] = True
                st.rerun()
            
            # Reset Filters Button
            if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
                st.session_state['filters_applied'] = False
                # Reset to defaults
                st.session_state['topic_filter'] = available_topics
                st.session_state['stance_filter'] = available_stances
                st.session_state['confidence_filter'] = 0.0
                st.rerun()
            
            st.sidebar.info("ℹ️ Filters apply to all visualizations below")

    if app_mode == "Analisis":
        # Asumsikan kolom: full_text (posts), created_at (timestamp), full_text_comments (comments)
        required_cols = ['full_text', 'created_at', 'full_text_comments']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dataset harus memiliki kolom: {', '.join(required_cols)}")
            st.stop()

        # Proses posts untuk topic modeling
        posts_df = df[['full_text', 'created_at']].dropna().drop_duplicates()
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
        posts_df = posts_df.sort_values(by='created_at')

        # Proses comments untuk stance analysis
        comment_cols = ['full_text_comments']
        if 'expert_stance' in df.columns:
            comment_cols.append('expert_stance')
        comments_df = df[comment_cols].dropna(subset=['full_text_comments'])

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
            
            # Setup results directory
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save original data
            df.to_csv(os.path.join(results_dir, f"original_data_{timestamp}.csv"), index=False)
            
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
                
                # Store model in session state immediately
                st.session_state['topic_model'] = topic_model
                
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
                
                # Assign topics to posts_df
                posts_df['Topik'] = topics
                
                st.markdown("---")
                
                # Progress 3: Topics Over Time
                status_text.text("🔄 Tahap 3/4: Menghitung Topics Over Time...")
                progress_bar.progress(0.75)
                logging.info("Calculating topics over time")
                topics_over_time = cached_topics_over_time(topic_model, docs, timestamps, _nr_bins=20)
                
                # Progress 4: Complete
                status_text.text("✅ Tahap 4/4: Selesai!")
                progress_bar.progress(1.0)

            st.success("✅ Topic Modeling Selesai!")

            # ========== APPLY FILTERS TO DATA ==========
            if st.session_state.get('filters_applied', False):
                # Get filter values
                selected_topics = st.session_state.get('topic_filter', [])
                selected_stances = st.session_state.get('stance_filter', [])
                min_confidence = st.session_state.get('confidence_filter', 0.0)
                
                # Apply filters
                filtered_posts_df = posts_df[posts_df['Topik'].isin(selected_topics)] if selected_topics else posts_df
                if selected_stances and 'sentiment' in comments_df.columns and 'confidence' in comments_df.columns:
                    filtered_comments_df = comments_df[
                        (comments_df['sentiment'].isin(selected_stances)) &
                        (comments_df['confidence'] >= min_confidence)
                    ]
                else:
                    filtered_comments_df = comments_df
                
                st.info(f"📊 **Filtered Results**: {len(filtered_posts_df)} posts, {len(filtered_comments_df)} comments")
            else:
                # Use unfiltered data
                filtered_posts_df = posts_df
                filtered_comments_df = comments_df

            st.subheader("📈 Topics Over Time")
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            st.plotly_chart(fig, use_container_width=True)
            
            # Save the plot
            fig.write_html(os.path.join(results_dir, f"topics_over_time_{timestamp}.html"))
            
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

            # ========== WORD CLOUDS PER TOPIC ==========
            st.subheader("☁️ Word Clouds per Topic")
            
            if topic_model is not None:
                # Get available topics for word clouds
                available_topics = [topic for topic in topic_model.get_topics().keys() if topic != -1]
                
                if available_topics:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        selected_topic_wc = st.selectbox(
                            "Select Topic for Word Cloud",
                            options=available_topics,
                            format_func=lambda x: f"Topic {x}: {topic_model.get_topic_info(x)['Name'].iloc[0] if x in topic_model.get_topic_info()['Topic'].values else f'Topic {x}'}",
                            key="wordcloud_topic_selector"
                        )
                    
                    with col2:
                        if selected_topic_wc is not None:
                            # Get topic words and weights
                            topic_words = topic_model.get_topic(selected_topic_wc)
                            if topic_words:
                                # Create word cloud
                                word_freq = {word: weight for word, weight in topic_words}
                                
                                # Generate word cloud
                                wordcloud = WordCloud(
                                    width=800, 
                                    height=400, 
                                    background_color='white',
                                    colormap='viridis',
                                    max_words=50
                                ).generate_from_frequencies(word_freq)
                                
                                # Display word cloud
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title(f'Word Cloud for Topic {selected_topic_wc}', fontsize=16, pad=20)
                                st.pyplot(fig)
                                
                                # Save the word cloud
                                fig.savefig(os.path.join(results_dir, f"wordcloud_topic_{selected_topic_wc}_{timestamp}.png"))
                                
                                # Show top words as text
                                with st.expander("📝 Top Words & Weights"):
                                    words_df = pd.DataFrame(topic_words, columns=['Word', 'Weight'])
                                    st.dataframe(words_df.head(20), use_container_width=True)
                            else:
                                st.info("No words available for this topic.")
                else:
                    st.info("No topics available for word clouds.")
            else:
                st.info("Topic model not available. Please run analysis first.")

            st.subheader("📌 Top Topics")
            if topic_model is not None:
                top_topics_df = topic_model.get_topic_info()
                st.dataframe(top_topics_df, use_container_width=True)
            else:
                st.error("Topic model is not available. Please run the analysis first.")
                top_topics_df = pd.DataFrame()  # Create empty dataframe to prevent further errors

            # Persist results for expert validation
            st.session_state['analysis_done'] = True
            st.session_state['comments_df'] = comments_df.copy()
            st.session_state['posts_df'] = posts_df.copy()
            st.session_state['topic_model'] = topic_model
            topic_validation_df = top_topics_df[top_topics_df['Topic'] != -1][['Topic', 'Name']].copy()
            topic_validation_df['Top Words'] = topic_validation_df['Topic'].apply(
                lambda topic_id: ", ".join([word for word, _ in topic_model.get_topic(int(topic_id))[:10]])
            )
            st.session_state['topic_validation_df'] = topic_validation_df
            topic_docs_mapping = {}
            for topic_id, group in posts_df.groupby('Topik'):
                if topic_id == -1:
                    continue
                topic_docs_mapping[int(topic_id)] = group['full_text_preprocessed'].head(5).tolist()
            st.session_state['topic_docs_mapping'] = topic_docs_mapping
            
            # Save all results to files
            
            # Save posts with topics
            posts_df.to_csv(os.path.join(results_dir, f"posts_with_topics_{timestamp}.csv"), index=False)
            
            # Save comments with stance
            comments_df.to_csv(os.path.join(results_dir, f"comments_with_stance_{timestamp}.csv"), index=False)
            
            # Save top topics
            top_topics_df.to_csv(os.path.join(results_dir, f"top_topics_{timestamp}.csv"), index=False)
            
            # Save topic validation
            topic_validation_df.to_csv(os.path.join(results_dir, f"topic_validation_{timestamp}.csv"), index=False)
            
            # Calculate and save coherence
            coherence_results = calculate_topic_coherence(topic_model, docs)
            import json
            with open(os.path.join(results_dir, f"coherence_results_{timestamp}.json"), "w") as f:
                json.dump(coherence_results, f)
            
            # Calculate and save topic metrics
            topic_metrics = calculate_topic_metrics(topic_model, docs)
            with open(os.path.join(results_dir, f"topic_metrics_{timestamp}.json"), "w") as f:
                json.dump(topic_metrics, f)
            
            st.success(f"✅ Semua hasil analisis telah disimpan ke folder '{results_dir}'!")
            
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
            st.dataframe(filtered_comments_df.head(20), use_container_width=True)
            
            # Evaluation metrics if expert ground truth is available
            if 'expert_stance' in comments_df.columns:
                y_true = comments_df['expert_stance'].astype(str)
                y_pred = comments_df['sentiment'].astype(str)

                accuracy = accuracy_score(y_true, y_pred)
                precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
                report_text = classification_report(y_true, y_pred, zero_division=0)

                st.subheader("📊 Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    st.metric("Precision (macro)", f"{precision_macro:.3f}")
                with col2:
                    st.metric("Recall (macro)", f"{recall_macro:.3f}")
                    st.metric("F1 Score (macro)", f"{f1_macro:.3f}")
                with col3:
                    st.metric("Precision (weighted)", f"{precision_weighted:.3f}")
                    st.metric("Recall (weighted)", f"{recall_weighted:.3f}")
                    st.metric("F1 Score (weighted)", f"{f1_weighted:.3f}")

                with st.expander("📄 Classification Report"):
                    st.text(report_text)
            else:
                st.info("Upload dataset dengan kolom 'expert_stance' untuk menampilkan evaluasi metrik (accuracy, precision, recall, F1).")
            
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
            if 'confidence' in filtered_comments_df.columns:
                st.subheader("📊 Distribusi Confidence Score")
                confidence_counts = pd.cut(filtered_comments_df['confidence'], bins=[0, 0.6, 0.7, 0.8, 1.0], 
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
            sentiment_counts = filtered_comments_df['sentiment'].value_counts()
            
            st.subheader("📊 Ringkasan Sentimen/Stance")
            
            # Create interactive Plotly bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Colors for negative, neutral, positive
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Click to see samples<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Sentiment Distribution",
                xaxis_title="Sentiment",
                yaxis_title="Count",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save the plot
            fig.write_html(os.path.join(results_dir, f"sentiment_distribution_{timestamp}.html"))
            
            # Sample Comments Display
            st.subheader("📝 Sample Comments by Sentiment")
            
            # Select sentiment to show samples
            selected_sentiment = st.selectbox(
                "Select sentiment to view sample comments:",
                options=sentiment_counts.index.tolist(),
                key="sentiment_sample_selector",
                help="Choose a sentiment category to see random sample comments"
            )
            
            if selected_sentiment:
                # Get sample comments for selected sentiment
                sentiment_comments = filtered_comments_df[
                    filtered_comments_df['sentiment'] == selected_sentiment
                ]['full_text_comments'].dropna()
                
                if not sentiment_comments.empty:
                    # Show random samples (up to 5)
                    num_samples = min(5, len(sentiment_comments))
                    sample_comments = sentiment_comments.sample(n=num_samples, random_state=42)
                    
                    st.markdown(f"**Showing {num_samples} random {selected_sentiment.lower()} comments:**")
                    
                    for idx, comment in enumerate(sample_comments, 1):
                        with st.expander(f"💬 Sample {idx}"):
                            st.write(comment)
                            # Show preprocessed version too
                            preprocessed = filtered_comments_df[
                                filtered_comments_df['full_text_comments'] == comment
                            ]['full_text_comments_preprocessed'].iloc[0] if len(filtered_comments_df[
                                filtered_comments_df['full_text_comments'] == comment
                            ]) > 0 else "N/A"
                            st.markdown(f"**Preprocessed:** {preprocessed}")
                else:
                    st.info(f"No {selected_sentiment.lower()} comments available with current filters.")
            
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
            
            # ========== TOPIC COHERENCE EVALUATION ==========
            st.subheader("🎯 Topic Coherence Evaluation")
            
            # Calculate coherence
            with st.spinner("Calculating topic coherence..."):
                coherence_results = calculate_topic_coherence(topic_model, docs)
            
            if "error" not in coherence_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "📊 Overall Coherence", 
                        f"{coherence_results['overall_coherence']:.4f}",
                        help=f"Coherence measure: {coherence_results['coherence_type']}"
                    )
                
                with col2:
                    st.metric(
                        "📈 Number of Topics", 
                        coherence_results['num_topics']
                    )
                
                with col3:
                    avg_topic_coherence = np.mean(coherence_results['topic_coherences'])
                    st.metric(
                        "📉 Average Topic Coherence", 
                        f"{avg_topic_coherence:.4f}"
                    )
                
                # Coherence interpretation
                st.subheader("📋 Coherence Interpretation")
                
                overall_score = coherence_results['overall_coherence']
                if overall_score >= 0.6:
                    coherence_quality = "Excellent"
                    color = "🟢"
                elif overall_score >= 0.4:
                    coherence_quality = "Good"
                    color = "🟡"
                elif overall_score >= 0.2:
                    coherence_quality = "Fair"
                    color = "🟠"
                else:
                    coherence_quality = "Poor"
                    color = "🔴"
                
                st.markdown(f"**{color} Topic Coherence Quality: {coherence_quality}**")
                
                with st.expander("📊 Detailed Coherence Scores per Topic"):
                    coherence_df = pd.DataFrame({
                        'Topic': [f'Topic {i}' for i in range(len(coherence_results['topic_coherences']))],
                        'Coherence Score': coherence_results['topic_coherences']
                    })
                    coherence_df = coherence_df.sort_values('Coherence Score', ascending=False)
                    st.dataframe(coherence_df, use_container_width=True)
                    
                    # Plot coherence distribution
                    fig = go.Figure(data=[
                        go.Bar(
                            x=coherence_df['Topic'],
                            y=coherence_df['Coherence Score'],
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Topic Coherence Scores",
                        xaxis_title="Topic",
                        yaxis_title="Coherence Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ About Topic Coherence"):
                    st.markdown("""
                    **Topic Coherence** mengukur seberapa baik kata-kata dalam sebuah topik saling berhubungan.
                    
                    **Interpretasi Score:**
                    - **0.6+**: Excellent - Topik sangat koheren
                    - **0.4-0.6**: Good - Topik cukup koheren  
                    - **0.2-0.4**: Fair - Topik perlu improvement
                    - **<0.2**: Poor - Topik kurang bermakna
                    
                    **Metode yang digunakan:** C_V (Context Vector)
                    """)
            else:
                st.error(f"Error calculating coherence: {coherence_results['error']}")
            
            # ========== ADDITIONAL TOPIC METRICS ==========
            st.subheader("📈 Additional Topic Modeling Metrics")
            
            # Calculate additional metrics
            with st.spinner("Calculating additional topic metrics..."):
                topic_metrics = calculate_topic_metrics(topic_model, docs)
            
            if "error" not in topic_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Topic Diversity", 
                        f"{topic_metrics['avg_topic_diversity']:.1f}",
                        help="Average unique words per topic (top 10)"
                    )
                
                with col2:
                    st.metric(
                        "📊 Document Coverage", 
                        f"{topic_metrics['doc_coverage']:.1%}",
                        help="Percentage of documents assigned to topics"
                    )
                
                with col3:
                    st.metric(
                        "📏 Topic Size CV", 
                        f"{topic_metrics['topic_size_cv']:.3f}",
                        help="Coefficient of variation in topic sizes"
                    )
                
                with col4:
                    st.metric(
                        "📚 Unique Words", 
                        f"{topic_metrics['total_unique_words']:,}",
                        help="Total unique words across all topics"
                    )
                
                # Topic size distribution
                with st.expander("📊 Topic Size Distribution"):
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Topic {i}' for i in range(len(topic_metrics['topic_sizes']))],
                            y=topic_metrics['topic_sizes'],
                            marker_color='lightgreen'
                        )
                    ])
                    fig.update_layout(
                        title="Document Distribution Across Topics",
                        xaxis_title="Topic",
                        yaxis_title="Number of Documents",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Topic diversity distribution
                with st.expander("🎨 Topic Diversity Distribution"):
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Topic {i}' for i in range(len(topic_metrics['topic_diversities']))],
                            y=topic_metrics['topic_diversities'],
                            marker_color='orange'
                        )
                    ])
                    fig.update_layout(
                        title="Topic Diversity (Unique Words per Topic)",
                        xaxis_title="Topic",
                        yaxis_title="Unique Words",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ About Additional Metrics"):
                    st.markdown("""
                    **Topic Diversity**: Rata-rata kata unik dalam top 10 kata setiap topik. Nilai tinggi menunjukkan topik yang beragam.
                    
                    **Document Coverage**: Persentase dokumen yang berhasil diklasifikasikan ke topik (bukan outlier).
                    
                    **Topic Size CV**: Koefisien variasi ukuran topik. Nilai rendah menunjukkan distribusi yang merata.
                    
                    **Unique Words**: Total kata unik yang muncul dalam semua topik.
                    """)
            else:
                st.error(f"Error calculating topic metrics: {topic_metrics['error']}")
            
            logging.info("Analysis completed successfully")
    elif app_mode == "Validasi Ahli Diplomasi":
        render_expert_validation_ui()