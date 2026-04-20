import streamlit as st
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="Dynamic Topic Modeling & Stance Analysis", layout="wide")

st.title("📊 Dynamic Topic Modeling & Stance Analysis")
st.write("Analisis topik dinamis pada unggahan dan analisis stance pada komentar")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Asumsikan kolom: full_text (posts), created_at (timestamp), full_text_comments (comments)
    required_cols = ['full_text', 'created_at', 'full_text_comments']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Dataset harus memiliki kolom: {', '.join(required_cols)}")
    else:
        # Proses posts untuk topic modeling
        posts_df = df[['full_text', 'created_at']].dropna().drop_duplicates()
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
        posts_df = posts_df.sort_values(by='created_at')

        docs = posts_df['full_text'].astype(str).tolist()
        timestamps = posts_df['created_at'].tolist()

        # Proses comments untuk stance analysis
        comments_df = df[['full_text_comments']].dropna()

        st.info("Memuat model...")
        embedding_model = load_embedding_model()
        sentiment_model = load_sentiment_model()

        if st.button("🚀 Jalankan Analisis"):
            with st.spinner("Processing Topic Modeling..."):
                topic_model = BERTopic(embedding_model=embedding_model)
                topics, probs = topic_model.fit_transform(docs)

                try:
                    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=20)
                except ValueError as e:
                    st.warning(f"Error dengan nr_bins=20: {e}. Menggunakan nr_bins=10.")
                    topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=10)

            st.success("Topic Modeling Selesai!")

            st.subheader("📈 Topics Over Time")
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 Top Topics")
            st.dataframe(topic_model.get_topic_info())

            # Stance Analysis pada komentar
            st.subheader("🗣️ Stance Analysis pada Komentar")
            with st.spinner("Analyzing sentiments..."):
                sentiments = sentiment_model(comments_df['full_text_comments'].tolist())
                comments_df['sentiment'] = [s['label'] for s in sentiments]
                comments_df['confidence'] = [s['score'] for s in sentiments]

            st.dataframe(comments_df.head(20))

            # Summary sentiment
            sentiment_counts = comments_df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)