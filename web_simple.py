#!/usr/bin/env python3
"""
Topic Modeling Web App dengan Gradio
Jalankan dengan: python web_simple.py
"""

import gradio as gr
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime

def preprocess_text(text):
    """Preprocess text untuk bahasa Indonesia"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_topics(file):
    """Analisis topik dari file CSV"""
    if file is None:
        return "❌ Upload file CSV dulu", "", ""

    try:
        # Load data
        df = pd.read_csv(file.name)
        print(f"📁 File loaded: {len(df)} rows")

        # Cari kolom text
        text_cols = [c for c in df.columns if 'text' in c.lower() or 'comment' in c.lower()]
        if not text_cols:
            return "❌ Tidak ada kolom text/comment", "", ""

        text_col = text_cols[0]
        print(f"📝 Using column: {text_col}")

        # Preprocess
        docs = df[text_col].fillna('').astype(str).apply(preprocess_text).tolist()
        docs = [d for d in docs if len(d.strip()) > 10]
        docs = docs[:200]  # Limit untuk performa

        if len(docs) < 20:
            return "❌ Butuh minimal 20 dokumen valid", "", ""

        print(f"📊 Processing {len(docs)} documents")

        # Load models
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from bertopic import BERTopic
        from bertopic.representation import MaximalMarginalRelevance

        print("🤖 Loading models...")
        embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        representation = MaximalMarginalRelevance(diversity=0.3)

        model = BERTopic(
            embedding_model=embedding,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            representation_model=representation,
            nr_topics="auto",
            min_topic_size=10,
            calculate_probabilities=True,
            verbose=True
        )

        print("🚀 Fitting BERTopic model...")
        topics, probs = model.fit_transform(docs)

        topics_info = model.get_topic_info()
        n_topics = len(topics_info[topics_info['Topic'] != -1])

        print(f"✅ Found {n_topics} topics")

        # Hitung coherence
        coherence_text = "⚠️ Coherence calculation skipped (web performance)"
        try:
            from gensim.models import CoherenceModel
            from gensim.corpora import Dictionary

            tokenized = [doc.split() for doc in docs]
            dictionary = Dictionary(tokenized)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized]

            topic_words = []
            for tid in sorted(model.get_topics().keys()):
                if tid != -1:
                    words = [w for w, _ in model.get_topic(tid)[:10]]
                    topic_words.append(words)

            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokenized,
                corpus=corpus,
                dictionary=dictionary,
                coherence='c_v'
            )

            coherence_score = coherence_model.get_coherence()
            coherence_text = f"🎯 Coherence (C_V): {coherence_score:.4f}"

        except Exception as e:
            print(f"⚠️ Coherence failed: {e}")

        # Summary
        summary = f"""
📊 **Analisis Topik Selesai!**

- Dokumen diproses: {len(docs)}
- Topik ditemukan: {n_topics}
- {coherence_text}

📌 **Topik Teratas:**
"""
        top_topics = topics_info[topics_info['Topic'] != -1].head(5)
        for idx, row in top_topics.iterrows():
            summary += f"\n• Topik {int(row['Topic'])}: {row['Name']} ({int(row['Count'])} dokumen)"

        # Topics table
        topics_table = topics_info[topics_info['Topic'] != -1][['Topic', 'Name', 'Count']].to_string(index=False)

        return summary, topics_table, f"✅ Berhasil! {n_topics} topik ditemukan"

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"❌ Error: {str(e)}", "", ""

# Gradio Interface
with gr.Blocks(title="Topic Modeling Web App") as demo:
    gr.Markdown("# 📊 Dynamic Topic Modeling & Stance Analysis")
    gr.Markdown("Upload file CSV untuk analisis topik dengan BERTopic (optimal config)")

    with gr.Row():
        file_input = gr.File(label="📁 Upload Dataset CSV", file_types=[".csv"])

    with gr.Row():
        analyze_btn = gr.Button("🚀 Analisis Topik", variant="primary", size="lg")

    with gr.Row():
        summary_output = gr.Textbox(label="📋 Ringkasan Analisis", lines=8, show_copy_button=True)

    with gr.Row():
        topics_output = gr.Textbox(label="📊 Tabel Topik", lines=12, show_copy_button=True)

    status_output = gr.Textbox(label="📢 Status", show_copy_button=True)

    analyze_btn.click(
        analyze_topics,
        inputs=[file_input],
        outputs=[summary_output, topics_output, status_output]
    )

    gr.Markdown("---")
    gr.Markdown("💡 **Tips:** Upload CSV dengan kolom text/comment. Minimal 20 dokumen untuk hasil terbaik.")

if __name__ == "__main__":
    print("🌐 Starting Gradio web app...")
    print("📡 Server akan berjalan di: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)