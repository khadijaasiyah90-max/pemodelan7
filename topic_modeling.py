#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Topic Modeling Script
Tidak perlu Streamlit, cukup run: python topic_modeling.py
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
from pathlib import Path

print("=" * 60)
print("📊 TOPIC MODELING - Standalone Script")
print("=" * 60)

# Create results folder
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    # ===== LOAD DATA =====
    print("\n📂 Loading dataset...")
    
    # Try common filenames
    csv_files = list(Path(".").glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found. Please upload a CSV file in this directory.")
        exit(1)
    
    csv_file = csv_files[0]
    print(f"✅ Found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} rows")
    
    # Find text column
    text_cols = [c for c in df.columns if 'text' in c.lower()]
    if not text_cols:
        print("❌ No text column found. Available columns:", df.columns.tolist())
        exit(1)
    
    text_col = text_cols[0]
    print(f"✅ Using column: {text_col}")
    
    # ===== PREPROCESSING =====
    print("\n🧹 Preprocessing...")
    
    # Custom stopwords for political content
    custom_stopwords = [
        "presiden", "prabowo", "indonesia", "rakyat", "negara",
        "pak", "bapak", "yg", "aja", "nya", "nih", "sih",
        "jadi", "yang", "dan", "atau", "dengan", "untuk", "dari",
        "dalam", "pada", "ke", "di", "itu", "ini", "adalah",
        "jika", "maka", "karena", "seperti", "sudah", "belum",
        "akan", "bisa", "harus", "ingin", "mau", "perlu"
    ]
    
    def preprocess(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    docs = df[text_col].fillna('').astype(str).apply(preprocess).tolist()
    docs = [d for d in docs if len(d.strip()) > 10]
    # docs = docs[:1000]  # Limit for speed - commented out for full dataset
    
    print(f"✅ Processed {len(docs)} documents")
    
    # ===== LOAD MODELS =====
    print("\n🤖 Loading models...")
    
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from bertopic.representation import MaximalMarginalRelevance
    
    print("  - Embedding model...")
    embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("    ✅ Loaded")
    
    print("  - Vectorizer...")
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), 
        min_df=5, 
        max_df=0.9,
        stop_words=custom_stopwords
    )
    print("    ✅ Loaded")
    
    print("  - UMAP...")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    print("    ✅ Loaded")
    
    print("  - HDBSCAN...")
    hdbscan_model = HDBSCAN(
        min_cluster_size=80,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    print("    ✅ Loaded")
    
    print("  - MaximalMarginalRelevance...")
    representation = MaximalMarginalRelevance(diversity=0.3)
    print("    ✅ Loaded")
    
    # ===== FIT BERTOPIC =====
    print("\n🔄 Fitting BERTopic model (this may take 1-5 minutes)...")
    
    topic_model = BERTopic(
        embedding_model=embedding,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation,
        nr_topics=12,
        min_topic_size=80,
        calculate_probabilities=True,
        verbose=False
    )
    
    topics, probs = topic_model.fit_transform(docs)
    print("✅ Model fitted!")
    
    # ===== GET RESULTS =====
    print("\n📊 Generating results...")
    
    topics_info = topic_model.get_topic_info()
    n_topics = len(topics_info[topics_info['Topic'] != -1])
    
    print(f"✅ Topics found: {n_topics}")
    
    # ===== COHERENCE =====
    print("\n🎯 Calculating coherence...")
    
    try:
        from gensim.models import CoherenceModel
        from gensim.corpora import Dictionary
        
        tokenized = [doc.split() for doc in docs]
        dictionary = Dictionary(tokenized)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized]
        
        # Extract topic words
        topic_words = []
        for tid in sorted(topic_model.get_topics().keys()):
            if tid != -1:
                words = [w for w, _ in topic_model.get_topic(tid)[:10]]
                topic_words.append(words)
        
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            corpus=corpus,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_score = coherence_model.get_coherence()
        print(f"✅ Overall Coherence (C_V): {coherence_score:.4f}")
        
    except Exception as e:
        print(f"⚠️  Coherence calculation failed: {e}")
        coherence_score = None
    
    # ===== SAVE RESULTS =====
    print("\n💾 Saving results...")
    
    # 1. Topics info
    topics_info.to_csv(f"results/topics_info_{timestamp}.csv", index=False)
    print(f"  ✅ topics_info_{timestamp}.csv")
    
    # 2. Topic distribution
    topic_dist = pd.DataFrame({
        'Topic': pd.Series(topics).value_counts().index,
        'Count': pd.Series(topics).value_counts().values
    })
    topic_dist.to_csv(f"results/topic_distribution_{timestamp}.csv", index=False)
    print(f"  ✅ topic_distribution_{timestamp}.csv")
    
    # 3. Metadata
    metadata = {
        "timestamp": timestamp,
        "n_documents": len(docs),
        "n_topics": n_topics,
        "coherence_score": float(coherence_score) if coherence_score else None,
        "model_config": {
            "embedding": "paraphrase-multilingual-MiniLM-L12-v2",
            "min_topic_size": 20,
            "nr_topics": "auto",
            "n_neighbors": 15,
            "min_cluster_size": 20
        }
    }
    
    with open(f"results/metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ metadata_{timestamp}.json")
    
    # ===== DISPLAY SUMMARY =====
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"Documents processed: {len(docs)}")
    print(f"Topics found: {n_topics}")
    if coherence_score:
        print(f"Coherence (C_V): {coherence_score:.4f}")
        if coherence_score >= 0.6:
            print("  → Quality: 🟢 EXCELLENT")
        elif coherence_score >= 0.4:
            print("  → Quality: 🟡 GOOD")
        elif coherence_score >= 0.2:
            print("  → Quality: 🟠 FAIR")
        else:
            print("  → Quality: 🔴 POOR")
    
    print("\n📌 Top Topics:")
    top_topics = topics_info[topics_info['Topic'] != -1].head(10)
    for idx, row in top_topics.iterrows():
        print(f"  Topic {int(row['Topic'])}: {row['Name']} ({int(row['Count'])} docs)")
    
    print("\n✅ Results saved to 'results/' folder")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
