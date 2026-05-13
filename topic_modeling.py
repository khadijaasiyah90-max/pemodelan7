import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN


def build_topic_model(
    posts_df: pd.DataFrame,
    text_column: str = "clean_text",
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    min_cluster_size: int = 20,
    nr_topics: str = "auto",
    min_topic_size: int = 20,
) -> tuple[pd.DataFrame, BERTopic, pd.DataFrame]:
    """Fit a BERTopic model on the cleaned post text and assign topic labels."""
    docs = posts_df[text_column].fillna("").astype(str).tolist()
    if not docs:
        raise ValueError("No documents available for topic modeling.")

    embedding_model = SentenceTransformer(embedding_model_name)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9)
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    representation_model = MaximalMarginalRelevance(diversity=0.3)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        calculate_probabilities=True,
        verbose=False,
    )

    topics, probabilities = topic_model.fit_transform(docs)
    probability_values = []
    for prob in probabilities:
        if isinstance(prob, (list, tuple)) and prob:
            probability_values.append(float(prob[0]))
        else:
            probability_values.append(float(prob))

    topic_info = topic_model.get_topic_info()
    topic_names = {}
    for topic_id in sorted(topic_info[topic_info.Topic != -1]["Topic"].unique()):
        words = topic_model.get_topic(int(topic_id))
        if words:
            topic_names[int(topic_id)] = ", ".join([word for word, _ in words[:5]])
        else:
            topic_names[int(topic_id)] = f"Topic {topic_id}"
    topic_names[-1] = "Outlier"

    assigned_names = [topic_names.get(int(topic_id), "Outlier") for topic_id in topics]

    output = posts_df.copy()
    output["topic_id"] = topics
    output["topic_name"] = assigned_names
    output["topic_probability"] = probability_values

    return output, topic_model, topic_info
