import pandas as pd
from transformers import pipeline
from typing import Optional


def _normalize_label(label: str, id2label: Optional[dict] = None) -> str:
    label = str(label).upper()
    if label.startswith("LABEL_") and id2label is not None:
        numeric = int(label.replace("LABEL_", ""))
        label = id2label.get(numeric, label)

    if any(token in label for token in ["NEG", "AGAINST", "CONTRA", "TIDAK", "NO"]):
        return "Against"
    if any(token in label for token in ["NEU", "NET", "NEUTRAL"]):
        return "Neutral"
    if any(token in label for token in ["POS", "FAVOR", "FOR", "SUPPORT", "SETUJU"]):
        return "Favor"
    return "Favor"


def run_stance_analysis(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int = 32,
    confidence_threshold: float = 0.70,
) -> pd.DataFrame:
    """Perform stance analysis on comments using the parent post context."""
    if comments_df.empty:
        return comments_df.copy()

    comments_df = comments_df.copy()
    comments_df["stance"] = "Neutral"
    comments_df["stance_confidence"] = 0.0

    model = pipeline("sentiment-analysis", model=model_name)
    id2label = getattr(model.model.config, "id2label", None)

    post_texts = posts_df.set_index("post_id")["clean_text"].to_dict()
    inputs = []
    for record in comments_df.itertuples(index=False):
        post_text = post_texts.get(str(record.post_id), "")
        comment_text = str(record.clean_comments or "")
        inputs.append(f"Post: {post_text} \nComment: {comment_text}")

    results = model(inputs, batch_size=batch_size)
    for idx, prediction in enumerate(results):
        label = prediction.get("label", "Neutral")
        score = float(prediction.get("score", 0.0))
        stance = _normalize_label(label, id2label)
        if score < confidence_threshold and stance != "Neutral":
            stance = "Neutral"
        comments_df.at[idx, "stance"] = stance
        comments_df.at[idx, "stance_confidence"] = score

    return comments_df
