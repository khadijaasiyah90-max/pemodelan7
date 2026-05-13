import pandas as pd


def merge_topic_and_stance(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
    """Merge post topic assignments with comment stance results."""
    merge_cols = ["post_id", "topic_id", "topic_name", "topic_probability"]
    if "full_text" in posts_df.columns:
        merge_cols.append("full_text")
    merged = comments_df.merge(
        posts_df[merge_cols],
        on="post_id",
        how="left",
        validate="many_to_one",
    )
    return merged


def stance_distribution_by_topic(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Return stance distribution (percentages) for each topic."""
    if merged_df.empty:
        return pd.DataFrame()

    pivot = (
        merged_df.groupby(["topic_name", "stance"]).size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
    )
    percentages = pivot.div(pivot.sum(axis=1), axis=0).fillna(0) * 100
    result = percentages.reset_index().sort_values(by="topic_name")
    return result


def topic_stance_counts(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Return raw count cross-tabulation of topic vs stance."""
    if merged_df.empty:
        return pd.DataFrame()
    return merged_df.groupby(["topic_name", "stance"]).size().unstack(fill_value=0)
