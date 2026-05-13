import pandas as pd
from pathlib import Path

REQUIRED_POST_COLUMNS = ["post_id", "full_text", "clean_text", "created_at"]
REQUIRED_COMMENT_COLUMNS = ["post_id", "comment_id", "full_text_comments", "clean_comments"]


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load a CSV or Excel dataset into a DataFrame."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if path.suffix.lower() in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or XLSX.")

    df.columns = [str(column).strip().lower() for column in df.columns]
    return df


def prepare_post_comment_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the loaded dataset into posts and comments DataFrames."""
    missing_post_cols = [col for col in REQUIRED_POST_COLUMNS if col not in df.columns]
    if missing_post_cols:
        raise ValueError(f"Missing required post columns: {missing_post_cols}")

    if "comment_id" not in df.columns:
        df = df.copy()
        df["comment_id"] = range(1, len(df) + 1)

    missing_comment_cols = [col for col in REQUIRED_COMMENT_COLUMNS if col not in df.columns]
    if missing_comment_cols:
        raise ValueError(f"Missing required comment columns: {missing_comment_cols}")

    posts_df = (
        df[["post_id", "full_text", "clean_text", "created_at"]]
        .drop_duplicates(subset=["post_id"], keep="first")
        .reset_index(drop=True)
    )

    comments_df = (
        df[["post_id", "comment_id", "full_text_comments", "clean_comments"]]
        .copy()
        .reset_index(drop=True)
    )

    comments_df["comment_id"] = comments_df["comment_id"].astype(str)
    posts_df["post_id"] = posts_df["post_id"].astype(str)
    comments_df["post_id"] = comments_df["post_id"].astype(str)

    return posts_df, comments_df
