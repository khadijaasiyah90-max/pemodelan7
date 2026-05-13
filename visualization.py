import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from pathlib import Path

sns.set(style="whitegrid")


def plot_topic_distribution(posts_df: pd.DataFrame, output_path: str | Path) -> plt.Figure:
    df = posts_df.copy()
    counts = df["topic_name"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.4)))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis", ax=ax)
    ax.set_title("Distribusi Topik pada Postingan")
    ax.set_xlabel("Jumlah Postingan")
    ax.set_ylabel("Topik")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return fig


def plot_stance_distribution(comments_df: pd.DataFrame, output_path: str | Path) -> plt.Figure:
    df = comments_df.copy()
    counts = df["stance"].value_counts(normalize=True).mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, palette="coolwarm", ax=ax)
    ax.set_title("Distribusi Stance pada Komentar")
    ax.set_xlabel("Persentase (%)")
    ax.set_ylabel("Stance")
    for index, value in enumerate(counts.values):
        ax.text(value + 0.5, index, f"{value:.1f}%", va="center")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return fig


def plot_heatmap_topic_stance(merged_df: pd.DataFrame, output_path: str | Path) -> plt.Figure:
    df = merged_df.copy()
    pivot = df.groupby(["topic_name", "stance"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Heatmap Topik vs Stance")
    ax.set_xlabel("Stance")
    ax.set_ylabel("Topik")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return fig


def plot_topic_over_time(posts_df: pd.DataFrame, output_path_html: str | Path, freq: str = "M") -> px.Figure:
    df = posts_df.copy()
    if "created_at" not in df.columns:
        raise ValueError("created_at column required for topic-over-time visualization.")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    if df.empty:
        raise ValueError("No valid timestamps available for topic over time visualization.")

    df["period"] = df["created_at"].dt.to_period(freq).dt.to_timestamp()
    counts = df.groupby(["period", "topic_name"]).size().reset_index(name="count")

    fig = px.line(
        counts,
        x="period",
        y="count",
        color="topic_name",
        markers=True,
        title="Topic Frequency Over Time",
        labels={"period": "Waktu", "count": "Jumlah Postingan", "topic_name": "Topik"},
    )
    fig.update_layout(legend_title_text="Topik", template="plotly_white")
    Path(output_path_html).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path_html)
    return fig


def plot_posts_per_topic(posts_df: pd.DataFrame, output_path: str | Path) -> plt.Figure:
    df = posts_df.copy()
    counts = df["topic_name"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.4)))
    sns.barplot(x=counts.values, y=counts.index, palette="magma", ax=ax)
    ax.set_title("Jumlah Postingan per Topik")
    ax.set_xlabel("Jumlah Postingan")
    ax.set_ylabel("Topik")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return fig
