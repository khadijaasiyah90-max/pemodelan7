import argparse
from pathlib import Path
import pandas as pd

from load_data import load_dataset, prepare_post_comment_data
from topic_modeling import build_topic_model
from stance_analysis import run_stance_analysis
from merge_analysis import merge_topic_and_stance, stance_distribution_by_topic, topic_stance_counts
from visualization import (
    plot_topic_distribution,
    plot_stance_distribution,
    plot_heatmap_topic_stance,
    plot_topic_over_time,
    plot_posts_per_topic,
)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main(dataset_path: str, results_dir: str, batch_size: int = 32) -> None:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    print("🔧 Preparing post and comment tables")
    posts_df, comments_df = prepare_post_comment_data(dataset)
    print(f"  - Posts: {len(posts_df)} rows")
    print(f"  - Comments: {len(comments_df)} rows")

    print("📊 Running Dynamic Topic Modeling")
    topic_posts_df, topic_model, topic_info = build_topic_model(posts_df)
    save_dataframe(topic_posts_df, results_path / "posts_with_topics.csv")
    print(f"  - Saved posts with topics to {results_path / 'posts_with_topics.csv'}")
    topic_info.to_csv(results_path / "topic_info.csv", index=False)

    print("🗣️ Running stance analysis on comments")
    stance_comments_df = run_stance_analysis(topic_posts_df, comments_df, batch_size=batch_size)
    save_dataframe(stance_comments_df, results_path / "comments_with_stance.csv")
    print(f"  - Saved comments with stance to {results_path / 'comments_with_stance.csv'}")

    print("🔗 Merging topic and stance results")
    merged_df = merge_topic_and_stance(topic_posts_df, stance_comments_df)
    save_dataframe(merged_df, results_path / "merged_topic_stance.csv")
    print(f"  - Saved merged results to {results_path / 'merged_topic_stance.csv'}")

    print("📈 Calculating stance distribution by topic")
    distribution_df = stance_distribution_by_topic(merged_df)
    save_dataframe(distribution_df, results_path / "stance_distribution_by_topic.csv")
    print(f"  - Saved distribution summary to {results_path / 'stance_distribution_by_topic.csv'}")

    print("📊 Creating visualizations")
    plot_topic_distribution(topic_posts_df, results_path / "topic_distribution.png")
    plot_posts_per_topic(topic_posts_df, results_path / "posts_per_topic.png")
    plot_stance_distribution(stance_comments_df, results_path / "stance_distribution.png")
    plot_heatmap_topic_stance(merged_df, results_path / "topic_stance_heatmap.png")
    plot_topic_over_time(topic_posts_df, results_path / "topic_over_time.html")

    print("✅ All analysis and visualizations saved in the results directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic Topic Modeling + Stance Analysis for Twitter/X research datasets"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the input CSV or XLSX dataset containing social media posts and comments.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Folder where analysis outputs and visualizations are saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for stance analysis inference.",
    )
    args = parser.parse_args()
    main(args.dataset_path, args.results_dir, batch_size=args.batch_size)
