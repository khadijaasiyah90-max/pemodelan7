"""
Enhanced Streamlit Dashboard for Granular Stance Analysis.

Features:
- Hierarchical display: Post → Comments → Stance Analysis
- Accordion/Expander for collapsible views
- Color-coding for stance labels
- Mini-summary statistics per post
- Interactive filter panel (stance, confidence, topic)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from pathlib import Path
import logging

# Import custom modules
from load_data import load_dataset, prepare_post_comment_data
from stance_analysis_granular import (
    initialize_gemini,
    run_granular_stance_analysis,
    aggregate_stance_by_post
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Granular Stance Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
/* Color-coding for stance labels */
.stance-mendukung {
    background-color: #90EE90;
    color: #000;
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
}

.stance-menolak {
    background-color: #FFB6C1;
    color: #000;
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
}

.stance-netral {
    background-color: #D3D3D3;
    color: #000;
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
}

/* Post container styling */
.post-container {
    border-left: 4px solid #1f77b4;
    padding: 16px;
    margin: 8px 0;
    background-color: #f8f9fa;
    border-radius: 4px;
}

/* Comment container styling */
.comment-container {
    border-left: 4px solid #17a2b8;
    padding: 12px;
    margin: 8px 0;
    background-color: #fff;
    border-radius: 4px;
}

/* Stats badge styling */
.stat-badge {
    display: inline-block;
    margin: 4px 8px 4px 0;
    padding: 4px 8px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

.stat-mendukung { background-color: #90EE90; }
.stat-menolak { background-color: #FFB6C1; }
.stat-netral { background-color: #D3D3D3; }
</style>
""", unsafe_allow_html=True)


def get_stance_badge_html(stance: str, weight: float) -> str:
    """Generate HTML badge for stance label."""
    stance_lower = stance.lower().strip()
    
    if stance_lower == "mendukung":
        color_class = "stance-mendukung"
        emoji = "🟩"
    elif stance_lower == "menolak":
        color_class = "stance-menolak"
        emoji = "🟥"
    else:  # Netral
        color_class = "stance-netral"
        emoji = "⬜"
    
    return f'<span class="{color_class}">{emoji} {stance} ({weight:.2f})</span>'


def render_mini_summary(post_stats: dict) -> None:
    """Render mini-summary statistics for a post."""
    col1, col2, col3 = st.columns(3)
    
    mendukung_pct = post_stats.get("mendukung_pct", 0)
    menolak_pct = post_stats.get("menolak_pct", 0)
    netral_pct = post_stats.get("netral_pct", 0)
    avg_weight = post_stats.get("avg_weight", 0)
    
    with col1:
        st.metric("🟩 Mendukung", f"{mendukung_pct:.1f}%")
    
    with col2:
        st.metric("🟥 Menolak", f"{menolak_pct:.1f}%")
    
    with col3:
        st.metric("⬜ Netral", f"{netral_pct:.1f}%")
    
    # Confidence average
    st.metric("Avg Confidence", f"{avg_weight:.2f}")


def render_stance_distribution_chart(post_stats: dict) -> None:
    """Render pie chart of stance distribution."""
    labels = ["Mendukung", "Menolak", "Netral"]
    values = [
        post_stats.get("mendukung_pct", 0),
        post_stats.get("menolak_pct", 0),
        post_stats.get("netral_pct", 0)
    ]
    colors = ["#90EE90", "#FFB6C1", "#D3D3D3"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textposition="inside",
        textinfo="label+percent"
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_post_with_comments(
    post_id: str,
    post_text: str,
    comments_data: pd.DataFrame,
    post_stats: dict,
    show_all: bool = False
) -> None:
    """Render a single post with its comments in accordion style or full vertical mode."""
    
    summary_line = (
        f"Distribusi: 🟩 {post_stats.get('mendukung_pct', 0):.0f}%  | "
        f"🟥 {post_stats.get('menolak_pct', 0):.0f}%  | "
        f"⬜ {post_stats.get('netral_pct', 0):.0f}%"
    )
    title_text = post_text if len(post_text) <= 120 else post_text[:120] + "..."

    if show_all:
        st.markdown(f"### 📌 Unggahan: {title_text}")
        st.markdown(f"**{summary_line}**")
        st.write("---")
        st.write("**Teks Unggahan:**")
        st.write(post_text)
        st.write("---")
    else:
        with st.expander(f"🔽 Unggahan: {title_text}", expanded=False):
            st.markdown(f"**{summary_line}**")
            st.write("---")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**Teks Unggahan:**")
                st.write(post_text)
            with col2:
                st.write("**Ringkasan Sikap**")
                st.markdown(f"- 🟩 Mendukung: **{post_stats.get('mendukung_pct', 0):.0f}%**")
                st.markdown(f"- 🟥 Menolak: **{post_stats.get('menolak_pct', 0):.0f}%**")
                st.markdown(f"- ⬜ Netral: **{post_stats.get('netral_pct', 0):.0f}%**")
                st.markdown(f"- 🎯 Avg Confidence: **{post_stats.get('avg_weight', 0):.2f}**")

            st.write("---")
            st.subheader("💬 Komentar dan Analisis Sikap")

            for idx, row in comments_data.iterrows():
                badge_html = get_stance_badge_html(row["stance_label"], row["stance_weight"])
                truncated_comment = row["full_text_comments"]
                if len(truncated_comment) > 220:
                    truncated_comment = truncated_comment[:220] + "..."

                st.markdown(f"**↳ Komentar {idx + 1}:** {truncated_comment}")
                st.markdown(f"- {badge_html}", unsafe_allow_html=True)
                st.markdown(f"- **Bobot:** {row['stance_weight']:.2f}")
                st.markdown(f"- **Alasan Singkat:** {row.get('stance_reasoning', 'Tidak tersedia')}")
                st.write("---")

            if st.checkbox(f"Tampilkan detail lengkap untuk {post_id}", key=f"details_{post_id}"):
                st.subheader("Detail Lengkap Komentar")
                for idx, row in comments_data.iterrows():
                    with st.expander(
                        f"Komentar {idx + 1}: {row['full_text_comments'][:60]}...",
                        expanded=False
                    ):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write("**Teks Komentar:**")
                            st.write(row["full_text_comments"])
                            
                            st.write("**Alasan Analisis:**")
                            st.write(row.get("stance_reasoning", "Tidak tersedia"))
                        
                        with col2:
                            st.write("**Sikap:**")
                            st.markdown(get_stance_badge_html(row["stance_label"], row["stance_weight"]), unsafe_allow_html=True)
                            
                            st.write("**Bobot:**")
                            st.metric("", f"{row['stance_weight']:.2f}")

        return

    st.subheader("💬 Komentar dan Analisis Sikap")

    for idx, row in comments_data.iterrows():
        badge_html = get_stance_badge_html(row["stance_label"], row["stance_weight"])
        truncated_comment = row["full_text_comments"]
        if len(truncated_comment) > 220:
            truncated_comment = truncated_comment[:220] + "..."

        st.markdown(f"**↳ Komentar {idx + 1}:** {truncated_comment}")
        st.markdown(f"- {badge_html}", unsafe_allow_html=True)
        st.markdown(f"- **Bobot:** {row['stance_weight']:.2f}")
        st.markdown(f"- **Alasan Singkat:** {row.get('stance_reasoning', 'Tidak tersedia')}")
        st.write("---")
    
    st.title("📊 Analisis Sikap Granular - Dashboard Interaktif")
    st.write("Tampilan hierarki: Unggahan Utama → Komentar → Analisis Sikap dengan Color-Coding & Filter Interaktif")
    
    # Sidebar: Filter controls
    st.sidebar.header("🎛️ Kontrol dan Filter")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Pilih sumber data:",
        options=["Upload File", "Sample Data"]
    )
    
    # Load data
    df = None
    posts_df = None
    comments_df = None
    
    if data_source == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_dataset(uploaded_file)
                st.sidebar.success("Dataset berhasil dimuat!")
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {e}")
                return
    else:
        # Try to load sample data
        sample_path = "sample_posts_comments.csv"
        if Path(sample_path).exists():
            try:
                df = load_dataset(sample_path)
                st.sidebar.success("Sample data berhasil dimuat!")
            except Exception as e:
                st.sidebar.error(f"Error loading sample data: {e}")
                return
        else:
            st.sidebar.error(f"Sample data not found at {sample_path}")
            return
    
    if df is None or df.empty:
        st.error("Tidak ada data untuk dianalisis.")
        return
    
    # Prepare post-comment structure
    try:
        posts_df, comments_df = prepare_post_comment_data(df)
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return
    
    # Check if stance analysis already exists
    has_stance_analysis = "stance_label" in comments_df.columns
    
    # Sidebar: Analysis controls
    st.sidebar.subheader("🔬 Kontrol Analisis")
    
    if not has_stance_analysis:
        st.sidebar.info("Jalankan analisis stance granular menggunakan Google Gemini API")
        
        run_analysis = st.sidebar.button("▶️ Jalankan Analisis Stance", key="run_analysis")
        
        if run_analysis:
            # Get API key from environment
            api_key = os.getenv("GOOGLE_API_KEY")
            
            if not api_key:
                st.sidebar.error("Google API Key tidak ditemukan di environment variable 'GOOGLE_API_KEY'")
                return
            
            # Show progress
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            try:
                status_text.write("Inisialisasi Gemini API...")
                initialize_gemini(api_key)
                
                status_text.write("Menjalankan analisis stance granular...")
                progress_bar.progress(50)
                
                comments_df_with_stance = run_granular_stance_analysis(
                    posts_df=posts_df,
                    comments_df=comments_df,
                    api_key=api_key,
                    model_name="gemini-1.5-flash",
                    sample_size=5  # Analyze first 5 posts for demo
                )
                
                progress_bar.progress(100)
                status_text.write("✅ Analisis selesai!")
                
                # Update in-memory dataframe
                comments_df = comments_df_with_stance
                has_stance_analysis = True
                
                # Option to save results
                if st.sidebar.button("💾 Simpan Hasil Analisis", key="save_results"):
                    output_path = f"results/stance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    Path("results").mkdir(exist_ok=True)
                    comments_df.to_csv(output_path, index=False)
                    st.sidebar.success(f"Hasil disimpan ke: {output_path}")
                
            except Exception as e:
                st.sidebar.error(f"Error dalam analisis: {e}")
                logger.error(f"Analysis error: {e}", exc_info=True)
    
    # Filter controls in sidebar
    if has_stance_analysis:
        st.sidebar.subheader("🔍 Filter Hasil Analisis")
        
        # Stance filter
        stance_options = ["Semua"] + list(comments_df["stance_label"].unique())
        selected_stance = st.sidebar.multiselect(
            "Filter Sikap:",
            options=stance_options,
            default=["Semua"]
        )
        
        # Confidence threshold
        min_confidence = st.sidebar.slider(
            "Bobot Keyakinan Minimum:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
        
        # Topic filter (if topic modeling exists)
        if "topic_id" in comments_df.columns:
            topic_options = ["Semua"] + sorted(comments_df["topic_id"].unique())
            selected_topics = st.sidebar.multiselect(
                "Filter Topik:",
                options=topic_options,
                default=["Semua"]
            )
        else:
            selected_topics = ["Semua"]
    
    show_all_posts = st.sidebar.checkbox(
        "Tampilkan semua unggahan secara vertikal (print friendly)",
        value=False
    )
    
    # Main content area
    if has_stance_analysis:
        # Apply filters
        filtered_comments = comments_df.copy()
        
        if "Semua" not in selected_stance:
            filtered_comments = filtered_comments[filtered_comments["stance_label"].isin(selected_stance)]
        
        if min_confidence > 0:
            filtered_comments = filtered_comments[filtered_comments["stance_weight"] >= min_confidence]
        
        if selected_topics and "Semua" not in selected_topics:
            filtered_comments = filtered_comments[filtered_comments["topic_id"].isin(selected_topics)]
        
        # Display summary statistics
        st.subheader("📈 Statistik Keseluruhan")
        
        total_posts = len(posts_df)
        total_comments = len(filtered_comments)
        avg_confidence = filtered_comments["stance_weight"].mean() if len(filtered_comments) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Unggahan", total_posts)
        with col2:
            st.metric("Total Komentar (Filtered)", total_comments)
        with col3:
            st.metric("Rata-rata Bobot", f"{avg_confidence:.2f}")
        with col4:
            mendukung_count = len(filtered_comments[filtered_comments["stance_label"] == "Mendukung"])
            st.metric("🟩 Mendukung", mendukung_count)
        
        # Display overall distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Distribusi Sikap Keseluruhan:**")
            stance_counts = filtered_comments["stance_label"].value_counts()
            fig_overall = go.Figure(data=[go.Bar(
                x=stance_counts.index,
                y=stance_counts.values,
                marker=dict(color=["#90EE90" if x == "Mendukung" else "#FFB6C1" if x == "Menolak" else "#D3D3D3" for x in stance_counts.index])
            )])
            fig_overall.update_layout(height=300, xaxis_title="Sikap", yaxis_title="Jumlah")
            st.plotly_chart(fig_overall, use_container_width=True)
        
        with col2:
            st.write("**Distribusi Bobot Keyakinan:**")
            fig_weight = go.Figure(data=[go.Histogram(x=filtered_comments["stance_weight"], nbinsx=20, marker_color="rgba(0, 100, 200, 0.7)")])
            fig_weight.update_layout(height=300, xaxis_title="Bobot Keyakinan", yaxis_title="Frekuensi")
            st.plotly_chart(fig_weight, use_container_width=True)
        
        # Display posts with comments
        st.divider()
        st.subheader("📑 Detail Per Unggahan")
        
        # Get aggregated stats per post
        post_stats_dict = {}
        for post_id in posts_df["post_id"].unique():
            post_comments = filtered_comments[filtered_comments["post_id"] == post_id]
            if len(post_comments) > 0:
                total = len(post_comments)
                post_stats_dict[post_id] = {
                    "mendukung_pct": (post_comments["stance_label"] == "Mendukung").sum() / total * 100,
                    "menolak_pct": (post_comments["stance_label"] == "Menolak").sum() / total * 100,
                    "netral_pct": (post_comments["stance_label"] == "Netral").sum() / total * 100,
                    "avg_weight": post_comments["stance_weight"].mean()
                }
        
        # Display each post
        for _, post_row in posts_df.iterrows():
            post_id = post_row["post_id"]
            post_text = post_row["full_text"]
            post_comments = filtered_comments[filtered_comments["post_id"] == post_id]
            
            if len(post_comments) > 0:
                post_stats = post_stats_dict.get(post_id, {})
                render_post_with_comments(
                    post_id=post_id,
                    post_text=post_text,
                    comments_data=post_comments,
                    post_stats=post_stats,
                    show_all=show_all_posts
                )
    else:
        st.info("📌 Jalankan analisis stance untuk melihat hasil detail.")


if __name__ == "__main__":
    main()
