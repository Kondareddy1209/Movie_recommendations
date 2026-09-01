"""
streamlit_app.py - Streamlit Interactive Exploration Dashboard
=============================================================
Provides an interactive data science and recommendation dashboard for the
9,718-film movie catalog using Content-Based k-NN and multi-criteria filtering.
"""

import os
import streamlit as st
from src.recommender import MovieRecommender

# Set page config
st.set_page_config(
    page_title="CineMatch ML | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_recommender():
    """Cached singleton of the MovieRecommender instance."""
    base_dir = os.path.dirname(__file__)
    movies_path = os.path.join(base_dir, "merged_movie_data.csv")
    posters_path = os.path.join(base_dir, "PosterPath.csv")
    return MovieRecommender(movies_path=movies_path, posters_path=posters_path)


recommender = get_recommender()

# Custom CSS styling
st.markdown(
    """
    <style>
    .main { background-color: #0a0e17; }
    .stApp { color: #f8fafc; }
    .movie-card-title { font-size: 16px; font-weight: 700; margin-top: 8px; margin-bottom: 2px; }
    .movie-meta { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
    .movie-score { color: #fbbf24; font-weight: 700; }
    .match-badge { background-color: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("🎬 CineMatch: Movie Recommendation System")
st.markdown(
    "**Content-Based k-Nearest Neighbors (k-NN) with Cosine Similarity** & Multi-Criteria Filtering across **9,718 curated films**."
)

# Sidebar navigation & filters
st.sidebar.header("🎯 Mode Selection")
mode = st.sidebar.radio(
    "Choose Recommendation Strategy:",
    ["Content Similarity (k-NN)", "Multi-Criteria Filter", "Dataset Explorer"],
)

if mode == "Content Similarity (k-NN)":
    st.subheader("🎯 Find Similar Movies via Nearest Neighbors")
    st.caption("Matches movies based on genre distribution, director, cast metadata, and user rating features.")

    popular_defaults = ["The Godfather", "Inception", "Pulp Fiction", "Interstellar", "Spirited Away", "The Dark Knight"]
    selected_movie_title = st.selectbox(
        "Select or type a film title:",
        options=popular_defaults + [t for t in recommender.df["title"].head(100) if t not in popular_defaults],
    )
    custom_title = st.text_input("Or enter any specific movie title from catalog:", value="")
    query_title = custom_title.strip() if custom_title.strip() else selected_movie_title

    n_recs = st.sidebar.slider("Number of recommendations:", min_value=3, max_value=15, value=6)

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Calculating nearest neighbors..."):
            target, recs = recommender.recommend_by_title(query_title, n_recommendations=n_recs)

        if target is None:
            st.error(f"Movie '{query_title}' not found in catalog. Please check the spelling or try another title.")
        else:
            # Display target movie
            st.markdown("### 📽️ Selected Source Movie")
            t_col1, t_col2 = st.columns([1, 4])
            with t_col1:
                if target.get("poster_path"):
                    st.image(target["poster_path"], width=160)
            with t_col2:
                st.markdown(f"## {target['title']} ⭐ **{target['user_score']} / 10**")
                st.markdown(f"**Release Date:** {target.get('release_date', 'N/A')} | **Runtime:** {target.get('runtime_hour', 0)}h {target.get('runtime_min', 0)}m | **Language:** {target.get('language', '').upper()}")
                st.markdown(f"**Genres:** `{target.get('genres', '')}`")
                st.markdown(f"**Director:** {target.get('director', 'Unknown')}")
                st.markdown(f"**Top Cast:** {target.get('top_billed', 'N/A')}")

            st.divider()
            st.markdown(f"### 🌟 Top {len(recs)} Nearest Neighbors (Cosine Similarity)")

            cols = st.columns(min(len(recs), 3))
            for i, rec in enumerate(recs):
                with cols[i % 3]:
                    if rec.get("poster_path"):
                        st.image(rec["poster_path"], use_container_width=True)
                    match_pct = int(rec.get("similarity_score", 0) * 100)
                    st.markdown(f"<span class='match-badge'>Match {match_pct}%</span> <span class='movie-score'>⭐ {rec['user_score']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<p class='movie-card-title'>{rec['title']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='movie-meta'>{rec.get('genres', '')}<br>Dir: {rec.get('director', 'Unknown')}</p>", unsafe_allow_html=True)

elif mode == "Multi-Criteria Filter":
    st.subheader("🎛️ Filter & Rank Movies by User Criteria")
    st.caption("Discover top-ranked movies filtered by genre, original language, and score threshold.")

    col1, col2, col3 = st.columns(3)
    with col1:
        genres = ["All Genres"] + recommender.get_genres()
        selected_genre = st.selectbox("Genre:", genres)
    with col2:
        languages = ["All Languages"] + recommender.get_languages()
        selected_lang = st.selectbox("Language:", languages)
    with col3:
        min_score = st.slider("Minimum User Score:", min_value=0.0, max_value=9.0, value=7.5, step=0.1)

    limit = st.sidebar.slider("Results limit:", min_value=4, max_value=30, value=12)
    sort_choice = st.sidebar.radio("Sort by:", ["user_score", "vote_count"])

    genre_arg = None if selected_genre == "All Genres" else selected_genre
    lang_arg = None if selected_lang == "All Languages" else selected_lang

    results = recommender.filter_recommendations(
        min_score=min_score,
        genre=genre_arg,
        language=lang_arg,
        sort_by=sort_choice,
        limit=limit,
    )

    st.markdown(f"### Found {len(results)} Matching Films")
    if not results:
        st.warning("No films match the specified criteria. Try relaxing the minimum score or genre.")
    else:
        cols = st.columns(min(len(results), 4))
        for i, m in enumerate(results):
            with cols[i % 4]:
                if m.get("poster_path"):
                    st.image(m["poster_path"], use_container_width=True)
                st.markdown(f"<span class='movie-score'>⭐ {m['user_score']}</span>", unsafe_allow_html=True)
                st.markdown(f"<p class='movie-card-title'>{m['title']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='movie-meta'>{m.get('genres', '')}<br>Dir: {m.get('director', 'Unknown')}</p>", unsafe_allow_html=True)

else:
    st.subheader("📊 Dataset Statistics & Catalog Overview")
    df = recommender.df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(df):,}")
    c2.metric("Unique Genres", f"{len(recommender.get_genres())}")
    c3.metric("Languages", f"{len(recommender.get_languages())}")
    c4.metric("Avg User Score", f"{df['user_score'].mean():.2f} / 10")

    st.markdown("### Movie Dataset Preview")
    st.dataframe(
        df[["id", "title", "genres", "language", "user_score", "release_date", "director", "vote_count"]].head(50),
        use_container_width=True,
    )
