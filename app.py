"""
app.py - Flask Web Application for Movie Recommendation System
=============================================================
Provides both a server-rendered web interface and RESTful JSON API endpoints
for content-based k-NN movie recommendations and multi-criteria preference filtering.
"""

import os
from typing import Any, Dict
from flask import Flask, jsonify, render_template, request
from src.recommender import MovieRecommender

app = Flask(__name__)

# Initialize recommender (loads dataset and fits k-NN cosine model)
recommender = MovieRecommender(
    movies_path=os.path.join(os.path.dirname(__file__), "merged_movie_data.csv"),
    posters_path=os.path.join(os.path.dirname(__file__), "PosterPath.csv"),
)


@app.route("/")
def home():
    """
    Renders the main recommendation portal with initial top-rated movie highlights,
    available genres, and language options.
    """
    genres = recommender.get_genres()
    languages = recommender.get_languages()
    featured_movies = recommender.filter_recommendations(min_score=8.5, limit=8)
    return render_template(
        "index.html",
        genres=genres,
        languages=languages,
        featured_movies=featured_movies,
    )


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    """
    Web form submission handler:
    Supports both:
    1. Title-based similarity recommendation (k-NN Cosine Similarity).
    2. Multi-criteria preference filtering (User Score, Genre, Language).
    """
    genres = recommender.get_genres()
    languages = recommender.get_languages()

    if request.method == "GET":
        title_query = request.args.get("title", "").strip()
        genre = request.args.get("genre", "").strip()
        language = request.args.get("language", "").strip()
        user_score_str = request.args.get("user_score", "0.0").strip()
    else:
        title_query = request.form.get("title", "").strip()
        genre = request.form.get("genre", "").strip()
        language = request.form.get("language", "").strip()
        user_score_str = request.form.get("user_score", "0.0").strip()

    try:
        user_score = float(user_score_str) if user_score_str else 0.0
    except ValueError:
        user_score = 0.0

    target_movie = None
    recommendations = []
    error = None
    search_type = "similarity" if title_query else "filter"

    try:
        if title_query:
            target_movie, recommendations = recommender.recommend_by_title(
                title=title_query,
                n_recommendations=8,
            )
            if target_movie is None:
                error = f"Movie '{title_query}' was not found in our catalog. Try searching with a different title or keyword."
        else:
            recommendations = recommender.filter_recommendations(
                min_score=user_score,
                genre=genre if genre else None,
                language=language if language else None,
                limit=12,
            )
            if not recommendations:
                error = "No movies matched your filter criteria. Try lowering the minimum user score or selecting another genre."

    except Exception as e:
        error = f"An error occurred while generating recommendations: {str(e)}"

    return render_template(
        "index.html",
        genres=genres,
        languages=languages,
        target_movie=target_movie,
        recommendations=recommendations,
        error=error,
        selected_title=title_query,
        selected_genre=genre,
        selected_language=language,
        selected_user_score=user_score_str,
        search_type=search_type,
    )


# ==========================================
# REST API Endpoints
# ==========================================


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    """
    REST API for movie recommendations:
    Parameters:
    - title: (optional) Movie title for content-based similarity.
    - genre: (optional) Genre filter.
    - language: (optional) Language filter.
    - min_score: (optional, float) Minimum score filter.
    - limit: (optional, int) Max number of recommendations (default: 10).
    """
    title = request.args.get("title", "").strip()
    genre = request.args.get("genre", "").strip()
    language = request.args.get("language", "").strip()
    min_score_str = request.args.get("min_score", "0.0").strip()
    limit_str = request.args.get("limit", "10").strip()

    try:
        min_score = float(min_score_str)
    except ValueError:
        min_score = 0.0

    try:
        limit = min(int(limit_str), 50)
    except ValueError:
        limit = 10

    if title:
        target, recs = recommender.recommend_by_title(title=title, n_recommendations=limit)
        if target is None:
            return jsonify({"status": "error", "message": f"Movie '{title}' not found."}), 404
        return jsonify({
            "status": "success",
            "type": "content_similarity",
            "target_movie": target,
            "count": len(recs),
            "recommendations": recs,
        })

    recs = recommender.filter_recommendations(
        min_score=min_score,
        genre=genre if genre else None,
        language=language if language else None,
        limit=limit,
    )
    return jsonify({
        "status": "success",
        "type": "criteria_filter",
        "count": len(recs),
        "recommendations": recs,
    })


@app.route("/api/movies", methods=["GET"])
def api_search_movies():
    """
    Autocomplete / typeahead movie title search endpoint.
    Query parameter: `q`
    """
    query = request.args.get("q", "").strip()
    matches = recommender.search_titles(query, limit=10)
    return jsonify({"status": "success", "results": matches})


@app.route("/api/metadata", methods=["GET"])
def api_metadata():
    """Returns available genres, languages, and dataset count."""
    return jsonify({
        "status": "success",
        "total_movies": len(recommender.df),
        "genres": recommender.get_genres(),
        "languages": recommender.get_languages(),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
