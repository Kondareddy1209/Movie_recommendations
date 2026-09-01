"""
test_recommender.py - Unit tests for MovieRecommender ML Engine
"""

import pytest
from src.recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender():
    """Initializes MovieRecommender singleton for test suite."""
    return MovieRecommender()


def test_dataset_loading(recommender):
    """Verify that dataset loaded properly with expected shape and metadata."""
    assert len(recommender.df) == 9718
    assert "title" in recommender.df.columns
    assert "user_score" in recommender.df.columns
    assert "genres" in recommender.df.columns
    assert "poster_path" in recommender.df.columns
    assert recommender._is_fitted is True


def test_genres_and_languages(recommender):
    """Verify genre and language extraction."""
    genres = recommender.get_genres()
    languages = recommender.get_languages()

    assert len(genres) > 0
    assert "Drama" in genres
    assert "Comedy" in genres
    assert "en" in languages


def test_recommend_by_title_valid(recommender):
    """Verify that k-NN recommendations return valid similar movies without self-match."""
    target_title = "The Godfather"
    target, recs = recommender.recommend_by_title(target_title, n_recommendations=5)

    assert target is not None
    assert target["title"] == target_title
    assert len(recs) == 5

    # Verify self is not in recommendations
    rec_titles = [r["title"] for r in recs]
    assert target_title not in rec_titles

    # Verify score structures
    for r in recs:
        assert "similarity_score" in r
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert "poster_path" in r


def test_recommend_by_title_not_found(recommender):
    """Verify that non-existent movie titles are handled gracefully."""
    target, recs = recommender.recommend_by_title("NonExistentMovieXYZ999", n_recommendations=5)
    assert target is None
    assert recs == []


def test_filter_recommendations_genre_and_score(recommender):
    """Verify multi-criteria filtering by genre and minimum score."""
    recs = recommender.filter_recommendations(min_score=8.5, genre="Drama", language="en", limit=5)
    assert len(recs) > 0
    for r in recs:
        assert r["user_score"] >= 8.5
        assert "Drama" in r["genres"]
        assert r["language"] == "en"


def test_filter_recommendations_empty_result(recommender):
    """Verify handling of filter conditions matching no records."""
    recs = recommender.filter_recommendations(min_score=9.9, genre="Western")
    assert recs == []


def test_search_titles(recommender):
    """Verify movie title autocomplete search."""
    results = recommender.search_titles("Inception", limit=5)
    assert len(results) >= 1
    assert any("Inception" in r["title"] for r in results)

    # Empty search query
    assert recommender.search_titles("") == []
