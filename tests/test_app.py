"""
test_app.py - Integration tests for Flask web app and REST API endpoints
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Provides Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_route(client):
    """Verify that home page loads successfully with featured movies."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "CineMatch ML" in html
    assert "9,718 Movies" in html


def test_recommend_post_similarity(client):
    """Verify similarity recommendation form submission."""
    response = client.post("/recommend", data={"title": "The Godfather"})
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "The Godfather" in html
    assert "Recommended Similar Movies" in html


def test_recommend_post_filter(client):
    """Verify criteria filter form submission."""
    response = client.post(
        "/recommend",
        data={"user_score": "8.0", "genre": "Drama", "language": "en"},
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Recommendations" in html or "Filtered Recommendations" in html


def test_recommend_post_not_found(client):
    """Verify handling of unknown movie title form post."""
    response = client.post("/recommend", data={"title": "UnknownMovieXYZ12345"})
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "was not found in our catalog" in html


def test_api_recommend_similarity(client):
    """Verify REST API similarity query endpoint."""
    response = client.get("/api/recommend?title=Inception&limit=5")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["type"] == "content_similarity"
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5


def test_api_recommend_filter(client):
    """Verify REST API filter query endpoint."""
    response = client.get("/api/recommend?genre=Comedy&min_score=7.5&limit=6")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["type"] == "criteria_filter"
    assert len(data["recommendations"]) <= 6


def test_api_recommend_not_found(client):
    """Verify REST API 404 response for unknown movie."""
    response = client.get("/api/recommend?title=RandomNonExistentFilm12345")
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "error"


def test_api_search_movies(client):
    """Verify autocomplete typeahead search API."""
    response = client.get("/api/movies?q=Godfather")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["results"]) > 0


def test_api_metadata(client):
    """Verify metadata API returns dataset stats and categories."""
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["total_movies"] == 9718
    assert len(data["genres"]) > 0
    assert len(data["languages"]) > 0
