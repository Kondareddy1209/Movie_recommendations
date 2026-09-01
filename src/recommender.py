"""
recommender.py - Machine Learning Movie Recommendation Engine
=============================================================
Provides content-based k-NN similarity recommendations and multi-criteria
preference filtering on the curated 9,718-film movie dataset.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class MovieRecommender:
    """
    Movie Recommender System supporting:
    1. Content-based k-Nearest Neighbors (k-NN) using Cosine Similarity on TF-IDF metadata.
    2. Multi-criteria preference filtering (genre, language, user score).
    """

    def __init__(
        self,
        movies_path: str = "merged_movie_data.csv",
        posters_path: str = "PosterPath.csv",
    ) -> None:
        self.movies_path = movies_path
        self.posters_path = posters_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.nn_model: Optional[NearestNeighbors] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.genres_list: List[str] = []
        self.languages_list: List[str] = []
        self._is_fitted: bool = False
        self.load_data()
        self.fit_model()

    def load_data(self) -> None:
        """Load and merge movie dataset with poster assets."""
        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(f"Movie dataset not found at: {self.movies_path}")

        movies_df = pd.read_csv(self.movies_path)
        required_cols = {"id", "title", "genres", "language", "user_score"}
        if not required_cols.issubset(movies_df.columns):
            raise ValueError(f"Dataset missing required columns: {required_cols - set(movies_df.columns)}")

        # Clean string columns
        movies_df["title"] = movies_df["title"].fillna("").astype(str).str.strip()
        movies_df["genres"] = movies_df["genres"].fillna("").astype(str).str.strip()
        movies_df["language"] = movies_df["language"].fillna("en").astype(str).str.strip()
        movies_df["director"] = movies_df["director"].fillna("Unknown").astype(str).str.strip() if "director" in movies_df.columns else "Unknown"
        movies_df["top_billed"] = movies_df["top_billed"].fillna("").astype(str).str.strip() if "top_billed" in movies_df.columns else ""
        movies_df["user_score"] = pd.to_numeric(movies_df["user_score"], errors="coerce").fillna(0.0)
        movies_df["vote_count"] = pd.to_numeric(movies_df.get("vote_count", 0), errors="coerce").fillna(0).astype(int)

        # Merge poster paths if available
        if os.path.exists(self.posters_path):
            posters_df = pd.read_csv(self.posters_path)
            if "id" in posters_df.columns and "poster_path" in posters_df.columns:
                movies_df = pd.merge(
                    movies_df,
                    posters_df[["id", "poster_path", "backdrop_path"] if "backdrop_path" in posters_df.columns else ["id", "poster_path"]],
                    on="id",
                    how="left",
                )

        if "poster_path" not in movies_df.columns:
            movies_df["poster_path"] = ""
        else:
            movies_df["poster_path"] = movies_df["poster_path"].fillna("")

        if "backdrop_path" not in movies_df.columns:
            movies_df["backdrop_path"] = ""
        else:
            movies_df["backdrop_path"] = movies_df["backdrop_path"].fillna("")

        self.df = movies_df.reset_index(drop=True)

        # Extract unique sorted genres and languages
        genres_set = set()
        for g_str in self.df["genres"]:
            for item in g_str.split(","):
                clean = item.strip()
                if clean:
                    genres_set.add(clean)
        self.genres_list = sorted(list(genres_set))
        self.languages_list = sorted(list(self.df["language"].unique()))

    def fit_model(self) -> None:
        """
        Builds feature representations and fits k-Nearest Neighbors model using Cosine Similarity.
        Feature representation combines:
        - TF-IDF on combined metadata (genres, director, top cast, language)
        - Scaled numerical features (user_score, log-transformed vote_count)
        """
        if self.df.empty:
            return

        # Combine text metadata with weighted importance
        combined_text = (
            self.df["genres"].apply(lambda g: " ".join([f"genre_{x.strip().replace(' ', '_')}" for x in g.split(",") if x.strip()])) + " " +
            self.df["director"].apply(lambda d: f"director_{d.replace(' ', '_')}") + " " +
            self.df["top_billed"].apply(lambda c: " ".join([f"actor_{x.strip().replace(' ', '_')}" for x in c.split(",") if x.strip()])) + " " +
            self.df["language"].apply(lambda l: f"lang_{l}")
        )

        self.tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_features=5000)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text).toarray()

        # Scaled numerical features
        scores = self.df[["user_score"]].values
        vote_counts = np.log1p(self.df[["vote_count"]].values)

        scaler = MinMaxScaler()
        num_features = scaler.fit_transform(np.hstack([scores, vote_counts]))

        # Weight text features vs numerical features (85% content metadata, 15% score/popularity)
        self.feature_matrix = np.hstack([tfidf_matrix * 0.85, num_features * 0.15])

        # Fit Nearest Neighbors with cosine distance
        self.nn_model = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
        self.nn_model.fit(self.feature_matrix)
        self._is_fitted = True

    def recommend_by_title(
        self,
        title: str,
        n_recommendations: int = 5,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Finds k-nearest movies similar to the given title using cosine similarity on content features.
        Returns (target_movie_dict, recommendations_list).
        """
        if not self._is_fitted or self.nn_model is None or self.feature_matrix is None:
            raise RuntimeError("Model is not fitted.")

        # Find movie index (case-insensitive search)
        matches = self.df[self.df["title"].str.lower() == title.strip().lower()]
        if matches.empty:
            # Try substring match
            matches = self.df[self.df["title"].str.lower().str.contains(title.strip().lower(), regex=False)]
            if matches.empty:
                return None, []

        target_idx = matches.index[0]
        target_movie = self.df.iloc[target_idx].to_dict()

        # Query k-NN (request extra neighbors to account for self match)
        k = min(n_recommendations + 5, len(self.df))
        distances, indices = self.nn_model.kneighbors(
            self.feature_matrix[target_idx].reshape(1, -1),
            n_neighbors=k,
        )

        distances = distances.flatten()
        indices = indices.flatten()

        recommendations = []
        for dist, idx in zip(distances, indices):
            if idx == target_idx:
                continue  # Skip self
            rec_dict = self.df.iloc[idx].to_dict()
            rec_dict["similarity_score"] = float(np.round(1.0 - dist, 4))
            rec_dict["cosine_distance"] = float(np.round(dist, 4))
            recommendations.append(rec_dict)
            if len(recommendations) >= n_recommendations:
                break

        return target_movie, recommendations

    def filter_recommendations(
        self,
        min_score: float = 0.0,
        genre: Optional[str] = None,
        language: Optional[str] = None,
        sort_by: str = "user_score",
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Multi-criteria preference filtering (the core original Flask functionality).
        Filters by minimum user_score, genre, and language, ordered by score or popularity.
        """
        filtered = self.df[self.df["user_score"] >= float(min_score)]

        if genre and genre.strip():
            genre_clean = genre.strip().lower()
            filtered = filtered[
                filtered["genres"].str.lower().apply(
                    lambda g: any(genre_clean == item.strip() for item in g.split(","))
                )
            ]

        if language and language.strip():
            lang_clean = language.strip().lower()
            filtered = filtered[filtered["language"].str.lower() == lang_clean]

        if filtered.empty:
            return []

        if sort_by == "vote_count" and "vote_count" in filtered.columns:
            filtered = filtered.sort_values(by=["vote_count", "user_score"], ascending=[False, False])
        else:
            filtered = filtered.sort_values(by=["user_score", "vote_count"], ascending=[False, False])

        return filtered.head(limit).to_dict(orient="records")

    def search_titles(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title query for autocomplete / typeahead."""
        if not query or not query.strip():
            return []
        q = query.strip().lower()
        matches = self.df[self.df["title"].str.lower().str.contains(q, regex=False)]
        matches = matches.sort_values(by="user_score", ascending=False).head(limit)
        return matches[["id", "title", "user_score", "genres", "language", "poster_path"]].to_dict(orient="records")

    def get_genres(self) -> List[str]:
        """Returns list of available genres."""
        return self.genres_list

    def get_languages(self) -> List[str]:
        """Returns list of available language codes."""
        return self.languages_list
