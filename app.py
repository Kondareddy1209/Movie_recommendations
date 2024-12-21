from flask import Flask, request, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Dummy data for demonstration
dummy_data = {
    "user_id": [1, 1, 2, 2, 3, 3],
    "movie_id": [1, 2, 2, 3, 1, 3],
    "rating": [5, 4, 3, 5, 2, 4]
}
user_ratings_df = pd.DataFrame(dummy_data)

# User-item interaction matrix
ratings_matrix = user_ratings_df.pivot(index="user_id", columns="movie_id", values="rating").fillna(0)

# NearestNeighbors model
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(ratings_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the user_id from the form
        user_id = int(request.form.get('user_id'))
        
        # Check if the user_id exists
        if user_id not in user_ratings_df['user_id'].values:
            return render_template('index.html', error="User ID not found in the dataset.")

        # Get the user's index in the ratings matrix
        user_index = ratings_matrix.index.get_loc(user_id)

        # Get recommendations
        distances, indices = model.kneighbors(ratings_matrix.iloc[user_index, :].values.reshape(1, -1), n_neighbors=3)

        # Generate recommendations
        recommendations = [ratings_matrix.index[i] for i in indices.flatten()[1:]]
        return render_template('index.html', user_id=user_id, recommendations=recommendations)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
