# Movie Recommendation System: Using Google Cloud Storage and CSV Data with Advanced Libraries for Personalized Insights

A simple **Movie Recommendation System** built using Flask, Python, and Scikit-learn, complete with a user-friendly interface for providing personalized movie suggestions based on collaborative filtering.

```
# 📖 Table of Contents

```
1. Introduction  
2. Features  
3. Technologies Used  
4. Setup and Installation  
5. How It Works  
6. Project Structure  
7. Data Source  
8. Sample Inputs  
9. CSS Styling  
10. Live Demo  
11. Future Enhancements  
12. Development  
13. License
14. Extra Colab Source for Data Merging
```
```
## 🎯 Introduction

This project demonstrates the implementation of a collaborative filtering-based recommendation system. Users provide their **User ID**, and the system generates personalized recommendations based on the **k-Nearest Neighbors (k-NN)** algorithm and similarity metrics.
```

```
## ✨ Features

- Interactive web interface built using Flask and HTML.  
- Machine Learning model integrated using Scikit-learn.  
- Dynamic and accurate movie recommendations.  
- Customizable and scalable for larger datasets.  
- Enhanced UI/UX with CSS for a polished experience.
```

```
## 🔧 Technologies Used

- **Backend:** Flask (Python)  
- **Frontend:** HTML5, CSS3  
- **Machine Learning:** Scikit-learn  
- **Dependencies:** pandas, NumPy
- **Data Storage : Buckets(GCP)
- **Hosting:** Flask built-in server
```

```
## 🛠️ Setup and Installation

### Prerequisites  

- Python 3.7 or above.  
- Virtual environment (optional but recommended).
- Some Familiar with Google Cloud [Buckets].
- Kaggle API key for downloading datasets.
- Using Pandas , Flask library.

### Steps  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/movie-recommendation-system.git  
   cd movie-recommendation-system  
   ```  

2. Download the dataset from Kaggle:  
   ```bash   

   https:https://www.kaggle.com/datasets/hassanelfattmi/which-movie-should-i-watch-today
   unzip the data set.zip -d data/
   ```  

3. Install required dependencies:  
   ```bash  
   pip install -r requirements.txt
   pip install scikit-learn
   pip install Flask
   ```  

4. Run the application:  
   ```bash  
   python app.py  
   ```
   For my Suggestion Use VS Code.
```

```
## 🚀 How It Works  

1. **Input:** User provides their Rating he want and Genres and Languages.  
2. **Processing:**  
   - A pivot table is generated from user-item interaction data.  
   - k-NN algorithm identifies similar users using cosine similarity.  
3. **Output:** A list of recommended movies is displayed.
```

```
## 📂 Project Structure  

```
movie-recommendation-system/  
│  
├── app.py                 # Main Flask application  
├── templates/  
│   └── index.html         # Frontend HTML  
├── static/  
│   └── styles.css         # CSS for styling  
├── data/  
│   ├── Movies.csv         # Movie dataset from Kaggle  
│   ├── FilmDetails.csv    # Additional movie details  
│   ├── PosterPath.csv     # Movie poster paths  
│   └── MoreInfo.csv       # Other movie information  
├── requirements.txt       # Dependencies  
├── README.md              # Documentation  
└── LICENSE                # Project license
```
```

```
## 📊 Data Source  

- **Dataset:** [Movies Dataset on Kaggle](https://www.kaggle.com/datasets/hassanelfattmi/which-movie-should-i-watch-today).
- Data use to Store in Google Cloud Buckets . 
- Ensure you have your Kaggle API key for dataset download.
```

```
## 📝 Sample Inputs  

- **Example Rating :**  
  Valid IDs include 7 , 8 , 9 .  

- **Example Output for  8:**  
  Recommended Movies: `Inception`, `The Dark Knight`, `Interstellar`.
```

```
## 🎨 CSS Styling  

The web application uses a minimalist and responsive design. The CSS file located in `static/styles.css` ensures seamless interaction across devices.
```
Which includes just Background image and interactive page.
```
## 🌐 Live Demo  

Check out the live application:  
[Movie Recommendation System Live Demo](#)  
(*Replace with the deployed link or keep as placeholder.*)
```

```
## 🔮 Future Enhancements  

- Integration with APIs like TMDb or OMDB for real-time movie data.  
- Advanced recommendation algorithms (e.g., matrix factorization).  
- User authentication for saving preferences.  
- Deployment on cloud platforms (e.g., AWS, GCP).
```

```
## 🛠️ Development  

1. Fork the repository:  
   ```bash  
   git fork https://github.com/your-username/movie-recommendation-system.git  
   ```  

2. Create a new branch:  
   ```bash  
   git checkout -b feature-name  
   ```  

3. Commit changes:  
   ```bash  
   git commit -m "Added a new feature"  
   ```  

4. Push the branch:  
   ```bash  
   git push origin feature-name  
   ```  

5. Open a Pull Request.
```

```
## ⚖️ License  

This project is licensed under the **MIT License**.
```

```
###Extra Colab Source for Data Merging
```
https://colab.research.google.com/drive/1Qdavm3oJDefyNdCNVK_Z0yM8s-ywF5vt?usp=sharing.
```



###
-
                                                 THANK💚YOU
