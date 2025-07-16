
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv('data/movies.csv')
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        recommended_movie_list = sorted(list(enumerate(distances)),
                                        reverse=True, key=lambda x: x[1])[1:6]
        recommendations = [movies.iloc[i[0]].title for i in recommended_movie_list]
    except IndexError:
        recommendations = ["Movie not found in database"]
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
