import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Vectorize movie descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Compute similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

# Recommendation Function
def recommend_movie(movie_name):
    if movie_name not in df['title'].values:
        return ["Movie not found in database"]

    # Find index of selected movie
    idx = df[df['title'] == movie_name].index[0]

    # Find similar movies
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity (highest first)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Top 5 recommendations (excluding the movie itself)
    top_movies = [df.iloc[i[0]].title for i in similarity_scores[1:6]]
    
    return top_movies

# User Input
while True:
    movie = input("\nEnter a movie name (or 'exit'): ")
    if movie.lower() == "exit":
        break
    
    recommendations = recommend_movie(movie)
    print("\nRecommended Movies:")
    for r in recommendations:
        print(" -", r)
