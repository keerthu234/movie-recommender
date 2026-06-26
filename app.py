import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch", layout="wide", page_icon="🎬")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.main-title {
    text-align: center;
    font-size: 4em;
    font-weight: 900;
    background: linear-gradient(90deg, #f953c6, #b91d73);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #aaaaaa;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.movie-card {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(249,83,198,0.3);
    color: white;
    transition: 0.3s;
}
.genre-badge {
    background: linear-gradient(90deg, #f953c6, #b91d73);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8em;
    color: white;
    margin-right: 5px;
}
.stat-box {
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
}
.stSelectbox label { color: white !important; font-size: 1.1em; }
.stButton button {
    background: linear-gradient(90deg, #f953c6, #b91d73);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 30px;
    font-size: 1.1em;
    width: 100%;
    cursor: pointer;
}
div[data-testid="stTab"] { color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    movies = pd.read_csv('ml-latest/movies.csv')
    ratings = pd.read_csv('ml-latest/ratings.csv')
    return movies, ratings
    movies = movies.head(5000).reset_index(drop=True)
    avg_ratings = ratings.groupby('movieId')['rating'].agg(['mean','count']).reset_index()
    avg_ratings.columns = ['movieId','avg_rating','num_ratings']
    movies = movies.merge(avg_ratings, on='movieId', how='left')
    return movies, ratings

@st.cache_data
def build_model(movies):
    tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return cosine_sim, movie_indices

movies, ratings = load_data()
cosine_sim, movie_indices = build_model(movies)

def recommend_movies(title, num=5):
    if title not in movie_indices:
        return []
    idx = movie_indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    return movies.iloc[[i[0] for i in sim_scores]][['title','genres','avg_rating','num_ratings']].values.tolist()

# Header
st.markdown("<div class='main-title'>🎬 CineMatch</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your Personal Movie Recommendation Engine</div>", unsafe_allow_html=True)

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='stat-box'><h2 style='color:#f953c6'>{len(movies)}</h2><p>Movies</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='stat-box'><h2 style='color:#f953c6'>{len(ratings)}</h2><p>Total Ratings</p></div>", unsafe_allow_html=True)
with col3:
    all_genres = set(g for genres in movies['genres'].str.split('|') for g in genres)
    st.markdown(f"<div class='stat-box'><h2 style='color:#f953c6'>{len(all_genres)}</h2><p>Genres</p></div>", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Recommend Movies", "🎭 Browse by Genre", "⭐ Top Rated"])

with tab1:
    st.markdown("<h3 style='color:white'>Search for a movie you love:</h3>", unsafe_allow_html=True)
    selected_movie = st.selectbox("", movies['title'].tolist(), label_visibility="collapsed")
    if st.button("✨ Find Similar Movies"):
        with st.spinner("Finding movies for you..."):
            recommendations = recommend_movies(selected_movie)
        st.markdown(f"<h3 style='color:white'>Because you liked <span style='color:#f953c6'>{selected_movie}</span>:</h3>", unsafe_allow_html=True)
        for i, (title, genre, rating, count) in enumerate(recommendations, 1):
            rating_str = f"⭐ {rating:.1f}/5" if pd.notna(rating) else "⭐ N/A"
            genres_html = " ".join([f"<span class='genre-badge'>{g}</span>" for g in genre.split('|')])
            st.markdown(f"""
            <div class='movie-card'>
                <h4 style='color:white;margin:0'>#{i} {title}</h4>
                <div style='margin:8px 0'>{genres_html}</div>
                <p style='color:#f953c6;margin:0'>{rating_str}</p>
            </div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("<h3 style='color:white'>Browse movies by genre:</h3>", unsafe_allow_html=True)
    selected_genre = st.selectbox("", sorted(all_genres), label_visibility="collapsed")
    if st.button("🔍 Show Movies"):
        genre_movies = movies[movies['genres'].str.contains(selected_genre)]
        genre_movies = genre_movies.sort_values('avg_rating', ascending=False).head(10)
        st.markdown(f"<h3 style='color:white'>Top <span style='color:#f953c6'>{selected_genre}</span> Movies:</h3>", unsafe_allow_html=True)
        for _, row in genre_movies.iterrows():
            rating_str = f"⭐ {row['avg_rating']:.1f}/5" if pd.notna(row['avg_rating']) else "⭐ N/A"
            st.markdown(f"""
            <div class='movie-card'>
                <h4 style='color:white;margin:0'>{row['title']}</h4>
                <p style='color:gray;margin:5px 0'>{row['genres']}</p>
                <p style='color:#f953c6;margin:0'>{rating_str}</p>
            </div>""", unsafe_allow_html=True)

with tab3:
    st.markdown("<h3 style='color:white'>⭐ Top 10 Highest Rated Movies:</h3>", unsafe_allow_html=True)
    top_movies = movies[movies['num_ratings'] > 50].sort_values('avg_rating', ascending=False).head(10)
    for i, (_, row) in enumerate(top_movies.iterrows(), 1):
        st.markdown(f"""
        <div class='movie-card'>
            <h4 style='color:white;margin:0'>#{i} {row['title']}</h4>
            <p style='color:gray;margin:5px 0'>{row['genres']}</p>
            <p style='color:#f953c6;margin:0'>⭐ {row['avg_rating']:.1f}/5 &nbsp;|&nbsp; {int(row['num_ratings'])} ratings</p>
        </div>""", unsafe_allow_html=True)
