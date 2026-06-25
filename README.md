# CineMatch - Movie Recommendation System

A content-based movie recommendation system built using Python and Machine Learning that suggests movies based on user preferences.

## Features

* Personalized movie recommendations
* Browse movies by genre
* View top-rated movies
* Interactive and user-friendly interface
* Dark theme UI

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

## Dataset

This project uses the MovieLens Dataset provided by GroupLens Research.

Dataset: https://grouplens.org/datasets/movielens/

Download the **ml-latest-small** dataset and place it in the project directory before running the application.

## Installation and Usage

1. Clone the repository.
2. Install the required dependencies:

pip install pandas numpy scikit-learn streamlit

3. Download and extract the MovieLens dataset.
4. Place the dataset folder inside the project directory.
5. Run the application:

streamlit run app.py

## Working Principle

This project uses a Content-Based Filtering approach.

* Movie genres are converted into numerical features using TF-IDF Vectorization.
* Cosine Similarity is used to measure similarity between movies.
* The system recommends movies that are most similar to the selected movie.

## Future Enhancements

* Hybrid Recommendation System
* User Authentication
* Movie Posters and Trailers
* Personalized User Profiles

## Developer

**Keerthana S**
Computer Science Engineering Student
