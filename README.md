# Social-medio_recommendation-system

# Overview

This project provides a recommendation system based on user interactions with posts (liked, viewed, rated). The system recommends the top N posts for a specific user based on their past interactions (likes and views) and computes two key performance metrics: Click-Through Rate (CTR) and Mean Average Precision (MAP). The recommendation is made using TF-IDF vectorization and cosine similarity to match the user's profile with the available posts.

# Features

Data Loading: Loads user interaction data (liked, viewed, rated posts) and post data.
Text Preprocessing: Preprocesses the text data (removes special characters, stopwords, applies stemming and lemmatization).
User Profile Creation: Builds a user profile based on the posts they have liked or viewed.
Post Recommendation: Recommends posts based on the cosine similarity between the user's profile and all available posts.
Evaluation Metrics: Calculates CTR and MAP to evaluate the quality of recommendations.

Key Functions
1. load_data(viewed_file, liked_file, rated_file, all_post_file)
This function loads the interaction data for viewed, liked, and rated posts along with all available posts. The data is returned as Pandas DataFrames for easy processing.

Parameters:
viewed_file: Path to the viewed posts JSON file.
liked_file: Path to the liked posts JSON file.
rated_file: Path to the rated posts JSON file.
all_post_file: Path to the all available posts JSON file.
Returns:
DataFrames for liked posts, viewed posts, and all posts.
2. utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None)
This function preprocesses the input text by removing special characters, stopwords, and optionally applying stemming or lemmatization.

Parameters:
text: The input text to preprocess.
flg_stemm: Flag to apply stemming (default: False).
flg_lemm: Flag to apply lemmatization (default: True).
lst_stopwords: List of stopwords to remove from the text.
Returns:
Preprocessed text as a string.
3. create_user_post_matrix(df_liked, df_viewed, username)
Creates a user-post interaction matrix, combining liked and viewed posts for a given username.

Parameters:
df_liked: DataFrame containing the liked posts data.
df_viewed: DataFrame containing the viewed posts data.
username: The username to filter interactions.
Returns:
DataFrame with the user's interactions and combined features (title and category description).
4. user_profile(username, df_interaction, vectorizer)
Generates a user profile by averaging the TF-IDF vectors of the posts the user has interacted with.

Parameters:
username: The username to compute the profile for.
df_interaction: DataFrame containing the user’s interactions.
vectorizer: The TF-IDF vectorizer to transform the post data.
Returns:
User profile vector as a sparse matrix.
5. recommend_posts_pipeline(df_liked, df_viewed, all_post_df, username, vectorizer, utils_preprocess_text, num_recommendations=20)
This function generates post recommendations based on cosine similarity between the user profile and all available posts.

Parameters:
df_liked: DataFrame containing liked posts.
df_viewed: DataFrame containing viewed posts.
all_post_df: DataFrame containing all available posts.
username: The username to generate recommendations for.
vectorizer: The TF-IDF vectorizer.
utils_preprocess_text: The text preprocessing function.
num_recommendations: The number of recommendations to return (default: 20).
Returns:
DataFrame of the top recommended posts.
6. calculate_ctr(recommended_posts, df_liked, df_viewed)
This function calculates the Click-Through Rate (CTR) for the recommended posts based on user interactions.

Parameters:
recommended_posts: DataFrame containing the recommended posts.
df_liked: DataFrame containing liked posts.
df_viewed: DataFrame containing viewed posts.
Returns:
CTR value (float).
7. mean_average_precision(recommended_posts, df_liked, df_viewed, num_recommendations=20)
This function calculates the Mean Average Precision (MAP) for the recommended posts.

Parameters:
recommended_posts: DataFrame containing the recommended posts.
df_liked: DataFrame containing liked posts.
df_viewed: DataFrame containing viewed posts.
num_recommendations: Number of top recommendations to consider (default: 20).
Returns:
MAP value (float).

Evaluation Metrics
1. Click-Through Rate (CTR)
CTR is calculated as the ratio of posts the user interacted with (liked or viewed) among the recommended posts.

2. Mean Average Precision (MAP)
MAP is the mean of precision scores at each rank in the top N recommended posts, providing a measure of how well the top recommendations align with the user’s preferences.

Click-Through Rate (CTR): 0.0500
Mean Average Precision (MAP): 0.0769
