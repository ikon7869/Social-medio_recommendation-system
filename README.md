# Social-medio_recommendation-system

# Overview

This project provides a recommendation system based on user interactions with posts (liked, viewed, rated). The system recommends the top N posts for a specific user based on their past interactions (likes and views) and computes two key performance metrics: Click-Through Rate (CTR) and Mean Average Precision (MAP). The recommendation is made using TF-IDF vectorization and cosine similarity to match the user's profile with the available posts.

# Features

Data Loading: Loads user interaction data (liked, viewed, rated posts) and post data.

Text Preprocessing: Preprocesses the text data (removes special characters, stopwords, applies stemming and lemmatization).

User Profile Creation: Builds a user profile based on the posts they have liked or viewed.

Post Recommendation: Recommends posts based on the cosine similarity between the user's profile and all available posts.

Evaluation Metrics: Calculates CTR and MAP to evaluate the quality of recommendations.


Evaluation Metrics
1. Click-Through Rate (CTR)
CTR is calculated as the ratio of posts the user interacted with (liked or viewed) among the recommended posts.

2. Mean Average Precision (MAP)
MAP is the mean of precision scores at each rank in the top N recommended posts, providing a measure of how well the top recommendations align with the user’s preferences.

Click-Through Rate (CTR): 0.0500
Mean Average Precision (MAP): 0.0769
