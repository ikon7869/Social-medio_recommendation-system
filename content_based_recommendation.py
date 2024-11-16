import json
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the user interaction data (liked, viewed, rated posts)
def load_data(viewed_file, liked_file, rated_file, all_post_file):
    """
    Load the JSON data from files and convert it to DataFrames.

    Parameters
    ----------
    viewed_file : str
        Path to the viewed posts data file.
    liked_file : str
        Path to the liked posts data file.
    rated_file : str
        Path to the user rating data file.
    all_post_file : str
        Path to the file containing all posts data.

    Returns
    -------
    tuple
        DataFrames for liked posts, viewed posts, and all posts.
    """
    with open(viewed_file, 'r') as file:
        viewed_data = json.load(file)['posts']

    with open(liked_file, 'r') as file:
        liked_data = json.load(file)['posts']

    all_post_data = json.load(open(all_post_file))['posts']
    all_post_df = pd.DataFrame(all_post_data)

    df_liked = pd.DataFrame(liked_data)
    df_viewed = pd.DataFrame(viewed_data)

    return df_liked, df_viewed, all_post_df

# 2. Preprocess the text data
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    """
    Preprocess text by removing special characters, stopwords, 
    and applying stemming or lemmatization.

    Parameters
    ----------
    text : str
        The input text to preprocess.
    flg_stemm : bool
        Whether to apply stemming.
    flg_lemm : bool
        Whether to apply lemmatization.
    lst_stopwords : list
        List of stopwords to remove.

    Returns
    -------
    str
        The preprocessed text.
    """
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()

    if lst_stopwords:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    if flg_lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    text = " ".join(lst_text)
    return text

# 3. Create a user-post interaction matrix
def create_user_post_matrix(df_liked, df_viewed, username):
    """
    Create a DataFrame with user interactions (liked and viewed posts) 
    filtered by username.

    Parameters
    ----------
    df_liked : pd.DataFrame
        DataFrame of liked posts.
    df_viewed : pd.DataFrame
        DataFrame of viewed posts.
    username : str
        The username to filter interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the user's interactions with combined features.
    """
    df_liked = df_liked[df_liked['username'] == username]
    df_viewed = df_viewed[df_viewed['username'] == username]
    df_liked['interaction_type'] = 'liked'
    df_viewed['interaction_type'] = 'viewed'
    df_viewed = df_viewed[df_viewed['upvoted']]
    df_interaction = pd.concat([df_liked, df_viewed])
    df_interaction['combined_features'] = df_interaction.apply(
        lambda x: x['title'] + " " + x['category']['description'], axis=1
    )
    return df_interaction

# 4. Compute user profile using TF-IDF vectorization
def user_profile(username, df_interaction, vectorizer):
    """
    Compute the user's profile vector by averaging the TF-IDF vectors 
    of posts the user interacted with.

    Parameters
    ----------
    username : str
        The username to compute the profile for.
    df_interaction : pd.DataFrame
        DataFrame of user interactions.
    vectorizer : TfidfVectorizer
        TF-IDF vectorizer.

    Returns
    -------
    csr_matrix
        User profile vector.
    """
    user = df_interaction[df_interaction['username'] == username]
    user_matrix = vectorizer.transform(user['combined_features'])
    user_profile = user_matrix.mean(axis=0)
    return user_profile

# 5. Generate post recommendations for the user
def recommend_posts_pipeline(df_liked, df_viewed, all_post_df, username, vectorizer, utils_preprocess_text, num_recommendations=20):
    """
    Create recommendations based on cosine similarity between 
    user profile and all available posts.

    Parameters
    ----------
    df_liked : pd.DataFrame
        DataFrame of liked posts.
    df_viewed : pd.DataFrame
        DataFrame of viewed posts.
    all_post_df : pd.DataFrame
        DataFrame of all posts.
    username : str
        Username for which recommendations are generated.
    vectorizer : TfidfVectorizer
        TF-IDF vectorizer.
    utils_preprocess_text : function
        Text preprocessing function.
    num_recommendations : int
        Number of top posts to recommend.

    Returns
    -------
    pd.DataFrame
        DataFrame of recommended posts.
    """
    df_interaction = create_user_post_matrix(df_liked, df_viewed, username)
    all_post_df['combined_features'] = all_post_df.apply(
        lambda x: utils_preprocess_text(x['title'] + " " + x['category']['description'], flg_stemm=True, flg_lemm=False), axis=1
    )
    post_matrix = vectorizer.transform(all_post_df['combined_features'])
    user_profile_vector = user_profile(username, df_interaction, vectorizer)
    similarity_scores = cosine_similarity(post_matrix, np.asarray(user_profile_vector)).flatten()
    all_post_df['similarity_score'] = similarity_scores

    # Return top N recommendations
    return all_post_df.sort_values(by='similarity_score', ascending=False).head(num_recommendations)


def calculate_ctr(recommended_posts, df_liked, df_viewed):
    """
    Calculate the Click-Through Rate (CTR) based on user interaction with recommended posts.

    Parameters
    ----------
    recommended_posts : pd.DataFrame
        The DataFrame containing the recommended posts.
    df_liked : pd.DataFrame
        DataFrame of liked posts.
    df_viewed : pd.DataFrame
        DataFrame of viewed posts.

    Returns
    -------
    float
        The click-through rate (CTR).
    """
    interacted_ids = pd.concat([df_liked['id'], df_viewed['id']])
    recommended_ids = recommended_posts['id']

    # Find how many of the recommended posts were interacted with (liked or viewed)
    clicks = len(interacted_ids[interacted_ids.isin(recommended_ids)])
    print(clicks)
    # CTR = (Number of Clicks / Total Recommended Posts)
    ctr = clicks / len(recommended_posts) if len(recommended_posts) > 0 else 0
    return ctr

def mean_average_precision(recommended_posts, df_liked, df_viewed, num_recommendations=20):
    """
    Calculate the Mean Average Precision (MAP) for the recommended posts.

    Parameters
    ----------
    recommended_posts : pd.DataFrame
        The DataFrame containing the recommended posts.
    df_liked : pd.DataFrame
        DataFrame of liked posts.
    df_viewed : pd.DataFrame
        DataFrame of viewed posts.
    num_recommendations : int
        Number of recommendations to consider.

    Returns
    -------
    float
        The mean average precision (MAP) score.
    """
    relevant_posts = pd.concat([df_liked, df_viewed])  # Posts that the user has interacted with
    relevant_ids = set(relevant_posts['id'])
    
    # Only consider the top N recommendations
    recommended_posts = recommended_posts.head(num_recommendations)

    # Calculate precision at each rank
    precision_at_k = []
    num_relevant = 0
    
    for rank, post_id in enumerate(recommended_posts['id'], start=1):
        if post_id in relevant_ids:
            num_relevant += 1
            precision_at_k.append(num_relevant / rank)

    # Calculate the mean average precision (MAP)
    map_score = np.mean(precision_at_k) if precision_at_k else 0
    return map_score

# 6. Main function to execute the recommendation pipeline
def main(viewed_file, liked_file, rated_file, all_post_file, username, num_recommendations=20):
    """
    Main function to load data, create user-post matrix, and generate recommendations.

    Parameters
    ----------
    viewed_file, liked_file, rated_file, all_post_file : str
        Paths to the JSON data files.
    username : str
        The username for which recommendations are generated.
    num_recommendations : int
        Number of posts to recommend.

    Returns
    -------
    pd.DataFrame
        DataFrame of recommended posts.
    """
    df_liked, df_viewed, all_post_df = load_data(viewed_file, liked_file, rated_file, all_post_file)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    all_post_df['combined_features'] = all_post_df.apply(
        lambda x: x['title'] + " " + x['category']['description'], axis=1
    )
    vectorizer.fit(all_post_df['combined_features'])

    # Generate recommendations
    recommended_posts = recommend_posts_pipeline(
        df_liked, df_viewed, all_post_df, username, vectorizer, utils_preprocess_text, num_recommendations
    )

        # Calculate metrics
    ctr = calculate_ctr(recommended_posts, df_liked, df_viewed)
    map_score = mean_average_precision(recommended_posts, df_liked, df_viewed, num_recommendations)
    
    print("Recommended Posts:\n", recommended_posts)
    print(f"Click-Through Rate (CTR): {ctr:.4f}")
    print(f"Mean Average Precision (MAP): {map_score:.4f}")

# Execute the main function
if __name__ == "__main__":
    main('viewed_post_data.json', 'liked_post_data.json', 'user_rating_data.json', 'all_post_data.json', 'kinha', 20)
