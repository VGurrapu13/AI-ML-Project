#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Load dataset
data = pd.read_csv("books_data.csv")
data.head()


# In[3]:


data.info()


# In[4]:


# Convert 'average_rating' to a numeric data type
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')


# In[5]:


# Create a new column 'book_content' by combining 'title' and 'authors'
data['book_content'] = data['title'] + ' ' + data['authors']


# In[6]:


data.head()


# In[7]:


# Vectorize text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])


# In[8]:


def recommend_books(user_input, data, tfidf_vectorizer, tfidf_matrix, top_n=5):
    """
    Recommends books based on a user's text description.
    :param user_input: str, user description of preferences.
    :param data: DataFrame, containing 'title' and 'book_content' columns.
    :param tfidf_vectorizer: Trained TF-IDF vectorizer.
    :param tfidf_matrix: Precomputed TF-IDF matrix.
    :param top_n: int, number of recommendations to return.
    :return: List of recommended book titles with similarity scores.
    """
    
    # Validate user input
    if not user_input.strip():
        print("Invalid input. Please enter a description.")
        return []
    
    # Transform user input only
    user_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Compute similarity
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get top N similar books
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Return recommended book titles with similarity scores and metadata
    recommendations = [(data.iloc[i]['title'], data.iloc[i]['authors'], data.iloc[i]['average_rating'], similarity_scores[i]) for i in top_indices]
    return recommendations


# In[9]:


while True:
    # Example user input
    user_input = input("Enter a short description of the type of books you like: ")
    
    # Get recommendations
    recommendations = recommend_books(user_input, data, tfidf_vectorizer, tfidf_matrix)
    
    # Display recommendations
    if recommendations:
        print("\nRecommended Books:")
        for title, author, rating, score in recommendations:
            print(f"{title} by {author} (Rating: {rating}, Similarity: {score:.2f})")
    
    # Ask if the user wants more recommendations
    more_recommendations = input("\nWould you like to enter input for more recommendations? (yes/no): ").strip().lower()
    if more_recommendations != 'yes':
        break
    
# Salary expectation per month (Mandatory)
salary_expectation = input("\nEnter your salary expectation per month: ")
print("Salary Expectation:", salary_expectation)


# In[ ]:




