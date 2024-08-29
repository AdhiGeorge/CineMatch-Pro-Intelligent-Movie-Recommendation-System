# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 23:28:20 2024

@author: Hp
"""
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the data
data = pd.read_csv('D:/Project/Movie Recommendation System/movies.csv')

# Selecting relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replacing the null values with empty strings
for i in selected_features:
    data[i] = data[i].fillna('')

# Combining all the selected features into a single string
combined_features = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data['director']

# Converting text data into feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Getting the similarity score using cosine similarity
similarity = cosine_similarity(feature_vectors)

# Creating a list with all the movie names given in the dataset
list_of_all_titles = data['title'].tolist()

while True:
    movie_name = input("Enter your favourite movie name as reference for suggestion: ")
    
    # Finding the closest match to the movie name given by the user
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        print(f"Did you mean '{close_match}'?")
        
        # Finding the index of the movie with the title
        index_of_the_movie = data[data.title == close_match]['index'].values[0]
        
        # Getting the list of similar movies
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        
        # Sorting the movies based on their similarity scores
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        # Print the names of similar movies based on the index
        print('Suggested movies: \n')
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = data[data.index == index]['title'].values[0]
            if i < 30:
                print(f"{i}. {title_from_index}")
                i += 1
        break
    else:
        print("Movie not found in the database. Please try again with a different movie name.")

import pickle
vectorname='MRS_Vectorizer.sav'
#modelname='MRS_CosineSimilarity.sav'

pickle.dump(vectorizer,open(vectorname,'wb'))
#pickle.dump(similarity,open(modelname,'wb'))