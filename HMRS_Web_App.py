import streamlit as st
import pickle
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved models
vectormodel = pickle.load(open("MRS_Vectorizer.sav", 'rb'))
#CosineSimilarity = pickle.load(open("MRS_CosineSimilarity.sav", "rb"))

data = pd.read_csv("D:/Project/Movie Recommendation System/movies.csv")

def movie_recommendation(movie_name):
    # Selecting relevant features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Replacing the null values with empty strings
    for feature in selected_features:
        data[feature] = data[feature].fillna('')

    # Combining all the selected features into a single string
    combined_features = data['genres'] + ' ' + data['keywords'] + ' ' + data['tagline'] + ' ' + data['cast'] + ' ' + data['director']

    # Converting text data into feature vectors
    feature_vectors = vectormodel.transform(combined_features)

    # Find the closest match to the movie name given by the user
    list_of_all_titles = data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = data[data.title == close_match].index.values[0]

        # Compute similarity scores
        similarity_scores = cosine_similarity(feature_vectors[index_of_the_movie], feature_vectors)

        # Sorting the movies based on their similarity scores
        similarity_score = list(enumerate(similarity_scores[0]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        # Prepare the suggestions
        suggestions = []
        i = 1
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = data.iloc[index]['title']
            if i < 30:
                suggestions.append(f"{i}. {title_from_index}")
                i += 1
        
        return '\n'.join(suggestions)
    else:
        return "Movie not found in the database. Please try again with a different movie name."

def main():
    # Add custom CSS for YouTube video background and delayed audio playback
    st.markdown("""
        <style>
        body {
            margin: 0;
            padding: 0;
            background: #000;
            color: #fff;
            overflow-x: hidden;
        }
        #video-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }
        .stApp {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 20px;
            z-index: 1;
        }
        h1 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 48px;
            text-align: center;
            color: #1DB954;
        }
        .stTextInput input {
            background-color: #333;
            color: #FFF;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #1DB954;
            color: #FFF;
            border-radius: 10px;
            height: 45px;
            width: 200px;
            font-size: 18px;
            margin-top: 10px;
        }
        .stButton button:hover {
            background-color: #1ED760;
        }
        .stSuccess {
            background-color: #1C1C1C;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
        <div id="video-background">
            <iframe width="100%" height="100%" src="https://www.youtube.com/embed/cvfVy_I5PrQ?autoplay=1&mute=1&loop=1&playlist=cvfVy_I5PrQ" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
        </div>
        <audio id="background-audio" loop>
            <source src="https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <script>
            setTimeout(function() {
                var audio = document.getElementById("background-audio");
                audio.play().catch(function(error) {
                    console.log('Audio playback was prevented by the browser:', error);
                });
            }, 5000); // Delay playback by 5 seconds
        </script>
        """, unsafe_allow_html=True)

    # Title with a fancy font and color
    st.title('🎬 Movie Recommendation System 🍿')
    
    # Input from the user
    movie_name = st.text_input("Enter the name of your favourite movie for suggestion:")

    # Button for prediction
    if st.button('Get Suggestions'):
        if movie_name:
            suggestions = movie_recommendation(movie_name)
            st.success(suggestions)
        else:
            st.error("Please enter a movie name.")

if __name__ == '__main__':
    main()
