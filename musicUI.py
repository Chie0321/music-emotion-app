import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytube import Search
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Global favorite list (in memory)
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

def load_data():
    file_path = r"C:\Users\87688\Desktop\music毕业设计\musicdata.csv"
    data = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")
    return data

def compute_arousal(data, alpha=1, beta=1, gamma=1):
    if not all(col in data.columns for col in ["energy", "tempo", "danceability"]):
        raise ValueError("Missing required columns.")
    max_tempo = data['tempo'].max()
    data['arousal'] = (
        alpha * data['energy'] +
        beta * (data['tempo'] / max_tempo) +
        gamma * data['danceability']
    )
    return data

def classify_emotions(data):
    scaler = MinMaxScaler()
    data[['valence', 'arousal']] = scaler.fit_transform(data[['valence', 'arousal']])
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['emotion_cluster'] = kmeans.fit_predict(data[['valence', 'arousal']])
    centers = kmeans.cluster_centers_

    cluster_map = {
        0: "Angry",
        1: "Calm",
        2: "Sad",
        3: "Happy"
    }
    data['emotion'] = data['emotion_cluster'].map(cluster_map)
    return data, centers

def recommend_similar_songs(data, song_name, num_recommendations=5):
    target_song = data[data['name'] == song_name]
    if target_song.empty:
        return pd.DataFrame()
    target_features = target_song[['valence', 'arousal']].values
    features = data[['valence', 'arousal']]
    similarity_scores = cosine_similarity(features, target_features)
    data['similarity'] = similarity_scores
    recommendations = data.sort_values('similarity', ascending=False).head(num_recommendations)
    return recommendations[['name', 'valence', 'arousal', 'emotion']]

def search_youtube(song_name):
    s = Search(song_name)
    if s.results:
        return s.results[0].watch_url
    return None

def display_song(song_name, emotion):
    youtube_url = search_youtube(song_name)
    st.markdown(f"🎵 [{song_name}]({youtube_url}) - Emotion: {emotion}")
    

def main():
    st.title("🎵 Emotion-Aware Music Recommendation System")
    st.markdown("This system recommends music based on emotional classification using unsupervised learning.")

    try:
        data = load_data()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        return

    try:
        data = compute_arousal(data)
        st.success("Arousal feature created.")
    except ValueError as e:
        st.error(str(e))
        return

    data, cluster_centers = classify_emotions(data)
    st.success("Emotion clustering completed.")

    st.markdown("### 📊 Emotion Distribution")
    st.dataframe(data['emotion'].value_counts().reset_index().rename(columns={'index': 'Emotion', 'emotion': 'Count'}))

    st.markdown("---")
    st.markdown("### 🎭 Emotion-Based Recommendation")
    emotion = st.selectbox("Choose your current emotion:", ['Happy', 'Angry', 'Calm', 'Sad'])
    if st.button("🎶 Recommend Songs"):
        selected_songs = data[data['emotion'] == emotion]
        for _, row in selected_songs.sample(min(5, len(selected_songs))).iterrows():
            display_song(row['name'], row['emotion'])

    st.markdown("---")
    st.markdown("### 🔍 Similar Songs Search")
    target_song = st.text_input("Enter a song title to find similar tracks:")
    if st.button("🔁 Find Similar Songs"):
        similar = recommend_similar_songs(data, target_song)
        if not similar.empty:
            for _, row in similar.iterrows():
                display_song(row['name'], row['emotion'])
        else:
            st.warning("No similar songs found.")

    
    st.markdown("### 📂 All Clustered Songs Table")
    if st.checkbox("Show Full Emotion Classification Table"):
        st.dataframe(data[['name', 'valence', 'arousal', 'emotion']])

if __name__ == "__main__":
    main()


#streamlit run C:\Users\87688\Desktop\music毕业设计\musicUI.py