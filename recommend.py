import os
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.neighbors import NearestNeighbors

def load_model():
    """Load trained model and metadata for recommendations."""
    model_path = os.path.join("models", "knn_model.pkl")
    metadata_path = os.path.join("models", "metadata.csv")
    scaled_path = os.path.join("models", "scaled_vectors.npy")

    knn_model = joblib.load(model_path)
    metadata = pd.read_csv(metadata_path)
    scaled_vectors = np.load(scaled_path)

    return knn_model, metadata, scaled_vectors

def rec_songs(song_query, metadata, scaled_vectors, knn_model, n_rec=5):
    """Get song recommendations based on a given song."""
    # Try to find the song by title first
    matches = metadata[metadata['track_name'].str.lower().str.contains(song_query.lower(), na=False)]
    if matches.empty:
        matches = metadata[
            (metadata['track_name'].str.lower().str.contains(song_query.lower(), na=False)) |
            (metadata['artist_name'].str.lower().str.contains(song_query.lower(), na=False))
        ]
    if matches.empty:
        return f"Error: '{song_query}' not found in the dataset"

    song_index = matches.index[0]
    genre = matches.iloc[0]['genre']

    # Filter by genre
    genre_mask = metadata['genre'].str.lower() == genre.lower()
    genre_metadata = metadata[genre_mask].reset_index(drop=True)
    genre_scaled = scaled_vectors[genre_mask]

    # Rebuild KNN on genre
    genre_model = NearestNeighbors(metric='cosine', algorithm='auto')
    genre_model.fit(genre_scaled)

    # Find song's index in genre_filtered data
    filtered_song = genre_metadata[
        (genre_metadata['track_name'].str.lower().str.contains(song_query.lower(), na=False))
    ]
    song_index_in_filtered = filtered_song.index[0]

    distances, indices = genre_model.kneighbors([genre_scaled[song_index_in_filtered]], n_neighbors=n_rec + 1)
    recommendations = genre_metadata.iloc[indices[0][1:]]
    recommendations = recommendations[['track_name', 'artist_name']].copy()
    recommendations['similarity'] = 1 - distances[0][1:]

    return recommendations

def format_recommendations(recommendations):
    """Format recommendations for display."""
    if isinstance(recommendations, str):
        return recommendations
    
    output = "\nRecommended songs:\n"
    for i, rec in recommendations.iterrows():
        similarity_percentage = round(rec['similarity'] * 100, 2)
        output += f"{i+1}. {rec['track_name']} by {rec['artist_name']} (Similarity: {similarity_percentage}%)\n"
    return output

def main():
    parser = argparse.ArgumentParser(description='Get music recommendations.')
    parser.add_argument('song', nargs='*', help='Song title')
    parser.add_argument('--num', '-n', type=int, default=5, help='Number of recommendations (default: 5)')
    
    args = parser.parse_args()
    
    knn_model, metadata, scaled_vectors = load_model()

    if not args.song:
        song_input = input("Enter a song: ").strip()
        if not song_input:
            print("Please enter a song title.")
            return
    else:
        song_input = ' '.join(args.song)

    recommendations = rec_songs(song_input, metadata, scaled_vectors, knn_model, n_rec=args.num)
    print(format_recommendations(recommendations))

if __name__ == "__main__":
    main()
