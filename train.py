import os
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors

def load_data():
    """Load and prepare the datasets."""
    script_dir = os.path.dirname(__file__)
    combined_path = os.path.join(script_dir, "data/final_combined_data.csv")
    scaled_path = os.path.join(script_dir, "data/scaled_data.csv")

    combined_df = pd.read_csv(combined_path)
    scaled_df = pd.read_csv(scaled_path)
    X_scaled = scaled_df.values

    return combined_df, X_scaled

def train_and_save_model():
    combined_df, X_scaled = load_data()
    
    # Save metadata and scaled features
    metadata = combined_df[['track_name', 'artist_name', 'genre']].copy()
    os.makedirs("models", exist_ok=True)
    metadata.to_csv("models/metadata.csv", index=False)
    np.save("models/scaled_vectors.npy", X_scaled)

    # Fit KNN model
    knn_model = NearestNeighbors(metric='cosine', algorithm='auto')
    knn_model.fit(X_scaled)
    joblib.dump(knn_model, "models/knn_model.pkl")
    
    print("Training complete. Model, metadata, and scaled vectors saved to 'models/'.")

if __name__ == "__main__":
    train_and_save_model()
