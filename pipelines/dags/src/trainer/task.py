import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import os
from google.cloud import storage

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def train_models(data_path, output_dir):
    # Load data
    df = pd.read_csv(data_path)
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence']
    X = df[features]
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    
    # KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Save models
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pca, f'{output_dir}/pca_model.joblib')
    joblib.dump(kmeans, f'{output_dir}/kmeans_model.joblib')
    
    # Add results to dataframe
    df['cluster'] = clusters
    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]
    
    # Map clusters to moods
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    mood_mapping = assign_moods_to_clusters(cluster_centers)
    df['mood'] = df['cluster'].map(mood_mapping)
    
    df.to_csv(f'{output_dir}/clustered_songs.csv', index=False)

def assign_moods_to_clusters(centers):
    mood_mapping = {}
    for cluster in range(len(centers)):
        center = centers.iloc[cluster]
        if center['valence'] > 0.6 and center['energy'] > 0.6:
            mood = 'Happy'
        elif center['valence'] < 0.4 and center['energy'] < 0.4:
            mood = 'Sad'
        elif center['energy'] > 0.7:
            mood = 'Energetic'
        elif center['valence'] > 0.5:
            mood = 'Relaxed'
        else:
            mood = 'Neutral'
        mood_mapping[cluster] = mood
    return mood_mapping

def main():
    args = parse_args()
    train_models(args.input_data, args.output_dir)

if __name__ == '__main__':
    main()