import os
from google.cloud import storage
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models from GCS
def load_models(bucket_name, model_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    pca_blob = bucket.blob(f"{model_path}/pca_model.joblib")
    kmeans_blob = bucket.blob(f"{model_path}/kmeans_model.joblib")
    
    pca_model = joblib.loads(pca_blob.download_as_bytes())
    kmeans_model = joblib.loads(kmeans_blob.download_as_bytes())
    
    return pca_model, kmeans_model

# Initialize models
BUCKET_NAME = "us-east1-vibeoncloud-89c595e0-bucket"
MODEL_PATH = "models/latest"
pca_model, kmeans_model = load_models(BUCKET_NAME, MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = ['danceability', 'energy', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence']
    
    df = pd.DataFrame([data], columns=features)
    pca_result = pca_model.transform(df)
    cluster = kmeans_model.predict(df)[0]
    
    # Map cluster to mood (same mapping as training)
    mood_mapping = {
        0: 'Happy',
        1: 'Sad',
        2: 'Energetic',
        3: 'Relaxed',
        4: 'Neutral'
    }
    
    mood = mood_mapping[cluster]
    
    return jsonify({
        'cluster': int(cluster),
        'mood': mood,
        'pca_coordinates': pca_result.tolist()[0]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))