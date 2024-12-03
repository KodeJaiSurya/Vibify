import joblib
from google.cloud import storage
import pandas as pd
import os.path

def save_final(model_outputs, execution_date, bucket_name=BUCKET_NAME):
    """Save final model outputs to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Get model directory from Vertex AI output
    model_dir = model_outputs['outputPaths']['modelDir'].replace(f'gs://{bucket_name}/', '')
    
    # Define blob paths using the execution date
    prefix = f"models/{execution_date}"
    results_path = os.path.join(prefix, 'clustered_songs.csv')
    pca_path = os.path.join(prefix, 'pca_model.joblib')
    kmeans_path = os.path.join(prefix, 'kmeans_model.joblib')

    # Download results from GCS
    results_blob = bucket.blob(results_path)
    pca_blob = bucket.blob(pca_path)
    kmeans_blob = bucket.blob(kmeans_path)

    # Load models and results
    pca_model = joblib.loads(pca_blob.download_as_bytes())
    kmeans_model = joblib.loads(kmeans_blob.download_as_bytes())
    results_df = pd.read_csv(results_blob.download_as_string().decode('utf-8'))

    # Save to local files if needed
    joblib.dump(pca_model, 'pca_model.joblib')
    joblib.dump(kmeans_model, 'kmeans_model.joblib')
    results_df.to_csv('clustered_songs.csv', index=False)

    return {
        'pca_model': pca_model,
        'kmeans_model': kmeans_model,
        'results': results_df
    }