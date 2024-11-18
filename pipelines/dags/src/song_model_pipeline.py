import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
 
def apply_pca(data: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Applies PCA to reduce data dimensions."""
 
    features = [i for i in data.columns if data[i].dtype != 'object']
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data[features])
    logging.info("PCA applied successfully. Explained variance ratio: %s", pca.explained_variance_ratio_)
    return principal_components
 
def visualize_clusters(pca_df: pd.DataFrame, df: pd.DataFrame) -> None:
    """Visualizes clusters in a 2D PCA plot."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_df[:,0], y=pca_df[:,1], hue=df['cluster'], palette='viridis', s=50)
    plt.title("KMeans Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()
    logging.info("Cluster visualization displayed successfully")
 
def apply_kmeans(df: pd.DataFrame, principal_components, n_clusters: int = 4) -> pd.DataFrame:
    """Applies KMeans clustering to the PCA data."""
    logging.info("Starting KMeans with %d clusters", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(principal_components)
    #visualize_clusters(data, df)
    logging.info("KMeans clustering applied with %d clusters", n_clusters)
    logging.info("Cluster distribution: %s", df['cluster'].value_counts())
    return df
 
def plotMoodBarChart(mood_mapping, cluster_means):
    sns.set(style="whitegrid")
    colors = sns.color_palette('viridis', 2)
    bar_width = 0.35
    index = cluster_means['cluster']
    plt.figure(figsize=(10, 6))
 
    plt.bar(index - bar_width/2, cluster_means['valence'], bar_width, color=colors[0], label='Valence')
    plt.bar(index + bar_width/2, cluster_means['energy'], bar_width, color=colors[1], label='Energy')
 
    for i, valence, energy in zip(index, cluster_means['valence'], cluster_means['energy']):
        plt.text(i - 0.15, valence + 0.01, f'{valence:.2f}', color='black', fontweight='bold')
        plt.text(i + 0.15, energy + 0.01, f'{energy:.2f}', color='black', fontweight='bold')
 
    plt.xlabel('Emotion')
    plt.ylabel('Average Value')
    plt.title('Average Valence and Energy for Each Emotion')
    emotion_labels = [mood_mapping[cluster] for cluster in index]
    plt.xticks(index, emotion_labels)
 
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    logging.info("Mood bar chart plotted successfully")
 
 
def assign_mood(df: pd.DataFrame) -> pd.DataFrame:
    cluster_means = df.groupby('cluster')[['valence', 'energy']].mean().reset_index()
    cluster_means = cluster_means.sort_values('valence', ascending=False)
    moods = []
    for i, row in cluster_means.iterrows():
        if i < 2:
            if row['energy'] > cluster_means['energy'].median():
                moods.append('Happy')
            else:
                moods.append('Sad')
        else:
            if row['energy'] > cluster_means['energy'].median():
                moods.append('Angry')
            else:
                moods.append('Calm')
 
    mood_mapping = dict(zip(cluster_means['cluster'], moods))
    df['mood'] = df['cluster'].map(mood_mapping)
 
    #plotMoodBarChart(mood_mapping, cluster_means)
    print(df['mood'].value_counts())
    logging.info("Mood assignment completed with distribution: %s", df['mood'].value_counts())
    return df
 
def save_final(df: pd.DataFrame, output_dir: str = "dags/data/final") -> None:
    """Saves the preprocessed dataframe"""
    logging.info("Saving final song data with cluster to local directory")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "clusters.csv")
    df.to_csv(output_path, index=False)
    
    del df
    logging.info(f"Final song data with  saved to {output_path}")
