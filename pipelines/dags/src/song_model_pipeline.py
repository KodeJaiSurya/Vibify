import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def apply_pca(data: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Applies PCA to reduce data dimensions."""

    features = [i for i in data.columns if data[i].dtype != 'object']
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data[features])
    print("PCA applied successfully.")
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

def apply_kmeans(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Applies KMeans clustering to the PCA data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data = apply_pca(df)
    df['cluster'] = kmeans.fit_predict(data)
    visualize_clusters(data, df)
    print(f"KMeans clustering applied with {n_clusters} clusters.")
    print(df['cluster'].value_counts())
    return df

def getCluster_Mood(df: pd.DataFrame) -> pd.DataFrame:
  sns.set(style="whitegrid")
  cluster_means = df.groupby('cluster')[['valence', 'energy']].mean().reset_index()
  colors = sns.color_palette('viridis', 2)
  bar_width = 0.35
  index = cluster_means['cluster']

  plt.figure(figsize=(10, 6))

  plt.bar(index - bar_width/2, cluster_means['valence'], bar_width, color=colors[0], label='Valence')
  plt.bar(index + bar_width/2, cluster_means['energy'], bar_width, color=colors[1], label='Energy')

  for i, valence, energy in zip(index, cluster_means['valence'], cluster_means['energy']):
    plt.text(i - 0.15, valence + 0.01, f'{valence:.2f}', color='black', fontweight='bold')
    plt.text(i + 0.15, energy + 0.01, f'{energy:.2f}', color='black', fontweight='bold')

  plt.xlabel('Cluster')
  plt.ylabel('Average Value')
  plt.title('Average Valence and Energy for Each Cluster')
  plt.xticks(index, cluster_means['cluster'])
  plt.legend()

  plt.grid(axis='y', linestyle='--', alpha=0.7)

  plt.tight_layout()
  plt.show()
  df['mood'] = np.where(df['cluster'] == 0, 'Sad', np.nan)
  df['mood'] = np.where(df['cluster'] == 1, 'Calm', df['mood'])
  df['mood'] = np.where(df['cluster'] == 2, 'Angry', df['mood'])
  df['mood'] = np.where(df['cluster'] == 3, 'Happy', df['mood'])
  return df

