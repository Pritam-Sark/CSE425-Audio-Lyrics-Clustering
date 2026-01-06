import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusterPipeline:
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()

    def run_kmeans(self, features):
        """Runs K-Means clustering."""
        features_scaled = self.scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        return labels

    def run_dbscan(self, features, eps=3.0, min_samples=5):
        """Runs DBSCAN clustering (for density)."""
        features_scaled = self.scaler.fit_transform(features)
        # PCA is often needed for DBSCAN to work well in high dimensions
        pca = PCA(n_components=10)
        feat_pca = pca.fit_transform(features_scaled)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(feat_pca)
        return labels

    def fuse_features(self, audio_features, lyrics_features):
        """Concatenates Audio and Lyrics features."""
        # Normalize separately before fusing
        aud_norm = self.scaler.fit_transform(audio_features)
        lyr_norm = self.scaler.fit_transform(lyrics_features)
        return np.hstack([aud_norm, lyr_norm])