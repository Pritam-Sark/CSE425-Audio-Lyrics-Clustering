import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def purity_score(y_true, y_pred):
    """Computes cluster purity."""
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_clustering(features, labels_pred, labels_true_str):
    """
    Calculates Silhouette, ARI, NMI, and Purity.
    """
    # Convert string labels (genres) to integers
    le = LabelEncoder()
    labels_true = le.fit_transform(labels_true_str)
    
    # Calculate Metrics
    sil = silhouette_score(features, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    pur = purity_score(labels_true, labels_pred)
    
    results = {
        "Silhouette": round(sil, 4),
        "ARI": round(ari, 4),
        "NMI": round(nmi, 4),
        "Purity": round(pur, 4)
    }
    return results