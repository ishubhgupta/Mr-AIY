"""
Clustering Module for AI Vision Suite
Provides various clustering algorithms for unsupervised learning
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
import os
from datetime import datetime

class ClusteringTrainer:
    """Trainer class for clustering algorithms"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.algorithm = None
        self.n_clusters = 3
        self.labels_ = None
        self.silhouette_score_ = None
        
    def set_algorithm(self, algorithm, **kwargs):
        """Set the clustering algorithm"""
        self.algorithm = algorithm
        
        if algorithm == "K-Means":
            self.model = KMeans(
                n_clusters=kwargs.get('n_clusters', 3),
                max_iter=kwargs.get('max_iter', 300),
                random_state=kwargs.get('random_state', 42)
            )
            self.n_clusters = kwargs.get('n_clusters', 3)
            
        elif algorithm == "DBSCAN":
            self.model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
            
        elif algorithm == "Agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=kwargs.get('n_clusters', 3),
                linkage=kwargs.get('linkage', 'ward')
            )
            self.n_clusters = kwargs.get('n_clusters', 3)
            
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def train(self, X, normalize=True):
        """Train the clustering model"""
        if self.model is None:
            raise ValueError("No algorithm set. Call set_algorithm() first.")
            
        # Preprocess data
        if normalize:
            X_processed = self.scaler.fit_transform(X)
        else:
            X_processed = X
            
        # Fit the model
        self.labels_ = self.model.fit_predict(X_processed)
        
        # Calculate silhouette score if we have more than one cluster
        if len(np.unique(self.labels_)) > 1:
            self.silhouette_score_ = silhouette_score(X_processed, self.labels_)
        else:
            self.silhouette_score_ = 0.0
            
        return self.labels_
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'algorithm': self.algorithm,
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score_
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def get_metrics(self):
        """Get training metrics"""
        return {
            'silhouette_score': self.silhouette_score_,
            'n_clusters': len(np.unique(self.labels_)) if self.labels_ is not None else 0,
            'algorithm': self.algorithm
        }

class ClusteringPredictor:
    """Predictor class for clustering models"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.algorithm = None
        self.n_clusters = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.algorithm = model_data['algorithm']
        self.n_clusters = model_data['n_clusters']
    
    def predict(self, X):
        """Predict cluster assignments for new data"""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
            
        # Preprocess data
        X_processed = self.scaler.transform(X)
        
        # For algorithms that support prediction
        if hasattr(self.model, 'predict'):
            return self.model.predict(X_processed)
        else:
            # For algorithms like DBSCAN, we need to refit
            return self.model.fit_predict(X_processed)
