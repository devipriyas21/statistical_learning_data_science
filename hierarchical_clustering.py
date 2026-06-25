EXPERIMENT: Hierarchical Clustering

# Step 1: Import Required Libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# Step 2: Generate Sample Dataset
np.random.seed(42)
X, y_true = make_blobs(
 n_samples=150,
 centers=3,
 cluster_std=1.2,
 random_state=42
)
print("Dataset shape:", X.shape)

# Step 3: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Hierarchical Clustering – Linkage Matrix
Z = linkage(X_scaled, method='ward')

# Step 5: Plot Dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Step 6: Agglomerative Clustering (Cut Dendrogram)
hc = AgglomerativeClustering(
 n_clusters=3,
 linkage='ward'
)
cluster_labels = hc.fit_predict(X_scaled)
print("Cluster labels assigned:", np.unique(cluster_labels))

# Step 7: Visualize Final Clusters
plt.figure(figsize=(7, 5))
plt.scatter(
 X_scaled[:, 0],
 X_scaled[:, 1],
 c=cluster_labels,
 cmap='viridis',
 s=50
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Hierarchical Clustering Results")
plt.tight_layout()
plt.show()
