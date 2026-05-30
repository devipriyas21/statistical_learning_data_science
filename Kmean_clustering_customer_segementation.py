# Customer Segmentation using K-Means Clustering 
# 1. Import required libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 

# 2. Load the dataset 
data = pd.read_csv("Mall_Customers.csv") 
# Display first few rows 
print(data.head()) 

# 3. Select features for clustering 
# Using Annual Income and Spending Score 
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 4. Feature scaling 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# 5. Elbow Method to find optimal number of clusters
wcss = [] 
for k in range(1, 11): 
 kmeans = KMeans(n_clusters=k, random_state=42) 
 kmeans.fit(X_scaled) 
 wcss.append(kmeans.inerƟ a_)
# Plot Elbow Curve 
plt.figure(figsize=(8, 5)) 
plt.plot(range(1, 11), wcss, marker='o') 
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters") 
plt.ylabel("WCSS") 
plt.show() 

# 6. Apply K-Means with opƟ mal K (usually K=5 for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42) 
clusters = kmeans.fit_predict(X_scaled) 

# 7. Add cluster labels to original data 
data['Cluster'] = clusters 

# 8. Visualize the clusters 
plt.figure(figsize=(8, 6)) 
for cluster in range(5): 
 plt.scatter(
 data[data['Cluster'] == cluster]['Annual Income (k$)'], 
 data[data['Cluster'] == cluster]['Spending Score (1-100)'], 
 label=f'Cluster {cluster}' 
 ) 
plt.scatter(
 scaler.inverse_transform(kmeans.cluster_centers_)[:, 0], 
 scaler.inverse_transform(kmeans.cluster_centers_)[:, 1], 
 s=200, 
 c='black', 
 marker='X', 
 label='Centroids' 
) 
plt.title("Customer SegmentaƟ on")
plt.xlabel("Annual Income (k$)") 
plt.ylabel("Spending Score (1-100)") 
plt.legend() 
plt.show() 

# 9. Cluster summary 
print("\nCluster-wise Summary:") 
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
