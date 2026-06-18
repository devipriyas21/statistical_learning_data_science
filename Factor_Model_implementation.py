# Experiment: Factor Model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Generate factor model data
np.random.seed(42)
n = 120 # samples
p = 300 # variables (high-dimensional)
k = 3 # latent factors
F = np.random.normal(0, 1, (n, k)) # latent factors
L = np.random.normal(0, 1, (p, k)) # loadings
E = np.random.normal(0, 0.5, (n, p)) # noise
X = F @ L.T + E # observed data

# Step 2: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Estimate factors using PCA
pca_factor = PCA(n_components=k)
F_hat = pca_factor.fit_transform(X_scaled)
L_hat = pca_factor.components_.T
print("Estimated factor shape:", F_hat.shape)
print("Estimated loading shape:", L_hat.shape)
Factor Scores Visualization
plt.figure(figsize=(6, 4))
plt.scatter(F_hat[:, 0], F_hat[:, 1], alpha=0.6)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Estimated Latent Factors")
plt.show() 
